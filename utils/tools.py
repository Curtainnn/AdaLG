import os
import numpy as np
import torch
import matplotlib.pyplot as plt
import pandas as pd
import torch.distributed as dist
plt.switch_backend('agg')


class ForecastCheckpointError(RuntimeError):
    """A checkpoint cannot restore the requested prediction architecture."""


def load_forecast_checkpoint(model, checkpoint_path, allow_missing_frozen=False):
    target = model.module if hasattr(model, 'module') else model
    checkpoint = _extract_state_dict(torch.load(checkpoint_path, map_location='cpu'))
    expected = target.state_dict()
    if allow_missing_frozen:
        for name, parameter in target.named_parameters():
            if not parameter.requires_grad and name not in checkpoint:
                checkpoint[name] = expected[name]
    missing = sorted(set(expected) - set(checkpoint))
    unexpected = sorted(set(checkpoint) - set(expected))
    mismatches = [f'{key}: saved={tuple(checkpoint[key].shape)} expected={tuple(expected[key].shape)}'
                  for key in sorted(set(expected) & set(checkpoint))
                  if checkpoint[key].shape != expected[key].shape]
    if missing or unexpected or mismatches:
        details = []
        if missing:
            details.append('missing=' + ', '.join(missing))
        if unexpected:
            details.append('unexpected=' + ', '.join(unexpected))
        if mismatches:
            details.append('shape_mismatch=' + '; '.join(mismatches))
        raise ForecastCheckpointError(
            f'Incompatible checkpoint {checkpoint_path}: ' + ' | '.join(details)
            + '. Missing trained branch weights cannot be reconstructed; this run must be retrained.'
        )
    target.load_state_dict(checkpoint, strict=True)


def _extract_state_dict(checkpoint):
    if not isinstance(checkpoint, dict):
        raise TypeError(f'Expected checkpoint to be a dict, got {type(checkpoint)}.')

    state_dict = checkpoint
    for key in ('state_dict', 'model_state_dict', 'model', 'net'):
        value = checkpoint.get(key)
        if isinstance(value, dict):
            state_dict = value
            break

    normalized_state_dict = {}
    for key, value in state_dict.items():
        if not torch.is_tensor(value):
            continue
        normalized_key = key[7:] if key.startswith('module.') else key
        normalized_state_dict[normalized_key] = value

    if not normalized_state_dict:
        raise ValueError('Checkpoint does not contain a usable tensor state_dict.')
    return normalized_state_dict


def load_pretrained_weights(model, checkpoint_path, load_mode='full', allowed_prefixes=None, verbose=True):
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    source_state_dict = _extract_state_dict(checkpoint)
    target_state_dict = model.state_dict()

    if load_mode == 'full':
        model.load_state_dict(source_state_dict, strict=True)
        if verbose:
            print(f"Loaded pretrained weights from {checkpoint_path} with mode=full ({len(source_state_dict)} tensors).")
        return {
            'loaded_keys': list(source_state_dict.keys()),
            'missing_keys': [],
            'unexpected_keys': [],
            'shape_mismatch_keys': [],
            'prefix_filtered_keys': [],
        }

    prefix_filtered_keys = []
    missing_keys = []
    shape_mismatch_keys = []
    loaded_state_dict = {}

    for key, value in source_state_dict.items():
        if allowed_prefixes is not None and not any(key.startswith(prefix) for prefix in allowed_prefixes):
            prefix_filtered_keys.append(key)
            continue
        if key not in target_state_dict:
            missing_keys.append(key)
            continue
        if target_state_dict[key].shape != value.shape:
            shape_mismatch_keys.append(
                (key, tuple(value.shape), tuple(target_state_dict[key].shape))
            )
            continue
        loaded_state_dict[key] = value

    if not loaded_state_dict:
        prefix_desc = '' if allowed_prefixes is None else f' and prefixes {tuple(allowed_prefixes)}'
        raise ValueError(
            f'No compatible pretrained weights found in {checkpoint_path} for mode={load_mode}{prefix_desc}.'
        )

    target_state_dict.update(loaded_state_dict)
    model.load_state_dict(target_state_dict, strict=True)

    if verbose:
        print(
            f"Loaded pretrained weights from {checkpoint_path} with mode={load_mode}: "
            f"loaded={len(loaded_state_dict)}, "
            f"prefix_filtered={len(prefix_filtered_keys)}, "
            f"missing={len(missing_keys)}, "
            f"shape_mismatch={len(shape_mismatch_keys)}."
        )
        if missing_keys:
            print(f"Skipped missing keys, first 5: {missing_keys[:5]}")
        if prefix_filtered_keys:
            print(f"Skipped filtered keys, first 5: {prefix_filtered_keys[:5]}")
        if shape_mismatch_keys:
            preview = [f'{key}: {src_shape} -> {dst_shape}' for key, src_shape, dst_shape in shape_mismatch_keys[:5]]
            print(f"Skipped shape mismatch keys, first 5: {preview}")

    return {
        'loaded_keys': list(loaded_state_dict.keys()),
        'missing_keys': missing_keys,
        'unexpected_keys': [],
        'shape_mismatch_keys': shape_mismatch_keys,
        'prefix_filtered_keys': prefix_filtered_keys,
    }


def adjust_learning_rate(optimizer, epoch, args):
    if args.lradj == 'type1':
        lr_adjust = {epoch: args.learning_rate * (0.1 ** epoch)}
    elif args.lradj == 'type2':
        lr_adjust = {epoch: args.learning_rate * (0.5 ** epoch)}
    elif args.lradj == 'type3':
        lr_adjust = {epoch: args.learning_rate * (0.9 ** epoch)}


    if epoch in lr_adjust.keys():
        lr = lr_adjust[epoch]
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        print('Updating learning rate to {}'.format(lr))


class EarlyStopping:
    def __init__(self, args, verbose=False, delta=0):
        self.patience = args.patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.inf
        self.delta = delta
        self.dp = args.dp
        self.ddp = args.ddp
        self.save_full_state = (str(getattr(args, 'model', '')).lower() == 'adalg'
                                or bool(getattr(args, 'point_gate_init_checkpoint', '')))
        if self.ddp:
            self.local_rank = args.local_rank
        else:
            self.local_rank = None

    def __call__(self, val_loss, model, path):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            if self.verbose:
                print(
                    f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).')
            self.val_loss_min = val_loss
            if self.ddp:
                if self.local_rank == 0:
                    self.save_checkpoint(val_loss, model, path)
                dist.barrier()
            else:
                self.save_checkpoint(val_loss, model, path)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(
                f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            if self.ddp:
                if self.local_rank == 0:
                    self.save_checkpoint(val_loss, model, path)
                dist.barrier()
            else:
                self.save_checkpoint(val_loss, model, path)
            if self.verbose:
                print(
                    f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).')
            self.val_loss_min = val_loss
            self.counter = 0

    def save_checkpoint(self, val_loss, model, path):
        if self.dp:
            model = model.module
        state_dict = model.state_dict()
        if not self.save_full_state:
            param_grad_dic = {
                k: v.requires_grad for (k, v) in model.named_parameters()
            }
            for k in list(state_dict.keys()):
                if k in param_grad_dic.keys() and not param_grad_dic[k]:
                    # delete parameters that do not require gradient
                    del state_dict[k]
        torch.save(state_dict, path + '/' + f'checkpoint.pth')

def visual(true, preds=None, name='./pic/test.pdf'):
    """
    Results visualization
    """
    plt.figure()
    plt.plot(true, label='GroundTruth', linewidth=2)
    if preds is not None:
        plt.plot(preds, label='Prediction', linewidth=2)
    plt.legend()
    plt.savefig(name, bbox_inches='tight')


def save_overfit_diagnostics(history, folder_path):
    if not history:
        return

    os.makedirs(folder_path, exist_ok=True)
    history_frame = pd.DataFrame(history)
    png_path = os.path.join(folder_path, 'loss_curve.png')

    epochs = history_frame['epoch'].to_numpy()
    best_row_idx = int(history_frame['val_loss'].idxmin())
    best_epoch = int(history_frame.loc[best_row_idx, 'epoch'])
    best_val_loss = float(history_frame.loc[best_row_idx, 'val_loss'])

    fig, axes = plt.subplots(2, 1, figsize=(9, 7), sharex=True)

    axes[0].plot(epochs, history_frame['train_loss'], marker='o', linewidth=1.8, label='Train')
    axes[0].plot(epochs, history_frame['val_loss'], marker='o', linewidth=1.8, label='Val')
    axes[0].plot(epochs, history_frame['test_loss'], marker='o', linewidth=1.8, label='Test')
    axes[0].axvline(best_epoch, linestyle='--', color='tab:red', linewidth=1.2, label=f'Best val epoch {best_epoch}')
    axes[0].scatter([best_epoch], [best_val_loss], color='tab:red', zorder=5)
    axes[0].set_ylabel('Selected loss / metric')
    axes[0].set_title('Overfitting diagnostics')
    axes[0].grid(True, linestyle='--', alpha=0.3)
    axes[0].legend()

    val_gap = history_frame['val_loss'] - history_frame['train_loss']
    test_gap = history_frame['test_loss'] - history_frame['train_loss']
    axes[1].plot(epochs, val_gap, marker='o', linewidth=1.8, label='Val - Train')
    axes[1].plot(epochs, test_gap, marker='o', linewidth=1.8, label='Test - Train')
    axes[1].axhline(0.0, linestyle='--', color='black', linewidth=1.0)
    axes[1].axvline(best_epoch, linestyle='--', color='tab:red', linewidth=1.2)
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Generalization gap')
    axes[1].grid(True, linestyle='--', alpha=0.3)
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(png_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
