"""Full, keyed query exports and split provenance for the raw-concat study."""
import hashlib
import json
from pathlib import Path
import numpy as np
import torch


def model_info(model):
    model = model.module if hasattr(model, 'module') else model
    digest = hashlib.sha256()
    encoder_elements = 0
    for key, value in sorted(model.state_dict().items()):
        if key.startswith('query_head.'):
            continue
        digest.update(key.encode())
        digest.update(value.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
        encoder_elements += value.numel()
    return dict(total_parameters=sum(p.numel() for p in model.parameters()),
                query_head_parameters=sum(p.numel() for p in model.query_head.parameters()),
                encoder_initial_sha256=digest.hexdigest(), encoder_elements=encoder_elements,
                matching=getattr(model.query_head, 'parameter_match', None),
                history_boundary_policy=getattr(model, 'history_boundary_policy', 'unspecified'))


def dataset_provenance(dataset):
    indices = np.asarray(dataset.index_map, dtype='<i8')
    groups = [str(dataset.sample_payloads[int(i)].get('group_id',
                  dataset.sample_payloads[int(i)].get('sample_id', int(i)))) for i in indices]
    scaler = np.asarray(dataset.scaler_mean, dtype='<f4').tobytes() + np.asarray(dataset.scaler_std, dtype='<f4').tobytes()
    return dict(samples=len(indices), index_sha256=hashlib.sha256(indices.tobytes()).hexdigest(),
                groups_sha256=hashlib.sha256(json.dumps(groups).encode()).hexdigest(),
                group_hashes=sorted({hashlib.sha256(group.encode()).hexdigest() for group in groups}),
                scaler_sha256=hashlib.sha256(scaler).hexdigest(), split_mode=dataset.point_split_mode,
                split_seed=int(dataset.SPLIT_SEED), seq_len=int(dataset.seq_len),
                num_vars=int(dataset.num_vars))


def assert_strict_forecast(batch):
    """A complete-history encoder is valid only if its input precedes every query."""
    valid_history = batch['x_mask'].bool().any(-1)
    valid_target = batch['y_mask'].bool().any(-1)
    end = batch['x_mark'][..., 0].masked_fill(~valid_history, -float('inf')).amax(1)
    start = batch['y_mark'][..., 0].masked_fill(~valid_target, float('inf')).amin(1)
    if torch.any(end >= start):
        raise ValueError('Raw-recent comparison rejected: Global history is not strictly before every valid query.')


def query_batch(batch, pred, target, mask, aux):
    mask = mask.detach().cpu().bool()
    idx = mask.nonzero(as_tuple=False)
    if len(idx) == 0:
        return None
    if pred.ndim != 3 or 'sample_index' not in batch:
        raise ValueError('Raw-recent export requires keyed dense point-forecast batches.')
    def cpu(x):
        return x.detach().cpu()
    def selected(x):
        return cpu(x)[mask].numpy()
    sample = cpu(batch['sample_index'])[idx[:, 0]].numpy()
    if (sample < 0).any():
        raise ValueError('Missing stable sample IDs; sync data_provider/data_loader.py.')
    result = dict(sample_index=sample, query_row=idx[:, 1].numpy(), variable=idx[:, 2].numpy(),
                  query_time=cpu(batch['y_mark'])[idx[:, 0], idx[:, 1], 0].numpy(),
                  true=selected(target), prediction=selected(pred))
    for key in ('global_pred', 'gate', 'recent_count', 'recent_gap', 'local_dynamics'):
        if key not in aux:
            raise ValueError(f'Missing query diagnostic {key}; sync model and training code.')
        result[key] = selected(aux[key])
    return result


def save_queries(chunks, dataset, folder, args, training_manifest):
    chunks = [c for c in chunks if c is not None]
    if not chunks:
        raise ValueError('No valid test queries to export.')
    values = {key: np.concatenate([c[key] for c in chunks]) for key in chunks[0]}
    for key in ('true', 'prediction', 'global_pred'):
        values[key + '_original'] = dataset.inverse_transform_point_values(values[key], values['variable'])
    order = np.lexsort((values['variable'], values['query_row'], values['sample_index']))
    values = {key: value[order] for key, value in values.items()}
    keys = np.stack([values[k] for k in ('sample_index', 'query_row', 'variable')], -1)
    if len(np.unique(keys, axis=0)) != len(keys):
        raise ValueError('Duplicate test query keys.')
    folder = Path(folder)
    np.savez_compressed(folder / 'raw_recent_queries.npz', **values)
    manifest = dict(variant=args.point_prediction_variant, seed=int(args.seed),
                    test=dataset_provenance(dataset), training=training_manifest,
                    query_count=len(keys), export_truncated=False,
                    metric_scale='dataset training-variable normalization, after model instance denormalization',
                    original_scale='inverse dataset training-variable scaler; *_original fields')
    (folder / 'raw_recent_export_manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    return manifest
