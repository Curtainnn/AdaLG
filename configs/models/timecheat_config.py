import os


def _env_value(env, name, default, cast=str):
    if name not in env or env[name] == '':
        return default
    return cast(env[name])


def load_model_config(dataset, env=None):
    env = os.environ if env is None else env
    key = dataset['key']
    defaults = {
        'batch_size': dataset['batch_size'],
        'train_epochs': dataset['train_epochs'],
        'learning_rate': '0.001',
        'latent_dim': 128,
        'n_layers': 2,
        'attn_heads': 4,
        'ref_points': 128,
        'n_patches': 8,
        'former_layers': 3,
        'former_heads': 8,
        'former_dff': 256,
        'dropout': '0.1',
        'use_norm': True,
    }
    if key == 'human_activity':
        defaults.update(batch_size=16, n_patches=8, ref_points=128)
    elif key == 'ushcn':
        defaults.update(batch_size=16, latent_dim=64, n_patches=5, ref_points=80, former_heads=4)
    elif key in {'physionet', 'p19'}:
        defaults.update(batch_size=32, train_epochs=100, n_patches=6, ref_points=96, former_heads=4)

    batch_size = _env_value(env, 'BATCH_SIZE', defaults['batch_size'], int)
    train_epochs = _env_value(env, 'TRAIN_EPOCHS', defaults['train_epochs'], int)
    learning_rate = _env_value(env, 'LEARNING_RATE', defaults['learning_rate'])
    latent_dim = _env_value(env, 'TIMECHEAT_LATENT_DIM', defaults['latent_dim'], int)
    n_layers = _env_value(env, 'TIMECHEAT_N_LAYERS', defaults['n_layers'], int)
    attn_heads = _env_value(env, 'TIMECHEAT_ATTN_HEADS', defaults['attn_heads'], int)
    ref_points = _env_value(env, 'TIMECHEAT_REF_POINTS', defaults['ref_points'], int)
    n_patches = _env_value(env, 'TIMECHEAT_N_PATCHES', defaults['n_patches'], int)
    former_layers = _env_value(env, 'TIMECHEAT_FORMER_LAYERS', defaults['former_layers'], int)
    former_heads = _env_value(env, 'TIMECHEAT_FORMER_HEADS', defaults['former_heads'], int)
    former_dff = _env_value(env, 'TIMECHEAT_FORMER_DFF', defaults['former_dff'], int)
    dropout = _env_value(env, 'DROPOUT', defaults['dropout'])
    use_norm = str(_env_value(env, 'TIMECHEAT_USE_NORM', '1' if defaults['use_norm'] else '0')).lower()

    model_des_suffix = (
        f'_ld{latent_dim}'
        f'_gl{n_layers}'
        f'_ah{attn_heads}'
        f'_rp{ref_points}'
        f'_np{n_patches}'
        f'_fl{former_layers}'
        f'_fh{former_heads}'
        f'_norm{1 if use_norm in {"1", "true", "yes", "on"} else 0}'
    )
    args = [
        '--dropout',
        str(dropout),
        '--nonautoregressive',
        '--timecheat_latent_dim',
        str(latent_dim),
        '--timecheat_n_layers',
        str(n_layers),
        '--timecheat_attn_heads',
        str(attn_heads),
        '--timecheat_ref_points',
        str(ref_points),
        '--timecheat_n_patches',
        str(n_patches),
        '--timecheat_former_layers',
        str(former_layers),
        '--timecheat_former_heads',
        str(former_heads),
        '--timecheat_former_dff',
        str(former_dff),
        '--point_align_val_with_test',
    ]
    if use_norm in {'1', 'true', 'yes', 'on'}:
        args.append('--use_norm')

    return {
        'model_name': 'timecheat',
        'model_id': f"{dataset['id']}_timecheat",
        'des': _env_value(env, 'DES', f"{dataset['slug']}_timecheat"),
        'optimizer': _env_value(env, 'OPTIMIZER', 'adam'),
        'learning_rate': learning_rate,
        'batch_size': batch_size,
        'train_epochs': train_epochs,
        'model_des_suffix': model_des_suffix,
        'args': args,
    }
