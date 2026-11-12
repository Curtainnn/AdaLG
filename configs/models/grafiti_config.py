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
        'use_norm': True,
    }
    if key == 'ushcn':
        defaults.update(batch_size=32, latent_dim=64)
    elif key in {'physionet', 'p19'}:
        defaults.update(batch_size=32, train_epochs=100, latent_dim=128)
    elif key == 'human_activity':
        defaults.update(batch_size=4, latent_dim=128)

    batch_size = _env_value(env, 'BATCH_SIZE', defaults['batch_size'], int)
    train_epochs = _env_value(env, 'TRAIN_EPOCHS', defaults['train_epochs'], int)
    learning_rate = _env_value(env, 'LEARNING_RATE', defaults['learning_rate'])
    latent_dim = _env_value(env, 'GRAFITI_LATENT_DIM', defaults['latent_dim'], int)
    n_layers = _env_value(env, 'GRAFITI_N_LAYERS', defaults['n_layers'], int)
    attn_heads = _env_value(env, 'GRAFITI_ATTN_HEADS', defaults['attn_heads'], int)
    use_norm = str(_env_value(env, 'GRAFITI_USE_NORM', '1' if defaults['use_norm'] else '0')).lower()

    model_des_suffix = f'_ld{latent_dim}_nl{n_layers}_ah{attn_heads}_norm{1 if use_norm in {"1", "true", "yes", "on"} else 0}'
    args = [
        '--nonautoregressive',
        '--grafiti_latent_dim',
        str(latent_dim),
        '--grafiti_n_layers',
        str(n_layers),
        '--grafiti_attn_heads',
        str(attn_heads),
        '--point_align_val_with_test',
    ]
    if use_norm in {'1', 'true', 'yes', 'on'}:
        args.append('--use_norm')

    return {
        'model_name': 'grafiti',
        'model_id': f"{dataset['id']}_grafiti",
        'des': _env_value(env, 'DES', f"{dataset['slug']}_grafiti"),
        'optimizer': _env_value(env, 'OPTIMIZER', 'adam'),
        'learning_rate': learning_rate,
        'batch_size': batch_size,
        'train_epochs': train_epochs,
        'model_des_suffix': model_des_suffix,
        'args': args,
    }
