import os


def _env_value(env, name, default, cast=str):
    if name not in env or env[name] == '':
        return default
    return cast(env[name])


def _env_flag(env, name, default='1'):
    value = str(_env_value(env, name, default)).strip().lower()
    return value in {'1', 'true', 'yes', 'on'}


def load_model_config(dataset, env=None):
    env = os.environ if env is None else env
    key = dataset['key']
    defaults = {
        'batch_size': dataset['batch_size'],
        'train_epochs': dataset['train_epochs'],
        'learning_rate': '0.001',
        'num_ref_points': 64,
        'latent_dim': 16,
        'rec_hidden': 64,
        'gen_hidden': 50,
        'embed_time': 128,
        'num_heads': 4,
        'k_samples': 1,
        'kl_weight': '0.0',
        'use_norm': True,
    }
    if key == 'human_activity':
        defaults.update(batch_size=16, num_ref_points=128)
    elif key == 'ushcn':
        defaults.update(batch_size=16, num_ref_points=64)
    elif key in {'physionet', 'p19'}:
        defaults.update(batch_size=32, train_epochs=100, num_ref_points=64)

    batch_size = _env_value(env, 'BATCH_SIZE', defaults['batch_size'], int)
    train_epochs = _env_value(env, 'TRAIN_EPOCHS', defaults['train_epochs'], int)
    learning_rate = _env_value(env, 'LEARNING_RATE', defaults['learning_rate'])
    num_ref_points = _env_value(env, 'MTAN_NUM_REF_POINTS', defaults['num_ref_points'], int)
    latent_dim = _env_value(env, 'MTAN_LATENT_DIM', defaults['latent_dim'], int)
    rec_hidden = _env_value(env, 'MTAN_REC_HIDDEN', defaults['rec_hidden'], int)
    gen_hidden = _env_value(env, 'MTAN_GEN_HIDDEN', defaults['gen_hidden'], int)
    embed_time = _env_value(env, 'MTAN_EMBED_TIME', defaults['embed_time'], int)
    num_heads = _env_value(env, 'MTAN_NUM_HEADS', defaults['num_heads'], int)
    k_samples = _env_value(env, 'MTAN_K_SAMPLES', defaults['k_samples'], int)
    kl_weight = _env_value(env, 'MTAN_KL_WEIGHT', defaults['kl_weight'])
    use_norm = _env_flag(env, 'MTAN_USE_NORM', '1' if defaults['use_norm'] else '0')

    model_des_suffix = (
        f'_rp{num_ref_points}'
        f'_ld{latent_dim}'
        f'_rh{rec_hidden}'
        f'_gh{gen_hidden}'
        f'_et{embed_time}'
        f'_nh{num_heads}'
        f'_ks{k_samples}'
        f'_kl{kl_weight}'
        f'_norm{1 if use_norm else 0}'
    )
    args = [
        '--nonautoregressive',
        '--mtan_num_ref_points',
        str(num_ref_points),
        '--mtan_latent_dim',
        str(latent_dim),
        '--mtan_rec_hidden',
        str(rec_hidden),
        '--mtan_gen_hidden',
        str(gen_hidden),
        '--mtan_embed_time',
        str(embed_time),
        '--mtan_num_heads',
        str(num_heads),
        '--mtan_k_samples',
        str(k_samples),
        '--mtan_kl_weight',
        str(kl_weight),
        '--point_align_val_with_test',
    ]
    if use_norm:
        args.append('--use_norm')

    return {
        'model_name': 'mtan',
        'model_id': f"{dataset['id']}_mtan",
        'des': _env_value(env, 'DES', f"{dataset['slug']}_mtan"),
        'optimizer': _env_value(env, 'OPTIMIZER', 'adam'),
        'learning_rate': learning_rate,
        'batch_size': batch_size,
        'train_epochs': train_epochs,
        'model_des_suffix': model_des_suffix,
        'args': args,
    }
