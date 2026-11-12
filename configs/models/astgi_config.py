import os
import hashlib


def _env_value(env, name, default, cast=str):
    if name not in env or env[name] == '':
        return default
    return cast(env[name])


def _short_hash(values):
    payload = '|'.join(f'{key}={values[key]}' for key in sorted(values))
    return hashlib.sha1(payload.encode('utf-8')).hexdigest()[:8]


def load_model_config(dataset, env=None):
    env = os.environ if env is None else env
    key = dataset['key']
    defaults = {
        'batch_size': dataset['batch_size'],
        'train_epochs': dataset['train_epochs'],
        'learning_rate': '0.001',
        'd_model': 64,
        'dropout': '0.1',
        'astgi_channel_dim': 24,
        'astgi_time_dim': 24,
        'astgi_mlp_ratio': '2.0',
    }
    if key == 'human_activity':
        defaults.update(
            astgi_channel_dim=64,
            astgi_time_dim=128,
            astgi_mlp_ratio='4.0',
        )
    elif key == 'physionet':
        defaults.update(astgi_channel_dim=64, astgi_time_dim=64, astgi_mlp_ratio='4.0')

    optimizer = _env_value(env, 'OPTIMIZER', 'adamw')
    learning_rate = _env_value(env, 'LEARNING_RATE', defaults['learning_rate'])
    batch_size = _env_value(env, 'BATCH_SIZE', defaults['batch_size'], int)
    train_epochs = _env_value(env, 'TRAIN_EPOCHS', defaults['train_epochs'], int)
    d_model = _env_value(env, 'D_MODEL', defaults['d_model'], int)
    dropout = _env_value(env, 'DROPOUT', defaults['dropout'])
    astgi_channel_dim = _env_value(env, 'ASTGI_CHANNEL_DIM', defaults['astgi_channel_dim'], int)
    astgi_time_dim = _env_value(env, 'ASTGI_TIME_DIM', defaults['astgi_time_dim'], int)
    astgi_mlp_ratio = _env_value(env, 'ASTGI_MLP_RATIO', defaults['astgi_mlp_ratio'])
    astgi_query_chunk_size = _env_value(env, 'ASTGI_QUERY_CHUNK_SIZE', 1024, int)

    causal_ode_state_dim = _env_value(env, 'CAUSAL_ODE_STATE_DIM', 32, int)
    causal_ode_hidden = _env_value(env, 'CAUSAL_ODE_HIDDEN', 128, int)
    causal_ode_constraint_strength = _env_value(env, 'CAUSAL_ODE_CONSTRAINT_STRENGTH', '0')
    causal_ode_h_weight = _env_value(env, 'CAUSAL_ODE_H_WEIGHT', '0.001')
    causal_ode_l1_weight = _env_value(env, 'CAUSAL_ODE_L1_WEIGHT', '0.0001')
    causal_ode_diag_weight = _env_value(env, 'CAUSAL_ODE_DIAG_WEIGHT', '0.001')
    causal_ode_laplacian_weight = _env_value(env, 'CAUSAL_ODE_LAPLACIAN_WEIGHT', '0.0001')
    causal_ode_adjacency_scale = _env_value(env, 'CAUSAL_ODE_ADJACENCY_SCALE', '1.0')
    causal_ode_gate_init = _env_value(env, 'CAUSAL_ODE_GATE_INIT', '-4.0')
    causal_ode_steps = _env_value(env, 'CAUSAL_ODE_STEPS', 8, int)

    causal_ode_hash = _short_hash({
        'adjacency_scale': causal_ode_adjacency_scale,
        'constraint_strength': causal_ode_constraint_strength,
        'diag_weight': causal_ode_diag_weight,
        'gate_init': causal_ode_gate_init,
        'h_weight': causal_ode_h_weight,
        'hidden': causal_ode_hidden,
        'l1_weight': causal_ode_l1_weight,
        'laplacian_weight': causal_ode_laplacian_weight,
        'state_dim': causal_ode_state_dim,
        'steps': causal_ode_steps,
    })
    model_des_suffix = (
        f'_cd{astgi_channel_dim}'
        f'_td{astgi_time_dim}'
        f'_cg{causal_ode_gate_init}'
        f'_cs{causal_ode_steps}'
        f'_oh{causal_ode_hash}'
    )
    return {
        'model_name': 'astgi',
        'model_id': f"{dataset['id']}_astgi",
        'des': _env_value(env, 'DES', f"{dataset['slug']}_astgi"),
        'optimizer': optimizer,
        'learning_rate': learning_rate,
        'batch_size': batch_size,
        'train_epochs': train_epochs,
        'model_des_suffix': model_des_suffix,
        'args': [
            '--d_model',
            str(d_model),
            '--dropout',
            str(dropout),
            '--use_norm',
            '--nonautoregressive',
            '--astgi_channel_dim',
            str(astgi_channel_dim),
            '--astgi_time_dim',
            str(astgi_time_dim),
            '--astgi_mlp_ratio',
            str(astgi_mlp_ratio),
            '--astgi_query_chunk_size',
            str(astgi_query_chunk_size),
            '--causal_ode_state_dim',
            str(causal_ode_state_dim),
            '--causal_ode_hidden',
            str(causal_ode_hidden),
            '--causal_ode_constraint_strength',
            str(causal_ode_constraint_strength),
            '--causal_ode_h_weight',
            str(causal_ode_h_weight),
            '--causal_ode_l1_weight',
            str(causal_ode_l1_weight),
            '--causal_ode_diag_weight',
            str(causal_ode_diag_weight),
            '--causal_ode_laplacian_weight',
            str(causal_ode_laplacian_weight),
            '--causal_ode_adjacency_scale',
            str(causal_ode_adjacency_scale),
            '--causal_ode_gate_init',
            str(causal_ode_gate_init),
            '--causal_ode_steps',
            str(causal_ode_steps),
            '--point_align_val_with_test',
        ],
    }
