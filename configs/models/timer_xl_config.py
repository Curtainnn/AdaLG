import os
import hashlib


def _env_value(env, name, default, cast=str):
    if name not in env or env[name] == '':
        return default
    return cast(env[name])


def _short_hash(values):
    payload = '|'.join(f'{key}={values[key]}' for key in sorted(values))
    return hashlib.sha1(payload.encode('utf-8')).hexdigest()[:8]


def _env_flag(env, name, default='0'):
    value = _env_value(env, name, default)
    return str(value).strip().lower() in {'1', 'true', 'yes', 'on'}


def load_model_config(dataset, env=None):
    env = os.environ if env is None else env
    key = dataset['key']
    defaults = {
        'optimizer': 'adamw',
        'learning_rate': '0.001',
        'batch_size': dataset['batch_size'],
        'train_epochs': dataset['train_epochs'],
        'd_model': 64,
        'd_ff': 256,
        'e_layers': 1,
    }
    if key == 'etth1':
        defaults.update(optimizer='adam', learning_rate='0.0001', d_model=1024, d_ff=2048)
    elif key == 'ecl':
        defaults.update(optimizer='adam', learning_rate='0.0005', d_model=512, d_ff=2048, e_layers=5)
    elif key == 'physionet':
        defaults.update(batch_size=32)

    optimizer = _env_value(env, 'OPTIMIZER', defaults['optimizer'])
    learning_rate = _env_value(env, 'LEARNING_RATE', defaults['learning_rate'])
    batch_size = _env_value(env, 'BATCH_SIZE', defaults['batch_size'], int)
    train_epochs = _env_value(env, 'TRAIN_EPOCHS', defaults['train_epochs'], int)
    d_model = _env_value(env, 'D_MODEL', defaults['d_model'], int)
    d_ff = _env_value(env, 'D_FF', defaults['d_ff'], int)
    e_layers = _env_value(env, 'E_LAYERS', defaults['e_layers'], int)
    n_heads = _env_value(env, 'N_HEADS', 8, int)
    dropout = _env_value(env, 'DROPOUT', '0.1')
    point_hidden_dim = _env_value(env, 'POINT_HIDDEN_DIM', 128, int)
    point_layers = _env_value(env, 'POINT_LAYERS', 3, int)
    point_knn = _env_value(env, 'POINT_KNN', 12, int)
    point_neighbor_window = _env_value(env, 'POINT_NEIGHBOR_WINDOW', -1, int)
    point_token_subbins = _env_value(env, 'POINT_TOKEN_SUBBINS', 4, int)
    point_time_dim = _env_value(env, 'POINT_TIME_DIM', 32, int)
    point_local_gate_bias = _env_value(env, 'POINT_LOCAL_GATE_BIAS', '-2.0')
    point_dual_gate_bias = _env_value(env, 'POINT_DUAL_GATE_BIAS', '-1.0')
    point_gate_max = _env_value(env, 'POINT_GATE_MAX', '1.0')
    point_local_fusion_mode = _env_value(env, 'POINT_LOCAL_FUSION_MODE', 'dual_gate')
    point_force_fusion = _env_value(env, 'POINT_FORCE_FUSION', 'auto')
    point_fixed_gate = str(_env_value(env, 'POINT_FIXED_GATE', '')).strip()
    point_local_fusion_k = _env_value(env, 'POINT_LOCAL_FUSION_K', point_knn, int)
    point_causal_context_mode = _env_value(env, 'POINT_CAUSAL_CONTEXT_MODE', 'normal')

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
    causal_prior_window = _env_value(env, 'CAUSAL_PRIOR_WINDOW', 30, int)
    causal_prior_mode = _env_value(env, 'CAUSAL_PRIOR_MODE', 'normal')
    causal_local_alpha = _env_value(env, 'CAUSAL_LOCAL_ALPHA', '0.0')
    causal_local_beta = _env_value(env, 'CAUSAL_LOCAL_BETA', '1.0')
    causal_local_expand_pool = _env_flag(env, 'CAUSAL_LOCAL_EXPAND_POOL', '0')

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
        f'_g{point_local_gate_bias}'
        f'_dg{point_dual_gate_bias}'
        f'_pgm{point_gate_max}'
        f'_lf{point_local_fusion_mode}'
        f'{"_ff" + point_force_fusion if point_force_fusion != "auto" else ""}'
        f'{"_fg" + point_fixed_gate if point_fixed_gate else ""}'
        f'_k{point_local_fusion_k}'
        f'_ctx{point_causal_context_mode}'
        f'_cg{causal_ode_gate_init}'
        f'_cs{causal_ode_steps}'
        f'_oh{causal_ode_hash}'
        f'_cla{causal_local_alpha}'
        f'_clb{causal_local_beta}'
        f'_cpw{causal_prior_window}'
        f'_cpm{causal_prior_mode}'
        f'_cpexp{1 if causal_local_expand_pool else 0}'
    )
    causal_local_args = [
        '--causal_prior_window',
        str(causal_prior_window),
        '--causal_prior_mode',
        str(causal_prior_mode),
        '--causal_local_alpha',
        str(causal_local_alpha),
        '--causal_local_beta',
        str(causal_local_beta),
    ]
    if causal_local_expand_pool:
        causal_local_args.append('--causal_local_expand_pool')
    fixed_gate_args = ['--point_fixed_gate', point_fixed_gate] if point_fixed_gate else []
    return {
        'model_name': 'timer_xl',
        'model_id': f"{dataset['id']}_timer_xl",
        'des': _env_value(env, 'DES', f"{dataset['slug']}_timer_xl"),
        'optimizer': optimizer,
        'learning_rate': learning_rate,
        'batch_size': batch_size,
        'train_epochs': train_epochs,
        'model_des_suffix': model_des_suffix,
        'args': [
            '--d_model',
            str(d_model),
            '--d_ff',
            str(d_ff),
            '--e_layers',
            str(e_layers),
            '--n_heads',
            str(n_heads),
            '--dropout',
            str(dropout),
            '--use_norm',
            '--nonautoregressive',
            '--use_point_representation',
            '--point_hidden_dim',
            str(point_hidden_dim),
            '--point_layers',
            str(point_layers),
            '--point_knn',
            str(point_knn),
            '--point_neighbor_window',
            str(point_neighbor_window),
            '--point_token_subbins',
            str(point_token_subbins),
            '--point_time_dim',
            str(point_time_dim),
            '--point_local_gate_bias',
            str(point_local_gate_bias),
            '--point_dual_gate_bias',
            str(point_dual_gate_bias),
            '--point_gate_max',
            str(point_gate_max),
            '--point_local_fusion_mode',
            str(point_local_fusion_mode),
            '--point_force_fusion',
            str(point_force_fusion),
            *fixed_gate_args,
            '--point_local_fusion_k',
            str(point_local_fusion_k),
            '--point_causal_context_mode',
            str(point_causal_context_mode),
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
            *causal_local_args,
            '--point_align_val_with_test',
        ],
    }
