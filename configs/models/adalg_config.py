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
    elif key in {'physionet', 'p19'}:
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
    prediction_variant = _env_value(env, 'POINT_PREDICTION_VARIANT', 'adalg')
    recent_strict_past = _env_flag(env, 'POINT_RECENT_STRICT_PAST', '0')
    point_gate_mode = _env_value(env, 'POINT_GATE_MODE', 'utility')
    point_train_stage = _env_value(env, 'POINT_TRAIN_STAGE', 'full')
    point_gate_holdout_ratio = _env_value(env, 'POINT_GATE_HOLDOUT_RATIO', '0.0')
    point_gate_split_seed = _env_value(env, 'POINT_GATE_SPLIT_SEED', 1729, int)
    point_gate_init_checkpoint = str(_env_value(env, 'POINT_GATE_INIT_CHECKPOINT', '')).strip()
    point_gate_initial_weight = str(_env_value(env, 'POINT_GATE_INITIAL_WEIGHT', '')).strip()
    point_frozen_global_local_training = _env_flag(env, 'POINT_FROZEN_GLOBAL_LOCAL_TRAINING', '0')
    if point_frozen_global_local_training and (point_train_stage != 'gate' or not point_gate_init_checkpoint):
        raise ValueError('Frozen Global Local training requires stage=gate and POINT_GATE_INIT_CHECKPOINT.')
    if point_train_stage not in {'full', 'branch', 'gate'}:
        raise ValueError('POINT_TRAIN_STAGE must be full, branch, or gate.')
    if point_train_stage != 'full' and not 0.0 < float(point_gate_holdout_ratio) < 1.0:
        raise ValueError('POINT_GATE_HOLDOUT_RATIO must be in (0, 1) for branch/gate stages.')
    if point_train_stage == 'branch' and point_gate_init_checkpoint:
        raise ValueError('Branch stage must not set POINT_GATE_INIT_CHECKPOINT.')
    if point_train_stage == 'gate' and not point_gate_init_checkpoint:
        raise ValueError('Gate stage requires POINT_GATE_INIT_CHECKPOINT from a branch-stage run.')
    point_gate_prediction_features = _env_flag(env, 'POINT_GATE_PREDICTION_FEATURES', '0')
    point_separate_utility_head = _env_flag(env, 'POINT_SEPARATE_UTILITY_HEAD', '0')
    point_disable_causal_gate_context = _env_flag(env, 'POINT_DISABLE_CAUSAL_GATE_CONTEXT', '0')
    point_disable_gate_availability_features = _env_flag(env, 'POINT_DISABLE_GATE_AVAILABILITY_FEATURES', '0')
    global_aux_loss_weight = _env_value(env, 'GLOBAL_AUX_LOSS_WEIGHT', '0.25')
    local_aux_loss_weight = _env_value(env, 'LOCAL_AUX_LOSS_WEIGHT', '0.25')
    utility_loss_weight = _env_value(env, 'UTILITY_LOSS_WEIGHT', '0.1')
    oracle_gate_loss_weight = _env_value(env, 'ORACLE_GATE_LOSS_WEIGHT', '0.1')
    local_drop_robustness = _env_flag(env, 'LOCAL_OBSERVATION_DROP_ROBUSTNESS', '0')
    input_drop_robustness = _env_flag(env, 'INPUT_OBSERVATION_DROP_ROBUSTNESS', '0')
    point_local_fusion_mode = _env_value(env, 'POINT_LOCAL_FUSION_MODE', 'dual_gate')
    point_force_fusion = _env_value(env, 'POINT_FORCE_FUSION', 'auto')
    point_disable_local_branch = _env_flag(env, 'POINT_DISABLE_LOCAL_BRANCH', '0')
    point_fixed_gate = str(_env_value(env, 'POINT_FIXED_GATE', '')).strip()
    point_local_fusion_k = _env_value(env, 'POINT_LOCAL_FUSION_K', point_knn, int)
    point_local_branch_source = _env_value(env, 'POINT_LOCAL_BRANCH_SOURCE', 'raw')
    point_local_encoder = _env_value(env, 'POINT_LOCAL_ENCODER', 'summary')
    if point_local_branch_source not in {'raw', 'shared_encoder'}:
        raise ValueError('POINT_LOCAL_BRANCH_SOURCE must be raw or shared_encoder.')
    if point_local_encoder not in {'summary', 'attention'}:
        raise ValueError('POINT_LOCAL_ENCODER must be summary or attention.')
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

    # Hash the full CausalODE config so setting names stay short while still
    # changing whenever hidden/state/scale/regularization options change.
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
        '_hbclosed2'
        f'_g{point_local_gate_bias}'
        f'_dg{point_dual_gate_bias}'
        f'_pgm{point_gate_max}'
        f'_gm{point_gate_mode}'
        f'{"_gi" + point_gate_initial_weight if point_gate_initial_weight else ""}'
        f'{"_frozenG_local" if point_frozen_global_local_training else ""}'
        f'{"_st" + point_train_stage + "_gh" + point_gate_holdout_ratio + "_gs" + str(point_gate_split_seed) if point_train_stage != "full" else ""}'
        f'{"_gateonly" + _short_hash({"source": point_gate_init_checkpoint}) if point_gate_init_checkpoint else ""}'
        f'{"_gpf1" if point_gate_prediction_features else ""}'
        f'{"_suh1" if point_separate_utility_head else ""}'
        f'_ctxoff{1 if point_disable_causal_gate_context else 0}'
        f'{"_avoff1" if point_disable_gate_availability_features else ""}'
        f'_gaux{global_aux_loss_weight}'
        f'_laux{local_aux_loss_weight}'
        f'_uaux{utility_loss_weight}'
        f'_oaux{oracle_gate_loss_weight}'
        f'_ldrop{1 if local_drop_robustness else 0}'
        f'_idrop{1 if input_drop_robustness else 0}'
        f'_lf{point_local_fusion_mode}'
        f'_ff{point_force_fusion}'
        f'{"_nolocal" if point_disable_local_branch else ""}'
        f'{"_fg" + point_fixed_gate if point_fixed_gate else ""}'
        f'_k{point_local_fusion_k}'
        f'_src{point_local_branch_source}_le{point_local_encoder}'
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
        'model_name': 'AdaLG',
        'model_id': f"{dataset['id']}_AdaLG",
        'des': _env_value(env, 'DES', f"{dataset['slug']}_AdaLG"),
        'optimizer': optimizer,
        'learning_rate': learning_rate,
        'batch_size': batch_size,
        'train_epochs': train_epochs,
        'model_des_suffix': model_des_suffix,
        'args': [
            *(['--point_prediction_variant', prediction_variant] if prediction_variant != 'adalg' else []),
            *(['--point_recent_strict_past'] if recent_strict_past else []),
            *(['--raw_recent_query_export'] if _env_flag(env, 'RAW_RECENT_QUERY_EXPORT', '0') else []),
            *(['--raw_concat_hidden', str(env['RAW_CONCAT_HIDDEN'])] if env.get('RAW_CONCAT_HIDDEN') else []),
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
            '--point_gate_mode',
            str(point_gate_mode),
            *(['--point_train_stage', point_train_stage,
               '--point_gate_holdout_ratio', str(point_gate_holdout_ratio),
               '--point_gate_split_seed', str(point_gate_split_seed)] if point_train_stage != 'full' else []),
            *(['--point_gate_init_checkpoint', point_gate_init_checkpoint] if point_gate_init_checkpoint else []),
            *(['--point_gate_initial_weight', point_gate_initial_weight] if point_gate_initial_weight else []),
            *(['--point_frozen_global_local_training'] if point_frozen_global_local_training else []),
            *(['--point_gate_prediction_features'] if point_gate_prediction_features else []),
            *(['--point_separate_utility_head'] if point_separate_utility_head else []),
            *(['--point_disable_causal_gate_context'] if point_disable_causal_gate_context else []),
            *(['--point_disable_gate_availability_features'] if point_disable_gate_availability_features else []),
            '--global_aux_loss_weight',
            str(global_aux_loss_weight),
            '--local_aux_loss_weight',
            str(local_aux_loss_weight),
            '--utility_loss_weight',
            str(utility_loss_weight),
            '--oracle_gate_loss_weight',
            str(oracle_gate_loss_weight),
            '--query_evidence_analysis',
            *(['--local_observation_drop_robustness'] if local_drop_robustness else []),
            *(['--input_observation_drop_robustness'] if input_drop_robustness else []),
            '--point_local_fusion_mode',
            str(point_local_fusion_mode),
            '--point_force_fusion',
            str(point_force_fusion),
            *(['--point_disable_local_branch'] if point_disable_local_branch else []),
            *fixed_gate_args,
            '--point_local_fusion_k',
            str(point_local_fusion_k),
            '--point_local_branch_source', point_local_branch_source,
            '--point_local_encoder', point_local_encoder,
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
            f'--causal_ode_gate_init={causal_ode_gate_init}',
            '--causal_ode_steps',
            str(causal_ode_steps),
            *causal_local_args,
            '--point_align_val_with_test',
        ],
    }
