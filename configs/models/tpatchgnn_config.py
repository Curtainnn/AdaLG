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
        'hid_dim': 64,
        'te_dim': 10,
        'node_dim': 10,
        'patch_size': dataset['input_token_len'],
        'stride': dataset['input_token_len'],
        'n_heads': 1,
        'tf_layer': 1,
        'n_layer': 1,
        'hop': 1,
        'outlayer': 'Linear',
    }
    if key == 'ushcn':
        defaults.update(hid_dim=32, patch_size=25, stride=25)
    elif key == 'human_activity':
        defaults.update(hid_dim=32, patch_size=300, stride=300)
    elif key == 'physionet':
        defaults.update(batch_size=32, train_epochs=100, hid_dim=64, patch_size=8, stride=8)

    batch_size = _env_value(env, 'BATCH_SIZE', defaults['batch_size'], int)
    train_epochs = _env_value(env, 'TRAIN_EPOCHS', defaults['train_epochs'], int)
    learning_rate = _env_value(env, 'LEARNING_RATE', defaults['learning_rate'])
    dropout = _env_value(env, 'DROPOUT', '0.1')
    hid_dim = _env_value(env, 'TPATCH_HID_DIM', defaults['hid_dim'], int)
    te_dim = _env_value(env, 'TPATCH_TE_DIM', defaults['te_dim'], int)
    node_dim = _env_value(env, 'TPATCH_NODE_DIM', defaults['node_dim'], int)
    patch_size = _env_value(env, 'TPATCH_PATCH_SIZE', defaults['patch_size'])
    stride = _env_value(env, 'TPATCH_STRIDE', defaults['stride'])
    n_heads = _env_value(env, 'TPATCH_N_HEADS', defaults['n_heads'], int)
    tf_layer = _env_value(env, 'TPATCH_TF_LAYER', defaults['tf_layer'], int)
    n_layer = _env_value(env, 'TPATCH_N_LAYER', defaults['n_layer'], int)
    hop = _env_value(env, 'TPATCH_HOP', defaults['hop'], int)
    outlayer = _env_value(env, 'TPATCH_OUTLAYER', defaults['outlayer'])

    model_des_suffix = (
        f'_ps{patch_size}'
        f'_st{stride}'
        f'_hd{hid_dim}'
        f'_td{te_dim}'
        f'_nd{node_dim}'
        f'_nh{n_heads}'
        f'_tf{tf_layer}'
        f'_nl{n_layer}'
        f'_hop{hop}'
        f'_ol{outlayer}'
    )
    return {
        'model_name': 'tpatchgnn',
        'model_id': f"{dataset['id']}_tpatchgnn",
        'des': _env_value(env, 'DES', f"{dataset['slug']}_tpatchgnn"),
        'optimizer': _env_value(env, 'OPTIMIZER', 'adam'),
        'learning_rate': learning_rate,
        'batch_size': batch_size,
        'train_epochs': train_epochs,
        'model_des_suffix': model_des_suffix,
        'args': [
            '--dropout',
            str(dropout),
            '--use_norm',
            '--nonautoregressive',
            '--tpatch_patch_size',
            str(patch_size),
            '--tpatch_stride',
            str(stride),
            '--tpatch_hid_dim',
            str(hid_dim),
            '--tpatch_te_dim',
            str(te_dim),
            '--tpatch_node_dim',
            str(node_dim),
            '--tpatch_n_heads',
            str(n_heads),
            '--tpatch_tf_layer',
            str(tf_layer),
            '--tpatch_n_layer',
            str(n_layer),
            '--tpatch_hop',
            str(hop),
            '--tpatch_outlayer',
            str(outlayer),
            '--point_align_val_with_test',
        ],
    }
