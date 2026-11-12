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
        'hidden_size': 100,
    }
    if key in {'physionet', 'p19'}:
        defaults.update(batch_size=128, train_epochs=100)

    batch_size = _env_value(env, 'BATCH_SIZE', defaults['batch_size'], int)
    train_epochs = _env_value(env, 'TRAIN_EPOCHS', defaults['train_epochs'], int)
    learning_rate = _env_value(env, 'LEARNING_RATE', defaults['learning_rate'])
    hidden_size = _env_value(env, 'GRUD_HIDDEN_SIZE', defaults['hidden_size'], int)

    model_des_suffix = f'_hd{hidden_size}'
    return {
        'model_name': 'gru_d',
        'model_id': f"{dataset['id']}_gru_d",
        'des': _env_value(env, 'DES', f"{dataset['slug']}_gru_d"),
        'optimizer': _env_value(env, 'OPTIMIZER', 'adam'),
        'learning_rate': learning_rate,
        'batch_size': batch_size,
        'train_epochs': train_epochs,
        'model_des_suffix': model_des_suffix,
        'args': [
            '--grud_hidden_size',
            str(hidden_size),
            '--nonautoregressive',
        ],
    }
