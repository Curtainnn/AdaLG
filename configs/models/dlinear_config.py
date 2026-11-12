import os


def _env_value(env, name, default, cast=str):
    if name not in env or env[name] == '':
        return default
    return cast(env[name])


def _env_flag(env, name, default='0'):
    value = _env_value(env, name, default)
    return str(value).strip().lower() in {'1', 'true', 'yes', 'on'}


def load_model_config(dataset, env=None):
    env = os.environ if env is None else env
    key = dataset['key']
    defaults = {
        'batch_size': dataset['batch_size'],
        'train_epochs': dataset['train_epochs'],
        'learning_rate': '0.001',
        'moving_avg': 25,
        'individual': False,
    }
    if key in {'physionet', 'p19'}:
        defaults.update(batch_size=256, train_epochs=100)

    batch_size = _env_value(env, 'BATCH_SIZE', defaults['batch_size'], int)
    train_epochs = _env_value(env, 'TRAIN_EPOCHS', defaults['train_epochs'], int)
    learning_rate = _env_value(env, 'LEARNING_RATE', defaults['learning_rate'])
    moving_avg = _env_value(env, 'DLINEAR_MOVING_AVG', defaults['moving_avg'], int)
    individual = _env_flag(env, 'DLINEAR_INDIVIDUAL', '1' if defaults['individual'] else '0')

    model_des_suffix = f'_ma{moving_avg}_ind{1 if individual else 0}'
    model_args = [
        '--dlinear_moving_avg',
        str(moving_avg),
        '--nonautoregressive',
    ]
    if individual:
        model_args.append('--dlinear_individual')

    return {
        'model_name': 'dlinear',
        'model_id': f"{dataset['id']}_dlinear",
        'des': _env_value(env, 'DES', f"{dataset['slug']}_dlinear"),
        'optimizer': _env_value(env, 'OPTIMIZER', 'adam'),
        'learning_rate': learning_rate,
        'batch_size': batch_size,
        'train_epochs': train_epochs,
        'model_des_suffix': model_des_suffix,
        'args': model_args,
    }
