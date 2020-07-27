import os

PROJECT_DIR = os.path.abspath(os.path.dirname(__file__))
CORE_DATA_DIR = PROJECT_DIR + '/{}'.format('data')


class Config:
    DATA_EXPERIMENT = 'google_trace'  # grid, traffic, google_trace

    PLT_ENV = 'Agg'  # TkAgg
    GOOGLE_TRACE_DATA_CONFIG = {
        'train_data_type': 'cpu',  # cpu_mem, uni_mem, uni_cpu
        'predict_data': 'cpu',
        'data_type': 'all_jobs',  # 1_job, all_jobs
        'time_interval': 5,
        'file_data_name': '/input_data/google_trace/{}/{}_mins.csv',
        'data_path': CORE_DATA_DIR + '{}'
    }

    GRID_DATA_CONFIG = {
        'time_interval': 10800,  # 600, 3600, 7200, 10800, 21600
        'file_data_name': '/input_data/grid_data/timeseries_anonjobs_{}Min.csv',
        'data_path': CORE_DATA_DIR + '{}',
        'colnames': ['job_id_data', 'n_proc_data', 'used_cpu_time_data', 'used_memory_data', 'user_id_data',
                     'group_id_data']
    }

    TRAFFIC_JAM_DATA_CONFIG = {
        'file_data_name': '/input_data/traffic/it_eu_5m.csv',
        'data_path': CORE_DATA_DIR + '{}',
        'colnames': ['timestamp', 'bit', 'byte', 'kilobyte', 'megabyte']
    }

    VISUALIZATION_CONFIG = {
        'options': False,
        'metrics': {
            'google_trace': 'disk_space',
            'grid': 'n_processes',  # 'n_processes', 'used_cpu_time', 'used_memory', 'users_id', 'groups_id'
            'traffic': 'eu'
        }
    }

    VISUALIZATION = True
    MODEL_EXPERIMENT = 'bnn'  # lstm, ann, bnn, gan
    METHOD_APPROACH = 'bp'  # pso, whale, bp, bp_pso, pso_bp

    METAHEURISTIC_METHOD = 'pso'  # pso, whale

    FITNESS_TYPE = 'bayesian_autoscaling'  # validation_error, bayesian_autoscaling

    LEARNING_RATE = 3e-4
    EPOCHS = 5
    EARLY_STOPPING = True
    PATIENCE = 20
    TRAIN_SIZE = 0.2
    VALID_SIZE = 0.8

    if DATA_EXPERIMENT == 'google_trace':
        INFO_SAVED_PATH = 'results/google_trace/{}/{}/{}/{}'.format(
            GOOGLE_TRACE_DATA_CONFIG['data_type'], MODEL_EXPERIMENT, GOOGLE_TRACE_DATA_CONFIG['train_data_type'],
            GOOGLE_TRACE_DATA_CONFIG['predict_data'])
        RESULTS_SAVE_PATH = CORE_DATA_DIR + '/{}/{}/{}/'.format(INFO_SAVED_PATH, METAHEURISTIC_METHOD, FITNESS_TYPE)
    else:
        INFO_SAVED_PATH = '{}/{}/{}'.format(MODEL_EXPERIMENT, METHOD_APPROACH, DATA_EXPERIMENT)
        RESULTS_SAVE_PATH = CORE_DATA_DIR + '/{}/'.format(INFO_SAVED_PATH)

    LSTM_CONFIG = {
        'sliding': [5],
        'batch_size': [8],
        'num_units': [[4]],
        'dropout_rate': [0.9],
        'variation_dropout': False,
        'activation': ['tanh'],  # 'sigmoid', 'relu', 'tanh', 'elu'
        'optimizers': ['adam'],  # 'momentum', 'adam', 'rmsprop'
    }

    ANN_CONFIG = {
        'sliding': [3],
        'batch_size': [8],
        'num_units': [[4]],
        'scalers': ['min_max_scaler', 'standard_scaler'],
        'activation': ['sigmoid', 'tanh', 'relu', 'elu'],  # 'sigmoid', 'relu', 'tanh', 'elu'
        'optimizers': ['momentum', 'adam', 'rmsprop'],  # 'momentum', 'adam', 'rmsprop'
        'domain': [
            {'name': 'scaler', 'type': 'discrete', 'domain': [1]},
            {'name': 'batch_size', 'type': 'discrete', 'domain': [64]},
            {'name': 'sliding', 'type': 'discrete', 'domain': [1, 2, 3, 4, 5, 6, 7, 8]},
            {'name': 'network_size', 'type': 'discrete', 'domain': [2, 3, 4, 5]},
            {'name': 'layer_size', 'type': 'discrete', 'domain': [16]},
            {'name': 'dropout', 'type': 'continuous', 'domain': (0.1, 0.5)},
            {'name': 'learning_rate', 'type': 'continuous', 'domain': (0.0003, 0.00031)},
            {'name': 'optimizer', 'type': 'discrete', 'domain': [2]},
            {'name': 'activation', 'type': 'discrete', 'domain': [2]}
        ]
    }

    BNN_CONFIG = {
        'sliding_encoder': [8, 9, 10],
        'sliding_inf': [2, 3, 4, 5],
        'batch_size': [8],
        'num_units': [[4]],
        'scalers': ['min_max_scaler'],
        'activation': ['sigmoid', 'tanh', 'relu', 'elu'],  # 'sigmoid', 'relu', 'tanh', 'elu'
        'optimizers': ['momentum', 'adam', 'rmsprop'],  # 'momentum', 'adam', 'rmsprop'
        'cell_type': ['lstm', 'gru'],
        'domain': [
            {'name': 'scaler', 'type': 'discrete', 'domain': [1, 2]},
            {'name': 'batch_size', 'type': 'discrete', 'domain': [8, 16, 32, 64, 128]},
            {'name': 'sliding_encoder', 'type': 'discrete', 'domain': [12, 15, 18, 21, 24]},
            {'name': 'sliding_decoder', 'type': 'discrete', 'domain': [6, 8, 10, 12]},
            {'name': 'sliding_inf', 'type': 'discrete', 'domain': [2, 3, 4, 5]},
            {'name': 'network_size_encoder', 'type': 'discrete', 'domain': [1, 2, 3, 4]},
            {'name': 'layer_size_encoder', 'type': 'discrete', 'domain': [4, 8, 16, 32, 64]},
            {'name': 'network_size_inf', 'type': 'discrete', 'domain': [1, 2, 3, 4]},
            {'name': 'layer_size_inf', 'type': 'discrete', 'domain': [4, 8, 16, 32, 64]},
            {'name': 'dropout', 'type': 'continuous', 'domain': (0.01, 0.2)},
            {'name': 'learning_rate', 'type': 'continuous', 'domain': (0.0001, 0.01)},
            {'name': 'optimizer', 'type': 'discrete', 'domain': [1, 2, 3]},
            {'name': 'activation', 'type': 'discrete', 'domain': [1, 2, 3, 4]},
            {'name': 'cell_type', 'type': 'discrete', 'domain': [1, 2]}
        ]
    }

    PSO_CONFIG = {
        'num_particles': [50]
    }

    PSO_BNN_CONFIG = {
        'num_particles': [50, 200]
    }
