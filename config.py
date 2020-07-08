import os

PROJECT_DIR = os.path.abspath(os.path.dirname(__file__))
CORE_DATA_DIR = PROJECT_DIR + '/{}'.format('data')


class Config:
    DATA_EXPERIMENT = 'google_trace'  # grid, traffic, google_trace
    PLT_ENV = 'TkAgg'  # TkAgg
    GOOGLE_TRACE_DATA_CONFIG = {
        'train_data_type': 'cpu_mem',  # cpu_mem, uni_mem, uni_cpu
        'predict_data': 'cpu',
        'data_type': '1_job',  # 1_job, all_jobs
        'time_interval': 5,
        'file_data_name': '/input_data/google_trace/{}/{}_mins.csv',
        'data_path': CORE_DATA_DIR + '{}',
        'colnames': ['cpu_rate', 'mem_usage', 'disk_io_time', 'disk_space'],
        'usecols': [3, 4, 9, 10]
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
    MODEL_EXPERIMENT = 'ann'  # lstm, ann, bnn
    METHOD_APPROACH = 'bp'  # pso, whale, bp, bp_pso, pso_bp

    LEARNING_RATE = 3e-4
    EPOCHS = 30
    EARLY_STOPPING = True
    PATIENCE = 20
    TRAIN_SIZE = 0.8
    VALID_SIZE = 0.2

    if DATA_EXPERIMENT == 'google_trace':
        INFO_PATH = 'results/{}/{}/{}'.format(
            MODEL_EXPERIMENT, GOOGLE_TRACE_DATA_CONFIG['train_data_type'], GOOGLE_TRACE_DATA_CONFIG['predict_data'])
        MODEL_SAVE_PATH = CORE_DATA_DIR + '/{}/model'.format(INFO_PATH)
        RESULTS_SAVE_PATH = CORE_DATA_DIR + '/{}/results/'.format(INFO_PATH)
        TRAIN_LOSS_PATH = CORE_DATA_DIR + '/{}/train_losses/'.format(INFO_PATH)
        EVALUATION_PATH = CORE_DATA_DIR + '/{}/evaluation.csv'.format(INFO_PATH)
    else:
        INFO_PATH = '{}/{}/{}'.format(MODEL_EXPERIMENT, METHOD_APPROACH, DATA_EXPERIMENT)
        MODEL_SAVE_PATH = CORE_DATA_DIR + '/{}/model/'.format(INFO_PATH)
        RESULTS_SAVE_PATH = CORE_DATA_DIR + '/{}/results/'.format(INFO_PATH)
        TRAIN_LOSS_PATH = CORE_DATA_DIR + '/{}/train_losses/'.format(INFO_PATH)
        EVALUATION_PATH = CORE_DATA_DIR + '/{}/evaluation.csv'.format(INFO_PATH)

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
        'scalers': ['min_max_scaler'],
        'activation': ['sigmoid', 'tanh', 'relu', 'elu'],  # 'sigmoid', 'relu', 'tanh', 'elu'
        'optimizers': ['momentum', 'adam', 'rmsprop'],  # 'momentum', 'adam', 'rmsprop'
        'domain': [
            {'name': 'scaler', 'type': 'discrete', 'domain': [1]},
            {'name': 'batch_size', 'type': 'discrete', 'domain': [4, 8, 16, 32, 64]},
            {'name': 'sliding', 'type': 'discrete', 'domain': [1, 2, 3, 4, 5, 6, 7, 8]},
            {'name': 'network_size', 'type': 'discrete', 'domain': [1, 2, 3, 4, 5]},
            {'name': 'layer_size', 'type': 'discrete', 'domain': [2, 4, 8, 16, 32]},
            {'name': 'dropout', 'type': 'continuous', 'domain': (0, 0.1)},
            {'name': 'learning_rate', 'type': 'continuous', 'domain': (0.0001, 0.01)},
            {'name': 'optimizer', 'type': 'discrete', 'domain': [1, 2, 3]},
            {'name': 'activation', 'type': 'discrete', 'domain': [1, 2, 3, 4]}
        ]
    }

    PSO_CONFIG = {
        'num_particles': [50]
    }

    PSO_BNN_CONFIG = {
        'num_particles': [50, 200]
    }

    BNN_CONFIG = {
        'sliding_encoder': [12, 24, 36, 48],
        'sliding_inference': [3, 4, 5, 6],
        'batch_size': [8],
        'num_units_lstm': [[4], [8, 4]],
        'num_units_inference': [[4], [8, 4]],
        'dropout_rate': [0.9],
        'variation_dropout': False,
        'activation': ['tanh'],
        'optimizer': ['momentum'],
        'variant': [False, True]
    }

# 'sliding_encoder': [10, 12],
# 'sliding_inference': [2, 3, 4, 5],
# 'batch_size': [8, 32, 128],
# 'num_units_lstm': [[32, 4], [8]],
# 'num_units_inference': [[4], [16, 4]],
# 'dropout_rate': [0.9],
# 'variation_dropout': False,
# 'activation': ['relu', 'elu'],
# 'optimizer': ['momentum', 'adam', 'rmsprop']
