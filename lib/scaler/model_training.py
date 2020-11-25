import math
import threading

# import tensorflow as tf
import numpy as np
import multiprocessing
from multiprocessing import Pool
from queue import Queue
from sklearn.model_selection import ParameterGrid, train_test_split
from sklearn import datasets
from sklearn.metrics import mean_squared_error
import matplotlib
import matplotlib.pyplot as plt

from config import *
from lib.preprocess.read_data import DataReader
from lib.scaler.preprocessing_data.data_preprocessor import DataPreprocessor
from lib.scaler.models.ann_model import AnnPredictor
from lib.scaler.models.lstm_model import LstmPredictor
from lib.scaler.models.bnn.bnn_model import BnnPredictor
# from lib.scaler.models.gan_model import GanPredictor
from lib.scaler.models.bnn.autoencoder import RnnAutoEncoder
# from lib.scaler.nets.base_net import GeneratorNet, DiscriminatorNet
from lib.includes.utility import *
from lib.evolution_algorithms.pso import Space
from lib.evaluation.fitness_manager import FitnessManager
from lib.scaler.model_loader import ModelLoader


class ModelTrainer:
    def __init__(self):

        self.lstm_config = Config.LSTM_CONFIG
        self.ann_config = Config.ANN_CONFIG
        self.bnn_config = Config.BNN_CONFIG
        self.method_experimet = Config.MODEL_EXPERIMENT

        self.learning_rate = Config.LEARNING_RATE
        self.epochs = Config.EPOCHS
        self.max_iter = Config.MAX_ITER
        self.early_stopping = Config.EARLY_STOPPING
        self.patience = Config.PATIENCE
        self.train_size = Config.TRAIN_SIZE
        self.valid_size = Config.VALID_SIZE
        self.results_save_path = Config.RESULTS_SAVE_PATH
        self.infor_save_path = Config.INFO_SAVED_PATH

        self.data_preprocessor = DataPreprocessor()
        self.model_loader = ModelLoader()

    def fit_with_ann(self, item, fitness_type=None):
        scaler_method = item['scaler']
        sliding = item['sliding']
        batch_size = item['batch_size']
        num_units = generate_units_size(item['network_size'], item['layer_size'])
        activation = item['activation']
        optimizer = item['optimizer']
        dropout = item['dropout']
        learning_rate = item['learning_rate']

        scaler_method = Config.SCALERS[scaler_method - 1]
        activation = Config.ACTIVATIONS[activation - 1]
        optimizer = Config.OPTIMIZERS[optimizer - 1]

        x_train, y_train, x_test, y_test, data_normalizer = \
            self.data_preprocessor.init_data_ann(sliding, scaler_method)

        input_shape = [x_train.shape[1]]
        output_shape = [y_train.shape[1]]
        model_name = create_name(input_shape=input_shape, output_shape=output_shape, batch_size=batch_size,
                                 num_units=num_units, activation=activation, optimizer=optimizer, dropout=dropout,
                                 learning_rate=learning_rate)

        folder_path = f'{self.results_save_path}models'
        gen_folder_in_path(folder_path)
        model_path = f'{folder_path}/{model_name}'

        ann_predictor = AnnPredictor(
            model_path=model_path,
            input_shape=input_shape,
            output_shape=output_shape,
            batch_size=batch_size,
            num_units=num_units,
            activation=activation,
            optimizer=optimizer,
            dropout=dropout,
            learning_rate=learning_rate
        )

        ann_predictor.fit(x_train, y_train, validation_split=Config.VALID_SIZE, batch_size=batch_size, epochs=Config.EPOCHS)

        fitness = ann_predictor.history.history['val_loss'][-1]

        return fitness, ann_predictor

    def train_with_ann(self):
        # item = {
        #     'scaler': 1,
        #     'sliding': 2,
        #     'batch_size': 8,
        #     'network_size': 3,
        #     'layer_size': 8,
        #     'activation': 1,
        #     'optimizer': 1,
        #     'dropout': 0.1,
        #     'learning_rate': 3e-4
        # }
        # fitness, ann_predictor = self.fit_with_ann(item)

        space = Space(self.fit_with_ann, Config.FITNESS_TYPE, Config.ANN_CONFIG['domain'])
        # early_stopping = False
        gbest_particle = space.optimize(Config.MAX_ITER)
        # queue = Queue()
        # for item in list(ParameterGrid(param_grid)):
            # queue.put_nowait(item)
        # summary = open(self.evaluation_path, 'a+')
        # summary.write('Model, MAE, RMSE\n')
        # print('>>> start experiment ANN model <<<')
        # pool = Pool(1)
        # pool.map(self.fit_with_ann, list(queue.queue))
        # pool.close()
        # pool.join()
        # pool.terminate()

    def fit_with_lstm(self, item, fitness_type=None):

        scaler_method = item['scaler']
        sliding = item['sliding']
        batch_size = item['batch_size']

        num_units = generate_units_size(item['network_size'], item['layer_size'])

        activation = item['activation']
        optimizer = item['optimizer']
        dropout = item['dropout']
        learning_rate = item['learning_rate']

        if type(scaler_method) == int and type(activation) == int and type(optimizer) == int:
            scaler_method = Config.SCALERS[scaler_method - 1]
            activation = Config.ACTIVATIONS[activation - 1]
            optimizer = Config.OPTIMIZERS[optimizer - 1]

        x_train, y_train, x_test, y_test, data_normalizer = \
            self.data_preprocessor.init_data_lstm(sliding, scaler_method)

        input_shape = [x_train.shape[1], x_train.shape[2]]
        output_shape = [y_train.shape[1]]
        model_name = create_name(input_shape=input_shape, output_shape=output_shape, batch_size=batch_size,
                                 num_units=num_units, activation=activation, optimizer=optimizer, dropout=dropout,
                                 learning_rate=learning_rate)
        folder_path = f'{self.results_save_path}models'
        gen_folder_in_path(folder_path)
        model_path = f'{folder_path}/{model_name}'

        lstm_predictor = LstmPredictor(
            model_path=model_path,
            input_shape=input_shape,
            output_shape=output_shape,
            batch_size=batch_size,
            num_units=num_units,
            activation=activation,
            optimizer=optimizer,
            dropout=dropout,
            learning_rate=learning_rate
        )

        lstm_predictor.fit(x_train, y_train, validation_split=Config.VALID_SIZE, batch_size=batch_size, epochs=Config.EPOCHS)
        if fitness_type == 'validation_error':
            fitness = lstm_predictor.history.history['val_loss'][-1]
        elif fitness_type == 'bayesian_autoscaling':
            n_train = int((1 - Config.VALID_SIZE) * len(x_train))
            x_valid = x_train[n_train:]
            y_valid = y_train[n_train:]
            fitness_manager = FitnessManager()
            fitness = fitness_manager.evaluate_fitness_scaling(
                lstm_predictor, data_normalizer, x_valid, y_valid)
        else:
            print(f'[ERROR] Do not support {fitness_type}')

        return fitness, lstm_predictor

    def train_with_lstm(self):
        # item = {
        #     'scaler': 'min_max_scaler',
        #     'sliding': 4,
        #     'batch_size': 32,
        #     'network_size': 2,
        #     'layer_size': 32,
        #     'activation': 'tanh',
        #     'optimizer': 'adam',
        #     'dropout': 0.5,
        #     'learning_rate': 3e-4
        # }
        # self.fit_with_lstm(item, fitness_type='bayesian_autoscaling')
        space = Space(self.fit_with_lstm, Config.FITNESS_TYPE, Config.LSTM_CONFIG['domain'])
        gbest_particle = space.optimize(Config.MAX_ITER)

    def fit_with_autoencoder(self, item):
        scaler_method = item['scaler_method']
        autoencoder_model_path = item['autoencoder_model_path']
        encoder_input_shape = item['encoder_input_shape']
        decoder_input_shape = item['decoder_input_shape']
        output_decoder_shape = item['output_decoder_shape']
        batch_size = item['batch_size']
        num_unit_autoencoder = item['num_unit_autoencoder']
        dropout = item['dropout']
        activation = item['activation']
        optimizer = item['optimizer']
        learning_rate = item['learning_rate']
        cell_type = item['cell_type']
        initial_state = True

        x_train_encoder = item['x_train_encoder']
        x_train_decoder = item['x_train_decoder']
        y_train_decoder = item['y_train_decoder']

        autoencoder_model = RnnAutoEncoder(
            model_path=autoencoder_model_path,
            encoder_input_shape=encoder_input_shape,
            decoder_input_shape=decoder_input_shape,
            output_shape=output_decoder_shape,
            batch_size=batch_size,
            num_units=num_unit_autoencoder,
            activation=activation,
            optimizer=optimizer,
            dropout=dropout,
            cell_type=cell_type,
            learning_rate=learning_rate,
            initial_state=initial_state
        )

        validation_split = 0.2

        autoencoder_model.fit(
            x_train_encoder, x_train_decoder, y_train_decoder, validation_split=validation_split, batch_size=batch_size,
            epochs=self.epochs)
        return autoencoder_model

    def fit_with_bnn(self, item, fitness_type):

        rate_real_value_in_prediction_interval = None
        real_scale_value_error = None
        validation_error = None

        if 'scaler' in item and 'sliding_encoder' in item and 'sliding_decoder' in item:
            scaler_method = item['scaler']
            sliding_encoder = item['sliding_encoder']
            sliding_decoder = item['sliding_decoder']
        else:
            scaler_method = self.scaler
            sliding_encoder = self.sliding_encoder
            sliding_decoder = self.sliding_decoder

        # Generate autoencoder units
        network_size_encoder = item['network_size_encoder']
        layer_size_encoder = item['layer_size_encoder']
        num_unit_autoencoder = generate_units_size(network_size_encoder, layer_size_encoder)

        # Generate inference units
        network_size_inf = item['network_size_inf']
        layer_size_inf = item['layer_size_inf']
        num_unit_inf = generate_units_size(network_size_inf, layer_size_inf)

        dropout = item['dropout']
        activation = item['activation']
        optimizer = item['optimizer']
        learning_rate = item['learning_rate']
        cell_type = item['cell_type']
        batch_size = item['batch_size']

        if 'save_mode' in item:
            save_mode = item['save_mode']
        else:
            save_mode = None

        if type(scaler_method) == int and type(activation) == int and type(optimizer) == int:
            scaler_method = Config.SCALERS[scaler_method - 1]
            activation = Config.ACTIVATIONS[activation - 1]
            optimizer = Config.OPTIMIZERS[optimizer - 1]
            cell_type = Config.BNN_CONFIG['cell_type'][cell_type - 1]

        x_train_encoder, x_train_decoder, y_train_decoder, y_train, x_test_encoder, y_test, data_normalizer = \
            self.data_preprocessor.init_data_bnn(sliding_encoder, sliding_decoder, scaler_method)

        encoder_input_shape = [x_train_encoder.shape[1], x_train_encoder.shape[2]]
        decoder_input_shape = [x_train_decoder.shape[1], x_train_decoder.shape[2]]
        output_decoder_shape = [y_train_decoder.shape[1], y_train_decoder.shape[2]]
        output_shape = [y_train.shape[1]]

        autoencoder_model_name = create_name(
            scaler=scaler_method, sli_enc=sliding_encoder, sli_dec=sliding_decoder, batch=batch_size,
            unit=num_unit_autoencoder, drop=dropout, act=activation, opt=optimizer, l_r=learning_rate, cell=cell_type)

        folder_path = f'{self.results_save_path}models/autoencoder'
        gen_folder_in_path(folder_path)
        autoencoder_model_path = f'{folder_path}/{autoencoder_model_name}'
        # autoencoder_model_path = f'{self.results_save_path}{autoencoder_model_name}'

        autoencoder_model = self.model_loader.load_autoencoder(
            autoencoder_model_path, encoder_input_shape, decoder_input_shape, output_decoder_shape)

        if autoencoder_model is None:
            item_autoencoder = {
                'scaler_method': scaler_method,
                'autoencoder_model_path': autoencoder_model_path,
                'encoder_input_shape': encoder_input_shape,
                'decoder_input_shape': decoder_input_shape,
                'output_decoder_shape': output_decoder_shape,
                'batch_size': batch_size,
                'num_unit_autoencoder': num_unit_autoencoder,
                'dropout': dropout,
                'activation': activation,
                'optimizer': optimizer,
                'learning_rate': learning_rate,
                'cell_type': cell_type,
                'x_train_encoder': x_train_encoder,
                'x_train_decoder': x_train_decoder,
                'y_train_decoder': y_train_decoder
            }

            autoencoder_model = self.fit_with_autoencoder(item_autoencoder)

        pretrained_encoder_net = autoencoder_model._encoder_model

        bnn_model_name = create_name(
            scaler=scaler_method, sli_enc=sliding_encoder, sli_dec=sliding_decoder, batch=batch_size,
            unit_ae=num_unit_autoencoder, unit_inf=num_unit_inf, drop=dropout, act=activation,
            opt=optimizer, l_r=learning_rate, cell=cell_type)

        bnn_model_path = f'{self.results_save_path}/{bnn_model_name}'

        bnn_model = BnnPredictor(
            model_path=bnn_model_path,
            pretrained_encoder_net=pretrained_encoder_net,
            encoder_input_shape=encoder_input_shape,
            output_shape=output_shape,
            batch_size=batch_size,
            num_units=num_unit_inf,
            activation=activation,
            optimizer=optimizer,
            dropout=dropout,
            learning_rate=learning_rate
        )

        bnn_model.fit(
            x_train_encoder, y_train, validation_split=Config.VALID_SIZE, batch_size=batch_size, epochs=self.epochs)

        if save_mode:
            gen_folder_in_path(autoencoder_model_path)
            autoencoder_model.save_model()
            gen_folder_in_path(bnn_model_path)
            bnn_model.save_model()

        n_train = int((1 - Config.VALID_SIZE) * len(x_train_encoder))
        x_valid_encoder = x_train_encoder[n_train:]
        y_valid = y_train[n_train:]

        fitness_manager = FitnessManager()

        if fitness_type == 'bayesian_autoscaling':
            fitness_value = fitness_manager.evaluate_fitness_bayesian(
                bnn_model, data_normalizer, x_valid_encoder, y_valid)
        elif fitness_type == 'validation_error':
            fitness_value = fitness_manager.evaluate_fitness_validation(
                bnn_model, data_normalizer, x_valid_encoder, y_valid)

        return fitness_value, bnn_model

    def _train_with_bnn(self, item):
        self.sliding_encoder = item['sliding_encoder']
        self.sliding_decoder = item['sliding_decoder']
        self.sliding_inf = item['sliding_inf']
        self.scaler = item['scaler']

        space = Space(self.fit_with_bnn, Config.FITNESS_TYPE, Config.BNN_CONFIG['domain_hyper_parameter'])
        pbest_particle = space.optimize(self.max_iter)

    def train_with_bnn(self):

        if Config.VALUE_OPTIMIZE == 'all_parameter':
            space = Space(self.fit_with_bnn, Config.FITNESS_TYPE, Config.BNN_CONFIG['domain'])
            pbest_particle = space.optimize(self.max_iter)
        elif Config.VALUE_OPTIMIZE == 'hyper_parameter':

            param_grid = {
                'sliding_encoder': Config.BNN_CONFIG['sliding_encoder'],
                'sliding_decoder': Config.BNN_CONFIG['sliding_decoder'],
                'sliding_inf': Config.BNN_CONFIG['sliding_inf'],
                'scaler': Config.BNN_CONFIG['scaler']
            }

            # Create combination of params.
            queue = Queue()
            for item in list(ParameterGrid(param_grid)):
                self._train_with_bnn(item)

            # pool = Pool(2)
            # pool.map(self._train_with_bnn, list(queue.queue))
            # pool.close()
            # pool.join()
            # pool.terminate()

    def train(self):
        print('[3] >>> Start choosing model and experiment')
        if Config.MODEL_EXPERIMENT.lower() == 'ann':
            self.train_with_ann()
        elif Config.MODEL_EXPERIMENT.lower() == 'lstm':
            self.train_with_lstm()
        elif Config.MODEL_EXPERIMENT.lower() == 'bnn':
            self.train_with_bnn()
        elif Config.MODEL_EXPERIMENT.lower() == 'gan':
            pass
        else:
            print('>>> Can not experiment your method <<<')
        print('[3] >>> Choosing model and experiment complete')
