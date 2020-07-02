import tensorflow as tf
import numpy as np
import multiprocessing
from multiprocessing import Pool
from queue import Queue
from sklearn.model_selection import ParameterGrid, train_test_split
from sklearn import datasets

from config import *
from lib.scaler.preprocessing_data import DataPreprocessor
from lib.scaler.models.lstm_model import LstmPredictor
from lib.scaler.models.ann_model import AnnPredictor
from lib.scaler.models.bnn.bnn_model import BnnPredictor


class ModelTrainer:
    def __init__(self, data, scaler):
        self.data = data
        self.scaler = scaler
        self.lstm_config = Config.LSTM_CONFIG
        self.ann_config = Config.ANN_CONFIG
        self.pso_config = Config.PSO_CONFIG
        self.pso_bnn_config = Config.PSO_BNN_CONFIG
        self.bnn_config = Config.BNN_CONFIG
        self.method_experimet = Config.MODEL_EXPERIMENT
        self.learning_rate = Config.LEARNING_RATE
        self.epochs = Config.EPOCHS
        self.early_stopping = Config.EARLY_STOPPING
        self.patience = Config.PATIENCE
        self.train_size = Config.TRAIN_SIZE
        self.valid_size = Config.VALID_SIZE
        self.optimizer_approach = Config.METHOD_APPROACH
        self.model_save_path = Config.MODEL_SAVE_PATH
        self.results_save_path = Config.RESULTS_SAVE_PATH

        self.train_loss_path = Config.TRAIN_LOSS_PATH
        self.evaluation_path = Config.EVALUATION_PATH

    def fit_with_ann(self, item):
        print('>>> start experiment with pool <<<')
        sliding = item["sliding"]
        batch_size = item["batch_size"]
        num_units = item["num_unit"]
        activation = item["activation"]
        optimizer = item["optimizer"]
        num_particle = item["num_particle"]
        model = AnnPredictor(self.data, self.scaler, sliding=sliding, batch_size=batch_size, num_units=num_units,
                             activation=activation, optimizer=optimizer, optimizer_approach=self.optimizer_approach,
                             learning_rate=self.learning_rate, epochs=self.epochs, early_stopping=self.early_stopping,
                             patience=self.patience, model_save_path=self.model_save_path,
                             results_save_path=self.results_save_path, train_size=self.train_size,
                             valid_size=self.valid_size, num_particle=num_particle,
                             train_loss_path=self.train_loss_path, evaluation_path=self.evaluation_path)
        model.fit()

    def train_with_ann(self):
        param_grid = {
            'sliding': self.ann_config['sliding'],
            'batch_size': self.ann_config['batch_size'],
            'num_unit': self.ann_config['num_units'],
            'activation': self.ann_config['activation'],
            'optimizer': self.ann_config['optimizers'],
            'num_particle': self.pso_config['num_particles']
        }
        queue = Queue()
        for item in list(ParameterGrid(param_grid)):
            queue.put_nowait(item)
        summary = open(self.evaluation_path, 'a+')
        summary.write('Model, MAE, RMSE\n')
        print('>>> start experiment <<<')
        pool = Pool(1)
        pool.map(self.fit_with_ann, list(queue.queue))
        pool.close()
        pool.join()
        pool.terminate()

    def fit_with_lstm(self, item):
        print('>>> start experiment with pool <<<')
        sliding = item["sliding"]
        batch_size = item["batch_size"]
        num_units = item["num_unit"]
        dropout_rate = item["dropout_rate"]
        variation_dropout = self.lstm_config['variation_dropout']
        activation = item["activation"]
        optimizer = item["optimizer"]
        num_particle = item["num_particle"]
        model = LstmPredictor(self.data, self.scaler, sliding=sliding, batch_size=batch_size, num_units=num_units,
                              dropout_rate=dropout_rate, variation_dropout=variation_dropout, activation=activation,
                              optimizer=optimizer, optimizer_approach=self.optimizer_approach,
                              learning_rate=self.learning_rate, epochs=self.epochs, early_stopping=self.early_stopping,
                              patience=self.patience, model_save_path=self.model_save_path,
                              results_save_path=self.results_save_path, train_size=self.train_size,
                              valid_size=self.valid_size, num_particle=num_particle,
                              train_loss_path=self.train_loss_path, evaluation_path=self.evaluation_path)
        model.fit()

    def train_with_lstm(self):
        param_grid = {
            'sliding': self.lstm_config['sliding'],
            'batch_size': self.lstm_config['batch_size'],
            'num_unit': self.lstm_config['num_units'],
            'dropout_rate': self.lstm_config['dropout_rate'],
            'activation': self.lstm_config['activation'],
            'optimizer': self.lstm_config['optimizers'],
            'num_particle': self.pso_config['num_particles']
        }
        queue = Queue()
        for item in list(ParameterGrid(param_grid)):
            queue.put_nowait(item)
        summary = open(self.evaluation_path, 'a+')
        summary.write('Model, MAE, RMSE, r2\n')
        print('>>> start experiment <<<')
        pool = Pool(1)
        pool.map(self.fit_with_lstm, list(queue.queue))
        pool.close()
        pool.join()
        pool.terminate()

    def fit_with_bnn(self, item):
        print('>>> start experiment bnn with pool <<<')
        sliding_encoder = item['sliding_encoder']
        sliding_inference = item['sliding_inference']
        batch_size = item['batch_size']
        num_units_lstm = item['num_units_lstm']
        num_units_inference = item['num_units_inference']
        dropout_rate = item['dropout_rate']
        variation_dropout = self.lstm_config['variation_dropout']
        activation = item['activation']
        optimizer = item['optimizer']
        variant = item['variant']
        num_particle = item['num_particle']

        model = BnnPredictor(self.data, self.scaler, train_size=self.train_size, valid_size=self.valid_size,
                             sliding_encoder=sliding_encoder, sliding_inference=sliding_inference,
                             batch_size=batch_size, num_units_lstm=num_units_lstm,
                             num_units_inference=num_units_inference, dropout_rate=dropout_rate,
                             variation_dropout=variation_dropout, activation=activation, optimizer=optimizer,
                             variant=variant, optimizer_approach=self.optimizer_approach,
                             learning_rate=self.learning_rate, epochs=self.epochs, patience=self.patience,
                             num_particle=num_particle)
        model.fit()

    def train_with_bnn(self):
        param_grid = {
            'sliding_encoder': self.bnn_config['sliding_encoder'],
            'sliding_inference': self.bnn_config['sliding_inference'],
            'batch_size': self.bnn_config['batch_size'],
            'num_units_lstm': self.bnn_config['num_units_lstm'],
            'num_units_inference': self.bnn_config['num_units_inference'],
            'dropout_rate': self.bnn_config['dropout_rate'],
            'activation': self.bnn_config['activation'],
            'optimizer': self.bnn_config['optimizer'],
            'variant': self.bnn_config['variant'],
            'num_particle': self.pso_bnn_config['num_particles']
        }
        queue = Queue()
        for item in list(ParameterGrid(param_grid)):
            self.fit_with_bnn(item)
        #     queue.put_nowait(item)
        # print('>>> start experiment <<<')
        # pool = Pool(3)
        # pool.map(self.fit_with_bnn, list(queue.queue))
        # pool.close()
        # pool.join()
        # pool.terminate()

    def train(self):
        print('[3] >>> Start choosing model and experiment')
        if Config.MODEL_EXPERIMENT.lower() == 'bnn':
            print(' >>> Choose bnn model <<<')
            self.train_with_bnn()
        elif Config.MODEL_EXPERIMENT.lower() == 'lstm':
            self.train_with_lstm()
        elif Config.MODEL_EXPERIMENT.lower() == 'ann':
            self.train_with_ann()
        else:
            print('>>> Can not experiment your method <<<')
        print('[3] >>> Choosing model and experiment complete')
