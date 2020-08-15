import matplotlib
import numpy as np
import pandas as pd
import tensorflow as tf
from pandas import read_csv
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error as MAE
from sklearn.metrics import mean_squared_error as MSE
from tensorflow.contrib import rnn
import time

from config import *
from lib.scaler.preprocessing_data.data_preprocessor import DataPreprocessor
from lib.evolution_algorithms.pso import *
from lib.scaler.nets.base_net import MlpNet
from lib.scaler.models.base_model import RegressionModel, BaseModel
from lib.evaluation.error_metrics import evaluate

matplotlib.use(Config.PLT_ENV)
import matplotlib.pyplot as plt


class BnnPredictor(BaseModel):
    def __init__(
        self,
        model_path=None,
        preprocess_name=None,
        pretrained_encoder_net=None,
        encoder_input_shape=None,
        inf_input_shape=None,
        output_shape=None,
        batch_size=None,
        num_units=None,
        activation=None,
        optimizer=None,
        dropout=None,
        cell_type=None,
        learning_rate=None,
        initial_state=True
    ):
        self.preprocess_name = preprocess_name
        self.num_units = num_units
        self.activation = activation
        self.dropout = dropout
        self.cell_type = cell_type
        self.pretrained_encoder_net = pretrained_encoder_net
        self.optimizer = optimizer
        self.learning_rate = learning_rate

        self.encoder_input_shape = encoder_input_shape
        self.inf_input_shape = inf_input_shape
        self.output_shape = output_shape
        self.cell_type = cell_type
        self.batch_size = batch_size
        self.epochs = Config.EPOCHS
        self.early_stopping = Config.EARLY_STOPPING
        self.patience = Config.PATIENCE

        super().__init__(model_path, initial_state)

    def initiate_state(self):
        params = {
            'num_units': self.num_units,
            'activation': self.activation,
            'dropout': self.dropout,
            'cell_type': self.cell_type
        }

        self.encoder_net = self.pretrained_encoder_net
        self.inf_net = MlpNet(params, 'inf_net')

        self.optimizer = get_optimizer(self.optimizer, self.learning_rate)

        tf.reset_default_graph()
        self.sess = tf.Session()

        self._build_model()
        init = tf.global_variables_initializer()
        self.saver = tf.train.Saver()
        self.sess.run(init)

    def _build_model(self):
        self._x_encoder = tf.placeholder(tf.float32, [None] + self.encoder_input_shape, 'x_encoder')

        self._output_encoder, self._hidden_state = self.encoder_net(self._x_encoder)
        if self.cell_type == 'lstm':
            self._hidden_state_vector = self._hidden_state[-1].h
        else:
            self._hidden_state_vector = self._hidden_state[-1]
        self._x_inf = tf.placeholder(tf.float32, [None] + self.inf_input_shape, 'x_inf')
        self._input_inf = input_inference = tf.concat([self._x_inf, self._hidden_state_vector], 1)
        self._pred = self.inf_net(self._input_inf)
        self._y = tf.placeholder(tf.float32, self._pred.shape, 'y')

        self._loss = tf.losses.mean_squared_error(self._y, self._pred)
        self._train_op = self.optimizer.minimize(self._loss)

    def load_model(self):
        metadata = '{}.meta'.format(self.model_path)
        model_graph_dir = self.model_path.rsplit(os.sep, 1)[0]

        # assert os.path.isfile(metadata), 'Not found classifier graph in path: {}'.format(metadata)
        if not os.path.isfile(metadata):
            print('Not found classifier graph in path: {}'.format(metadata))
            return
        # run on its own graph session
        self.graph = tf.Graph()
        self.sess = tf.Session(graph=self.graph)
        with self.graph.as_default():

            checkpoint = tf.train.import_meta_graph(metadata)
            checkpoint.restore(self.sess, tf.train.latest_checkpoint(model_graph_dir))
            f = open('nodes.txt', 'a+')
            for v in self.graph.as_graph_def().node:
                f.write(v.name + '\n')
            self._x_encoder = self.graph.get_tensor_by_name('x_encoder:0')
            self._x_inf = self.graph.get_tensor_by_name('x_inf:0')
            self._pred = self.graph.get_tensor_by_name('inf_net/prediction/Sigmoid:0')

    def _step(self, x_encoder, x_inf, y=None, mode='predict'):
        if mode == 'predict':
            return self.sess.run(self._pred, feed_dict={self._x_encoder: x_encoder, self._x_inf: x_inf})
        elif mode == 'train':
            l, _ = self.sess.run(
                [self._loss, self._train_op], feed_dict={self._x_encoder: x_encoder, self._x_inf: x_inf, self._y: y})
            return l

    def _step_batch(self, x_encoder, x_inf, y=None, batch_size=1, mode='predict'):
        n_batch = math.ceil(len(x_encoder) / batch_size)
        results = []
        for batch in range(n_batch):
            x_e_b = x_encoder[batch * batch_size: (batch + 1) * batch_size]
            x_i_b = x_inf[batch * batch_size: (batch + 1) * batch_size]
            if y is not None:
                y_b = y[batch * batch_size: (batch + 1) * batch_size]
            else:
                y_b = None
            res = self._step(x_e_b, x_i_b, y_b, mode)
            results.append(res)
        if mode == 'train':
            return results
        if mode == 'predict':
            return np.concatenate(results, axis=0)

    def fit(self, x_encoder, x_inf, y, validation_split=0, batch_size=1, epochs=2000, verbose=1, step_print=1,
            early_stopping=True, patience=20):

        # Create x_train and y_train
        do_validation = False
        if 0 < validation_split < 1:
            do_validation = True
            n_train = int((1 - validation_split) * len(x_encoder))
            x_encoder_valid = x_encoder[n_train:]
            x_inf_valid = x_inf[n_train:]
            y_valid = y[n_train:]
            x_encoder_train = x_encoder[:n_train]
            x_inf_train = x_inf[:n_train]
            y_train = y[:n_train]
        else:
            x_encoder_train = x_encoder
            x_inf_train = x_inf
            y_train = y

        # Start optimization process
        self.train_loss_arr = []
        self.valid_loss_arr = []
        for epoch in range(epochs):
            _train_loss = self._step_batch(x_encoder_train, x_inf_train, y_train, batch_size, mode='train')
            avg_loss_train = average(_train_loss)
            self.train_loss_arr.append(avg_loss_train)
            if (epoch + 1) % step_print == 0:
                validation_state = ''
                if do_validation:
                    result_eval = self.evaluate(x_encoder_valid, x_inf_valid, y_valid)
                    validation_state = ', validation - {}'.format(result_eval['mse'])
                    self.valid_loss_arr.append(result_eval['mse'])
                # print('Epoch {}/{}'.format(epoch + 1, epochs), end=': ')
                # print(f'mean_squared_error inf: training - {round(avg_loss_train, 7)}{validation_state}')

            if early_stopping:
                if epoch > patience:
                    if not early_stopping_decision(self.train_loss_arr, self.patience):
                        print('|>>> Early stopping training ...')
                        break

    def predict(self, x_encoder, x_inf):
        return self._step_batch(x_encoder, x_inf, mode='predict', batch_size=len(x_encoder))

    def evaluate(self, x_encoder, x_inf, y):
        pred = self.predict(x_encoder, x_inf)
        return evaluate(y, pred, ('mae', 'rmse', 'mse', 'mape', 'smape'))
