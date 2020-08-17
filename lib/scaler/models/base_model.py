
import math
import os

import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import plot_model
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt

from lib.evaluation.error_metrics import evaluate
from lib.includes.utility import *


class BaseModel:
    def __init__(self, model_path):
        self.model_path = model_path
        self._build_model()

    def load_model(self):
        self.model = load_model(self.model_path)

    def save_model(self, model_saved_path=None):
        if model_saved_path is None:
            # Save model by self.model_path
            self.model.save(self.model_path)
        else:
            # Save model by model_saved_path
            self.model.save(model_saved_path)

    def _build_model(self):
        pass

    def get_model_description(self, infor_path):
        try:
            self.model.summary()
            plot_model(self.model, infor_path, show_shapes=True)
        except Exception as ex:
            print('[ERROR] Can not get description of the model')

    def plot_learning_curves(self):
        try:
            # plot learning curves
            plt.title('Learning Curves')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.plot(self.history.history['loss'], label='train')
            plt.plot(self.history.history['val_loss'], label='val')
            plt.legend()
            plt.show()
        except Exception as ex:
            print('[ERROR] Can not plot learning curves of the model')

    def fit(self, x, y, validation_split=0, batch_size=1, epochs=1, verbose=2, early_stopping=False, patience=20):
        callbacks = []
        if early_stopping:
            es = EarlyStopping(monitor='val_loss', patience=patience)
            callbacks = [es]

        self.history = self.model.fit(
            x, y, validation_split=validation_split, epochs=epochs, batch_size=batch_size, verbose=verbose, shuffle=False,
            callbacks=callbacks)

    def predict(self, x):
        return self.model.predict(x)

    def evaluate(self, x, y, data_normalizer):
        pred = self.predict(x)
        pred = data_normalizer.invert_tranform(pred)
        return evaluate(y, pred, ('mae', 'rmse', 'mse', 'mape', 'smape'))


class RegressionModel(BaseModel):
    def __init__(self, net, input_shape, output_shape, optimizer, model_path):
        self.net = net
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.optimizer = optimizer
        super().__init__(model_path)

    def _build_model(self):
        self._x = tf.placeholder(tf.float32, [None] + self.input_shape, 'x')
        self._pred = self.net(self._x)
        self._pred = tf.reshape(self._pred, [-1] + self.output_shape)
        self._y = tf.placeholder(tf.float32, self._pred.shape, 'y')

        self._loss = tf.losses.mean_squared_error(self._y, self._pred)
        # self._loss = tf.reduce_mean(tf.square(self._y - self._pred))
        self._train_op = self.optimizer.minimize(self._loss)

    def _step(self, x, y=None, mode='predict'):
        if mode == 'predict':
            return self.sess.run(self._pred, feed_dict={self._x: x})
        elif mode == 'train':
            l, _ = self.sess.run([self._loss, self._train_op], feed_dict={self._x: x, self._y: y})
            return l


class UnsupervisedPretrainModel(BaseModel):
    def __init__(self, encoder_net, decoder_net, encoder_input_shape, decoder_input_shape, output_shape, optimizer, 
                 model_path, initial_state):
        self.encoder_net = encoder_net
        self.decoder_net = decoder_net
        self.encoder_input_shape = encoder_input_shape
        self.decoder_input_shape = decoder_input_shape
        self.output_shape = output_shape
        self.optimizer = optimizer
        super().__init__(model_path, initial_state)

    def _build_model(self):

        self._x_encoder = tf.placeholder(tf.float32, [None] + self.encoder_input_shape, 'x_encoder')
        self._x_decoder = tf.placeholder(tf.float32, [None] + self.decoder_input_shape, 'x_decoder')
        self._output_encoder, self._embedding_state = self.encoder_net(self._x_encoder)
        self._output_decoder, self._decoder_hidden_state = \
            self.decoder_net(self._x_decoder, initial_state=self._embedding_state)

        self._pred = self._output_decoder[:, :, -1]

        self._y = tf.placeholder(tf.float32, [None] + self.output_shape, 'y')
        self._loss = tf.reduce_mean(tf.square(self._y - self._pred))
        self._train_op = self.optimizer.minimize(self._loss)

    def _step(self, x_encoder, x_decoder=None, y=None, mode='predict'):
        if mode == 'predict':
            return self.sess.run(self._embedding_state, feed_dict={self._x_encoder: x_encoder})
        elif mode == 'train':
            l, _ = self.sess.run([self._loss, self._train_op],
                                 feed_dict={self._x_encoder: x_encoder, self._x_decoder: x_decoder, self._y: y})
            return l

    def _step_batch(self, x_encoder, x_decoder, y=None, batch_size=1, mode='predict'):

        n_batch = math.ceil(len(x_encoder) / batch_size)
        results = []
        for batch in range(n_batch):
            x_e_b = x_encoder[batch * batch_size: (batch + 1) * batch_size]
            x_d_b = x_decoder[batch * batch_size: (batch + 1) * batch_size]
            if y is not None:
                y_b = y[batch * batch_size: (batch + 1) * batch_size]
            else:
                y_b = None
            res = self._step(x_e_b, x_d_b, y_b, mode)
            results.append(res)
        if mode == 'train':
            return results
        if mode == 'predict':
            return np.concatenate(results, axis=0)

    def fit(self, x_encoder, x_decoder, y, validation_split=0, batch_size=1, epochs=2000, verbose=1, step_print=1,
            early_stopping=True, patience=20):

        # Create x_train and y_train
        do_validation = False
        if 0 < validation_split < 1:
            do_validation = True
            n_train = int((1 - validation_split) * len(x_encoder))
            x_valid_encoder = x_encoder[n_train:]
            x_valid_decoder = x_decoder[n_train:]
            y_valid = y[n_train:]
            x_train_encoder = x_encoder[:n_train]
            x_train_decoder = x_decoder[:n_train]
            y_train = y[:n_train]
        else:
            x_train_encoder = x_encoder
            x_train_decoder = x_decoder
            y_train = y

        # Start optimization process
        self.train_loss_arr = []
        self.valid_loss_arr = []
        for epoch in range(epochs):
            _train_loss = self._step_batch(x_train_encoder, x_train_decoder, y_train, batch_size, mode='train')

            avg_loss_train = average(_train_loss)
            self.train_loss_arr.append(avg_loss_train)
            if (epoch + 1) % step_print == 0:
                validation_state = ''
                # if do_validation:
                #     result_eval = self.evaluate(x_valid_encoder, x_valid_decoder, y_valid)
                #     validation_state = ', validation - {}'.format(result_eval['mse'])
                #     self.valid_loss_arr.append(result_eval['mse'])

                # print('Epoch {}/{}'.format(epoch + 1, epochs), end=': ')
                # print(f'mean_squared_error: training - {round(avg_loss_train, 7)}{validation_state}')

            if early_stopping:
                if epoch > patience:
                    if not early_stopping_decision(self.train_loss_arr, self.patience):
                        print('|>>> Early stopping training ...')
                        break

    def predict(self, x_encoder, x_decoder):
        return self._step_batch(x_encoder, x_decoder, mode='predict', batch_size=len(x_encoder))

    def evaluate(self, x_encoder, x_decoder, y):
        pred = self.predict(x_encoder, x_decoder)
        return evaluate(y, pred, ('mae', 'rmse', 'mse', 'mape', 'smape'))