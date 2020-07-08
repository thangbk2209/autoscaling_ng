
import math
import os

import numpy as np
import tensorflow as tf

from lib.evaluation.error_metrics import evaluate
from lib.includes.utility import *


class BaseModel:
    def __init__(self, model_path):

        self.model_path = model_path

        # if not os.path.exists(self.model_path):
        #     os.mkdir(self.model_path)

        tf.reset_default_graph()
        self.sess = tf.Session()

        self._build_model()
        init = tf.global_variables_initializer()
        self.saver = tf.train.Saver()
        self.sess.run(init)
        # self.saver.save(self.sess, self.model_path)

    def close_session(self):
        self.sess.close()

    def load_model(self):
        print('self.model_path: ', self.model_path)
        # metadata = '{}/model.meta'.format(self.model_path)

        # model_graph_dir = self.model_path
        metadata = '{}.meta'.format(self.model_path)
        model_graph_dir = '/Users/thangnguyen/hust_project/master_course/thesis_implementation/cloud_autoscaling/data/results/ann/cpu_mem/cpu/model'
        # model_graph_dir = model_graph_dir.rsplit(os.sep, 1)[0]
        print('=====')
        print(metadata, model_graph_dir)

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
            self._x = self.graph.get_tensor_by_name('x:0')
            self._pred = self.graph.get_tensor_by_name('mlp_net/prediction/Tanh:0')

    def save_model(self):
        self.saver.save(self.sess, self.model_path)
        # self.close_session()

    def _build_model(self):
        pass

    def _step(self, x, y=None, mode='predict'):
        pass

    def _step_batch(self, x, y=None, batch_size=1, mode='predict'):
        n_batch = math.ceil(len(x) / batch_size)
        results = []
        for batch in range(n_batch):
            x_b = x[batch * batch_size: (batch + 1) * batch_size]
            if y is not None:
                y_b = y[batch * batch_size: (batch + 1) * batch_size]
            else:
                y_b = None
            res = self._step(x_b, y_b, mode)
            results.append(res)
        if mode == 'train':
            return results
        if mode == 'predict':
            return np.concatenate(results, axis=0)

    def fit(self, x, y, validation_split=0, batch_size=1, epochs=1, verbose=1, step_print=1, early_stopping=False, 
            patience=20):

        # Create x_train and y_train
        do_validation = False
        if 0 < validation_split < 1:
            do_validation = True
            n_train = int((1 - validation_split) * len(x))
            x_valid = x[n_train:]
            y_valid = y[n_train:]
            x_train = x[:n_train]
            y_train = y[:n_train]
        else:
            x_train = x
            y_train = y

        # Start optimization process
        self.train_loss_arr = []
        self.valid_loss_arr = []
        for epoch in range(epochs):
            _train_loss = self._step_batch(x_train, y_train, batch_size, mode='train')
            avg_loss_train = average(_train_loss)
            self.train_loss_arr.append(avg_loss_train)
            if (epoch + 1) % step_print == 0:
                validation_state = ''
                if do_validation:
                    result_eval = self.evaluate(x_valid, y_valid)
                    validation_state = ', validation - {}'.format(result_eval['mse'])
                    self.valid_loss_arr.append(result_eval['mse'])
                # print('Epoch {}/{}'.format(epoch + 1, epochs), end=': ')
                # print(f'mean_squared_error: training - {round(avg_loss_train, 7)}{validation_state}')

            if early_stopping:
                if epoch > patience:
                    if not early_stopping(train_loss_arr, self.patience):
                        print('|>>> Early stopping training ...')
                        break

    def predict(self, x):
        return self._step_batch(x, mode='predict', batch_size=len(x))

    def evaluate(self, x, y):
        pred = self.predict(x)
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
