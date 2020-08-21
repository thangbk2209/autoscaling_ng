import time

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from pandas import read_csv
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error as MAE
from sklearn.metrics import mean_squared_error as MSE
import matplotlib
import matplotlib.pyplot as plt

from config import *
from lib.scaler.preprocessing_data.data_preprocessor import DataPreprocessor
from lib.evolution_algorithms.pso import *
from lib.scaler.models.base_model import BaseModel
from lib.evaluation.error_metrics import evaluate


class BnnPredictor(BaseModel):
    def __init__(self,
                 model_path=None,
                 pretrained_encoder_net=None,
                 encoder_input_shape=None,
                 output_shape=None,
                 batch_size=None,
                 num_units=None,
                 activation=None,
                 optimizer=None,
                 dropout=None,
                 learning_rate=None,
                 initial_state=True):

        self.pretrained_encoder_net = pretrained_encoder_net
        self.encoder_input_shape = encoder_input_shape
        self.output_shape = output_shape
        self.batch_size = batch_size
        self.num_units = num_units
        self.activation = activation
        self.optimizer = optimizer
        self.dropout = dropout
        self.learning_rate = learning_rate
        super().__init__(model_path, initial_state)

    def save_model(self, model_saved_path=None):
        if model_saved_path is None:
            # Save model by self.model_path
            self.model.save(self.model_path)
        else:
            # Save model by model_saved_path
            self.model.save(model_saved_path)

    def _build_model(self):
        self.encoder_input_layer = Input(shape=self.encoder_input_shape)
        self.encoder_state = self.pretrained_encoder_net(self.encoder_input_layer)
        for i, _num_unit in enumerate(self.num_units):
            self.hidden_state = Dense(_num_unit, activation=self.activation, kernel_initializer='he_normal')(self.encoder_state)
            self.hidden_state = Dropout(self.dropout)(self.hidden_state)
        self.output = Dense(1)(self.hidden_state)
        self.model = Model(self.encoder_input_layer, self.output)
        self.model.compile(optimizer=self.optimizer, loss='mse')
        print(self.model.summary())
        # print('=------------------=')
        # print(self.encoder_state)
        # print('=========')
        # print(self.pretrained_encoder_net.summary())
        # exit(0)

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
