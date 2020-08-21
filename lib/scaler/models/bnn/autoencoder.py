import os
import time

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras import Model
from pandas import read_csv
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

from config import *
from lib.scaler.preprocessing_data.data_preprocessor import DataPreprocessor
from lib.scaler.models.base_model import UnsupervisedPretrainModel
from lib.evolution_algorithms.pso import *


class RnnAutoEncoder(UnsupervisedPretrainModel):
    def __init__(
        self,
        model_path=None,
        encoder_input_shape=None,
        decoder_input_shape=None,
        output_shape=None,
        num_units=None,
        cell_type=None,
        dropout=None,
        activation=None,
        optimizer=None,
        learning_rate=None,
        batch_size=None,
        initial_state=True
    ):
        self.encoder_input_shape = encoder_input_shape
        self.decoder_input_shape = decoder_input_shape
        self.output_shape = output_shape
        self.num_units = num_units
        self.cell_type = cell_type
        self.dropout = dropout
        self.activation = activation
        self.optimizer = optimizer

        super().__init__(model_path)

        self.batch_size = batch_size
        self.epochs = Config.EPOCHS
        self.early_stopping = Config.EARLY_STOPPING
        self.patience = Config.PATIENCE

    def _build_encoder(self):
        self.encoder_input_layer = Input(shape=self.encoder_input_shape)

        if len(self.num_units) == 1:
            self.output, self.state_h, self.state_c = LSTM(
                units=self.num_units[0],
                activation=self.activation,
                recurrent_activation=self.activation,
                dropout=self.dropout,
                recurrent_dropout=self.dropout,
                return_sequences=False,
                return_state=True
            )(self.encoder_input_layer)
            self.encoder_state = [self.state_h, self.state_c]
        else:
            self.encoder_hidden_layer = LSTM(
                units=self.num_units[0],
                activation=self.activation,
                recurrent_activation=self.activation,
                dropout=self.dropout,
                recurrent_dropout=self.dropout,
                return_sequences=True
            )(self.encoder_input_layer)
            for i in range(1, len(self.num_units) - 1, 1):
                self.encoder_hidden_layer = LSTM(
                    units=self.num_units[i],
                    activation=self.activation,
                    recurrent_activation=self.activation,
                    dropout=self.dropout,
                    recurrent_dropout=self.dropout,
                    return_sequences=True
                )(self.encoder_hidden_layer)
            self.output, self.state_h, self.state_c = LSTM(
                units=self.num_units[-1],
                activation=self.activation,
                recurrent_activation=self.activation,
                dropout=self.dropout,
                recurrent_dropout=self.dropout,
                return_sequences=False,
                return_state=True
            )(self.encoder_hidden_layer)
            self.encoder_state = [self.state_h, self.state_c]
        self._encoder_model = Model([self.encoder_input_layer], self.state_h)

    def _build_decoder(self):
        self.decoder_input_layer = Input(shape=self.decoder_input_shape)

        if len(self.num_units) == 1:
            self.output_decoder = LSTM(
                units=self.num_units[0],
                activation=self.activation,
                recurrent_activation=self.activation,
                dropout=self.dropout,
                recurrent_dropout=self.dropout,
                return_sequences=True
            )(self.decoder_input_layer, initial_state=self.encoder_state)
        else:
            self.decoder_hidden_layer = LSTM(
                units=self.num_units[0],
                activation=self.activation,
                recurrent_activation=self.activation,
                dropout=self.dropout,
                recurrent_dropout=self.dropout,
                return_sequences=True
            )(self.decoder_input_layer)
            for i in range(1, len(self.num_units) - 1, 1):
                self.decoder_hidden_layer = LSTM(
                    units=self.num_units[i],
                    activation=self.activation,
                    recurrent_activation=self.activation,
                    dropout=self.dropout,
                    recurrent_dropout=self.dropout,
                    return_sequences=True
                )(self.decoder_hidden_layer)
            self.output_decoder = LSTM(
                units=self.num_units[-1],
                activation=self.activation,
                recurrent_activation=self.activation,
                dropout=self.dropout,
                recurrent_dropout=self.dropout,
                return_sequences=True
            )(self.decoder_hidden_layer, initial_state=self.encoder_state)
            # self.decoder_state = [self.state_h, self.state_c]

        self.output = self.output_decoder[:, :, -1]
        self.model = Model([self.encoder_input_layer, self.decoder_input_layer], self.output)

        self.model.compile(optimizer=self.optimizer, loss='mse')

    def _build_model(self):
        self._build_encoder()

        self._build_decoder()
