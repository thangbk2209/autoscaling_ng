import os
import time

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import LSTM, GRU
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

    def gen_rnn_layer(self, num_units, activation, dropout=0, return_sequences=False, return_state=False):
        if self.cell_type.lower() == 'lstm':
            return LSTM(
                units=num_units,
                activation=activation,
                recurrent_activation=activation,
                dropout=dropout,
                recurrent_dropout=dropout,
                return_sequences=return_sequences,
                return_state=return_state
            )
        elif self.cell_type.lower() == 'gru':
            return GRU(
                units=num_units,
                activation=activation,
                recurrent_activation=activation,
                dropout=dropout,
                recurrent_dropout=dropout,
                return_sequences=return_sequences,
                return_state=return_state
            )

    def _build_encoder(self):
        self.encoder_input_layer = Input(shape=self.encoder_input_shape)

        if len(self.num_units) == 1:
            rnn_layer = self.gen_rnn_layer(self.num_units[0], self.activation, self.dropout, False, True)
            if self.cell_type.lower() == 'lstm':
                self.output, self.state_h, self.state_c = rnn_layer(self.encoder_input_layer)
                self.encoder_state = [self.state_h, self.state_c]
            elif self.cell_type.lower() == 'gru':
                self.output, self.state_h = rnn_layer(self.encoder_input_layer)
                self.encoder_state = [self.state_h]
        else:
            self.encoder_state = []
            # Layer 1
            rnn_layer = self.gen_rnn_layer(self.num_units[0], self.activation, self.dropout, True, True)
            if self.cell_type.lower() == 'lstm':
                self.encoder_hidden_layer, self.state_h, self.state_c = rnn_layer(self.encoder_input_layer)
                self.encoder_state.append([self.state_h, self.state_c])
            elif self.cell_type.lower() == 'gru':
                self.encoder_hidden_layer, self.state_h = rnn_layer(self.encoder_input_layer)
                self.encoder_state.append([self.state_h])

            # Layer 2 to n-1
            for i in range(1, len(self.num_units) - 1, 1):
                rnn_layer = self.gen_rnn_layer(self.num_units[i], self.activation, self.dropout, True, True)
                if self.cell_type.lower() == 'lstm':
                    self.encoder_hidden_layer, self.state_h, self.state_c = rnn_layer(self.encoder_input_layer)
                    self.encoder_state.append([self.state_h, self.state_c])
                elif self.cell_type.lower() == 'gru':
                    self.encoder_hidden_layer, self.state_h = rnn_layer(self.encoder_input_layer)
                    self.encoder_state.append([self.state_h])

            # layer n
            rnn_layer = self.gen_rnn_layer(self.num_units[-1], self.activation, self.dropout, False, True)
            if self.cell_type.lower() == 'lstm':
                self.output, self.state_h, self.state_c = rnn_layer(self.encoder_input_layer)
                self.encoder_state.append([self.state_h, self.state_c])
            elif self.cell_type.lower() == 'gru':
                self.output, self.state_h = rnn_layer(self.encoder_input_layer)
                self.encoder_state.append([self.state_h])
        self._encoder_model = Model([self.encoder_input_layer], self.state_h)

    def _build_decoder(self):
        self.decoder_input_layer = Input(shape=self.decoder_input_shape)

        if len(self.num_units) == 1:
            rnn_layer = self.gen_rnn_layer(self.num_units[0], self.activation, self.dropout, True, False)
            self.output_decoder = rnn_layer(self.decoder_input_layer, initial_state=self.encoder_state)
        else:
            rnn_layer = self.gen_rnn_layer(self.num_units[0], self.activation, self.dropout, True)
            self.decoder_hidden_layer = rnn_layer(self.decoder_input_layer, initial_state=self.encoder_state[0])
            for i in range(1, len(self.num_units) - 1, 1):
                rnn_layer = self.gen_rnn_layer(self.num_units[i], self.activation, self.dropout, True)
                self.decoder_hidden_layer = rnn_layer(self.decoder_hidden_layer, initial_state=self.encoder_state[i])
            rnn_layer = self.gen_rnn_layer(self.num_units[-1], self.activation, self.dropout, True)
            self.output_decoder = rnn_layer(self.decoder_hidden_layer, initial_state=self.encoder_state[-1])

        self.output = self.output_decoder[:, :, -1]
        self.model = Model([self.encoder_input_layer, self.decoder_input_layer], self.output)

        self.model.compile(optimizer=self.optimizer, loss='mse')

    def _build_model(self):
        self._build_encoder()

        self._build_decoder()
