import time

import matplotlib
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dense
from sklearn.metrics import mean_absolute_error as MAE
from sklearn.metrics import mean_squared_error as MSE
import matplotlib.pyplot as plt

from lib.scaler.preprocessing_data.data_preprocessor import DataPreprocessor
from lib.scaler.models.base_model import BaseModel
from lib.evolution_algorithms.pso import *
from config import *
from lib.includes.utility import *


class LstmPredictor(BaseModel):
    def __init__(
        self,
        model_path=None,
        input_shape=None,
        output_shape=None,
        batch_size=None,
        num_units=None,
        activation=None,
        optimizer=None,
        dropout=None,
        cell_type=None,
        learning_rate=None
    ):

        self.input_shape = input_shape
        self.output_shape = output_shape
        self.model_path = model_path

        self.num_units = num_units
        self.activation = activation
        self.dropout = dropout
        self.cell_type = cell_type

        self.optimizer = optimizer
        super().__init__(model_path)

        self.batch_size = batch_size
        self.epochs = Config.EPOCHS
        self.early_stopping = Config.EARLY_STOPPING
        self.patience = Config.PATIENCE

    def _build_model(self):
        self.model = Sequential()
        for i, _num_unit in enumerate(self.num_units):
            if not i:
                self.model.add(LSTM(_num_unit, activation=self.activation, recurrent_activation=self.activation,
                                    dropout=self.dropout, recurrent_dropout=self.dropout,
                                    kernel_initializer='he_normal', input_shape=self.input_shape, return_sequences=True))
            else:
                self.model.add(LSTM(_num_unit, activation=self.activation, recurrent_activation=self.activation,
                                    dropout=self.dropout, recurrent_dropout=self.dropout,
                                    kernel_initializer='he_normal'))
        self.model.add(Dense(1))
        self.model.compile(optimizer=self.optimizer, loss='mse')
