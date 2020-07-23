import time

import matplotlib
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import mean_absolute_error as MAE
from sklearn.metrics import mean_squared_error as MSE

from lib.scaler.preprocessing_data.data_preprocessor import DataPreprocessor
from lib.scaler.models.base_model import RegressionModel
from lib.scaler.nets.base_net import RnnNet
from lib.evolution_algorithms.pso import *
from config import *
from lib.includes.utility import *

matplotlib.use(Config.PLT_ENV)
import matplotlib.pyplot as plt


class LstmPredictor(RegressionModel):
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

        params = {
            'num_units': num_units,
            'activation': activation,
            'dropout': dropout,
            'cell_type': cell_type
        }

        net = RnnNet(params, 'rnn_net', mode='predictor')
        optimizer = get_optimizer(optimizer, learning_rate)

        super().__init__(net, input_shape, output_shape, optimizer, model_path)

        self.batch_size = batch_size
        self.epochs = Config.EPOCHS
        self.early_stopping = Config.EARLY_STOPPING
        self.patience = Config.PATIENCE
