import os
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from pandas import read_csv
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.contrib import rnn
import time

from config import *
from lib.scaler.preprocessing_data.data_preprocessor import DataPreprocessor
from lib.scaler.models.base_model import UnsupervisedPretrainModel
from lib.scaler.nets.base_net import RnnNet
from lib.evolution_algorithms.pso import *

matplotlib.use(Config.PLT_ENV)


class RnnAutoEncoder(UnsupervisedPretrainModel):
    def __init__(
        self,
        model_path=None,
        encoder_input_shape=None,
        decoder_input_shape=None,
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
        params = {
            'num_units': num_units,
            'activation': activation,
            'dropout': dropout,
            'cell_type': cell_type
        }

        # encoder_net
        encoder_net = RnnNet(params, 'encoder_net', mode='pretrain')
        # decoder_net
        decoder_net = RnnNet(params, 'decoder_net', mode='pretrain')
        optimizer = get_optimizer(optimizer, learning_rate)

        super().__init__(
            encoder_net, decoder_net, encoder_input_shape, decoder_input_shape, output_shape, optimizer, model_path,
            initial_state)

        self.batch_size = batch_size
        self.epochs = Config.EPOCHS
        self.early_stopping = Config.EARLY_STOPPING
        self.patience = Config.PATIENCE
