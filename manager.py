import os
import tensorflow as tf
import time
import numpy as np
import pickle as pk
from multiprocessing import Pool
from queue import Queue
from sklearn.model_selection import ParameterGrid, train_test_split
from sklearn import datasets
import pandas as pd
import matplotlib

from config import *
from lib.scaler.model_training import ModelTrainer
from lib.data_visualization.read_grid_data import *
from lib.data_visualization.visualize import *
from lib.preprocess.read_data import DataReader
matplotlib.use(Config.PLT_ENV)
import matplotlib.pyplot as plt


def init_model():
    print('[1] >>> Start init model')
    data_reader = DataReader()
    normalized_data, scaler = data_reader.read_data()
    model_trainer = ModelTrainer(normalized_data, scaler)
    model_trainer.train()
    print('[1] >>> Init model complete')


if __name__ == "__main__":
    init_model()
