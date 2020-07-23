import os
import sys
from subprocess import Popen, PIPE
import time
import pickle as pk

import tensorflow as tf
import numpy as np
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
    # data_reader = DataReader()
    # normalized_data, scaler = data_reader.read_data()
    model_trainer = ModelTrainer()
    model_trainer.train()
    print('[1] >>> Init model complete')


def download():

    print(' === start_download === ')
    link_path = f'{CORE_DATA_DIR}/input_data/azure/link_v2.txt'
    # curl -o abc.csv.gz https://azurecloudpublicdataset2.blob.core.windows.net/azurepublicdatasetv2/trace_data/vm_cpu_readings/vm_cpu_readings-file-1-of-195.csv.gz
    folder_save_path = '/root/thangbk2209/thesis/datasets/azure'
    # folder_save_path = '/Users/thangnguyen/hust_project/master_course/thesis_implementation/cloud_autoscaling/data/input_data/azure'
    with open(link_path, 'r') as f:
        lines = f.readlines()
        for i, line in enumerate(lines):
            line = line.rstrip('\n')
            file_name = line.split('/')[-1]

            if file_name not in os.listdir(folder_save_path):
                continue

            print(f' === downloading {file_name} === ')
            file_save_path = f'{folder_save_path}/{file_name}'

            process = Popen(
                ['curl', '-o', file_save_path, line], stdout=PIPE, stderr=PIPE)
            stderr, stdout = process.communicate()
            print(f'===> stderr: {stderr}')
            print(f' === download {file_name} complete === ')


if __name__ == "__main__":
    # init_model()
    download()
