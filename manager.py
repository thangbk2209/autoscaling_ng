import os
import sys
from subprocess import Popen, PIPE
import time
import pickle as pkl

import tensorflow as tf
import numpy as np
import multiprocessing
from multiprocessing import Pool
from queue import Queue
from sklearn.model_selection import ParameterGrid, train_test_split
from sklearn import datasets
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

from config import *
from lib.scaler.model_training import ModelTrainer
from lib.data_visualization.read_grid_data import *
from lib.data_visualization.visualize import *
from lib.preprocess.read_data import DataReader
from lib.scaler.preprocessing_data.data_preprocessor import DataPreprocessor
from lib.evaluation.error_metrics import *
# from lib.evaluation.model_evaluator import ModelEvaluator


def init_model():
    print('[1] >>> Start init model')
    print(f'=== model information: {Config.RESULTS_SAVE_PATH}')
    model_trainer = ModelTrainer()
    model_trainer.train()
    print('[1] >>> Init model complete')


# def evaluate_model():

#     try:
#         iteration = sys.argv[2]
#         value_optimize = sys.argv[3]
#     except Exception as ex:
#         value_optimize = ''
#         print('[ERROR] Can not define your iteration')

#     model_evaluator = ModelEvaluator()
#     preprocess_item = None
#     if value_optimize == 'hyper_parameter':
#         preprocess_item = {
#             'scaler': 'min_max_scaler',
#             'sliding_encoder': 6,
#             'sliding_decoder': 4,
#             'sliding_inf': 2
#         }

#     model_evaluator.evaluate_bnn(iteration, preprocess_item, visualize_option=True)


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
    if sys.argv[1] == 'training':
        init_model()
    elif sys.argv[1] == 'evaluate':
        evaluate_model()
    else:
        print(f'[ERROR] Not support: {sys.argv[1]}')
    # download()
    # evaluate_model()
