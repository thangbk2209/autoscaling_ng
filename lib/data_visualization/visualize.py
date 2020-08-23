"""
  Author:  thangbk2209
  Project: Autoscaling
  Created: 3/15/19 11:00
  Purpose:
"""
import matplotlib
import matplotlib.pyplot as plt

from config import *
from lib.includes.utility import *


def visualize(data, data_experiment, metric):
    file_name = PROJECT_DIR + '/{}/{}/{}/{}'.format('data', 'visualization', data_experiment, metric)
    title = data_experiment
    x_label = 'Time stamp'
    y_label = metric
    draw_time_series(data, title, x_label, y_label, file_name)


def visualize_data():
    official_data = read_data()
    data2visualize = official_data[Config.VISUALIZATION_CONFIG['metrics'][Config.DATA_EXPERIMENT]]
    print('>>> Start visualization <<<')
    visualize(data2visualize, Config.DATA_EXPERIMENT, Config.VISUALIZATION_CONFIG['metrics'][Config.DATA_EXPERIMENT])
    print('>>> Visualize complete <<<')
