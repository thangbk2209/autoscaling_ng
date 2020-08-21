import numpy as np
import matplotlib
from config import *
matplotlib.use(Config.PLT_ENV)
import matplotlib.pyplot as plt


class DataVisualizer:
    def __init__(self):
        pass

    def visualize_prediction(self, actual, prediction, save_path):
        plt.plot(actual, label='Actual')
        plt.plot(prediction, label='Prediction')
        plt.legend()
        # plt.show()
        plt.savefig(f'{save_path}.png')
        plt.savefig(f'{save_path}.pdf')
        plt.close()

    def visualize_fitness(self, fitness, label, save_path):
        for _fitness, _label in zip(fitness, label):
            plt.plot(_fitness, label=_label)
        plt.legend()
        plt.savefig(f'{save_path}.png')
        plt.savefig(f'{save_path}.pdf')
        plt.close()

    def visualize_uncertainty(self, actual, prediction, uncertainty, save_path):

        prediction = np.reshape(prediction, (prediction.shape[0]))
        uncertainty = np.array(uncertainty)
        # uncertainty = np.reshape(uncertainty, (uncertainty.shape[0], 1))
        # print(prediction.shape)
        # print(uncertainty.shape)
        pred_lower = prediction - uncertainty
        pred_upper = prediction + uncertainty
        plt.plot(actual, label='Actual')
        plt.plot(prediction, label='Prediction')
        plt.fill_between(range(prediction.shape[0]), pred_lower, pred_upper, label='Prediction Interval', color='gray')
        plt.legend()
        # plt.show()
        plt.savefig(f'{save_path}.png')
        plt.savefig(f'{save_path}.pdf')
        plt.close()
