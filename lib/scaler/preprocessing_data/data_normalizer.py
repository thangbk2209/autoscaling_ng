import numpy as np

from lib.includes.utility import *


class DataNormalizer:
    def __init__(self, scaler_method=None):
        self.x_scaler = get_scaler(scaler_method)
        self.y_scaler = get_scaler(scaler_method)

    def normalize(self, x_data, y_data):
        x_data = np.concatenate((x_data[0], x_data[1]), axis=1)
        x_data_normal = self.x_scaler.fit_transform(x_data)
        y_data_normal = self.y_scaler.fit_transform(y_data)
        return x_data_normal, y_data_normal, self.y_scaler
