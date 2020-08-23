
import math
import os

import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import plot_model
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import matplotlib.pyplot as plt

from lib.evaluation.error_metrics import evaluate
from lib.includes.utility import *


class BaseModel:
    def __init__(self, model_path, initial_state=True):
        self.model_path = model_path
        if initial_state:
            self._build_model()

    def _build_model(self):
        pass

    def load_model(self, model_path=None):
        if model_path is None:
            self.model = load_model(self.model_path)
        else:
            self.model = load_model(model_path)

    def save_model(self, model_saved_path=None):
        if model_saved_path is None:
            # Save model by self.model_path
            self.model.save(self.model_path)
        else:
            # Save model by model_saved_path
            self.model.save(model_saved_path)

    def get_model_description(self, infor_path):
        try:
            self.model.summary()
            plot_model(self.model, infor_path, show_shapes=True)
        except Exception as ex:
            print('[ERROR] Can not get description of the model')

    def plot_learning_curves(self):
        try:
            # plot learning curves
            plt.title('Learning Curves')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.plot(self.history.history['loss'], label='train')
            plt.plot(self.history.history['val_loss'], label='val')
            plt.legend()
            plt.show()
        except Exception as ex:
            print('[ERROR] Can not plot learning curves of the model')

    def fit(self, x, y, validation_split=0, batch_size=1, epochs=1, early_stopping=True, patience=40):
        callbacks = []

        if early_stopping:
            es = EarlyStopping(monitor='val_loss', patience=patience)
            model_checkpoint = ModelCheckpoint(f'{self.model_path}.h5', monitor='val_loss', mode='min',
                                               verbose=Config.VERBOSE, save_best_only=True)
            callbacks = [es, model_checkpoint]

        self.history = self.model.fit(
            x, y, validation_split=validation_split, epochs=epochs, batch_size=batch_size, verbose=Config.VERBOSE,
            shuffle=False, callbacks=callbacks)

        self.model.load_weights(f'{self.model_path}.h5')
        os.remove(f'{self.model_path}.h5')

    def predict(self, x):
        return self.model.predict(x)

    def evaluate(self, x, y, data_normalizer):
        pred = self.predict(x)
        pred = data_normalizer.invert_tranform(pred)
        print(evaluate(y, pred, ('mae', 'rmse', 'mse', 'mape', 'smape')))
        # self.load_model('model_best.h5')
        # best_pred = self.predict(x)
        # best_pred = data_normalizer.invert_tranform(best_pred)
        # plt.plot(pred, label='pred')
        # plt.plot(best_pred, label='best pred')
        # plt.legend()
        # plt.show()
        return evaluate(y, best_pred, ('mae', 'rmse', 'mse', 'mape', 'smape'))


class UnsupervisedPretrainModel(BaseModel):
    def __init__(self, model_path, initial_state=True):
        super().__init__(model_path, initial_state)

    def _build_encoder(self):
        pass

    def _build_decoder(self):
        pass

    def _build_model(self):
        self._build_encoder()
        self._build_decoder()

    def fit(self, x_encoder, x_decoder, y, validation_split=0, batch_size=1, epochs=1, early_stopping=True,
            patience=40):
        callbacks = []

        if early_stopping:
            es = EarlyStopping(monitor='val_loss', patience=patience)
            model_checkpoint = ModelCheckpoint(
                f'{self.model_path}.h5', monitor='val_loss', mode='min', verbose=Config.VERBOSE, save_best_only=True)
            callbacks = [es, model_checkpoint]
        self.history = self.model.fit(
            [x_encoder, x_decoder], y, validation_split=validation_split, epochs=epochs, batch_size=batch_size,
            verbose=Config.VERBOSE, shuffle=False, callbacks=callbacks)

        self.model.load_weights(f'{self.model_path}.h5')
        # os.remove(f'{self.model_path}.h5')

    def get_embedding(self, x):
        return self._encoder_model.predict(x)
