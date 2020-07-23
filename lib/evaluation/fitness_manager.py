import math

import numpy as np
from sklearn.metrics import mean_squared_error

from lib.includes.utility import *
from lib.evaluation.error_metrics import *


class FitnessManager:
    def __init__(self):
        pass

    def get_model_prediction(self, bnn_model, data_nomalizer, x_valid_encoder, x_valid_inf, number_of_time_to_evaluate):
        y_inf_valid_predict = []
        for i in range(number_of_time_to_evaluate):
            print(f'=== i: {i}')
            _normaled_y_inf_valid_predict = bnn_model.predict(x_valid_encoder, x_valid_inf)
            _y_inf_valid_predict = data_nomalizer.invert_tranform(_normaled_y_inf_valid_predict)
            y_inf_valid_predict.append(_y_inf_valid_predict)
        y_inf_valid_predict = np.array(y_inf_valid_predict)

        mean_y_inf_valid_predict = np.mean(y_inf_valid_predict, axis=0)
        std_y_inf_valid_predict = np.std(y_inf_valid_predict, axis=0)
        # y_inf_valid_predict_95 = mean_y_inf_valid_predict + 1.96 * std_y_inf_valid_predict
        return mean_y_inf_valid_predict, y_inf_valid_predict

    def evaluate_fitness_validation(self, bnn_model, data_nomalizer, x_valid_encoder, x_valid_inf, y_valid_inf,
                                    number_of_time_to_evaluate=50):
        mean_y_inf_valid_predict, y_inf_valid_predict = self.get_model_prediction(
            bnn_model, data_nomalizer, x_valid_encoder, x_valid_inf, number_of_time_to_evaluate)
        validation_error = evaluate(mean_y_inf_valid_predict, y_valid_inf)['mse']
        return validation_error

    def evaluate_uncertainty(self, bnn_model, data_normalizer, x_valid_encoder, x_valid_inf, x_test_encoder, x_test_inf,
                             y_valid_inf, number_of_time_to_evaluate=50):

        mean_test_y_inf_predict, y_test_inf_predict = self.get_model_prediction(
            bnn_model, data_normalizer, x_test_encoder, x_test_inf, number_of_time_to_evaluate)
        mean_valid_y_inf_predict, y_valid_inf_predict = self.get_model_prediction(
            bnn_model, data_normalizer, x_valid_encoder, x_valid_inf, number_of_time_to_evaluate)

        evaluate_result = evaluate(mean_valid_y_inf_predict, y_valid_inf)
        inherence_noise = evaluate_result['mse']

        model_miss_uncertainty = []
        overall_uncertainty = []
        prediction_interval = []

        for i in range(mean_test_y_inf_predict.shape[0]):
            err = []
            for j in range(number_of_time_to_evaluate):
                _err = mean_squared_error(mean_test_y_inf_predict[i], y_test_inf_predict[j][i])
                err.append(_err)
            _model_miss_uncertainty = sum(err) / number_of_time_to_evaluate
            _overall_uncertainty = math.sqrt(_model_miss_uncertainty + inherence_noise)
            model_miss_uncertainty.append(_model_miss_uncertainty)
            overall_uncertainty.append(_overall_uncertainty)

        return mean_test_y_inf_predict, overall_uncertainty

    def evaluate_fitness_bayesian(self, bnn_model, data_nomalizer, x_valid_encoder, x_valid_inf, y_valid_inf,
                                  number_of_time_to_evaluate=50):

        mean_y_inf_valid_predict, y_inf_valid_predict = self.get_model_prediction(
            bnn_model, data_nomalizer, x_valid_encoder, x_valid_inf, number_of_time_to_evaluate)
        evaluate_result = evaluate(mean_y_inf_valid_predict, y_valid_inf)
        validation_error = evaluate_result['smape']
        inherence_noise = evaluate_result['mse']

        model_miss_uncertainty = []
        overall_uncertainty = []
        prediction_interval = []

        for i in range(mean_y_inf_valid_predict.shape[0]):
            err = []
            for j in range(number_of_time_to_evaluate):
                _err = mean_squared_error(mean_y_inf_valid_predict[i], y_inf_valid_predict[j][i])
                err.append(_err)
            _model_miss_uncertainty = sum(err) / number_of_time_to_evaluate
            _overall_uncertainty = math.sqrt(_model_miss_uncertainty + inherence_noise)
            model_miss_uncertainty.append(_model_miss_uncertainty)
            overall_uncertainty.append(_overall_uncertainty)

            _prediction_interval = \
                [mean_y_inf_valid_predict[i][0] - _overall_uncertainty,
                 mean_y_inf_valid_predict[i][0] + _overall_uncertainty]
            prediction_interval.append(_prediction_interval)

        overall_uncertainty = np.array(overall_uncertainty)
        rate_real_value_in_prediction_interval, real_scale_value_error = \
            compute_scale_fitness_value(prediction_interval, y_valid_inf)

        fitness = (1 - rate_real_value_in_prediction_interval + real_scale_value_error + validation_error) / 3

        return fitness
