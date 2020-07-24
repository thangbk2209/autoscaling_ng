import math

import numpy as np
from sklearn.metrics import mean_squared_error

from lib.includes.utility import *
from lib.evaluation.error_metrics import *
from config import *


class FitnessManager:
    def __init__(self):
        pass

    def get_model_prediction(self, bnn_model, data_normalizer, x_valid_encoder, x_valid_inf, number_of_time_to_evaluate):
        y_inf_valid_predict = []
        for i in range(number_of_time_to_evaluate):
            _normaled_y_inf_valid_predict = bnn_model.predict(x_valid_encoder, x_valid_inf)
            _y_inf_valid_predict = data_normalizer.invert_tranform(_normaled_y_inf_valid_predict)
            y_inf_valid_predict.append(_y_inf_valid_predict)
        y_inf_valid_predict = np.array(y_inf_valid_predict)

        mean_y_inf_valid_predict = np.mean(y_inf_valid_predict, axis=0)
        std_y_inf_valid_predict = np.std(y_inf_valid_predict, axis=0)
        # y_inf_valid_predict_95 = mean_y_inf_valid_predict + 1.96 * std_y_inf_valid_predict
        return mean_y_inf_valid_predict, y_inf_valid_predict

    def evaluate_fitness_validation(self, bnn_model, data_normalizer, x_valid_encoder, x_valid_inf, y_valid_inf,
                                    number_of_time_to_evaluate=50):
        mean_y_inf_valid_predict, y_inf_valid_predict = self.get_model_prediction(
            bnn_model, data_normalizer, x_valid_encoder, x_valid_inf, number_of_time_to_evaluate)
        validation_error = evaluate(mean_y_inf_valid_predict, y_valid_inf)['rmse']
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

    def get_prediction_interval(
            self, bnn_model, data_normalizer, x_valid_encoder, x_valid_inf, y_valid_inf, number_of_time_to_evaluate=50):
        mean_y_inf_valid_predict, y_inf_valid_predict = self.get_model_prediction(
            bnn_model, data_normalizer, x_valid_encoder, x_valid_inf, number_of_time_to_evaluate)
        inherence_noise = evaluate(mean_y_inf_valid_predict, y_valid_inf)['mse']

        model_miss_uncertainty = []
        upper_prediction = []
        lower_prediction = []

        for i in range(mean_y_inf_valid_predict.shape[0]):
            err = []
            for j in range(number_of_time_to_evaluate):
                _err = mean_squared_error(mean_y_inf_valid_predict[i], y_inf_valid_predict[j][i])
                err.append(_err)
            _model_miss_uncertainty = sum(err) / number_of_time_to_evaluate
            _overall_uncertainty = math.sqrt(_model_miss_uncertainty + inherence_noise)
            model_miss_uncertainty.append(_model_miss_uncertainty)

            _upper_prediction = mean_y_inf_valid_predict[i][0] + _overall_uncertainty
            _lower_prediction = mean_y_inf_valid_predict[i][0] - _overall_uncertainty

            upper_prediction.append(_upper_prediction)
            lower_prediction.append(_lower_prediction)
        return mean_y_inf_valid_predict, lower_prediction, upper_prediction

    def evaluate_fitness_bayesian_normalized(
            self, bnn_model, data_normalizer, x_valid_encoder, x_valid_inf, y_valid_inf, number_of_time_to_evaluate=50):
        mean_y_inf_valid_predict, lower_prediction, upper_prediction = \
            self.get_prediction_interval(bnn_model, data_normalizer, x_valid_encoder, x_valid_inf, y_valid_inf,
                                         number_of_time_to_evaluate=50)

        rate_real_value_in_prediction_interval = \
            compute_scale_fitness_value(upper_prediction, lower_prediction, y_valid_inf)  # rate

        y_valid_inf = data_normalizer.y_tranform(y_valid_inf)
        mean_y_inf_valid_predict = np.array(mean_y_inf_valid_predict).reshape(-1, 1)
        mean_y_inf_valid_predict = data_normalizer.y_tranform(mean_y_inf_valid_predict)
        evaluate_validation_prediction = evaluate(mean_y_inf_valid_predict, y_valid_inf)
        validation_error = evaluate_validation_prediction['rmse']  # validation error

        upper_prediction = np.array(upper_prediction).reshape(-1, 1)
        upper_prediction = data_normalizer.y_tranform(upper_prediction)
        evaluate_real_scale = evaluate(upper_prediction, y_valid_inf)
        real_scale_error = evaluate_real_scale['rmse']  # real scale error

        fitness = (1 - rate_real_value_in_prediction_interval + real_scale_error + validation_error) / 3
        return fitness

    def evaluate_fitness_bayesian_based_smape(
            self, bnn_model, data_normalizer, x_valid_encoder, x_valid_inf, y_valid_inf, number_of_time_to_evaluate=50):

        mean_y_inf_valid_predict, lower_prediction, upper_prediction = \
            self.get_prediction_interval(bnn_model, data_normalizer, x_valid_encoder, x_valid_inf, y_valid_inf,
                                         number_of_time_to_evaluate=50)

        rate_real_value_in_prediction_interval = \
            compute_scale_fitness_value(upper_prediction, lower_prediction, y_valid_inf)  # rate

        evaluate_validation_prediction = evaluate(mean_y_inf_valid_predict, y_valid_inf)
        validation_error = evaluate_validation_prediction['smape']  # validation error

        evaluate_real_scale = evaluate(upper_prediction, y_valid_inf)
        real_scale_error = evaluate_real_scale['smape']  # real scale error

        fitness = (1 - rate_real_value_in_prediction_interval + real_scale_error + validation_error) / 3

        return fitness

    def evaluate_fitness_bayesian(self, bnn_model, data_normalizer, x_valid_encoder, x_valid_inf, y_valid_inf,
                                  number_of_time_to_evaluate=50):

        if Config.FITNESS_NORMALIZE_METHOD == 'smape':
            return self.evaluate_fitness_bayesian_based_smape(
                bnn_model, data_normalizer, x_valid_encoder, x_valid_inf, y_valid_inf, number_of_time_to_evaluate=50)
        elif Config.FITNESS_NORMALIZE_METHOD == 'normalized_value':
            return self.evaluate_fitness_bayesian_normalized(
                bnn_model, data_normalizer, x_valid_encoder, x_valid_inf, y_valid_inf, number_of_time_to_evaluate=50)
        else:
            print(f'[ERROR] Do not support method: {Config.FITNESS_NORMALIZE_METHOD} in evaluate fitness bayesian')
