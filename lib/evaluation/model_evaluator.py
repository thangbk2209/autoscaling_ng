import pickle as pkl

from lib.scaler.preprocessing_data.data_preprocessor import DataPreprocessor
from lib.scaler.model_loader import ModelLoader
from lib.evaluation.error_metrics import *
from lib.evaluation.data_visualizer import DataVisualizer
from lib.evaluation.fitness_manager import FitnessManager
from config import *


class ModelEvaluator:
    def __init__(self):
        self.data_preprocessor = DataPreprocessor()
        self.model_loader = ModelLoader()
        self.data_visualizer = DataVisualizer()
        self.fitness_manager = FitnessManager()

    def get_model_infor(self, model_path):
        print(model_path)
        try:
            with open(model_path, 'rb') as f:
                gbest_value = pkl.load(f)
                gbest_position = pkl.load(f)
                gbest_attribute = pkl.load(f)
                model_path = pkl.load(f)
                iteration = pkl.load(f)
                optimize_loss = pkl.load(f) 
            return gbest_value, gbest_position, gbest_attribute, model_path, iteration, optimize_loss
        except Exception as ex:
            print(ex)
            print(f'[ERROR] Do not run experiment at iteration')
            return

    def evaluate_ann(self, iteration, preprocess_item, visualize_option=True):

        if preprocess_item is None:
            saved_path = f'{Config.RESULTS_SAVE_PATH}best_model/iter_{iteration}'

        item, model_path, optimize_loss = self.get_model_infor(f'{saved_path}/optimize_infor.pkl')
        optimize_process = [optimize_loss]

        label = ['pso']
        visualize_folder_save_path = f'{saved_path}/visualize/'
        if not os.path.exists(visualize_folder_save_path):
            os.mkdir(visualize_folder_save_path)

        scaler_method = item['scaler']
        scaler_method = Config.SCALERS[scaler_method - 1]
        sliding = item['sliding']
        x_train, y_train, x_test, y_test, data_normalizer = \
            self.data_preprocessor.init_data_ann(sliding, scaler_method)

        validation_split = Config.VALID_SIZE
        n_train = int((1 - validation_split) * len(x_train))
        x_valid = x_train[n_train:]
        y_valid = y_train[n_train:]

        x_train = x_train[:n_train]
        y_train = y_train[:n_train]

        input_shape = [x_train.shape[1]]
        output_shape = [y_train.shape[1]]

        model_path = model_path.split('data')
        model_path = CORE_DATA_DIR + model_path[1]

        ann_predictor = self.model_loader.load_ann(model_path, input_shape, output_shape)

        if ann_predictor is None:
            return
        y_test_predict = ann_predictor.predict(x_test)
        y_test_predict = data_normalizer.invert_tranform(y_test_predict)
        test_error = evaluate(y_test, y_test_predict)

        y_valid_predict = ann_predictor.predict(x_valid)
        y_valid_predict = data_normalizer.invert_tranform(y_valid_predict)
        y_valid = data_normalizer.invert_tranform(y_valid)
        valid_error = evaluate(y_valid, y_valid_predict)
        uncertainty = valid_error['rmse']

        if visualize_option:
            optimize_process_path = f'{visualize_folder_save_path}/optimize_process'
            prediction_path = f'{visualize_folder_save_path}/prediction'
            uncertainty_path = f'{visualize_folder_save_path}/uncertainty'
            self.data_visualizer.visualize_fitness(optimize_process, label, optimize_process_path)
            self.data_visualizer.visualize_prediction(y_test, y_test_predict, prediction_path)
            self.data_visualizer.visualize_uncertainty(y_test, y_test_predict, uncertainty, uncertainty_path)

    def evaluate_lstm(self, iteration, preprocess_item, visualize_option=True):
        if preprocess_item is None:
            saved_path = f'{Config.RESULTS_SAVE_PATH}iter_{iteration}'

        item, model_path, optimize_loss = self.get_model_infor(f'{saved_path}/optimize_infor.pkl')
        optimize_process = [optimize_loss]

        label = ['pso']
        visualize_folder_save_path = f'{saved_path}/visualize/'
        if not os.path.exists(visualize_folder_save_path):
            os.mkdir(visualize_folder_save_path)

        scaler_method = item['scaler']
        scaler_method = Config.SCALERS[scaler_method - 1]
        sliding = item['sliding']
        x_train, y_train, x_test, y_test, data_normalizer = \
            self.data_preprocessor.init_data_lstm(sliding, scaler_method)

        validation_split = Config.VALID_SIZE
        n_train = int((1 - validation_split) * len(x_train))
        x_valid = x_train[n_train:]
        y_valid = y_train[n_train:]

        x_train = x_train[:n_train]
        y_train = y_train[:n_train]

        input_shape = [x_train.shape[1], x_train.shape[2]]
        output_shape = [y_train.shape[1]]

        model_path = model_path.split('data')
        model_path = CORE_DATA_DIR + model_path[1]

        lstm_predictor = self.model_loader.load_lstm(model_path, input_shape, output_shape)

        if lstm_predictor is None:
            print('[ERROR] Can not load LSTM model')
            return

        y_test_predict = lstm_predictor.predict(x_test)
        y_test_predict = data_normalizer.invert_tranform(y_test_predict)
        test_error = evaluate(y_test, y_test_predict)

        y_valid_predict = lstm_predictor.predict(x_valid)
        y_valid_predict = data_normalizer.invert_tranform(y_valid_predict)
        y_valid = data_normalizer.invert_tranform(y_valid)
        valid_error = evaluate(y_valid, y_valid_predict)
        uncertainty = valid_error['rmse']

        if visualize_option:
            optimize_process_path = f'{visualize_folder_save_path}/optimize_process'
            prediction_path = f'{visualize_folder_save_path}/prediction'
            uncertainty_path = f'{visualize_folder_save_path}/uncertainty'
            self.data_visualizer.visualize_fitness(optimize_process, label, optimize_process_path)
            self.data_visualizer.visualize_prediction(y_test, y_test_predict, prediction_path)
            self.data_visualizer.visualize_uncertainty(y_test, y_test_predict, uncertainty, uncertainty_path)

    def evaluate_bnn(self, iteration, preprocess_item=None, visualize_option=True):
        
        if preprocess_item is None:
            saved_path = f'{Config.RESULTS_SAVE_PATH}best_model/iter_{iteration}'
        else:
            scaler_method = preprocess_item['scaler']
            sliding_enc = preprocess_item['sliding_encoder']
            sliding_dec = preprocess_item['sliding_decoder']

            preprocess_name = f'scaler_{scaler_method}-sli_enc_{sliding_enc}-sli_dec_{sliding_dec}'
            saved_path = f'{Config.RESULTS_SAVE_PATH}{preprocess_name}/iter_{iteration}'

        gbest_value, gbest_position, item, model_path, iteration, optimize_loss = \
            self.get_model_infor(f'{saved_path}/optimize_infor.pkl')

        optimize_process = [optimize_loss]

        label = ['pso']
        visualize_folder_save_path = f'{saved_path}/visualize/'
        if not os.path.exists(visualize_folder_save_path):
            os.mkdir(visualize_folder_save_path)

        if preprocess_item is None:
            scaler_method = item['scaler']
            scaler_method = Config.SCALERS[scaler_method - 1]
            sliding_encoder = item['sliding_encoder']
            sliding_decoder = item['sliding_decoder']
        else:
            scaler_method = preprocess_item['scaler']
            sliding_encoder = preprocess_item['sliding_encoder']
            sliding_decoder = preprocess_item['sliding_decoder']

        batch_size = item['batch_size']

        x_train_encoder, x_train_decoder, y_train_decoder, y_train, x_test_encoder, y_test, data_normalizer = \
            self.data_preprocessor.init_data_bnn(sliding_encoder, sliding_decoder, scaler_method)

        validation_split = Config.VALID_SIZE
        n_train = int((1 - validation_split) * len(x_train_encoder))
        x_valid_encoder = x_train_encoder[n_train:]
        y_valid = y_train[n_train:]

        x_train_encoder = x_train_encoder[:n_train]
        y_train = y_train[:n_train]

        encoder_input_shape = [x_train_encoder.shape[1], x_train_encoder.shape[2]]
        decoder_input_shape = [x_train_decoder.shape[1], x_train_decoder.shape[2]]
        output_decoder_shape = [y_train_decoder.shape[1], y_train_decoder.shape[2]]
        output_shape = [y_train.shape[1]]
        model_path = model_path.split('data')
        print(model_path)

        model_path = CORE_DATA_DIR + model_path[-1]

        bnn_model = self.model_loader.load_bnn(model_path, encoder_input_shape, output_shape)

        if bnn_model is None:
            return

        mean_predict, uncertainty = \
            self.fitness_manager.evaluate_uncertainty(
                bnn_model, data_normalizer, x_valid_encoder, x_test_encoder, y_valid)

        if visualize_option:
            optimize_process_path = f'{visualize_folder_save_path}/optimize_process'
            prediction_path = f'{visualize_folder_save_path}/prediction'
            uncertainty_path = f'{visualize_folder_save_path}/uncertainty'
            self.data_visualizer.visualize_fitness(optimize_process, label, optimize_process_path)
            self.data_visualizer.visualize_prediction(y_test, mean_predict, prediction_path)
            self.data_visualizer.visualize_uncertainty(y_test, mean_predict, uncertainty, uncertainty_path)
        error = evaluate(y_test, mean_predict)
        print(error)
