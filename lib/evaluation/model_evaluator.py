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

    def evaluate_bnn(self, iteration, preprocess_item=None, visualize_option=False):

        if preprocess_item is None:
            saved_path = f'{Config.RESULTS_SAVE_PATH}iter_{iteration}'
        else:
            scaler_method = preprocess_item['scaler']
            sliding_enc = preprocess_item['sliding_encoder']
            sliding_dec = preprocess_item['sliding_decoder']
            sliding_inf = preprocess_item['sliding_inf']

            preprocess_name = f'scaler_{scaler_method}-sli_enc_{sliding_enc}-sli_dec_{sliding_dec}-sli_inf_{sliding_inf}'
            saved_path = f'{Config.RESULTS_SAVE_PATH}{preprocess_name}/iter_{iteration}'

        try:
            with open(f'{saved_path}/optimize_infor.pkl', 'rb') as f:
                item = pkl.load(f)
                model_path = pkl.load(f)
                optimize_loss = pkl.load(f)
        except Exception as ex:
            print(f'[ERROR] Do not run experiment at iteration: {iteration}')
            return

        optimize_process = [optimize_loss]

        label = ['pso']
        visualize_folder_save_path = f'{saved_path}/visualize/'
        if not os.path.exists(visualize_folder_save_path):
            os.mkdir(visualize_folder_save_path)

        if preprocess_item is None:
            scaler_method = item['scaler']
            scaler_method = Config.BNN_CONFIG['scalers'][scaler_method - 1]
            sliding_encoder = item['sliding_encoder']
            sliding_decoder = item['sliding_decoder']
            sliding_inf = item['sliding_inf']
        else:
            scaler_method = preprocess_item['scaler']
            sliding_encoder = preprocess_item['sliding_encoder']
            sliding_decoder = preprocess_item['sliding_decoder']
            sliding_inf = preprocess_item['sliding_inf']

        batch_size = item['batch_size']

        x_train_encoder, x_train_decoder, y_train_decoder, x_test_encoder = \
            self.data_preprocessor.init_data_autoencoder(sliding_encoder, sliding_decoder, scaler_method)

        x_train_inf, y_train_inf, x_test_inf, y_test_inf, data_normalizer = \
            self.data_preprocessor.init_data_inf(sliding_encoder, sliding_inf, scaler_method)

        validation_split = 0.1
        n_train = int((1 - validation_split) * len(x_train_encoder))
        x_valid_encoder = x_train_encoder[n_train:]
        x_valid_inf = x_train_inf[n_train:]
        y_valid_inf = y_train_inf[n_train:]

        x_train_encoder = x_train_encoder[:n_train]
        x_train_inf = x_train_inf[:n_train]
        y_train_inf = y_train_inf[:n_train]

        encoder_input_shape = [x_train_encoder.shape[1], x_train_encoder.shape[2]]
        decoder_input_shape = [x_train_decoder.shape[1], x_train_decoder.shape[2]]
        inf_input_shape = [x_train_inf.shape[1]]
        output_shape = [y_train_decoder.shape[1], y_train_decoder.shape[2]]
        model_path = model_path.split('data')

        model_path = CORE_DATA_DIR + model_path[1]

        bnn_model = self.model_loader.load_bnn(model_path, encoder_input_shape, inf_input_shape, output_shape)

        if bnn_model is None:
            return

        mean_predict, uncertainty = \
            self.fitness_manager.evaluate_uncertainty(
                bnn_model, data_normalizer, x_valid_encoder, x_valid_inf, x_test_encoder, x_test_inf, y_valid_inf)
        # mean_predict = data_normalizer.invert_tranform(mean_predict)
        y_test_inf = data_normalizer.invert_tranform(y_test_inf)
        if visualize_option:
            optimize_process_path = f'{visualize_folder_save_path}/optimize_process'
            prediction_path = f'{visualize_folder_save_path}/prediction'
            uncertainty_path = f'{visualize_folder_save_path}/uncertainty'
            self.data_visualizer.visualize_fitness(optimize_process, label, optimize_process_path)
            self.data_visualizer.visualize_prediction(y_test_inf, mean_predict, prediction_path)
            self.data_visualizer.visualize_uncertainty(y_test_inf, mean_predict, uncertainty, uncertainty_path)
        error = evaluate(y_test_inf, mean_predict)
        print(error)
