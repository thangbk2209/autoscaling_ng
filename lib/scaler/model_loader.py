
from lib.scaler.models.ann_model import AnnPredictor
from lib.scaler.models.lstm_model import LstmPredictor
from lib.scaler.models.bnn.autoencoder import RnnAutoEncoder
from lib.scaler.models.bnn.bnn_model import BnnPredictor


class ModelLoader:
    def __init__(self):
        pass

    def load_ann(self, model_path, input_shape, output_shape):
        try:
            ann_predictor = AnnPredictor(
                model_path=model_path,
                input_shape=input_shape,
                output_shape=output_shape,
                initial_state=False
            )

            ann_predictor.load_model()
        except Exception:
            ann_predictor = None

        return ann_predictor

    def load_lstm(self, model_path, input_shape, output_shape):
        try:
            lstm_predictor = LstmPredictor(
                model_path=model_path,
                input_shape=input_shape,
                output_shape=output_shape,
                initial_state=False
            )

            lstm_predictor.load_model()
        except Exception:
            lstm_predictor = None

        return lstm_predictor

    def load_autoencoder(self, autoencoder_model_path, encoder_input_shape, decoder_input_shape, output_shape):
        try:
            autoencoder_model = RnnAutoEncoder(
                model_path=autoencoder_model_path,
                encoder_input_shape=encoder_input_shape,
                decoder_input_shape=decoder_input_shape,
                output_shape=output_shape,
                initial_state=False
            )

            autoencoder_model.load_model()
        except Exception:
            autoencoder_model = None

        return autoencoder_model

    def load_bnn(self, model_path, encoder_input_shape, output_shape):
        print(model_path)
        try:
            bnn_model = BnnPredictor(
                model_path=model_path,
                encoder_input_shape=encoder_input_shape,
                output_shape=output_shape,
                initial_state=False
            )
            bnn_model.load_model()
        except Exception as ex:
            print(ex)
            print('[ERROR] Can not load bnn model')
            bnn_model = None

        return bnn_model
