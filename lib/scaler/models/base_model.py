
import tensorflow as tf


class BaseModel:
    def __init__(self):

        tf.reset_default_graph()
        self.sess = tf.Session()

        self.__build_model()

        self.sess.run(tf.global_variables_initializer())

    def close_session(self):
        self.sess.close()

    def __build_model(self):
        pass

    def fit(self):
        pass

    def predict(self, x):
        pass

    def evaluate(self, x, y):
        pass


class RegressionModel:
    def __init__(self):
        pass


class AnnModel:
    def __init__(self):
        pass
