
import tensorflow as tf

from lib.includes.utility import *


class Net:
    def __init__(self, params, scope):
        self.params = params
        self.scope = scope

    def __call__(self):
        pass


class MlpNet(Net):
    def __init__(self, params, scope):
        super().__init__(params, scope)
        self.num_units = params['num_units']
        self.activation = params['activation']
        self.dropout = params['dropout']

    def __call__(self, x, reuse=False, *args, **kwargs):

        activation = get_activation(self.activation)

        with tf.variable_scope(self.scope) as scope:
            num_layers = len(self.num_units)
            prev_layer = x
            for i in range(num_layers):
                prev_layer = tf.layers.dense(prev_layer, self.num_units[i], activation=activation, name='layer' + str(i))
                prev_layer = tf.layers.dropout(prev_layer, rate=self.dropout)

            net = tf.layers.dense(inputs=prev_layer, units=1, activation='sigmoid', name='prediction')
        return net


class RnnNet(Net):
    def __init__(self, params, scope, mode='predictor'):
        super().__init__(params, scope)
        self.num_units = params['num_units']
        self.activation = params['activation']
        self.dropout = params['dropout']
        self.cell_type = self._get_cell(params['cell_type'])
        self.mode = mode
        # self.concat_noise = None
        # if 'concat_noise' in params:
        #     self.concat_noise = params['concat_noise']

    def _get_cell(self, type_cell):
        if type_cell == 'lstm':
            return tf.nn.rnn_cell.LSTMCell
        elif type_cell == 'gru':
            return tf.nn.rnn_cell.GRUCell
        else:
            raise NotImplementedError

    def __call__(self, x, initial_state=None, reuse=False, *args, **kwargs):

        activation = get_activation(self.activation)

        with tf.variable_scope(self.scope) as scope:
            # if reuse:
            #     scope.reuse_variables()
            # if z is not None and self.concat_noise == 'before':
            #     x = tf.concat([x, z], axis=1)
            cells = []
            for i, units in enumerate(self.num_units):
                cell = self.cell_type(self.num_units[i], activation=activation)
                if self.dropout != 0:
                    if i == 0:
                        cell = tf.nn.rnn_cell.DropoutWrapper(
                            cell, input_keep_prob=1.0, output_keep_prob=self.dropout, state_keep_prob=self.dropout,
                            variational_recurrent=True, dtype=tf.float32)  # , input_size=self.input_dim
                    else:
                        cell = tf.nn.rnn_cell.DropoutWrapper(
                            cell, input_keep_prob=self.dropout, output_keep_prob=self.dropout, state_keep_prob=self.dropout,
                            variational_recurrent=True, input_size=self.num_units[i - 1], dtype=tf.float32)
                cells.append(cell)

            cells = tf.contrib.rnn.MultiRNNCell(cells, state_is_tuple=True)
            if initial_state is not None:
                outputs, hidden_state = tf.nn.dynamic_rnn(cells, x, dtype="float32", initial_state=initial_state)
            else:
                outputs, hidden_state = tf.nn.dynamic_rnn(cells, x, dtype="float32")

        if self.mode == 'predictor':
            # outputs = tf.identity(outputs, name='outputs')
            net = tf.layers.dense(
                outputs[:, :, -1], 1, activation=activation, use_bias=True, name='prediction')
            return net
        elif self.mode == 'pretrain':
            # new_state = tf.identity(new_state, name='new_state')
            return outputs, hidden_state


class GeneratorNet(Net):
    def __init__(self, params, scope):
        super().__init__(params, scope)
        self.embedding_num_units = params['embedding_num_units']
        self.mlp_num_units = params['mlp_num_units']
        self.activation = params['activation']
        self.dropout = params['dropout']
        self.cell_type = self._get_cell(params['cell_type'])

        self.params_mlp_net = {
            'num_units': params['mlp_num_units'],
            'activation': params['activation'],
            'dropout': params['dropout']
        }

    def _get_cell(self, type_cell):
        if type_cell == 'lstm':
            return tf.nn.rnn_cell.LSTMCell
        elif type_cell == 'gru':
            return tf.nn.rnn_cell.GRUCell
        else:
            raise NotImplementedError

    def __call__(self, x, z=None, initial_state=None, reuse=False, *args, **kwargs):

        activation = get_activation(self.activation)

        with tf.variable_scope(self.scope) as scope:
            cells = []
            for i, units in enumerate(self.embedding_num_units):
                cell = self.cell_type(self.embedding_num_units[i], activation=activation)
                cells.append(cell)

            cells = tf.contrib.rnn.MultiRNNCell(cells, state_is_tuple=True)
            if initial_state is not None:
                outputs, hidden_state = tf.nn.dynamic_rnn(cells, x, dtype="float32", initial_state=initial_state)
            else:
                outputs, hidden_state = tf.nn.dynamic_rnn(cells, x, dtype="float32")

            z = tf.keras.layers.Flatten()(z)

            pred_input = tf.keras.layers.concatenate([outputs[:, :, 1], z], axis=1)
            mlp_net = MlpNet(self.params_mlp_net, 'mlp')
            predict = mlp_net(pred_input)
        return predict


class DiscriminatorNet(Net):
    def __init__(self, params, scope):
        super().__init__(params, scope)
        self.embedding_num_units = params['embedding_num_units']
        self.mlp_num_units = params['mlp_num_units']
        self.activation = params['activation']
        self.dropout = params['dropout']
        self.cell_type = self._get_cell(params['cell_type'])

        self.params_mlp_net = {
            'num_units': params['mlp_num_units'],
            'activation': params['activation'],
            'dropout': params['dropout']
        }

    def _get_cell(self, type_cell):
        if type_cell == 'lstm':
            return tf.nn.rnn_cell.LSTMCell
        elif type_cell == 'gru':
            return tf.nn.rnn_cell.GRUCell
        else:
            raise NotImplementedError

    def __call__(self, x, initial_state=None, reuse=False, *args, **kwargs):

        activation = get_activation(self.activation)

        with tf.variable_scope(self.scope) as scope:
            if reuse:
                scope.reuse_variables()
            cells = []
            for i, units in enumerate(self.embedding_num_units):
                cell = self.cell_type(self.embedding_num_units[i], activation=activation)
                cells.append(cell)

            cells = tf.contrib.rnn.MultiRNNCell(cells, state_is_tuple=True)
            if initial_state is not None:
                outputs, hidden_state = tf.nn.dynamic_rnn(cells, x, dtype="float32", initial_state=initial_state)
            else:
                outputs, hidden_state = tf.nn.dynamic_rnn(cells, x, dtype="float32")

            mlp_net = MlpNet(self.params_mlp_net, 'mlp')
            predict = mlp_net(outputs)
        return predict
