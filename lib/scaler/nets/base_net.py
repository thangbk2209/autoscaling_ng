
import tensorflow as tf


class Net:
    def __init__(self, params, scope):
        self.params = params
        self.scope = scope

    def __call__(self):
        pass


class MlpNet(Net):
    def __init__(self, params, scope):
        super().__init__(params, scope)
        self.layer_size = params['layer_size']
        self.activation = params['activation']
        self.dropout = params['dropout']
        self.output_activation = params['output_activation']
    
    def __call__(self, x, reuse=False, *args, **kwargs):
        with tf.variable_scope(self.scope) as scope:
            if reuse:
                scope.reuse_variables()
            # if z is not None:
            #     x = tf.concat([x, z], axis=1)
            net = tf.keras.layers.Flatten()(x)
            for i, units in enumerate(self.layer_size[:-1]):
                net = tf.keras.layers.Dense(units, self.activation)(net)
                net = tf.keras.layers.Dropout(self.dropout)(net)
            net = tf.keras.layers.Dense(self.layer_size[-1], activation=self.output_activation)(net)
        return net


class RnnNet(Net):
    def __init__(self, params, scope):
        super().__init__(params, scope)
        self.layer_size = params['layer_size']
        self.activation = params['activation']
        self.dropout = params['dropout']
        self.output_activation = params['output_activation']
        self.cell_type = self._get_cell(params['cell_type'])
        self.concat_noise = None
        # if 'concat_noise' in params:
        #     self.concat_noise = params['concat_noise']

    def _get_cell(self, type_cell):
        if type_cell == 'lstm':
            return tf.keras.layers.LSTMCell
        elif type_cell == 'gru':
            return tf.keras.layers.GRUCell
        else:
            raise NotImplementedError

    def __call__(self, x, reuse=False, *args, **kwargs):
        with tf.variable_scope(self.scope) as scope:
            # if reuse:
            #     scope.reuse_variables()
            # if z is not None and self.concat_noise == 'before':
            #     x = tf.concat([x, z], axis=1)
            cells = []
            for i, units in enumerate(self.layer_size[:-1]):
                cell = self.cell_type(units, activation=self.activation, dropout=self.dropout)
                cells.append(cell)
            net = tf.keras.layers.RNN(cells, return_sequences=False)(x)
            if z is not None and self.concat_noise == 'after':
                z = tf.keras.layers.Flatten()(z)
                net = tf.concat([net, z], axis=1)
            net = tf.keras.layers.Dense(self.layer_size[-1], activation=self.output_activation)(net)
        return net
