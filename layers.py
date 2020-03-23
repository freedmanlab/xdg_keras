import tensorflow as tf
import numpy as np

class XDG_Layer(tf.keras.layers.Layer):
    def __init__(self, num_gates, gate_frac, trainable=False, **kwargs):
        """
        Args:
        - num_gates: The initial number of different gates you can select from.
        - gate_frac: In [0,1], the fraction of units to gate (i.e. set to zero).
        """
        super(XDG_Layer, self).__init__(trainable, **kwargs)
        self.num_gates = num_gates
        self.gate_frac = gate_frac

    def build(self, input_shape):
        """
        Args:
        - input_shape: When constructing a model, this will be given.
        """
        numpy_gates = np.random.binomial(1, 1-self.gate_frac, [self.num_gates, input_shape[-1]])
        self.gates = tf.Variable(numpy_gates, trainable=False, shape=[None, input_shape[-1]], dtype=tf.float32, name='gates')

    def call(self, input, gate_index):
        """
        Args:
        - input: Has shape (batch_size, n_input).
        - gate_index: Has shape (batch_size, 1).
        """
        onehots = tf.one_hot(tf.cast(gate_index, tf.int32), self.num_gates) # (batch_size, num_gates)
        onehots = tf.squeeze(onehots, axis=1)
        selected_gates = onehots @ self.gates # (B, n_input)
        return input * selected_gates # (B, n_input)

    def get_config(self):
        config = super(XDG_Layer, self).get_config()
        config['num_gates'] = self.num_gates
        config['gate_frac'] = self.gate_frac
        return config

    def add_gates(self, num_gates_to_add):
        """
        Generates new gates and concats them to self.gates, updates self.num_gates.
        """
        pass


class LSTMCell_XDG(tf.keras.layers.Layer):
    def __init__(self, units, num_gates, gate_frac, **kwargs):
        super(LSTMCell_XDG, self).__init__(**kwargs)
        self.lstm_cell = tf.keras.layers.LSTMCell(units)
        self.xdg = XDG_Layer(num_gates, gate_frac)

    def call(self, input, gate_index, states):
        """
        Args:
        - input: Has shape (B, I).
        - states: List [h,c] each of shape (B, H).
        - gate_index: Has shape (B, 1)

        Returns: gated_h, [gated_h, c]

        NB: Calling LSTMCell returns h, [h,c]
        """
        _, [h, c] = self.lstm_cell(input, states) # (B, H)
        gated_h = self.xdg(h, gate_index) # (B, H)
        return gated_h, [gated_h, c]


def LSTM_XDG(units, num_gates, gate_frac, **kwargs):
    cell = LSTMCell_XDG(units, num_gates, gate_frac, **kwargs)
    return tf.keras.layers.RNN(cell)
