import tensorflow as tf
import numpy as np
from layers import LSTMCell_XDG

"""
Defined using the Functional API.
"""
def build_lstm_xdg_agent(obs_dim, hidden_dim, pol_dim, num_gates=20, gate_frac=0.8, **kwargs):
    """
    An RL agent which acts upon a single time-step.
    """
    obs_input = tf.keras.layers.Input(shape=[obs_dim,], name='obs_input')
    h_input = tf.keras.layers.Input(shape=[hidden_dim,], name='h_input')
    c_input = tf.keras.layers.Input(shape=[hidden_dim,], name='c_input')
    gate_index_input = tf.keras.layers.Input(shape=[1,], name='gate_index_input')

    _, [gated_h, c] = LSTMCell_XDG(hidden_dim, num_gates, gate_frac, name='lstm_cell')(obs_input, gate_index_input, [h_input, c_input])
    logits = tf.keras.layers.Dense(pol_dim, activation=None, name='pol_dense')(gated_h)
    val = tf.keras.layers.Dense(1, activation=None, name='val_dense')(gated_h)

    action_index = tf.random.categorical(logits, 1) # (B, 1)
    action = tf.one_hot(tf.squeeze(action_index), pol_dim) # (B, pol_dim)
    pol = tf.nn.softmax(logits, 1) # (B, pol_dim)

    return tf.keras.Model(inputs=[obs_input, gate_index_input, h_input, c_input],
        outputs=[action, pol, val, gated_h, c])


"""
Defined via Model subclassing.
"""
class LSTM_XDG_Agent(tf.keras.Model):
    """
    An RL agent which acts upon a single time-step.
    """
    def __init__(self, hidden_dim, pol_dim, num_gates=10, gate_frac=0.8, **kwargs):
        super(LSTM_XDG_Agent, self).__init__(name='lstm_xdg_agent', **kwargs)
        self.lstm_cell_gated = LSTMCell_XDG(hidden_dim, num_gates, gate_frac)
        self.policy_dense = tf.keras.layers.Dense(pol_dim, activation=None)
        self.value_dense = tf.keras.layers.Dense(1, activation=None)

    def call(self, obs, state):
        """
        Args:
        - obs: [stimulus, gate_index]
            - stimulus: Has shape (B, obs_dim).
            - gate_index: Has shaoe (B, 1).
        - state: [h, c], state for LSTM cell.
        - gate_index: Has shape (B, 1).

        Returns:
        - Output: List of [action, pol, val].
        - State: List of [gated_h, c] (to be fed to next t-step if trial alive).
        """
        _, [gated_h, c] = self.lstm_cell_gated(obs, state)
        logits = self.policy_dense(gated_h) # (B, pol_dim)
        val = self.value_dense(gated_h) # (B, 1)

        action_index = tf.random.categorical(logits, 1) # (B, 1)
        action = tf.one_hot(tf.squeeze(action_index), self.n_pol) # (B, pol_dim)
        pol = tf.nn.softmax(logits, 1) # (B, pol_dim)

        return [action, pol, val], [gated_h, c]
