import tensorflow as tf

"""
All the components for A2C RL loss (advantage-actor-critic).
"""

def negative_policy_entropy_loss(pols, mask=None, epsilon=1e-8):
    """
    Args:
    - pols: (T, B, pol_dim).
    - mask: (T, B, 1) or None.

    Returns:
    - loss: scalar
    """
    if mask is None:
        mask = tf.ones([v_s.shape[0], q_sa.shape[1], 1])

    mask = tf.stop_gradient(mask)

    loss_per_tstep_batch = tf.reduce_sum(mask*pols*tf.math.log(pols+epsilon), axis=2) # (T, B)
    loss_per_batch = tf.reduce_sum(loss_per_tstep_batch, axis=0) # (B)
    return tf.reduce_mean(loss_per_batch) # scalar


def value_loss(v_s, q_sa, mask=None):
    """
    Args:
    - v_s: V(s) as output by agent. (T, B, 1).
    - q_sa: Q(s,a) = r(s,a) + V(s'). (T, B, 1).
    - mask: (T, B, 1) or None

    Returns:
    - loss: scalar
    """
    if mask is None:
        mask = tf.ones([v_s.shape[0], q_sa.shape[1], 1])

    q_sa = tf.stop_gradient(q_sa)
    mask = tf.stop_gradient(mask)

    value_loss_per_batch = tf.reduce_sum(mask*tf.square(v_s - q_sa), axis=0) # (B, 1)
    return 0.5 * tf.reduce_mean(value_loss_per_batch) # scalar


def advantage_policy_gradient_loss(pols, actions, advantages, mask=None, epsilon=1e-8):
    """
    Args:
    - pols: (T, B, pol_dim)
    - actions: (T, B, pol_dim)
    - advantages: (T, B, 1)
    - mask: (T, B, 1) or None

    Returns:
    - loss: scalar
    """
    if mask is None:
        mask = tf.ones([pols.shape[0], pols.shape[1], 1])

    actions = tf.stop_gradient(actions)
    advantages = tf.stop_gradient(advantages)
    mask = tf.stop_gradient(mask)

    actions_logps = tf.reduce_sum(actions*tf.math.log(pols+epsilon), axis=-1, keepdims=True) # (T, B, 1)
    actor_loss_per_batch = tf.reduce_sum(mask*advantages*-actions_logps, axis=0) # (B, 1)
    return tf.reduce_mean(actor_loss_per_batch) # scalar
