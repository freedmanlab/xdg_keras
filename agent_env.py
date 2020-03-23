import tensorflow as tf
from rl_env import TaskEnvironment
from parameters import par

"""
Code for interactions between:
- Any RL agent specified as a Keras Model which at each t-step:
    - Takes in (obs, state) - can have multiple of each, as list e.g. [obs_1, obs_2].
    - Returns (output, state) - output is typically [act, pol, val], must include action.
- RL environments.
"""

def do_trial(agent, task_name_or_index):
    """
    Args:
    - agent: See specification above. Furthermore - obs is [stim, gate_index],
        state is [h, c], output is [act, pol, val].
    - task_name_or_index: Str if name, int if index.

    Returns:
    - oars:
        - obss: [T, B, obs_dim]
        - as: [T, B, pol_dim]
        - rs: [T, B, 1]
    - pols_and_vals:
        - pols: [T, B, pol_dim]
        - vals: [T, B, val_dim]
    - masks:
        - done_mask: (T, B, 1)
        - dead_time_mask: (T, B, 1)
        - full_mask: (T, B, 1)
    """
    env = TaskEnvironment(task_name_or_index)
    [obs, trial_over] = env.begin_trial()

    vals = tf.TensorArray(tf.float32, size=env.total_tsteps)
    pols = tf.TensorArray(tf.float32, size=env.total_tsteps)
    batch_size = obs.shape[0]
    gate_index = tf.fill([batch_size, 1], tf.cast(env.task_index, tf.int32))
    state_dim = agent.get_layer('h_input').output_shape[0][1]
    h = tf.zeros([batch_size, state_dim])
    c = tf.zeros([batch_size, state_dim])
    while not trial_over:
        action, pol, val, h, c = agent([obs, gate_index, h, c])

        # tf.print(tf.math.count_nonzero(h[0]))

        vals = vals.write(env.tstep, val)
        pols = pols.write(env.tstep, pol)
        [_, _, _], [obs, trial_over] = env.step(action)
    # oars, masks = [obss, as, rs], [done_mask, dead_time_mask, full_mask] = env.trial_summary()
    oars, masks = env.trial_summary()
    return oars, [pols.stack(), vals.stack()], masks
