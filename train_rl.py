import tensorflow as tf
import numpy as np
from agent_env import do_trial
from losses import advantage_policy_gradient_loss, value_loss, negative_policy_entropy_loss


def mean_accuracy(rs):
    return tf.reduce_mean(tf.reduce_sum(tf.cast(tf.greater(rs, 0.), dtype=tf.float32), axis=0))

def mean_reward(rs):
    return tf.reduce_mean(tf.reduce_sum(rs, axis=0))

def print_metrics(metrics_dict):
    to_print = []
    for metric_name, metric in metrics_dict.items():
        string = '| ' + metric_name + ': '
        to_print.append(string)
        to_print.append(metric)
    to_print.append('\n')
    tf.print(*to_print)

@tf.function
def get_grads_from_trial(agent, task_name_or_index, q_discount, critic_scale,
    entropy_scale, stabilisation_scale, prev_params, params_importance, get_grads=True, print_diag=False):
    """
    Args:
    - prev_params: A 1-D vector of concat'ed params. Required if stabilisation_scale > 0.
    - params_importance: Same shape as prev_params. Required if stabilisation_scale > 0.

    Returns:
    - loss: Scalar.
    - losses: Dictionary containing all losses.
    - metrics: Dictionary containing any metrics.
    """
    losses = {}
    with tf.GradientTape(watch_accessed_variables=get_grads) as tape:
        [_, acts, rs], [pols, vals], [_, _, full_mask] = do_trial(agent, task_name_or_index)

        vals = tf.concat([vals, tf.zeros([1, vals.shape[1], vals.shape[2]])], axis=0)
        q_sa = rs + q_discount*vals[1:, :, :]*full_mask
        v_s = vals[:-1, :, :]
        advantages = q_sa - v_s # (T, B, 1)

        losses['actor_loss'] = advantage_policy_gradient_loss(pols, acts, advantages, mask=full_mask)
        losses['critic_loss'] = value_loss(v_s, q_sa, mask=full_mask)
        losses['neg_ent_loss'] = negative_policy_entropy_loss(pols, mask=full_mask)
        losses['stabilisation_loss'] = 0.
        if stabilisation_scale > 0:
            curr_params = tf.concat([tf.reshape(var, shape=[-1]) for var in agent.trainable_variables], axis=0)
            losses['stabilisation_loss'] += tf.reduce_sum(params_importance * tf.square(curr_params - prev_params))

        loss = losses['actor_loss'] + critic_scale*losses['critic_loss'] + \
            entropy_scale*losses['neg_ent_loss'] + \
            stabilisation_scale*losses['stabilisation_loss']

    metrics = {}
    metrics['mean_acc'] = mean_accuracy(rs)
    metrics['mean_r'] = mean_reward(rs)

    if get_grads:
        grads = tape.gradient(loss, agent.trainable_variables)
        return grads, loss, losses, metrics
    return loss, losses, metrics


def compute_params_importance(final_params, init_params, final_reward, init_reward, damping_term=1e-2):
    """
    Args:
    - final_params: Flattened params after end of train step (after update).
    - init_params: Flattened params from begin of train step (before update).
    - final_reward: Scalar, mean across batch, summed across episode.
    - init_reward: Scalar, mean across batch, summed across episode.

    Returns:
    - params_importance: Same shape as final_params, init_params.
    """
    abs_params_change = tf.abs(final_params - init_params)
    reward_change = final_reward - init_reward
    little_omega = abs_params_change * reward_change
    normalising_little_omega = tf.abs(little_omega)
    return tf.nn.relu(little_omega / (normalising_little_omega + damping_term))


def train_loop(agent, state_dim, tasks, n_iters, lr, q_discount, critic_scale,
    entropy_scale, stabilisation_scale):
    tasks_trained_on = []
    params_importance = tf.constant(tf.concat([tf.reshape(tf.zeros_like(var), shape=[-1]) for var in agent.trainable_variables], axis=0))
    for task in tasks:
        accs = []
        init_params = tf.concat([tf.reshape(var, shape=[-1]) for var in agent.trainable_variables], axis=0)
        prev_params = init_params
        opt = tf.keras.optimizers.Adam(learning_rate=lr)
        for i in range(n_iters):
            print_diag = i % 50 == 0 # and len(tasks_trained_on) > 1
            grads, loss, losses, metrics = get_grads_from_trial(agent, task,
                prev_params=prev_params,
                params_importance=params_importance,
                q_discount=q_discount,
                critic_scale=critic_scale,
                entropy_scale=entropy_scale,
                stabilisation_scale=stabilisation_scale,
                print_diag=print_diag)
            prev_params = tf.concat([tf.reshape(var, shape=[-1]) for var in agent.trainable_variables], axis=0)
            opt.apply_gradients(zip(grads, agent.trainable_variables))

            accs.append(metrics['mean_acc'])
            if np.mean(accs[-1000:]) > 0.99: break

            if i == 0: init_r = metrics['mean_r']
            print_metrics_bool = False
            if i % 50 == 0: print_metrics_bool = True
            if print_metrics_bool:
                metrics_dict = {}
                metrics_dict['Task'] = task
                metrics_dict['Iter'] = i
                metrics_dict['r'] = metrics['mean_r']
                metrics_dict['acc'] = metrics['mean_acc']
                metrics_dict['AL'] = losses['actor_loss']
                metrics_dict['CL'] = losses['critic_loss']
                metrics_dict['SL'] = losses['stabilisation_loss']
                print_metrics(metrics_dict)
        final_params = tf.concat([tf.reshape(var, shape=[-1]) for var in agent.trainable_variables], axis=0)
        add_params_imp = compute_params_importance(final_params, init_params, metrics['mean_r'], init_r)
        params_importance = params_importance + add_params_imp

        tasks_trained_on.append(task)
        tf.print('\n=== BEGIN TEST ===')
        for task_ in tasks_trained_on:
            loss, losses, metrics = get_grads_from_trial(agent, task_,
                prev_params=prev_params,
                params_importance=params_importance,
                q_discount=q_discount,
                critic_scale=critic_scale,
                entropy_scale=entropy_scale,
                stabilisation_scale=stabilisation_scale,
                get_grads=False)
            metrics_dict = {}
            metrics_dict['Task'] = task_
            metrics_dict['r'] = metrics['mean_r']
            metrics_dict['acc'] = metrics['mean_acc']
            print_metrics(metrics_dict)
        tf.print('=== END TEST === \n')
