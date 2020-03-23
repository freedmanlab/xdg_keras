import tensorflow as tf
import stimulus
from parameters import par

class TaskEnvironment:
    """
    A RL environment for eye saccade tasks.
    Built on-top of stimulus.py.
    """

    def __init__(self, task_name_or_index):
        task_dict = {
            'go': 0,
            'rt_go': 1,
            'dly_go': 2,
            'anti-go': 3,
            'anti-rt_go': 4,
            'anti-dly_go': 5,
            'dm1': 6,
            'dm2': 7,
            'ctx_dm1': 8,
            'ctx_dm2': 9,
            'multsen_dm': 10,
            'dm1_dly': 11,
            'dm2_dly': 12,
            'ctx_dm1_dly': 13,
            'ctx_dm2_dly': 14,
            'multsen_dm_dly': 15,
            'dms': 16,
            'dmc': 17,
            'dnms': 18,
            'dnmc': 19
            }

        if isinstance(task_name_or_index, str):
            self.task_index = task_dict[task_name_or_index]
        else:
            self.task_index = task_name_or_index

        self.stim = stimulus.MultiStimulus()

    def begin_trial(self):
        """
        Returns:
        - List from next t-step:
            - obs: Has shape (B, obs_dim). Returns None trial_over.
            - trial_over: True if tstep at max tsteps.
        """
        self.tstep = 0
        self.name, self.input_data, _, self.dead_time_mask, self.reward_data = \
            self.stim.generate_trial(self.task_index)
        self.dead_time_mask = tf.expand_dims(self.dead_time_mask, axis=-1)
        self.total_tsteps = self.input_data.shape[0]
        self.batch_size = self.input_data.shape[1]
        self.done_mask = tf.ones([par['batch_size'], 1])
        self.done_masks = tf.TensorArray(tf.float32, size=self.total_tsteps)
        self.actions = tf.TensorArray(tf.float32, size=self.total_tsteps)
        self.rewards = tf.TensorArray(tf.float32, size=self.total_tsteps+1, clear_after_read=False)
        self.rewards = self.rewards.write(0, tf.zeros([self.batch_size, 1]))
        self.trial_over = False
        return [self.input_data[self.tstep, :, :], self.trial_over]

    def step(self, action):
        """
        Args:
        - action: Has shape (B, pol_dim).

        Returns:
        - List from curr t-step:
            - reward: Has shape (B, 1).
            - done_mask: Has shape (B, 1). Boolean (0 if trial done).
            - dead_time_mask: Has shape (B, 1). Boolean (0 if dead time).
        - List from next t-step:
            - obs: Has shape (B, obs_dim). Returns None trial_over.
            - trial_over: True if tstep at max tsteps.
        """
        curr_dead_time_mask = self.dead_time_mask[self.tstep, :, :]
        last_trial_zero = tf.cast(tf.equal(self.rewards.read(self.tstep), 0.), tf.float32)
        self.done_mask *= last_trial_zero
        r_t = tf.reduce_sum(action * self.reward_data[self.tstep, :, :], axis=1, keepdims=True) \
            * self.done_mask * curr_dead_time_mask

        self.done_masks = self.done_masks.write(self.tstep, self.done_mask)
        self.rewards = self.rewards.write(self.tstep+1, r_t)
        self.actions = self.actions.write(self.tstep, action)

        if self.tstep == self.total_tsteps-1:
            self.trial_over = True
            obs = None
        else:
            self.tstep += 1
            obs = self.input_data[self.tstep, :, ]

        return [r_t, self.done_mask, curr_dead_time_mask], [obs, self.trial_over]

    def random_step(self):
        """
        Step the env forward with a random action.
        """
        pass

    def trial_summary(self):
        """
        Returns:
        - o,a,r:
            - observations: (T, B, obs_dim)
            - actions: (T, B, pol_dim)
            - rewards: (T, B, 1)
        - masks:
            - done_masks: (T, B, 1)
            - dead_time_masks: (T, B, 1)
            - full_mask: (T, B, 1)
        """
        done_masks = self.done_masks.stack()
        full_mask = done_masks * self.dead_time_mask
        masks = [done_masks, self.dead_time_mask, full_mask]
        oar = [self.input_data, self.actions.stack(), self.rewards.stack()[1:, :, :]]
        return oar, masks

    def trial_animation(self):
        """
        Renders an animation of the trial.
        """
        pass
