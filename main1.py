import os, sys
import tensorflow as tf
from train_rl import train_loop
from agent import LSTM_XDG_Agent, build_lstm_xdg_agent
from parameters import par

# Match GPU IDs to nvidia-smi command
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# Ignore Tensorflow startup warnings
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
gpus = tf.config.experimental.list_physical_devices('GPU')
if len(gpus) > 0 and len(sys.argv) > 1:
    tf.config.experimental.set_visible_devices(gpus[int(sys.argv[1])], 'GPU')

state_dim = 256
hparams = {
        'lr': 5e-4,
        'q_discount': 0.,
        'entropy_scale': 1e-3,
        'critic_scale': 1e-2,
        'stabilisation_scale': 1e2
}
tasks = ['go', 'rt_go', 'dly_go', 'anti-go', 'anti-rt_go', 'anti-dly_go']
n_iters = 4000

# agent = LSTM_XDG_Agent(state_dim, par['n_pol'], num_gates=20, gate_frac=0.8)
agent = build_lstm_xdg_agent(par['n_input'], state_dim, par['n_pol'], num_gates=20, gate_frac=0.8)

print('state_dim: ', state_dim)
print(hparams)
train_loop(agent, state_dim, tasks, n_iters, **hparams)
