import os
import sys
import torch
import torch.nn as nn
from tqdm import tqdm
import numpy as np
import time
from train_val_neural_dynamic_model import simNet
import torch.optim as op
device = torch.device("cpu")
data_path = '../data'
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt


class APG(object):
    def __init__(self, dynamics_name='single_track', horizon=5, checkpoint_path=None):
        self.action_dim = 2
        self.horizon = horizon
        if dynamics_name == 'single_track':
            self.state_dim = 4
            self.ref_dim = 4
            from env_dynamics import SingleTrackDynamics
            self.dynamics = SingleTrackDynamics()
        else:
            self.state_dim = 29
            self.ref_dim = 6
            from env_dynamics import MultiBodyDynamics
            self.dynamics = MultiBodyDynamics()
        self.model = simNet(self.state_dim, horizon, self.ref_dim, self.action_dim)
        self.chk_path = checkpoint_path
        self.model.load(self.chk_path)

    def __call__(self, trajectory, state, dims, factors, mins, noise=0, dt=0.01,
                 retrain=False, threshold=10, start_lr=1e-2, epochs=5):
        states = []
        refs = []
        actions = []
        com_ts = []
        pa = nn.Parameter(torch.ones((1, self.action_dim)), requires_grad=True)
        trainable_parameters = [pa]
        optimizer = op.Adam(trainable_parameters, lr=start_lr)
        for k in range(trajectory.shape[0]//self.horizon):
            current_state = state[-1].reshape((1, -1))
            ref = trajectory[k*self.horizon:(k+1)*self.horizon]
            err = torch.mean((current_state - ref[[0]])**2)
            print('time step: {} / {}, err: {}'.format(k + 1, trajectory.shape[0] // self.horizon, err.item()), end='\r',
                  flush=True)
            if  err > threshold:
                if retrain:
                    if len(states) > 0:
                        stime = time.time()
                        for e in range(epochs):
                            loss = self.train(optimizer, pa, torch.stack(states)[:, 0, :].detach().clone().requires_grad_(False),
                                              torch.stack(refs), factors, mins, dims, noise=noise)
                            print('epoch: {}, loss: {}'.format(e, loss))
                            if loss < err:
                                self.model.save(self.chk_path.replace('.pt', '_retrain.pt'))
                                err = loss
                        re_t = time.time() - stime
                        print(re_t/epochs)
                    self.model.load(self.chk_path.replace('.pt', '_retrain.pt'))
                    trainable_parameters = [p for p in self.model.parameters() if p.requires_grad]
                    optimizer = op.Adam(trainable_parameters, lr=start_lr)

            state, action, com_t = self.predict_per_step(current_state, factors, mins, dims, noise=noise, ref=ref, dt=dt)
            states.append(state)
            refs.append(ref)
            actions.append(action)
            com_ts.append(com_t)
        states_seq = torch.cat(states)
        actions_seq = torch.cat(actions)
        com_ts = np.array(com_ts)
        return states_seq, actions_seq, com_ts

    def predict_per_step(self, states, factors, mins, dims, noise=0, ref=None, dt=0.01):
        states = (states - mins.reshape(1, -1)) / factors.reshape(1, -1)
        states += noise * torch.randn_like(states)
        ref = (ref - mins[dims].reshape(1, -1)) / factors[dims].reshape(1, -1)
        _stime = time.time()
        actions = self.model(states, ref.unsqueeze(0)).squeeze(0)
        com_t = time.time() - _stime
        next_states = []
        states = states * factors.reshape((1, -1)) + mins.reshape(1, -1)
        for i in range(self.horizon):
            states = self.dynamics(states, actions[[i]], dt=dt) + noise * torch.randn_like(states) * factors.reshape(1, -1)
            next_states.append(states)
        return torch.cat(next_states), actions, com_t

    def train(self, optimizer, pa, state, ref, factors, mins, dims, noise=0):
        optimizer.zero_grad()
        current_state = (state.clone() - mins.reshape(1, -1)) / factors.reshape(1, -1)
        ref = (ref - mins.reshape(1, -1)) / factors[dims].reshape(1, -1)
        current_state += noise * torch.randn_like(current_state)
        actions = self.model(current_state, ref)
        actions = actions * pa
        current_state = current_state * factors.reshape(1, -1) + mins.reshape(1, -1)
        ref = ref * factors[dims].reshape(1, -1) + mins.reshape(1, -1)
        _state = current_state.detach().clone().requires_grad_(False)
        for k in range(horizon):
            _state = self.dynamics(_state, actions[:, k, :], dt)

        loss = torch.mean((_state[:, dims] - ref[:, -1, :])**2)
        torch.autograd.set_detect_anomaly(True)
        # Backprop
        loss.backward()
        optimizer.step()
        return loss.item()


def get_factor(data):
    min_data = data.copy()
    max_data = data.copy()
    for i in range(len(data.shape) - 1):
        min_data = np.min(min_data, axis=0)
        max_data = np.max(max_data, axis=0)
    factor = max_data - min_data

    return factor, min_data


def load_data(file_x, file_u, selected_inds, factory=None, miny=None):

    Y = np.load(file_x)
    factory, miny = get_factor(Y) if factory is None else factory, miny
    U = np.load(file_u)
    state_tensor = torch.tensor(Y[:, 0, :], dtype=torch.float32)
    ref_tensor = torch.tensor(Y[:, :, selected_inds], dtype=torch.float32)
    U_tensor = torch.tensor(U, dtype=torch.float32)
    if not isinstance(factory, torch.Tensor):
        factory = torch.tensor(factory, dtype=torch.float32)
    if not isinstance(miny, torch.Tensor):
        miny = torch.tensor(miny, dtype=torch.float32)

    return state_tensor, ref_tensor, U_tensor, factory, miny


def plot_signals(t, y, y_pred, u, u_pred=None, labels=None, n_columns=3, n_rows=2, filepath=None):
    fig, ax = plt.subplots(n_rows, n_columns, figsize=(n_columns * 4, n_rows * 4))
    ax = ax.flat
    for i in range(y.shape[-1]):
        ax[i].plot(t, y[:, i])
        ax[i].plot(t, y_pred[:, i], color='#e55e5e')
        ax[i].set_title(labels[i])

    for i in range(u.shape[-1]):
        ax[y.shape[-1] + i].plot(t, u[:, i])
        ax[y.shape[-1] + i].set_title(labels[y.shape[-1] + i])
        if u_pred is not None:
            ax[y.shape[-1] + i].plot(t, u_pred[:, i], color='#e55e5e')

    if filepath is not None:
        fig.savefig(filepath)


dynamics_name = 'single_track'
horizon = 5
noise = 0
dt = 0.01

if dynamics_name == 'single_track':
    selected_inds = [0, 1, 2, 3]
    labels = ["sx", "sy", "velocity [m/s]", "yaw angle [rad]",
              "steering angle", "long. acceleration"]
else:
    selected_inds = [0, 1, 2, 3, 4, 5]
    labels = ["sx", "sy", "steering angle [rad]", "velocity [m/s]", "yaw angle [rad]", "yaw rate [rad/s]",
              "steering angle velocity", "long. acceleration"]
    checkpoint_path = f'apg_dynamic_model_rnn_{dynamics_name}_horizon_{horizon}.pt'
    if noise > 0:
        noise_str = str(noise).replace('.', '')
        checkpoint_path = checkpoint_path.replace('.pt', f'_noise_{noise_str}.pt')
folder = os.path.join('..', 'results', checkpoint_path.replace('.pt', ''))
factors, mins = get_factor(np.load(os.path.join(data_path, f"{dynamics_name}_4000_x_train.npy")))
factors = torch.tensor(factors, dtype=torch.float32)
mins = torch.tensor(mins, dtype=torch.float32)
exp = APG(dynamics_name=dynamics_name, horizon=horizon, checkpoint_path=os.path.join(folder, checkpoint_path))

states_test, ref_test, action_test, factors, mins = load_data(file_x=os.path.join(data_path, f"{dynamics_name}_500_x_test.npy"),
                                                  file_u=os.path.join(data_path, f"{dynamics_name}_500_u_test.npy"),
                                                  selected_inds=selected_inds, factory=factors, miny=mins)
inds = 0
_ref = ref_test[inds]
_state0 = states_test[inds]
_states, _actions, _com_t = exp(_ref, _state0.reshape(1, -1), selected_inds, factors, mins, noise=noise, dt=dt, 
                                retrain=True, start_lr=1e-3, epochs=5)
_states = _states.detach().numpy()
_actions = _actions.detach().numpy()
np.save(os.path.join(folder, 'apg_states_{inds}.npy'), _states)
np.save(os.path.join(folder, 'apg_actions_{inds}.npy'), _actions)
t = np.linspace(0, (_states.shape[0] - 1) * dt, _states.shape[0])

plot_signals(t, _ref[:-1, selected_inds].detach().numpy(), _states[:, selected_inds], _actions,
             labels=labels, n_columns=3, n_rows=3 if dynamics_name == 'multi_body' else 2,
             filepath=os.path.join(folder, 'apg_test_{inds}.png'))