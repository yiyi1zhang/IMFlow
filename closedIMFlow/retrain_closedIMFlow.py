import sys
import numpy as np
import os
import math
import torch
from torch.optim import Adam
import torch.nn as nn
import itertools
from xitorch.interpolate import Interp1D
from scipy.integrate import odeint
import time
import torchdiffeq
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt


if torch.cuda.is_available():
    device = torch.device("cuda:0")
    torch.backends.cudnn.benchmark = True
else:
    print('cuda is not available')
    device = torch.device("cpu")

sys.path.append("..")
from env_dynamics import model_parameters
from train_val_closedIMFlow import SIFlow


class SIFlow_off(nn.Module):
    def __init__(self, u_dim, x_dim, y_dim, hidden_layer_size_s, n_hidden_s, h_rnn_dim_s, h_linear_dim_s, summary_dim,
                 n_blocks, hidden_layer_size_f, n_hidden_f, h_rnn_dim_f, h_linear_dim_f, horizon, win, deltat=0.01):
        super().__init__()
        self.u_dim = u_dim
        self.x_dim = x_dim
        self.y_dim = y_dim
        self.horizon = horizon
        self.win = win
        # self.l1 = nn.Linear(y_dim, y_dim)
        # self.l2 = nn.Linear(y_dim, y_dim)
        # self.init_weight(self.l1)
        # self.init_weight(self.l2)
        self.siflow = SIFlow(u_dim, x_dim, y_dim, hidden_layer_size_s, n_hidden_s, h_rnn_dim_s, h_linear_dim_s, summary_dim,
                 n_blocks, hidden_layer_size_f, n_hidden_f, h_rnn_dim_f, h_linear_dim_f, horizon, win, deltat=deltat)

    def init_weight(self, m, init_bias=0.001):
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(init_bias)

    def parameters(self, recurse: bool = True):
        params = [
                  # self.l1.parameters(),
                  # self.l2.parameters(),
                  # self.siflow.subnet.rnn1.parameters(),
                  # self.siflow.subnet.bn1.parameters(),
                  # self.siflow.subnet.l1.parameters(),
                  self.siflow.subnet.bn2.parameters(),
                  self.siflow.subnet.l2.parameters(),
                  # self.siflow.cnf.rnny.parameters(),
                  # self.siflow.cnf.bn1y.parameters(),
                  # self.siflow.cnf.l1y.parameters(),
                  # self.siflow.cnf.bn2y.parameters(),
                  # self.siflow.cnf.l2y.parameters(),
                  # self.siflow.cnf.bn1t.parameters(),
                  # self.siflow.cnf.l1t.parameters(),
                  # self.siflow.cnf.bn2t.parameters(),
                  # self.siflow.cnf.l2t.parameters()
                  ]
        return itertools.chain(*params)

    def train(self, mode: bool = True):
        # self.l1.train(mode)
        # self.l2.train(mode)
        # self.siflow.subnet.rnn1.train(mode)
        # self.siflow.subnet.bn1.train(mode)
        # self.siflow.subnet.l1.train(mode)
        self.siflow.subnet.bn2.train(mode)
        self.siflow.subnet.l2.train(mode)
        # self.siflow.cnf.rnny.train(mode)
        # self.siflow.cnf.bn1y.train(mode)
        # self.siflow.cnf.l1y.train(mode)
        # self.siflow.cnf.bn2y.train(mode)
        # self.siflow.cnf.l2y.train(mode)
        # self.siflow.cnf.bn1t.train(mode)
        # self.siflow.cnf.l1t.train(mode)
        # self.siflow.cnf.bn2t.train(mode)
        # self.siflow.cnf.l2t.train(mode)

    def forward(self, y_past, y, n_samples):
        # y_past = self.l1(y_past)
        # y = self.l2(y)
        us, y_pred = self.test(y_past, y, n_samples)
        return us[0], y_pred

    def test(self, y_past, y, n_samples):
        x0 = self.siflow.subnet.encoder(y_past)
        us = self.siflow.cnf.sample_together(x0, y, n_samples)
        y_pred = self.siflow.subnet.test(x0, us.mean(0))
        return us, y_pred

    def load(self, name):
        self.siflow.load(name)
        # print(f"load model from {name}")
        # state_dicts = torch.load(name, map_location=torch.device('cpu'))
        # self.l1.load_state_dict(state_dicts['l1'])
        # self.l2.load_state_dict(state_dicts['l2'])
        # self.siflow.load_state_dict(state_dicts['siflow'])
        # with torch.no_grad():
        #     self.l1.eval()
        #     self.l2.eval()
        #     self.siflow.eval()

    def save(self, name):
        self.siflow.save(name)
        # torch.save({
        #     'l1': self.l1.state_dict(),
        #     'l2': self.l2.state_dict(),
        #     'siflow': self.siflow.state_dict()
        # }, name)


def normalize(data, factor=None, min_data=None):
    if isinstance(data, torch.Tensor):
        if factor is None:
            min_data = data.clone()
            max_data = data.clone()
            for i in range(len(data.shape) - 1):
                min_data = torch.min(min_data, dim=0)
                max_data = torch.max(max_data, dim=0)
            factor = max_data - min_data
        else:
            factor = torch.tensor(factor, dtype=torch.float32, device=data.device)
            min_data = torch.tensor(min_data, dtype=torch.float32, device=data.device)

    else:
        if factor is None:
            min_data = data.copy()
            max_data = data.copy()
            for i in range(len(data.shape) - 1):
                min_data = np.min(min_data, axis=0)
                max_data = np.max(max_data, axis=0)
            factor = max_data - min_data

    return (data-min_data)/factor, factor, min_data


def denormalize(data, factor, min_data):
    if isinstance(data, torch.Tensor):
        factor = torch.tensor(factor, dtype=torch.float32, device=data.device)
        min_data = torch.tensor(min_data, dtype=torch.float32, device=data.device)
    factor = factor.reshape(*([1] * (len(data.shape) - 1)), -1)
    min_data = min_data.reshape(*([1] * (len(data.shape) - 1)), -1)
    data = data * factor + min_data
    return data


def prediction_iteration(model, y_past, y, n_samples):

    y_past = torch.tensor(y_past, dtype=torch.float32)
    y_past = torch.cat([y_past]*2)
    y = torch.cat([y]*2)
    stime = time.time()
    us, _ = model.test(y_past, y, n_samples)
    print(time.time() - stime)
    u_mean = us.mean(0)[0].mean(0, keepdims=True)
    return us[:, 0, :, :], u_mean


def prediction_iteration2(model, y, n_samples):
    us = model.sample_together(y, n_samples)
    u_mean = us.mean(0).squeeze(0).mean(0, keepdims=True)
    return us.squeeze(1), u_mean


def plant_ode(t, x, u, p):

    u1, u2 = u[0], u[1]

    if u1 < p.steering.min:
        u1 = torch.tensor(p.steering.min, dtype=torch.float32, device=device)
    elif u1 > p.steering.max:
        u1 = torch.tensor(p.steering.max, dtype=torch.float32, device=device)
    if x[2] > p.longitudinal.v_switch:
        posLimit = p.longitudinal.a_max * p.longitudinal.v_switch / x[2]
    else:
        posLimit = p.longitudinal.a_max
    if (x[2] <= p.longitudinal.v_min and u2 <= 0) or (x[2] >= p.longitudinal.v_max and u2 >= 0):
        u2 = torch.tensor(0.0, dtype=torch.float32, device=device)
    elif u2 <= -p.longitudinal.a_max:
        u2 = torch.tensor(-p.longitudinal.a_max, dtype=torch.float32, device=device)
    elif u2 >= posLimit:
        u2 = torch.tensor(posLimit, dtype=torch.float32, device=device)
    f = [x[2] * torch.cos(x[3]),
         x[2] * torch.sin(x[3]),
         u2,
         x[2] * torch.tan(u1) / p.l_wb]

    return torch.tensor(f)


def input_interp(t, timepoints, input_vals):

    # Get list indicating if t <= tp fpr all tp in timepoints
    t_smaller_time = [1 if t <= tp else 0 for tp in timepoints]

    # Return value corresponding to first tp that fulfills t <= tp
    if any(t_smaller_time):
        idx_last_value = t_smaller_time.index(1)
        val_interp = input_vals[idx_last_value]
    # Return last value if there is no tp that fulfills t <= tp
    else:
        val_interp = input_vals[len(input_vals) - 1]

    return val_interp


def update_plant(x, u, p, factor, min):
    # Prepare lambda function of ODE system, using interpolation of controls u
    if len(u) > 1:
        u = denormalize(torch.cat(u, dim=0), factor[y_dim:], min[y_dim:])
        u_fun = lambda t: input_interp(t, [i * DT for i in range(u.shape[0])], u)
        rhs_fun = lambda t, x: plant_ode(t, x, u_fun(t), p)
        x = torchdiffeq.odeint(rhs_fun, x, torch.linspace(0, DT*(u.shape[0]-1), u.shape[0],
                                                          dtype=torch.float32, device=device), method="dopri5")
    else:
        u = u[0]
        u = denormalize(u, factor[y_dim:], min[y_dim:])
        rhs_fun = lambda t, x: plant_ode(t, x, u[0], p)
        x = torchdiffeq.odeint(rhs_fun, x, torch.linspace(0, DT, 2, dtype=torch.float32, device=device), method="dopri5")
    return torch.tensor(x[-1], dtype=torch.float32, device=device, requires_grad=u.requires_grad)



def do_simulation(model, optimizer, y0, u0, states_ref, factor, min):
    states_tensor = [normalize(y0, factor[:y_dim], min[:y_dim])[0].unsqueeze(0)]
    inputs_samples = []
    inputs_tensor = [torch.zeros(1, 2)]
    cts = []
    seq_len = states_ref.shape[0]
    xref_interp = Interp1D(torch.linspace(0, (seq_len - 1) * DT, seq_len, device=device), states_ref.transpose(1, 0), method='linear')
    xref = xref_interp(torch.linspace(0, (seq_len - 1) * DT, seq_len * T, device=device)).transpose(1, 0)
    _y = update_plant(y0, [inputs_tensor[0], u0], p, factor, min)
    states_tensor.append(normalize(_y.data, factor[:y_dim], min[:y_dim])[0].unsqueeze(0))
    inputs_tensor.append(u0)
    k = 1

    while k < seq_len:
        print(k)
        if 0 < k < model.win // T:
            _y_past_interp = Interp1D(torch.linspace(0, k * DT, k+1, device=device),
                                      torch.cat(states_tensor[:(k+1)], dim=0).transpose(1, 0), method='linear')
            _y_past1 = _y_past_interp(torch.linspace(0, k * DT - tSim, k * T, device=device)).transpose(1, 0)
            _y_past = torch.cat([torch.repeat_interleave(states_tensor[0], repeats=model.win - k * T, dim=0),
                                 _y_past1], dim=0)
        else:
            _y_past_interp = Interp1D(torch.linspace(k * DT - tSim * model.win, k * DT, model.win // T + 1, device=device),
                                      torch.cat(states_tensor[(k - model.win // T):(k+1)], dim=0).transpose(1, 0), method='linear')
            _y_past = _y_past_interp(torch.linspace(k * DT - tSim * model.win, k * DT - tSim, model.win, device=device)).transpose(1, 0)

        stime = time.time()
        _us, _up = prediction_iteration(model, _y_past.unsqueeze(0), xref[k * T:(k + 1) * T].unsqueeze(0),
                                        n_samples)
        _y = update_plant(denormalize(states_tensor[-1][0], factor[:y_dim], min[:y_dim]), [_up], p, factor, min)
        loss = torch.sum((_y - denormalize(states_ref[k], factor[:y_dim], min[:y_dim])) ** 2, dim=-1)
        if loss > 1.0:
            for i in range(n_iter):
                print(i, loss.item())
                loss.retain_grad()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                _us, _up = prediction_iteration(model, _y_past.unsqueeze(0), xref[k*T:(k+1)*T].unsqueeze(0), n_samples)
                _y = update_plant(denormalize(states_tensor[-1][0], factor[:y_dim], min[:y_dim]), [_up], p, factor, min)
                # loss = (_y[0]-denormalize(states_ref[k, 0], factor[0], min[0]))**2 + (_y[1]-denormalize(states_ref[k, 1], factor[1], min[1]))**2
                loss = torch.sum((_y - denormalize(states_ref[k], factor[:y_dim], min[:y_dim])) ** 2, dim=-1)
        etime = time.time()
        cts.append(etime - stime)
        states_tensor.append(normalize(_y.data, factor[:y_dim], min[:y_dim])[0].unsqueeze(0))
        inputs_samples.append(_us)
        inputs_tensor.append(_up.data)
        k += 1

    return torch.cat(states_tensor, dim=0), torch.cat(inputs_tensor, dim=0), torch.cat(inputs_samples, dim=1), np.array(cts)

if __name__ == '__main__':

    T = int(sys.argv[1]) if len(sys.argv) > 1 else 5

    tSim = 0.01
    DT = tSim * T  # [s] time tick

    selected_inds = [0, 1, 2, 3]
    u_dim = 2
    x_dim = 32
    y_dim = len(selected_inds)
    n_blocks = 4
    summary_dim = 128
    hidden_layer_size_s = 64
    n_hidden_s = 2
    h_linear_dim_s = 64
    h_rnn_dim_s = 64
    hidden_layer_size_f = 64
    n_hidden_f = 2
    h_linear_dim_f = 64
    h_rnn_dim_f = 64
    n_samples = 50
    win_size = 10
    n_iter = 1
    p = model_parameters(3)

    checkpoint_path = f'Bayesian_conditional_normalizing_flow_RNN_RealNVP_SUBNET_transpose_single_track_horizon_{T}.pt'

    factor = np.array([ 95.66950259, 105.66003855,   3.89112207,  13.05282436,   0.99978093,   1.99837415])
    min = np.array([-34.13587075, -51.52120045,  18.2785546,   -4.20497803,  -0.49990596,  -0.9983958 ])

    data_path = '../data'
    Y = np.load(os.path.join(data_path, 'single_track_500_x_test.npy'))[:, :, selected_inds]
    num = 105
    t0 = 200
    U = np.load(os.path.join(data_path, 'single_track_500_u_test.npy'))
    Y, _, _ = normalize(Y, factor[:len(selected_inds)], min[:len(selected_inds)])
    U, _, _ = normalize(U, factor[len(selected_inds):], min[len(selected_inds):])
    n_test = U.shape[0]
    folder = os.path.join(os.getcwd(), 'results', checkpoint_path.replace('.pt', ''))
    y0 = torch.tensor(normalize(Y[num][t0], factor[:y_dim], min[:y_dim])[0], dtype=torch.float32, device=device).unsqueeze(0)
    u0 = torch.tensor(normalize(U[num][t0], factor[y_dim:], min[y_dim:])[0], dtype=torch.float32, device=device).unsqueeze(0)

    model_test = SIFlow_off(u_dim, x_dim, y_dim, hidden_layer_size_s, n_hidden_s, h_rnn_dim_s, h_linear_dim_s,
                            summary_dim, n_blocks, hidden_layer_size_f, n_hidden_f, h_rnn_dim_f, h_linear_dim_f,
                            T, win_size, deltat=tSim)
    model_test.siflow.load(os.path.join(folder, checkpoint_path))
    trainable_parameters = [p for p in model_test.parameters() if p.requires_grad]
    optimizer = Adam(trainable_parameters, 1e-3, weight_decay=1e-5)

    states_ref = torch.tensor(Y[num, t0::T, :], dtype=torch.float32, device=device)
    states_tensor, inputs_tensor, inputs_samples, cts = \
        do_simulation(model_test.to(device), optimizer, y0, u0, states_ref, factor, min)
    states_retrain = states_tensor.detach().numpy()
    y_plant_c = np.load(os.path.join(folder, 'y_plant.npy'))


    np.save(os.path.join(folder, f'retrain_states_{num}.npy'), denormalize(states_tensor.detach().numpy(), factor=factor[:y_dim], min_data=min[:y_dim]))
    np.save(os.path.join(folder, f'retrain_inputs_{num}.npy'), denormalize(inputs_tensor.detach().numpy(), factor=factor[y_dim:], min_data=min[y_dim:]))
    np.save(os.path.join(folder, f'retrain_inputs_samples_{num}.npy'), denormalize(inputs_samples.detach().numpy(), factor=factor[y_dim:], min_data=min[y_dim:]))
    np.save(os.path.join(folder, f'retrain_time_per_step_{num}.npy'), cts)
    model_test.save(os.path.join(folder, checkpoint_path.replace('.pt', f'_retrain_{num}.pt')))

    subtitles = [r"x-position [$m$]", r"y-position [$m$]", r"velocity [$m/s$]", r"yaw angle [$rad$]", 
                 r"steering angle [$rad$]", r"long. acceleration [$rad/s^2$]"]
    colors = ['#59c6eb', '#2596be', '#14b521']
    labels = ['ref', 'closed SIFlow', 'online SIFlow']

    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    ax.plot(denormalize(Y[num, :, 0], factor[0], min[0]), denormalize(Y[num, :, 1], factor[1], min[1]),
               color=colors[0], alpha=0.8, linewidth=5, label=labels[0])
    ax.plot(denormalize(y_plant_c[num, :, 0], factor[0], min[0]),
               denormalize(y_plant_c[num, :, 1], factor[1], min[1]), color=colors[3], label=labels[3])
    ax.plot(states_retrain[:, 0], states_retrain[:, 1], color=colors[4], label=labels[4])
    ax.grid(True)
    ax.axis("equal")
    ax.legend()
    ax.set_xlabel(subtitles[0])
    ax.set_ylabel(subtitles[1])

    plt.savefig(os.path.join(folder, 'retrain', f'tracking_compare_{num}.png'))