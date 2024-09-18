import sys
import numpy as np
import scipy.stats as st
import os
import math
import torch
from scipy.integrate import odeint
from xitorch.interpolate import Interp1D
import time
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt

sys.path.append("..")
from train_val_cNF import RNNcNF
from env_dynamics import model_parameters, ST_ode, MB_ode


def prediction(model, plant_ode, y0, t, p, y, factor, mins, n_samples):
    """

    Args:
        model (): SIFlow
        y0 (): list of initial value of y
        t (): torch tensor of time steps of y
        p (): parameters of plant
        y (): torch tensor in the shape (seq_len, y_dim)
        n_samples (): int number

    Returns:

    """
    tStart = t[0].item()
    tStop = t[-1].item()
    seq_len = int((tStop - tStart) / tSim + 1)
    t1 = torch.linspace(tStart+1e-6, tStop, seq_len)
    y_interp = Interp1D(t, y.transpose(1, 0), method='linear')(t1).transpose(1, 0)
    us, ct = prediction_iteration(model, y_interp.unsqueeze(0), n_samples)
    u_pred = us.mean(0)
    y_plant = update_plant(plant_ode, y0, torch.linspace(tStart+1e-6, tStop, seq_len+1), p,
                           torch.cat([torch.zeros(1, model.u_dim), u_pred]), factor, mins, [i*tPred for i in range(seq_len+1)])

    return us.detach().numpy(), u_pred.detach().numpy(), y_plant[1:].detach().numpy(), np.array(ct)


def prediction_iteration(model, y, n_samples):
    stime = time.time()
    us = model.sample(y, n_samples)
    etime = time.time()
    return us.squeeze(1), etime-stime


def update_plant(plant_ode, x0, t, p, u, factor, mins, time_points):
    u = u.detach().numpy()
    u = denormalize(u, factor[y_dim:], mins[y_dim:])
    sol = odeint(plant_ode, x0, t, args=(p, u, time_points))
    # ynew = sol[:, :y_dim]
    ynew = sol[:, selected_inds]
    ynew, _, _ = normalize(ynew, factor[:y_dim], mins[:y_dim])
    ynew = torch.tensor(ynew, dtype=torch.float32)
    return ynew


def confidence_interval(data, confidence=0.95):
    return st.norm.interval(alpha=confidence, loc=np.mean(data, axis=0), scale=np.std(data, axis=0))


def normalize(data, factor=None, min_data=None):
    if factor is None:
        min_data = data.copy()
        max_data = data.copy()
        for i in range(len(data.shape) - 1):
            min_data = np.min(min_data, axis=0)
            max_data = np.max(max_data, axis=0)
        factor = max_data - min_data

    return (data-min_data)/factor, factor, min_data


def denormalize(data, factor, min_data):
    factor = factor.reshape(*([1] * (len(data.shape) - 1)), -1)
    min_data = min_data.reshape(*([1] * (len(data.shape) - 1)), -1)
    data = data * factor + min_data
    return data


def plot_signals(t, y, y_plant, u, u_pred, t1, u_approx, factor, mins, labels, n_columns=3, n_rows=2, filepath=None):
    y = denormalize(y, factor[:y.shape[-1]], mins[:y.shape[-1]])
    y_plant = denormalize(y_plant, factor[:y.shape[-1]], mins[:y.shape[-1]])
    u = denormalize(u, factor[y.shape[-1]:], mins[y.shape[-1]:])
    u_pred = denormalize(u_pred, factor[y.shape[-1]:], mins[y.shape[-1]:])
    u_approx = denormalize(u_approx, factor[y.shape[-1]:], mins[y.shape[-1]:])
    u_mean = u_approx.mean(axis=0)
    CI = confidence_interval(u_approx)
    u_approx_min = CI[0]
    u_approx_max = CI[1]

    fig, ax = plt.subplots(n_rows, n_columns, figsize=(n_columns * 4, n_rows * 4))
    ax = ax.flat

    for i in range(y.shape[-1]):
        ax[i].plot(t, y[:, i], 'b--', label='ref')
        ax[i].plot(t1, y_plant[:, i], 'g', label='flow')
        ax[i].set_title(labels[i])
        ax[i].legend()
    for i in range(u.shape[-1]):
        ax[y.shape[-1] + i].plot(t, u[:, i], 'b--', label='ref')
        ax[y.shape[-1] + i].plot(t1, u_pred[:, i], 'g', label='flow')
        ax[y.shape[-1] + i].plot(t1, u_mean[:, i], 'g--', label='flow_mu')
        ax[y.shape[-1] + i].fill_between(t1, u_approx_min[:, i], u_approx_max[:, i], alpha=0.8, color='#96eba7', label='flow_CI')
        ax[y.shape[-1] + i].set_title(labels[y.shape[-1] + i])
        ax[y.shape[-1] + i].legend()
    ax[-1].plot(y[:, 0], y[:, 1], 'b--', label='ref')
    ax[-1].plot(y_plant[:, 0], y_plant[:, 1], 'g', label='flow')
    ax[-1].axis('equal')
    ax[-1].legend()

    if filepath is not None:
        fig.savefig(filepath)


if __name__ == '__main__':

    tStart = 0  # start time
    tFinal = 3  # final time
    tSim = 1e-2  # step size
    dynamics = sys.argv[1] if len(sys.argv) > 1 else 'single_track' # 'multi_body'
    horizon = int(sys.argv[2]) if len(sys.argv) > 2 else 5

    if dynamics == 'multi_body':
        selected_inds = [0, 1, 2, 3, 4, 5]
        labels = [r"x-position [$m$]", r"y-position [$m$]", r"steering angle [$rad$]", r"velocity [$m/s$]", r"yaw angle [$rad$]", 
                  r"yaw rate [$rad/s$]", r"steering velocity [$rad/s$]", r"long. acceleration [$rad/s^2$]"]
        plant_ode = MB_ode
    else:
        selected_inds = [0, 1, 2, 3]
        labels = [r"x-position [$m$]", r"y-position [$m$]", r"velocity [$m/s$]", r"yaw angle [$rad$]", 
                  r"steering angle [$rad$]", r"long. acceleration [$rad/s^2$]"]
        plant_ode = ST_ode
    
    data_path = '../data'

    Y = np.load(os.path.join(data_path, f'{dynamics}_500_x_test.npy'))
    y0 = Y[:, 0, :]
    Y = Y[:, :, selected_inds]
    U = np.load(os.path.join(data_path, f'{dynamics}_500_u_test.npy'))

    Y_train = np.load(os.path.join(data_path, f'{dynamics}_4000_x_train.npy'))[:, :, selected_inds]
    U_train = np.load(os.path.join(data_path, f'{dynamics}_4000_u_train.npy'))
    factory, miny = normalize(Y_train)
    factoru, minu = normalize(U_train)
    factor = np.concatenate([factory, factoru], axis=-1)
    mins = np.concatenate([miny, minu], axis=-1)
    Y, _, _ = normalize(Y, factor[:len(selected_inds)], mins[:len(selected_inds)])
    U, _, _ = normalize(U, factor[len(selected_inds):], mins[len(selected_inds):])
    n_test = Y.shape[0]
    Y_tensor = torch.tensor(Y, dtype=torch.float32)
    U_tensor = torch.tensor(U, dtype=torch.float32)

    win_size = 10
    horizon = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    tPred = horizon * tSim
    u_dim = 2
    x_dim = 29
    y_dim = len(selected_inds)
    n_blocks = 4
    summary_dim = 64
    hidden_layer_size_s = 64
    n_hidden_s = 2
    h_linear_dim_s = 64
    h_rnn_dim_s = 64
    hidden_layer_size_f = 64
    n_hidden_f = 2
    h_linear_dim_f = 64
    h_rnn_dim_f = 64
    n_samples = 50
    vehicle = 3
    p = model_parameters(vehicle)
    seq_len = int((tFinal-tStart)/tPred+1)
    seq_len_1 = int((tFinal-tStart)/tSim+1)
    t = torch.linspace(tStart, tFinal, seq_len)
    checkpoint_path = f'Bayesian_conditional_normalizing_flow_RNN_RealNVP_l_single_step_{dynamics}.pt'

    folder = os.path.join('..', 'results', checkpoint_path.replace('.pt', ''))
    model_test = RNNcNF(u_dim, y_dim, summary_dim, n_blocks, hidden_layer_size_f, n_hidden_f, h_rnn_dim_f, h_linear_dim_f)
    
    model_test.load(os.path.join(folder, checkpoint_path))
    t = torch.linspace(tStart, tFinal, seq_len)
    
    u_samplesl, u_predl, y_plantl, ctsl = [], [], [], []
    for index in range(n_test):
        print(index)
        y = Y_tensor[index, ::horizon, :]
        u_samples, u_pred, y_plant, cts = prediction(model_test, plant_ode, y0[index], t, p, y, factor, mins, n_samples)
        u_samplesl += [u_samples]
        u_predl += [u_pred]
        y_plantl += [y_plant]
        ctsl += [cts]

        plot_signals(np.linspace(tStart, 3, seq_len), Y[index, ::horizon, :], y_plant[index],
                        U[index, ::horizon, :], u_pred[index], np.linspace(tStart, 3, seq_len_1),
                        u_samples[index], factor, labels, n_columns=3, n_rows=3,
                        filepath=os.path.join(folder, f'prediction_{index}.png'))
        plt.close()

    np.save(os.path.join(folder, f'u_samples.npy'), np.stack(u_samplesl, dim=0))
    np.save(os.path.join(folder, f'u_pred.npy'), np.stack(u_predl, dim=0))
    np.save(os.path.join(folder, f'y_plant.npy'), np.stack(y_plantl, dim=0))
    np.save(os.path.join(folder, f'cts.npy'), np.stack(ctsl, axis=0))