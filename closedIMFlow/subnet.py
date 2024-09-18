import sys
import os
import torch
import torch.nn as nn
import torch.optim as op
import numpy as np
from tqdm import tqdm
import itertools
import scipy.stats as st
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
sys.path.append('..')
from utils.pts_flows import RealNVP
from utils.early_stop import EarlyStopping


if torch.cuda.is_available():
    device = torch.device("cuda:0")
    torch.backends.cudnn.benchmark = True
else:
    print('cuda is not available')
    device = torch.device("cpu")


class MLP_res_block(nn.Module):
    def __init__(self, n_in, n_out, activation=nn.Tanh):
        super(MLP_res_block, self).__init__()
        if n_in == n_out:
            self.skipper = True
        else:
            self.skipper = False
            self.res = nn.Linear(n_in, n_out)
        self.nonlin = nn.Linear(n_in, n_out)
        self.activation = activation()

    def forward(self, x):
        if self.skipper:
            return x + self.activation(self.nonlin(x))
        else:
            return self.res(x) + self.activation(self.nonlin(x))


class SUBNET(nn.Module):
    def __init__(self, u_dim, x_dim, y_dim, hidden_layer_size_s, n_hidden_s, h_rnn_dim_s, h_linear_dim_s,
                 horizon, win):
        super().__init__()
        self.u_dim = u_dim
        self.x_dim = x_dim
        self.y_dim = y_dim
        self.horizon = horizon
        self.win = win
        self.rnn1 = nn.GRU(y_dim, h_rnn_dim_s, batch_first=True)
        self.bn1 = nn.BatchNorm1d(h_rnn_dim_s)
        self.l1 = nn.Linear(h_rnn_dim_s, h_linear_dim_s)
        self.bn2 = nn.BatchNorm1d(h_linear_dim_s)
        self.l2 = nn.Sequential(nn.ELU(), nn.Linear(h_linear_dim_s, x_dim))
        f_layers = [nn.Linear(x_dim + u_dim, hidden_layer_size_s)] + \
                   [MLP_res_block(hidden_layer_size_s, hidden_layer_size_s)] * n_hidden_s + \
                   [nn.Linear(hidden_layer_size_s, x_dim)]
        self.f_net = nn.Sequential(*f_layers)
        self.h_net = nn.Linear(x_dim, y_dim)

    def encoder(self, x):
        x_en, hx = self.rnn1(x)
        x_en = self.bn1(x_en[:, -1, :])
        x_en = self.l1(x_en)
        x_en = self.bn2(x_en)
        x_en = self.l2(x_en)
        return x_en

    def forward(self, y_past, u, y):
        _x = self.encoder(y_past)
        y_pred = []
        for i in range(self.horizon):
            _y = self.h_net(_x)
            y_pred.append(_y)
            _e = y[:, i, :] - _y
            _x = self.f_net(torch.cat([_x, u[:, i, :]], dim=-1)) + \
                 torch.repeat_interleave(_e.mean(dim=-1, keepdims=True), repeats=self.x_dim, dim=1)
        y_pred = torch.stack(y_pred, dim=1)
        loss_subnet = nn.MSELoss()(y_pred, y)

        return loss_subnet

    def test(self, y_past, u):
        _x = self.encoder(y_past)
        y_pred = []
        x_pred = []
        for i in range(u.shape[1]):
            _y = self.h_net(_x)
            y_pred.append(_y)
            _x = self.f_net(torch.cat([_x, u[:, i, :]], dim=-1))
            x_pred.append(_x)
        return torch.stack(y_pred, dim=1), torch.stack(x_pred, dim=1)

    def save(self, name):
        torch.save(self.state_dict(), name)

    def load(self, name):
        print(f"load model from {name}")
        state_dicts = torch.load(name, map_location=torch.device('cpu'))
        self.load_state_dict(state_dicts)
        with torch.no_grad():
            self.eval()


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


def split_and_roll_data(data, win_size, horizon, times=1):
    length = seq_len + 1
    datas = []
    datas_1 = []
    for i in range(times):
        _data = np.concatenate([data[:, i + 1:, :], data[:, 1:i + 1, :]], axis=1)
        datas += np.array_split(_data, length // horizon, axis=1)[:(length // horizon - i)]
        _data_1 = []
        for j in range(length // horizon - i):
            start = j * horizon - win_size + i + 1
            end = j * horizon + i + 1
            datas_1 += [data[:, start - max(0, end - length):min(end, length), :] if start >= 0 else
                        np.concatenate([np.zeros((data.shape[0], win_size - end, data.shape[-1])), data[:, :end, :]],
                                       axis=1)]
    datas = np.concatenate(datas, axis=0)
    datas_1 = np.concatenate(datas_1, axis=0)
    return datas, datas_1


def load_data(file_x, file_u, selected_inds, win_size, horizon, n_times=5, split=False, factor=None, mins=None):

    Y = np.load(file_x)[:, :, selected_inds]
    Y, factory, miny = normalize(Y, factor=factor[:Y.shape[-1]] if factor is not None else None,
                                 min_data=mins[:Y.shape[-1]] if mins is not None else None)

    U = np.load(file_u)
    U, factoru, minu = normalize(U, factor=factor[Y.shape[-1]:] if factor is not None else None,
                                 min_data=mins[Y.shape[-1]:] if mins is not None else None)
    if split:
        Y, Y_past = split_and_roll_data(Y, win_size, horizon, times=n_times)
        U, U_past = split_and_roll_data(U, win_size, horizon, times=n_times)
        Y_tensor = torch.tensor(Y, dtype=torch.float32)
        Y_past_tensor = torch.tensor(Y_past, dtype=torch.float32)
        U_tensor = torch.tensor(U, dtype=torch.float32)
        U_past_tensor = torch.tensor(U_past, dtype=torch.float32)

        return Y_tensor, U_tensor, Y_past_tensor, U_past_tensor, \
               np.concatenate([factory, factoru], axis=-1), np.concatenate([miny, minu], axis=-1)
    else:
        Y_tensor = torch.tensor(Y, dtype=torch.float32)
        U_tensor = torch.tensor(U, dtype=torch.float32)
        return Y_tensor, U_tensor, np.concatenate([factory, factoru], axis=-1), np.concatenate([miny, minu], axis=-1)


def train_one_epoch(model, optimizer, train_loader, val_loader, iterations, ep, pbar,
                    lr_scheduler, grad_clip, init_epoch, max_explr_epoch):
    # training
    model.train(True)
    for it in range(iterations[0]):
        y_past, u, y = next(iter(train_loader))
        y_past, u, y = y_past.to(device), u.to(device), y.to(device)
        train_loss = model(y_past, u, y)
        train_loss.retain_grad()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()
        pbar.update(1)
    model.train(False)
    for it in range(iterations[1]):
        y_past_val, u_val, y_val = next(iter(val_loader))
        y_past_val, u_val, y_val = y_past_val.to(device), u_val.to(device), y_val.to(device)
        val_loss = model(y_past_val, u_val, y_val)
    if ep <= init_epoch:
        lr_scheduler[0].step()
    elif init_epoch < ep < 3*max_explr_epoch:
        lr_scheduler[1].step()

    return train_loss.item(), val_loss.item()


def confidence_interval(data, confidence=0.95):
    return st.norm.interval(alpha=confidence, loc=np.mean(data, axis=0), scale=np.std(data, axis=0))


def plot_signals(t, y, y_pred, u, factor, mins, labels, n_columns=3, n_rows=2, filepath=None):
    y = denormalize(y, factor[:y.shape[-1]], mins[:y.shape[-1]])
    y_pred = denormalize(y_pred, factor[:y.shape[-1]], mins[:y.shape[-1]])
    u = denormalize(u, factor[y.shape[-1]:], mins[y.shape[-1]:])

    fig, ax = plt.subplots(n_rows, n_columns, figsize=(n_columns * 4, n_rows * 4))
    ax = ax.flat

    for i in range(y.shape[-1]):
        ax[i].plot(t, y[:, i])
        ax[i].plot(t, y_pred[:, i], color='#e55e5e')
        ax[i].set_title(labels[i+1])
    for i in range(u.shape[-1]):
        ax[y.shape[-1] + i].plot(t, u[:, i])

    if filepath is not None:
        fig.savefig(filepath)


if __name__ == '__main__':

    seq_len = 300
    tfinal = 3
    deltat = 0.01
    win_size = 10
    dynamics = sys.argv[1] if len(sys.argv) > 1 else 'single_track' # 'multi_body'
    horizon = int(sys.argv[2]) if len(sys.argv) > 2 else 5

    if dynamics == 'multi_body':
        selected_inds = [0, 1, 2, 3, 4, 5]
        labels = [r"x-position [$m$]", r"y-position [$m$]", r"steering angle [$rad$]", r"velocity [$m/s$]", r"yaw angle [$rad$]", 
                  r"yaw rate [$rad/s$]", r"steering velocity [$rad/s$]", r"long. acceleration [$rad/s^2$]"]
    else:
        selected_inds = [0, 1, 2, 3]
        labels = [r"x-position [$m$]", r"y-position [$m$]", r"velocity [$m/s$]", r"yaw angle [$rad$]", 
                  r"steering angle [$rad$]", r"long. acceleration [$rad/s^2$]"]
    u_dim = 2
    x_dim = 32
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

    checkpoint_path = f'SUBNET_{dynamics}_horizon_{horizon}.pt'
    folder = os.path.join('..', 'results', checkpoint_path.replace('.pt', ''))
    os.mkdir(folder)

    data_path = '../data'

    y_train, u_train, y_past_train, u_past_train, factors, mins = \
        load_data(data_path + os.sep + f'{dynamics}_4000_x_train.npy',
                  data_path + os.sep + f'{dynamics}_4000_u_train.npy',
                  selected_inds, win_size, horizon, split=True, n_times=5)
    y_val, u_val, y_past_val, u_past_val, _, _ = \
        load_data(data_path + os.sep + f'{dynamics}_500_x_val.npy',
                  data_path + os.sep + f'{dynamics}_500_u_val.npy',
                  selected_inds, win_size, horizon, split=True, n_times=1, factor=factors, mins=mins)
    y_test, u_test, _, _ = \
        load_data(data_path + os.sep + f'{dynamics}_500_x_test.npy',
                  data_path + os.sep + f'{dynamics}_500_u_test.npy',
                  selected_inds, win_size, horizon, split=False, n_times=1, factor=factors, mins=mins)
    y_test_split, u_test_split, y_past_test_split, u_past_test_split, _, _ = \
        load_data(data_path + os.sep + f'{dynamics}_500_x_test.npy',
                  data_path + os.sep + f'{dynamics}_500_u_test.npy',
                  selected_inds, win_size, horizon, split=True, n_times=1, factor=factors, mins=mins)


    batch_size = 10000
    n_test = y_test.shape[0]
    init_epoch = 100
    epochs = 1000
    iterations = [int(y_train.shape[0]/batch_size), int(y_val.shape[0]/min(batch_size, y_val.shape[0]))]
    n_samples = 50
    grad_clip = 1e+2
    l2_reg = 1e-5
    init_bias = 0.001
    kernel_gamma = 0.5
    n_exp = int(np.log(0.1)/np.log(kernel_gamma))
    max_explr_epoch = 500
    start_lr = 5e-4
    ratio = 0.1

    train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(
        y_past_train, u_train, y_train), batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(
        y_past_val, u_val, y_val), batch_size=min(batch_size, u_val.shape[0]), shuffle=True)

    def init_weights(m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(init_bias)

    model = SUBNET(u_dim, x_dim, y_dim, hidden_layer_size_s, n_hidden_s, h_rnn_dim_s, h_linear_dim_s, horizon, win_size)

    model.apply(init_weights)

    early_stopping = EarlyStopping(patience=100, verbose=True, delta=0.0001, path=os.path.join(folder, checkpoint_path))

    # train flow 1
    gamma = kernel_gamma ** (1 / max_explr_epoch)
    trainable_parameters = [p for p in itertools.chain(model.parameters()) if p.requires_grad]
    optimizer = op.Adam(trainable_parameters, start_lr, weight_decay=l2_reg)
    lr_scheduler1 = op.lr_scheduler.StepLR(optimizer, init_epoch, gamma=0.2)
    lr_scheduler2 = op.lr_scheduler.ExponentialLR(optimizer, gamma=gamma)
    model.to(device)

    for ep in range(1, epochs + 1):
        with tqdm(total=iterations[0], desc='Training epoch {}'.format(ep), unit="epoch") as p_bar:
            train_loss, val_loss = \
                train_one_epoch(model, optimizer, train_loader, val_loader, iterations, ep, p_bar,
                                [lr_scheduler1, lr_scheduler2], grad_clip, init_epoch, max_explr_epoch)
            p_bar.set_postfix_str("t {0:.5f}, v {1:.5f}".format(
                train_loss, val_loss))
            p_bar.update(ep)

        early_stopping(train_loss, model)
        if early_stopping.early_stop:
            print("Early stopping")
            break

    model_test = SUBNET(u_dim, x_dim, y_dim, hidden_layer_size_s, n_hidden_s, h_rnn_dim_s, h_linear_dim_s,
                        horizon, win_size)
    model_test.load(os.path.join(folder, checkpoint_path))
    y_pred_test_split, _ = model_test.test(y_past_test_split, u_test_split).detach().numpy()
    y_pred_test_split = np.concatenate(
        [y_pred_test_split[(i * n_test):((i + 1) * n_test), :, :] for i in range(seq_len // horizon)], axis=1)
    y_test_split = torch.cat(
        [y_test_split[(i * n_test):((i + 1) * n_test), :, :] for i in range(seq_len // horizon)], dim=1).detach().numpy()
    u_test_split = torch.cat(
        [u_test_split[(i * n_test):((i + 1) * n_test), :, :] for i in range(seq_len // horizon)], dim=1).detach().numpy()
    
    np.save(os.path.join(folder, f'check_test_inference_y_pred.npy'), y_pred_test_split)
    
    t = np.linspace(0, tfinal, int(tfinal/deltat) + 1)
    for index in list(np.arange(0, n_test, 5)):
        plot_signals(t[:-1], y_test_split[index], y_pred_test_split[index], u_test_split[index], 
                     factors, mins, labels, n_columns=3, n_rows=2, filepath=os.path.join(folder, f'check_test_split_inference_{index}.png'))
        plt.close()