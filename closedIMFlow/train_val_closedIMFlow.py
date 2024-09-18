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

sys.path.append("..")
from utils.pts_flows import RealNVP
from utils.early_stop import EarlyStopping
from subnet import SUBNET

if torch.cuda.is_available():
    device = torch.device("cuda:1")
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


class TcNF(nn.Module):
    """combine conditional INN blocks (RNVP Coupling Block) with conditional network"""

    def __init__(self, u_dim, x_dim, y_dim, summary_dim, n_blocks, hidden_layer_size, n_hidden, h_rnn_dim, h_linear_dim,
                 horizon, deltat=0.01):
        super().__init__()
        self.summary_dim = summary_dim
        self.u_dim = u_dim
        self.x_dim = x_dim
        self.y_dim = y_dim
        self.h_rnn_dim = h_rnn_dim
        self.horizon = horizon
        self.deltat = deltat
        self.cINN = RealNVP(n_blocks, u_dim, hidden_layer_size, n_hidden, cond_label_size=summary_dim, batch_norm=True)
        self.rnny = nn.GRU(y_dim, h_rnn_dim, batch_first=True)
        self.bn1y = nn.BatchNorm1d(h_rnn_dim)
        self.l1y = nn.Linear(h_rnn_dim, h_linear_dim)
        self.bn2y = nn.BatchNorm1d(h_linear_dim)
        self.l2y = nn.Sequential(nn.ELU(), nn.Linear(h_linear_dim, int(summary_dim//2) - x_dim))
        self.bn1t = nn.BatchNorm1d(horizon)
        self.l1t = nn.Linear(horizon, h_linear_dim)
        self.bn2t = nn.BatchNorm1d(h_linear_dim)
        self.l2t = nn.Sequential(nn.ELU(), nn.Linear(h_linear_dim, horizon))

    def forward(self, x, y, u):
        # x N X x_dim
        x_en = torch.repeat_interleave(x.unsqueeze(1), repeats=self.horizon, dim=1)
        # y N*h*y_dim, u N*h*u_dim
        y_en, hy = self.rnny(y)
        y_en = self.bn1y(y_en.transpose(1, 2))
        y_en = self.l1y(y_en.transpose(1, 2))
        y_en = self.bn2y(y_en.transpose(1, 2))
        y_en = self.l2y(y_en.transpose(1, 2))
        cond = torch.cat([x_en, y_en], dim=-1)

        # transpose (N, dim, win_size)
        condt = cond.transpose(1, 2)
        condt = self.bn1t(condt.transpose(1, 2))
        condt = self.l1t(condt.transpose(1, 2))
        condt = self.bn2t(condt.transpose(1, 2))
        condt = self.l2t(condt.transpose(1, 2))

        z, jac = self.cINN(u, torch.cat([cond, condt.transpose(1, 2)], dim=-1))
        z_loss = 0
        for t in range(self.horizon):
            z_loss += 0.5 * torch.pow(z[:, t, :] / (t + 1), 2) / self.deltat
        log_det_J = torch.sum(jac, dim=-2)

        loss = torch.mean(torch.sum((z_loss - log_det_J) / self.horizon, dim=1))

        return loss

    def inverse(self, x, y):
        device = x.device
        x_en = torch.repeat_interleave(x.unsqueeze(1), repeats=self.horizon, dim=1)
        # y N*h*y_dim, u N*h*u_dim
        y_en, hy = self.rnny(y)
        y_en = self.bn1y(y_en.transpose(1, 2))
        y_en = self.l1y(y_en.transpose(1, 2))
        y_en = self.bn2y(y_en.transpose(1, 2))
        y_en = self.l2y(y_en.transpose(1, 2))
        cond = torch.cat([x_en, y_en], dim=-1)

        # transpose (N, dim, win_size)
        condt = cond.transpose(1, 2)
        condt = self.bn1t(condt.transpose(1, 2))
        condt = self.l1t(condt.transpose(1, 2))
        condt = self.bn2t(condt.transpose(1, 2))
        condt = self.l2t(condt.transpose(1, 2))

        z = torch.stack(
            [(t + 1) * np.sqrt(self.deltat) * torch.randn((x.shape[0], self.input_dim), device=device) for t in
             range(self.horizon)], dim=1)
        p, _ = self.cINN.inverse(z, torch.cat([cond, condt.transpose(1, 2)], dim=-1))
        return p

    def sample_together(self, x, y, n_samples):
        device = x.device
        x_en = torch.repeat_interleave(x.unsqueeze(1), repeats=self.horizon, dim=1)
        # y N*h*y_dim, u N*h*u_dim
        y_en, hy = self.rnny(y)
        y_en = self.bn1y(y_en.transpose(1, 2))
        y_en = self.l1y(y_en.transpose(1, 2))
        y_en = self.bn2y(y_en.transpose(1, 2))
        y_en = self.l2y(y_en.transpose(1, 2))
        cond = torch.cat([x_en, y_en], dim=-1)

        # transpose (N, dim, win_size)
        condt = cond.transpose(1, 2)
        condt = self.bn1t(condt.transpose(1, 2))
        condt = self.l1t(condt.transpose(1, 2))
        condt = self.bn2t(condt.transpose(1, 2))
        condt = self.l2t(condt.transpose(1, 2))

        conds = torch.cat([cond, condt.transpose(1, 2)], dim=-1)
        z = torch.stack(
            [(t + 1) * np.sqrt(self.deltat) * torch.randn((n_samples, x.shape[0], self.u_dim), device=device) for t in
             range(self.horizon)], dim=2)
        p, _ = self.cINN.inverse(z, torch.stack([conds]*n_samples, dim=0))
        return p


class SIFlow(nn.Module):
    def __init__(self, u_dim, x_dim, y_dim, hidden_layer_size_s, n_hidden_s, h_rnn_dim_s, h_linear_dim_s, summary_dim,
                 n_blocks, hidden_layer_size_f, n_hidden_f, h_rnn_dim_f, h_linear_dim_f, horizon, win, deltat=0.01):
        super().__init__()
        self.u_dim = u_dim
        self.x_dim = x_dim
        self.y_dim = y_dim
        self.horizon = horizon
        self.win = win
        self.subnet = SUBNET(u_dim, x_dim, y_dim, hidden_layer_size_s, n_hidden_s, h_rnn_dim_s, h_linear_dim_s,
                             horizon, win)
        self.cnf = TcNF(u_dim, x_dim, y_dim, summary_dim, n_blocks, hidden_layer_size_f, n_hidden_f, h_rnn_dim_f,
                        h_linear_dim_f, horizon, deltat=deltat)

    def parameters(self, recurse: bool = True):
        return itertools.chain(self.subnet.rnn1.parameters(), self.subnet.bn1.parameters(),
                               self.subnet.l1.parameters(), self.subnet.bn2.parameters(),
                               self.subnet.l2.parameters(), self.cnf.parameters())

    def forward(self, y_past, u, y, noise=0.0):
        y_past += max(noise, 0.01) * torch.randn_like(y_past)
        x0, y_pred = self.subnet(y_past, u, y)
        loss_cnf = self.cnf(x0, y, u)
        loss_subnet = nn.MSELoss()(y, y_pred)

        return loss_cnf, loss_subnet

    def test(self, y_past, y, n_samples):
        try:
            x0 = self.subnet.encoder(y_past)
            us = self.cnf.sample_together(x0, y, n_samples)
            y_pred = self.subnet.test(x0, us.mean(0))
        except:
            us = []
            y_pred = []
            for i in range(y_past.shape[0] // 100):
                _x0 = self.subnet.encoder(y_past[i*100:(i+1)*100])
                _us = self.cnf.sample_together(_x0, y[i*100:(i+1)*100], n_samples)
                _y_pred = self.subnet.test(_x0, _us.mean(0))
                us.append(_us)
                y_pred.append(_y_pred)
                us = torch.cat(_us)
                y_pred = torch.cat(_y_pred)
        return us, y_pred

    def save(self, name):
        torch.save({
            'subnet': self.subnet.state_dict(),
            'cnf': self.cnf.state_dict()
        }, name)

    def load_subnets(self, name):
        print(f"load model from {name}")
        state_dicts = torch.load(name, map_location=torch.device('cpu'))
        self.subnet.load_state_dict(state_dicts)
        with torch.no_grad():
            self.subnet.eval()

    def load(self, name):
        print(f"load model from {name}")
        state_dicts = torch.load(name, map_location=torch.device('cpu'))
        self.subnet.load_state_dict(state_dicts['subnet'])
        self.cnf.load_state_dict(state_dicts['cnf'])
        with torch.no_grad():
            self.subnet.eval()
            self.cnf.eval()


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
    length = 301
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

        return Y_tensor, U_tensor, Y_past_tensor, U_past_tensor, np.concatenate([factory, factoru], axis=-1), \
            np.concatenate([miny, minu], axis=-1)
    else:
        Y_tensor = torch.tensor(Y, dtype=torch.float32)
        U_tensor = torch.tensor(U, dtype=torch.float32)
        return Y_tensor, U_tensor, np.concatenate([factory, factoru], axis=-1), np.concatenate([miny, minu], axis=-1)


def train_one_epoch(model, optimizer, train_loader, val_loader, iterations, ep, pbar,
                    lr_scheduler, grad_clip, init_epoch, max_explr_epoch, ratio=1.0):
    # training
    model.train(True)
    for it in range(iterations[0]):
        y_past, u, y = next(iter(train_loader))
        y_past, u, y = y_past.to(device), u.to(device), y.to(device)
        train_loss_cnf, train_loss_subnet = model(y_past, u, y, noise=noise)
        train_loss = ratio * train_loss_cnf + train_loss_subnet
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
        val_loss_cnf, val_loss_subnet = model(y_past_val, u_val, y_val, noise=noise)
        val_loss = ratio * val_loss_cnf + val_loss_subnet
    if ep <= init_epoch:
        lr_scheduler[0].step()
    elif init_epoch < ep < 3*max_explr_epoch:
        lr_scheduler[1].step()

    return train_loss.item(), train_loss_cnf.item(), train_loss_subnet.item(),\
           val_loss.item(), val_loss_cnf.item(), val_loss_subnet.item()


def confidence_interval(data, confidence=0.95):
    return st.norm.interval(alpha=confidence, loc=np.mean(data, axis=0), scale=np.std(data, axis=0))


def plot_signals(t, y, y_pred, u, u_approx, factor, mins, labels, n_columns=3, n_rows=2, filepath=None):
    y = denormalize(y, factor[:y.shape[-1]], mins[:y.shape[-1]])
    y_pred = denormalize(y_pred, factor[:y.shape[-1]], mins[:y.shape[-1]])
    u = denormalize(u, factor[y.shape[-1]:], mins[:y.shape[-1]])
    u_approx = denormalize(u_approx, factor[y.shape[-1]:], mins[:y.shape[-1]])
    u_mean = u_approx.mean(axis=0)
    CI = confidence_interval(u_approx)
    u_approx_min = CI[0]
    u_approx_max = CI[1]

    fig, ax = plt.subplots(n_rows, n_columns, figsize=(n_columns * 4, n_rows * 4))
    ax = ax.flat

    for i in range(y.shape[-1]):
        ax[i].plot(t, y[:, i])
        ax[i].plot(t, y_pred[:, i], color='#e55e5e')
        ax[i].set_title(labels[i])
    for i in range(u.shape[-1]):
        ax[y.shape[-1] + i].plot(t, u[:, i])
        ax[y.shape[-1] + i].set_title(labels[y.shape[-1] + i])
        ax[y.shape[-1] + i].plot(t, u_mean[:, i], color='#e55e5e')
        ax[y.shape[-1] + i].fill_between(t, u_approx_min[:, i], u_approx_max[:, i], alpha=0.2, color='#e55e5e')

    if filepath is not None:
        fig.savefig(filepath)


if __name__ == '__main__':

    tfinal = 3
    deltat = 0.01
    win_size = 10
    u_dim = 2
    x_dim = 32
    dynamics = sys.argv[1] if len(sys.argv) > 1 else 'single_track' # 'multi_body'
    horizon = int(sys.argv[2]) if len(sys.argv) > 2 else 5
    noise = float(sys.argv[3]) if len(sys.argv) > 3 else 0.0

    if dynamics == 'multi_body':
        selected_inds = [0, 1, 2, 3, 4, 5]
        labels = [r"x-position [$m$]", r"y-position [$m$]", r"steering angle [$rad$]", r"velocity [$m/s$]", r"yaw angle [$rad$]", 
                  r"yaw rate [$rad/s$]", r"steering velocity [$rad/s$]", r"long. acceleration [$rad/s^2$]"]
    else:
        selected_inds = [0, 1, 2, 3]
        labels = [r"x-position [$m$]", r"y-position [$m$]", r"velocity [$m/s$]", r"yaw angle [$rad$]", 
                  r"steering angle [$rad$]", r"long. acceleration [$rad/s^2$]"]
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

    checkpoint_path = f'Bayesian_conditional_normalizing_flow_RNN_RealNVP_SUBNET_transpose_{dynamics}_horizon_{horizon}.pt'
    if noise > 0:
        noise_str = str(noise).replace('.', '')
        checkpoint_path = checkpoint_path.replace('.pt', f'_noise_{noise_str}.pt')

    folder = os.path.join('..', 'results', checkpoint_path.replace('.pt', ''))
    os.mkdir(folder)

    data_path = '../data'

    y_train, u_train, y_past_train, u_past_train, factor, mins = \
        load_data(data_path + os.sep + f'{dynamics}_4000_x_train.npy',
                  data_path + os.sep + f'{dynamics}_4000_u_train.npy',
                  selected_inds, win_size, horizon, split=True, n_times=5)
    y_val, u_val, y_past_val, _, _ = \
        load_data(data_path + os.sep + f'{dynamics}_500_x_val.npy',
                  data_path + os.sep + f'{dynamics}_500_u_val.npy',
                  selected_inds, win_size, horizon, split=True, n_times=1, factor=factor, mins=mins)
    y_test, u_test, _, _ = \
        load_data(data_path + os.sep + f'{dynamics}_500_x_test.npy',
                  data_path + os.sep + f'{dynamics}_500_u_test.npy',
                  selected_inds, win_size, horizon, split=False, n_times=1, factor=factor, mins=mins)
    y_test_split, u_test_split, y_past_test_split, _, _ = \
        load_data(data_path + os.sep + f'{dynamics}_500_x_test.npy',
                  data_path + os.sep + f'{dynamics}_500_u_test.npy',
                  selected_inds, win_size, horizon, split=True, n_times=1, factor=factor, mins=mins)

    batch_size = 10000
    n_test = 500
    init_epoch = 100
    epochs = 5000
    iterations = [int(y_train.shape[0]/batch_size), int(y_val.shape[0]/min(batch_size, y_val.shape[0]))]
    n_samples = 50
    grad_clip = 1e+2
    l2_reg = 1e-5
    init_bias = 0.001
    kernel_gamma = 0.5
    n_exp = int(np.log(0.1)/np.log(kernel_gamma))
    max_explr_epoch = 500
    start_lr = 5e-4
    ratio = 1.0

    train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(y_past_train, u_train, y_train),
                                               batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(y_past_val, u_val, y_val),
                            batch_size=min(batch_size, u_val.shape[0]), shuffle=True)
    
    def init_weights(m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(init_bias)
    
    model = SIFlow(u_dim, x_dim, y_dim, hidden_layer_size_s, n_hidden_s, h_rnn_dim_s, h_linear_dim_s, summary_dim,
             n_blocks, hidden_layer_size_f, n_hidden_f, h_rnn_dim_f, h_linear_dim_f, horizon, win_size, deltat=0.01)
    
    model.cnf.apply(init_weights)
    
    subnet_path = os.path.join('..', 'results', f'SUBNET_{dynamics}_horizon_{horizon}', f'SUBNET_{dynamics}_horizon_{horizon}.pt')
    model.load_subnets(subnet_path)
    
    early_stopping = EarlyStopping(patience=500, verbose=True, delta=0.0001, path=os.path.join(folder, checkpoint_path))
    
    # train flow 1
    gamma = kernel_gamma ** (1 / max_explr_epoch)
    trainable_parameters = [p for p in itertools.chain(model.parameters()) if p.requires_grad]
    optimizer = op.Adam(trainable_parameters, start_lr, weight_decay=l2_reg)
    lr_scheduler1 = op.lr_scheduler.StepLR(optimizer, init_epoch, gamma=0.2)
    lr_scheduler2 = op.lr_scheduler.ExponentialLR(optimizer, gamma=gamma)
    model.to(device)
    
    for ep in range(1, epochs + 1):
        with tqdm(total=iterations[0], desc='Training epoch {}'.format(ep), unit="epoch") as p_bar:
            train_loss, train_loss_cnf, train_loss_subnet, val_loss, val_loss_cnf, val_loss_subnet = \
                train_one_epoch(model, optimizer, train_loader, val_loader, iterations, ep, p_bar,
                                [lr_scheduler1, lr_scheduler2], grad_clip, init_epoch, max_explr_epoch, ratio=ratio)
            p_bar.set_postfix_str("t {0:.5f}, tf {1:.5f}, ts {2:.5f}, v {3:.5f}, vf {4:.5f}, vs {5:.5f}".format(
                train_loss, train_loss_cnf, train_loss_subnet, val_loss, val_loss_cnf, val_loss_subnet))
            p_bar.update(ep)
    
        early_stopping(train_loss, model)
        if early_stopping.early_stop:
            if ep < 2000:
                early_stopping.reset()
            else:
                print("Early stopping")
                break

    model_test = SIFlow(u_dim, x_dim, y_dim, hidden_layer_size_s, n_hidden_s, h_rnn_dim_s, h_linear_dim_s,
                        summary_dim, n_blocks, hidden_layer_size_f, n_hidden_f, h_rnn_dim_f, h_linear_dim_f,
                        horizon, win_size, deltat=deltat)
    model_test.load(os.path.join(folder, checkpoint_path))
    if noise > 0:
        y_past_test_split += noise * torch.randn_like(y_past_test_split)
    u_samples_test, y_pred_test = model_test.test(y_past_test_split, y_test_split, n_samples)

    u_samples_test = np.concatenate([u_samples_test[:, i*n_test:(i+1)*n_test, :, :] for i in range(u_samples_test.shape[1]//n_test)], axis=2)
    y_pred_test = np.concatenate([y_pred_test[i * n_test:(i + 1) * n_test, :, :] for i in range(y_pred_test.shape[0] // n_test)], axis=1)
    y_test = np.concatenate([y_test[i * n_test:(i + 1) * n_test, :, :] for i in range(y_test.shape[0] // n_test)], axis=1)
    u_test = np.concatenate([u_test[i * n_test:(i + 1) * n_test, :, :] for i in range(u_test.shape[0] // n_test)], axis=1)

    t = np.linspace(0, tfinal, int(tfinal/deltat) + 1)

    for index in list(np.arange(0, n_test, 5)):
        plot_signals(t[:-1], y_test[index][:-1], y_pred_test[index], u_test[index][:-1], u_samples_test[:, index, :, :],
                        factor, mins, labels, n_columns=3, n_rows=3 if dynamics == 'multi_body' else 2,
                        filepath=os.path.join(folder, f'check_test_inference_{index}.png'))
        plt.close()