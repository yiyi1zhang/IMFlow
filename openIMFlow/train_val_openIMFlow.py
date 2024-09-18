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


class TcNF(nn.Module):
    """combine conditional INN blocks (RNVP Coupling Block) with conditional network"""

    def __init__(self, u_dim, cond_dim, summary_dim, n_blocks, hidden_layer_size, n_hidden, h_rnn_dim, h_linear_dim,
                 horizon, deltat=0.01):
        super().__init__()
        self.summary_dim = summary_dim
        self.u_dim = u_dim
        self.h_rnn_dim = h_rnn_dim
        self.horizon = horizon
        self.deltat = deltat
        self.cINN = RealNVP(n_blocks, u_dim, hidden_layer_size, n_hidden, cond_label_size=summary_dim, batch_norm=True)
        self.rnn1 = nn.GRU(cond_dim, summary_dim//2, batch_first=True)
        self.bn1_l = nn.BatchNorm1d(summary_dim//2)
        self.l1_l = nn.Linear(summary_dim//2, h_linear_dim)
        self.bn2_l = nn.BatchNorm1d(h_linear_dim)
        self.l2_l = nn.Sequential(nn.ELU(), nn.Linear(h_linear_dim, summary_dim//2))
        self.bn1_s = nn.BatchNorm1d(horizon)
        self.l1_s = nn.Linear(horizon, h_linear_dim)
        self.bn2_s = nn.BatchNorm1d(h_linear_dim)
        self.l2_s = nn.Sequential(nn.ELU(), nn.Linear(h_linear_dim, horizon))

    def forward(self, x, p):
        x_en, hx = self.rnn1(x)
        # long (N, horizon, dim)
        x_en_l = self.bn1_l(x_en.transpose(1, 2))
        x_en_l = self.l1_l(x_en_l.transpose(1, 2))
        x_en_l = self.bn2_l(x_en_l.transpose(1, 2))
        x_en_l = self.l2_l(x_en_l.transpose(1, 2))
        # short (N, dim, horizon)
        x_en_s = self.bn1_s(x_en)
        x_en_s = self.l1_s(x_en_s.transpose(1, 2))
        x_en_s = self.bn2_s(x_en_s.transpose(1, 2))
        x_en_s = self.l2_s(x_en_s.transpose(1, 2))
        x_en = torch.cat([x_en_l, x_en_s.transpose(1, 2)], dim=-1)
        z, jac = self.cINN(p, x_en)
        z_loss = 0
        for t in range(self.horizon):
            z_loss += 0.5 * torch.pow(z[:, t, :] / (t + 1), 2) / self.deltat
        log_det_J = torch.sum(jac, dim=-2)

        loss = torch.mean(torch.sum((z_loss - log_det_J) / self.horizon, dim=1))

        return loss

    def inverse(self, x):
        device = x.device
        x_en, hx = self.rnn1(x)
        # long (N, horizon, dim)
        x_en_l = self.bn1_l(x_en.transpose(1, 2))
        x_en_l = self.l1_l(x_en_l.transpose(1, 2))
        x_en_l = self.bn2_l(x_en_l.transpose(1, 2))
        x_en_l = self.l2_l(x_en_l.transpose(1, 2))
        # short (N, dim, horizon)
        x_en_s = self.bn1_s(x_en)
        x_en_s = self.l1_s(x_en_s.transpose(1, 2))
        x_en_s = self.bn2_s(x_en_s.transpose(1, 2))
        x_en_s = self.l2_s(x_en_s.transpose(1, 2))
        x_en = torch.cat([x_en_l, x_en_s.transpose(1, 2)], dim=-1)
        z = torch.stack(
            [(t + 1) * np.sqrt(self.deltat) * torch.randn((x.shape[0], self.u_dim), device=device) for t in
             range(self.horizon)], dim=1)
        p, _ = self.cINN.inverse(z, x_en)
        return p

    def test(self, loader, n_samples):
        p = []
        for it in range(n_iter_test):
            x, = next(iter(loader))
            x_en, hx = self.rnn1(x)
            # long (N, horizon, dim)
            x_en_l = self.bn1_l(x_en.transpose(1, 2))
            x_en_l = self.l1_l(x_en_l.transpose(1, 2))
            x_en_l = self.bn2_l(x_en_l.transpose(1, 2))
            x_en_l = self.l2_l(x_en_l.transpose(1, 2))
            # short (N, dim, horizon)
            x_en_s = self.bn1_s(x_en)
            x_en_s = self.l1_s(x_en_s.transpose(1, 2))
            x_en_s = self.bn2_s(x_en_s.transpose(1, 2))
            x_en_s = self.l2_s(x_en_s.transpose(1, 2))
            x_en = torch.cat([x_en_l, x_en_s.transpose(1, 2)], dim=-1)
            z = torch.stack(
                [(t + 1) * np.sqrt(self.deltat) * torch.randn((n_samples, x.shape[0], self.u_dim), device=x.device) for t in
                 range(self.horizon)], dim=2)
            _p, _ = self.cINN.inverse(z, torch.stack([x_en]*n_samples, dim=0))
            p.append(_p)
        return torch.cat(p)

    def sample_together(self, x, n_samples):
        device = x.device
        x_en, hx = self.rnn1(x)
        # long (N, horizon, dim)
        x_en_l = self.bn1_l(x_en.transpose(1, 2))
        x_en_l = self.l1_l(x_en_l.transpose(1, 2))
        x_en_l = self.bn2_l(x_en_l.transpose(1, 2))
        x_en_l = self.l2_l(x_en_l.transpose(1, 2))
        # short (N, dim, horizon)
        x_en_s = self.bn1_s(x_en)
        x_en_s = self.l1_s(x_en_s.transpose(1, 2))
        x_en_s = self.bn2_s(x_en_s.transpose(1, 2))
        x_en_s = self.l2_s(x_en_s.transpose(1, 2))
        x_en = torch.cat([x_en_l, x_en_s.transpose(1, 2)], dim=-1)
        z = torch.stack(
            [(t + 1) * np.sqrt(self.deltat) * torch.randn((n_samples, x.shape[0], self.u_dim), device=device) for t in
             range(self.horizon)], dim=2)
        p, _ = self.cINN.inverse(z, torch.stack([x_en]*n_samples, dim=0))
        return p

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


def split_data(data, horizon, times=1):
    length = data.shape[1]
    datas = []
    for i in range(times):
        _data = np.concatenate([data[:, i:(horizon * (length // horizon)), :], data[:, :i, :]], axis=1)
        datas += np.array_split(_data, length // horizon, axis=1)
    return np.concatenate(datas, axis=0)


def load_data(file_x, file_u, selected_inds, length=0, split=False, times=1, factor=None, mins=None):

    Y = np.load(file_x)[:, :, selected_inds][:, 1:, :]
    Y, factory, miny = normalize(Y, factor=factor[:Y.shape[-1]] if factor is not None else None,
                                 min_data=mins[:Y.shape[-1]] if mins is not None else None)
    if split:
        Y = split_data(Y, length, times=times)

    U = np.load(file_u)[:, 1:, :]
    U, factoru, minu = normalize(U, factor=factor[Y.shape[-1]:] if factor is not None else None,
                                 min_data=mins[Y.shape[-1]:] if mins is not None else None)
    if split:
        U = split_data(U, length, times=times)

    Y_tensor = torch.tensor(Y, dtype=torch.float32)
    U_tensor = torch.tensor(U, dtype=torch.float32)

    return Y_tensor, U_tensor, np.concatenate([factory, factoru], axis=-1), np.concatenate([miny, minu], axis=-1)


def train_one_epoch(model, optimizer, train_loader, val_loader, iterations, ep, pbar, lr_scheduler):
    # training
    model.train(True)
    for it in range(iterations[0]):
        y, u = next(iter(train_loader))
        y, u = y.to(device), u.to(device)
        train_loss = model(y, u)
        train_loss.retain_grad()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()
        pbar.update(1)
    model.train(False)
    for it in range(iterations[1]):
        y_val, u_val = next(iter(val_loader))
        y_val, u_val = y_val.to(device), u_val.to(device)
        val_loss = model(y_val, u_val)
    if ep <= init_epoch:
        lr_scheduler[0].step()
    elif init_epoch < ep < 3*max_explr_epoch:
        lr_scheduler[1].step()

    return train_loss.item(), val_loss.item()


def confidence_interval(data, confidence=0.95):
    return st.norm.interval(alpha=confidence, loc=np.mean(data, axis=0), scale=np.std(data, axis=0))


def plot_signals(t, y, u, u_approx, factor, mins, labels, n_columns=3, n_rows=2, filepath=None):
    y = denormalize(y, factor[:y.shape[-1]], mins[:y.shape[-1]])
    u = denormalize(u, factor[y.shape[-1]:], mins[y.shape[-1]:])
    u_approx = denormalize(u_approx, factor[y.shape[-1]:], min[y.shape[-1]:])
    u_mean = u_approx.mean(axis=0)
    CI = confidence_interval(u_approx)
    u_approx_min = CI[0]
    u_approx_max = CI[1]

    fig, ax = plt.subplots(n_rows, n_columns, figsize=(n_columns * 4, n_rows * 4))
    ax = ax.flat

    for i in range(y.shape[-1]):
        ax[i].plot(t, y[:, i])
        ax[i].set_title(labels[i])
    for i in range(u.shape[-1]):
        ax[y.shape[-1] + i].plot(t, u[:, i])
        ax[y.shape[-1] + i].set_title(labels[y.shape[-1] + i])
        ax[y.shape[-1] + i].plot(t, u_mean[:, i])
        ax[y.shape[-1] + i].fill_between(t, u_approx_min[:, i], u_approx_max[:, i], alpha=0.8, color='#96eba7')

    if filepath is not None:
        fig.savefig(filepath)


if __name__ == '__main__':

    seq_len = 300
    tfinal = 3
    deltat = 0.01
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
    step_size = horizon
    u_dim = 2
    y_dim = len(selected_inds)
    n_blocks = 4
    batch_size = 10000
    n_test = 500
    init_epoch = 100
    epochs = 2500
    summary_dim = 64
    hidden_layer_size = 64
    n_layers = 2
    h_linear_dim = 64
    h_rnn_dim = 64
    n_samples = 50
    grad_clip = 1e+2
    l2_reg = 1e-5
    init_bias = 0.001
    kernel_gamma = 0.5
    n_exp = int(np.log(0.1)/np.log(kernel_gamma))

    max_explr_epoch = 500
    start_lr = 5e-4
    dropout_rate = 0.0

    checkpoint_path = f'Bayesian_conditional_normalizing_flow_RNN_RealNVP_transpose_{dynamics}.pt'

    folder = os.path.join('..', 'results', checkpoint_path.replace('.pt', ''))
    os.mkdir(folder)
    data_path = '../data'
    
    y_train, u_train, factor, mins = \
        load_data(data_path + os.sep + f'{dynamics}_4000_x_train.npy',
                  data_path + os.sep + f'{dynamics}_4000_u_train.npy',
                  selected_inds, horizon, split=True, times=5)
    y_val, u_val, _, _ = \
        load_data(data_path + os.sep + f'{dynamics}_500_x_val.npy',
                  data_path + os.sep + f'{dynamics}_500_u_val.npy',
                  selected_inds, horizon, split=True, times=1, factor=factor, min_data=mins)
    y_test, u_test, _, _ = \
        load_data(data_path + os.sep + f'{dynamics}_500_x_test.npy',
                  data_path + os.sep + f'{dynamics}_500_u_test.npy',
                  selected_inds, horizon, split=True, times=1, factor=factor, min_data=mins)
    model = TcNF(u_dim, y_dim, summary_dim, n_blocks, hidden_layer_size, n_layers, h_rnn_dim, h_linear_dim, horizon)

    train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(y_train, u_train), batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(y_val, u_val),
                                            batch_size=min(batch_size, u_val.shape[0]), shuffle=True)

    def init_weights(m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(init_bias)

    model.apply(init_weights)

    early_stopping = EarlyStopping(patience=500, verbose=True, delta=0.0001, path=os.path.join(folder, checkpoint_path))

    # train flow 1
    gamma = kernel_gamma ** (1 / max_explr_epoch)
    trainable_parameters = [p for p in itertools.chain(model.parameters()) if p.requires_grad]
    optimizer = op.Adam(trainable_parameters, start_lr, weight_decay=l2_reg)
    lr_scheduler1 = op.lr_scheduler.StepLR(optimizer, init_epoch, gamma=0.2)
    lr_scheduler2 = op.lr_scheduler.ExponentialLR(optimizer, gamma=gamma)
    model.to(device)
    iterations = [int(y_train.shape[0]/batch_size), int(y_val.shape[0]/min(batch_size, y_val.shape[0]))]
    for ep in range(1, epochs + 1):
        with tqdm(total=epochs, desc='Training epoch {}'.format(ep), unit="epoch") as p_bar:
            train_loss, val_loss = train_one_epoch(model, optimizer, train_loader, val_loader,
                                                    iterations, ep, p_bar, [lr_scheduler1, lr_scheduler2])
            p_bar.set_postfix_str("train_loss {0:.5f}, val_loss {1:.5f}".format(train_loss, val_loss))
            p_bar.update(ep)

        early_stopping(train_loss, model)
        if early_stopping.early_stop:
            print("Early stopping")
            break
    
    n_iter_test = 100
    test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(y_test), batch_size=y_test.shape[0]//n_iter_test, shuffle=True)
    model_test = TcNF(u_dim, y_dim, summary_dim, n_blocks, hidden_layer_size, n_layers, h_rnn_dim, h_linear_dim, horizon)
    model_test.load(folder, checkpoint_path)
    u_samples_test = model_test.test(test_loader, n_samples).detach().numpy()
    u_samples_test = np.concatenate([u_samples_test[:, i * n_test:(i + 1) * n_test, :, :] 
                                     for i in range(u_samples_test.shape[1] // n_test)], axis=2)
    np.save(os.path.join(folder, f'check_test_inference_u_samples.npy'), u_samples_test)

    y_test = np.concatenate([y_test[i * n_test:(i + 1) * n_test, :, :] for i in range(y_test.shape[0] // n_test)], axis=1)
    u_test = np.concatenate([u_test[i * n_test:(i + 1) * n_test, :, :] for i in range(u_test.shape[0] // n_test)], axis=1)

    t = np.linspace(0, tfinal, int(tfinal/deltat) + 1)

    for index in list(np.arange(0, n_test, 5)):
        plot_signals(t[:-1], y_test[index], u_test[index], u_samples_test[:, index, :, :], factor, mins, labels, 
                     n_columns=3, n_rows=3 if dynamics == 'multi_body' else 2, 
                     filepath=os.path.join(folder, f'check_test_inference_{index}.png'))
        plt.close()