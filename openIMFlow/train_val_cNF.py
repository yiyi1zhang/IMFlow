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



class RNNcNF(nn.Module):
    """combine conditional INN blocks (RNVP Coupling Block) with conditional network"""

    def __init__(self, u_dim, cond_dim, summary_dim, n_blocks, hidden_layer_size, n_hidden, h_rnn_dim, h_linear_dim):
        super().__init__()
        self.summary_dim = summary_dim
        self.u_dim = u_dim
        self.h_rnn_dim = h_rnn_dim
        self.cINN = RealNVP(n_blocks, u_dim, hidden_layer_size, n_hidden, cond_label_size=summary_dim, batch_norm=True)
        self.l1 = nn.Linear(cond_dim, h_linear_dim)
        self.bn2 = nn.BatchNorm1d(h_linear_dim)
        self.l2 = nn.Sequential(nn.ELU(), nn.Linear(h_linear_dim, summary_dim))

    def forward(self, x, p):
        x = x.reshape(-1, x.shape[-1])
        p = p.reshape(-1, p.shape[-1])
        x_en = self.l1(x)
        x_en = self.bn2(x_en)
        x_en = self.l2(x_en)
        z, jac = self.cINN(p, x_en)
        loss = torch.mean(torch.sum((0.5 * torch.pow(z, 2) - jac) / win_size, dim=1))

        return loss

    def inverse(self, x, device='cpu'):
        x = x.reshape(-1, x.shape[-1])
        x_en = self.l1(x)
        x_en = self.bn2(x_en)
        x_en = self.l2(x_en)
        z = torch.randn((x.shape[0], self.u_dim), device=device)
        _p_hat, _ = self.cINN.inverse(z, x_en)
        return _p_hat.reshape(n_test, seq_len, u_dim)

    def sample(self, x, n_samples):
        device = x.device
        _n_test, seq_len, _ = x.shape
        x = x.reshape(-1, x.shape[-1])
        x_en = self.l1(x)
        x_en = self.bn2(x_en)
        x_en = self.l2(x_en)
        z = torch.randn((n_samples, x.shape[0], self.u_dim), device=device)
        _p_hat, _ = self.cINN.inverse(z, torch.stack([x_en]*n_samples))
        return _p_hat.reshape(n_samples, _n_test, seq_len, self.u_dim)

    def test(self, loader, n_samples):
        p = []
        for it in range(n_iter_test):
            x, = next(iter(loader))
            _n_test = x.shape[0]
            x = x.reshape(-1, x.shape[-1])
            x_en = self.l1(x)
            x_en = self.bn2(x_en)
            x_en = self.l2(x_en)
            z = torch.randn((n_samples, x.shape[0], self.u_dim), device=x.device)
            _p_hat, _ = self.cINN.inverse(z, torch.stack([x_en] * n_samples))
            p.append(_p_hat.reshape(n_samples, _n_test, seq_len, u_dim))
        return torch.cat(p, dim=1)

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


def load_data(file_sig, file_p, factor=None):
    """
    load actuator data
    split them into train, validation and test data
    :return: tensors
    """

    Y = np.load(file_sig)[:, :, selected_inds][:, 1:, :]
    Y, factory, miny = normalize(Y, factor=factor[:Y.shape[-1]] if factor is not None else None,
                                 min_data=mins[:Y.shape[-1]] if mins is not None else None)

    U = np.load(file_p)[:, 1:, :]
    U, factoru, minu = normalize(U, factor=factor[Y.shape[-1]:] if factor is not None else None,
                                 min_data=mins[Y.shape[-1]:] if mins is not None else None)

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
    u_approx = denormalize(u_approx, factor[y.shape[-1]:], mins[y.shape[-1]:])
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
        ax[y.shape[-1] + i].plot(t, u_mean[:, i], color='#e55e5e')
        ax[y.shape[-1] + i].fill_between(t, u_approx_min[:, i], u_approx_max[:, i], alpha=0.2, color='#e55e5e')

    if filepath is not None:
        fig.savefig(filepath)
        plt.close()



if __name__ == '__main__':
    seq_len = 300
    tfinal = 3
    deltat = 0.01
    win_size = 1
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
    batch_size = 100
    n_test = 500
    init_epoch = 100
    epochs = 5000
    summary_dim = 64
    hidden_layer_size = 64
    n_layers = 2
    h_linear_dim = 64
    h_rnn_dim = 64
    iteration_per_epoch = 18
    n_samples = 50
    grad_clip = 1e+2
    l2_reg = 1e-5
    init_bias = 0.001
    kernel_gamma = 0.5
    n_exp = int(np.log(0.1)/np.log(kernel_gamma))

    max_explr_epoch = 500
    start_lr = 5e-4
    dropout_rate = 0.0

    checkpoint_path = f'Bayesian_conditional_normalizing_flow_RNN_RealNVP_single_step_{dynamics}.pt'
    folder = os.path.join('..', 'results', checkpoint_path.replace('.pt', ''))
    os.mkdir(folder)

    data_path = '../data'
    
    y_train, u_train, factor, mins = \
        load_data(os.path.join(data_path, f'{dynamics}_4000_x_train.npy'),
                  os.path.join(data_path, f'{dynamics}_4000_u_train.npy'))
    y_val, u_val, _, _ = \
        load_data(os.path.join(data_path, f'{dynamics}_500_x_val.npy'),
                  os.path.join(data_path, f'{dynamics}_500_u_val.npy'),
                  factor=factor, min_data=mins)
    y_test, u_test, _, _ = \
        load_data(os.path.join(data_path, f'{dynamics}_500_x_test.npy'),
                  os.path.join(data_path, f'{dynamics}_500_u_test.npy'),
                  factor=factor, min_data=mins)
    model = RNNcNF(u_dim, y_dim, summary_dim, n_blocks, hidden_layer_size, n_layers, h_rnn_dim, h_linear_dim)

    # train cNF
    
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

    # validate cNF 

    n_iter_test = 1
    test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(y_test),
                                                batch_size=y_test.shape[0]//n_iter_test, shuffle=True)
    model_test = RNNcNF(u_dim, y_dim, summary_dim, n_blocks, hidden_layer_size, n_layers, h_rnn_dim, h_linear_dim)
    model_test.load(os.path.join(folder, checkpoint_path))
    u_samples_test = model_test.test(test_loader, n_samples).detach().numpy()

    np.save(os.path.join(folder, f'check_test_inference_u_samples.npy'), u_samples_test)
    
    t = np.linspace(0, tfinal, int(tfinal/deltat) + 1)
    for index in list(np.arange(0, n_test, 5)):
        plot_signals(t[:-1], y_test[index], u_test[index], u_samples_test[:, index, :, :], factor, mins, labels, 
                     n_columns=3, n_rows=3 if dynamics == 'multi_body' else 2,
                     filepath=os.path.join(folder, f'check_test_inference_{index}.png'))