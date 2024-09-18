import sys
import torch
import torch.nn as nn
import numpy as np
import os
from tqdm import tqdm
import torch.optim as op
import wandb
import time
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
sys.path.append('..')
from utils.early_stop import EarlyStopping

if torch.cuda.is_available():
    device = torch.device("cuda:0")
    torch.backends.cudnn.benchmark = True
else:
    print('cuda is not available')
    device = torch.device("cpu")


class simNet(nn.Module):

    def __init__(
            self, state_dim, horizon, ref_dim, action_dim, rnn_hidden=16, linear_hidden=64
    ):
        super(simNet, self).__init__()
        self.state_dim = state_dim
        self.ref_dim = ref_dim
        self.horizon = horizon
        self.action_dim = action_dim
        self.rnn_hidden = rnn_hidden
        # normal logic for processing the reference trajectory
        self.rnny = nn.GRU(ref_dim, rnn_hidden, batch_first=True)
        self.bn1y = nn.BatchNorm1d(rnn_hidden)
        self.l1y = nn.Linear(rnn_hidden, linear_hidden // 2)
        self.l1x = nn.Sequential(nn.ELU(), nn.Linear(state_dim, linear_hidden // 2))
        self.flat = nn.Flatten()
        self.l_out = nn.Sequential(nn.ELU(), nn.Linear(linear_hidden * horizon, action_dim * horizon))

    def forward(self, state, ref):
        ref_out, _ = self.rnny(ref)
        ref_out = self.bn1y(ref_out.transpose(-1, -2))
        ref_out = self.l1y(ref_out.transpose(-1, -2))
        state_out = self.l1x(state)
        state_out = torch.repeat_interleave(state_out.unsqueeze(1), dim=1, repeats=self.horizon)
        x = torch.concatenate((state_out, ref_out), dim=-1)
        out = self.l_out(self.flat(x))
        return out.reshape((-1, self.horizon, self.action_dim))

    def save(self, name):
        torch.save(self.state_dict(), name)

    def load(self, name):
        # print(f"load model from {name}")
        state_dicts = torch.load(name, map_location=torch.device('cpu'))
        self.load_state_dict(state_dicts)


def train_recurrent_model(net, train_loader, val_loader, optimizer, dynamics, dt, dims, factors, mins, iterations, pbar):
    net.train(True)
    for it in range(iterations[0]):
        current_state, ref, act = next(iter(train_loader))
        current_state, ref, act = current_state.to(device), ref.to(device), act.to(device)
        current_state = (current_state - mins.reshape(1, -1)) / factors.reshape(1, -1)
        ref = (ref - mins[dims].reshape(1, 1, -1)) / factors[dims].reshape(1, 1, -1)
        optimizer.zero_grad()
        batch_size, horizon, obs_dim = ref.shape
        state_dim = current_state.shape[-1]
        current_state += noise * torch.randn_like(current_state)
        actions = net(current_state, ref)
        current_state = current_state * factors.reshape(1, -1) + mins.reshape(1, -1)
        ref = ref * factors[dims].reshape(1, 1, -1) + mins[dims].reshape(1, -1)
        states = torch.zeros((batch_size, horizon, state_dim), device=device)
        for k in range(horizon):
            current_state = dynamics(current_state, actions[:, k, :], dt)
            states[:, k, :] = current_state

        loss = dynamics.loss(current_state, actions.sum(dim=1), ref[:, -1, :], dims, factors)
        
        torch.autograd.set_detect_anomaly(True)
        # Backprop
        loss.backward()
        optimizer.step()

        pbar.update(1)
    net.train(False)
    for it in range(iterations[1]):
        current_state_val, ref_val, act_val = next(iter(val_loader))
        current_state_val, ref_val, act_val = current_state_val.to(device), ref_val.to(device), act_val.to(device)
        current_state_val = (current_state_val - mins.reshape(1, -1)) / factors.reshape(1, -1)
        ref_val = (ref_val - mins[dims].reshape(1, -1)) / factors[dims].reshape(1, 1, -1)
        current_state_val += noise * torch.randn_like(current_state_val)
        actions_val = net(current_state_val, ref_val)
        ref_val = ref_val * factors[dims].reshape(1, 1, -1) + mins[dims].reshape(1, 1, -1)
        current_state_val = current_state_val * factors.reshape(1, -1) + mins.reshape(1, -1)
        states_val = torch.zeros((batch_size, horizon, state_dim), device=device)
        for k in range(horizon):
            current_state_val = dynamics(current_state_val, actions_val[:, k, :], dt) + \
                                noise * torch.randn_like(current_state_val) * factors.reshape(1, -1)
            states_val[:, k, :] = current_state_val

        loss_val = dynamics.loss(current_state_val, actions_val.sum(1), ref_val[:, -1, :], dims, factors)
    
    return loss.item(), loss_val.item()

def test_recurrent_model(net, current_state, ref, dynamics, dt, dims, factors, mins):
    current_state = (current_state - mins.reshape(1, -1)) / factors.reshape(1, -1)
    ref = (ref - mins[dims].reshape(1, -1)) / factors[dims].reshape(1, 1, -1)
    batch_size, horizon, obs_dim = ref.shape
    state_dim = current_state.shape[-1]
    stime = time.time()
    current_state += noise * torch.randn_like(current_state)
    action = net(current_state, ref)
    ct = time.time() - stime
    current_state = current_state * factors.reshape(1, -1) + mins.reshape(1, -1)
    ref = ref * factors[dims].reshape(1, 1, -1) + mins[dims].reshape(1, -1)
    states = torch.zeros((batch_size, horizon, state_dim))
    for k in range(horizon):
        current_state = dynamics(current_state, action[:, k, :], dt) + \
                        noise * torch.randn_like(current_state) * factors.reshape(1, -1)
        states[:, k, :] = current_state
    nrmse = torch.mean(torch.sqrt(((states[..., dims] - ref)/factors[dims].reshape(1, 1, -1))**2))
    return states, action, nrmse, ct

def predict_concurrent_model(net, current_state, ref, dynamics, dt, horizon, dims, factors, mins):

    ref = (ref - mins[dims].reshape(1, 1, -1)) / factors[dims].reshape(1, 1, -1)
    batch_size, seq_len, obs_dim = ref.shape
    states = []
    actions = []
    for k in range(seq_len - horizon):
        current_state = (current_state - mins.reshape(1, -1)) / factors.reshape(1, -1)
        current_state += noise * torch.randn_like(current_state)
        action = net(current_state, ref[:, k:k+horizon])
        current_state = current_state * factors.reshape(1, -1) + mins.reshape(1, -1)
        current_state = dynamics(current_state, action[:, 0, :], dt) + \
                        noise * torch.randn_like(current_state) * factors.reshape(1, -1)
        states.append(current_state)
        actions.append(action[:, 0, :])
    stime = time.time()
    current_state = (current_state - mins.reshape(1, -1)) / factors.reshape(1, -1)
    action = net(current_state, ref[:, seq_len - horizon:seq_len])
    current_state = current_state * factors.reshape(1, -1) + mins.reshape(1, -1)
    ct = time.time() - stime
    actions.append(action)
    for k in range(horizon):
        current_state = dynamics(current_state, action[:, k, :], dt) + \
                        noise * torch.randn_like(current_state) * factors.reshape(1, -1)
        states.append(current_state)
    states = torch.stack(states, dim=1)
    actions = torch.cat([torch.stack(actions[:-1], dim=1), actions[-1]], dim=1)
    nrmse = torch.mean(torch.sqrt((states[..., dims]/factors[dims].reshape(1, 1, -1) - ref)**2))
    return states, actions, nrmse, ct

def get_factor(data):
    min_data = data.copy()
    max_data = data.copy()
    for i in range(len(data.shape) - 1):
        min_data = np.min(min_data, axis=0)
        max_data = np.max(max_data, axis=0)
    factor = max_data - min_data

    return factor, min_data


def split_and_roll_data(data, horizon, times=1):
    length = 301
    datas = []
    for i in range(times):
        _data = np.concatenate([data[:, i + 1:, :], data[:, 1:i + 1, :]], axis=1)
        datas += np.array_split(_data, length // horizon, axis=1)[:(length // horizon - i)]
    datas = np.concatenate(datas, axis=0)
    return datas


def reconstruct_data(data, n_test):
    data = np.concatenate([data[i * n_test:(i + 1) * n_test, :, :] for i in range(data.shape[0] // n_test)], axis=1)
    return data

def load_data(file_x, file_u, selected_inds, horizon, times=1):

    Y = np.load(file_x)
    factory, miny = get_factor(Y)
    U = np.load(file_u)
    Y = split_and_roll_data(Y, horizon, times=times)
    state_tensor = torch.tensor(Y[:, 0, :], dtype=torch.float32)
    ref_tensor = torch.tensor(Y[:, :, selected_inds], dtype=torch.float32)
    U = split_and_roll_data(U, horizon, times=times)
    U_tensor = torch.tensor(U, dtype=torch.float32)
    factory = torch.tensor(factory, dtype=torch.float32)
    miny = torch.tensor(miny, dtype=torch.float32)

    return state_tensor, ref_tensor, U_tensor, factory, miny


def plot_signals(t, y, y_pred, u, u_pred, labels, n_columns=3, n_rows=2, filepath=None):
    fig, ax = plt.subplots(n_rows, n_columns, figsize=(n_columns * 4, n_rows * 4))
    ax = ax.flat
    for i in range(y.shape[-1]):
        ax[i].plot(t, y[:, i])
        ax[i].plot(t, y_pred[:, i], color='#e55e5e')
        ax[i].set_title(labels[i])

    for i in range(u.shape[-1]):
        ax[y.shape[-1] + i].plot(t, u[:, i])
        ax[y.shape[-1] + i].set_title(labels[y.shape[-1] + i])
        ax[y.shape[-1] + i].plot(t, u_pred[:, i], color='#e55e5e')

    if filepath is not None:
        fig.savefig(filepath)

if __name__ == "__main__":
    dynamics_name = sys.argv[1] if len(sys.argv)> 1 else 'single_track'
    horizon = int(sys.argv[2]) if len(sys.argv)> 2 else 5
    noise = float(sys.argv[3]) if len(sys.argv)> 3 else 0.0
    if dynamics_name == 'single_track':
        state_dim = 4
    else:
        state_dim = 29
    if dynamics_name == 'multi_body':
        selected_inds = [0, 1, 2, 3, 4, 5]
    else:
        selected_inds = [i for i in range(state_dim)]
    ref_dim = len(selected_inds)
    action_dim = 2
    dt = 0.01
    tfinal = 3
    checkpoint_path = f'apg_dynamic_model_rnn_{dynamics_name}_horizon_{horizon}.pt'
    if noise > 0:
        noise_str = str(noise).replace('.', '')
        checkpoint_path = checkpoint_path.replace('.pt', f'_noise_{noise_str}.pt')
    
    
    folder = os.path.join('..', 'results', checkpoint_path.replace('.pt', ''))
    os.mkdir(folder)

    data_path = '../data'
        
    states_train, ref_train, action_train, factors, mins = load_data(file_x=os.path.join(data_path, f"{dynamics_name}_4000_x_train.npy"),
                                                    file_u=os.path.join(data_path, f"{dynamics_name}_4000_u_train.npy"),
                                                    selected_inds=selected_inds, horizon=horizon,
                                                    times=horizon)
    states_test, ref_test, action_test, _, _ = load_data(file_x=os.path.join(data_path, f"{dynamics_name}_500_x_test.npy"),
                                                      file_u=os.path.join(data_path, f"{dynamics_name}_500_u_test.npy"),
                                                      selected_inds=selected_inds, horizon=horizon, times=1)
    if dynamics_name == 'single_track':
        from env_dynamics import SingleTrackDynamics
        dynamics = SingleTrackDynamics()
    else:
        from env_dynamics import MultiBodyDynamics
        dynamics = MultiBodyDynamics()

    net = simNet(state_dim, horizon, ref_dim, action_dim)

    wandb.init(
        # Set the project where this run will be logged
        project="model_predictive_control",
        # We pass a run name (otherwise it’ll be randomly assigned, like sunshine-lollypop-10)
        name=f"experiment_train_neural_dynamic_model_{dynamics_name}_horizon_{horizon}_noise_{noise}")
    batch_size = 10000
    epochs = 5000
    start_lr = 1e-3

    train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(states_train, ref_train, action_train),
                                                batch_size=batch_size, shuffle=True)
    states_val, ref_val, action_val, _, _ = load_data(
        file_x=os.path.join(data_path, f"{dynamics_name}_500_x_val.npy"),
        file_u=os.path.join(data_path, f"{dynamics_name}_500_u_val.npy"),
        selected_inds=selected_inds, horizon=horizon, times=horizon)

    val_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(states_val, ref_val, action_val),
                                                batch_size=batch_size, shuffle=True)

    iterations = [int(states_train.shape[0] / batch_size), int(states_val.shape[0] / min(batch_size, states_val.shape[0]))]

    early_stopping = EarlyStopping(patience=100, verbose=True, delta=0.0001, path=os.path.join(folder, checkpoint_path))

    trainable_parameters = [p for p in net.parameters() if p.requires_grad]
    optimizer = op.Adam(trainable_parameters, lr=start_lr)

    net.to(device)

    for ep in range(1, epochs + 1):
        with tqdm(total=iterations[0], desc='Training epoch {}'.format(ep), unit="epoch") as p_bar:
            train_loss, val_loss = train_recurrent_model(net, train_loader, val_loader, optimizer,
                                                dynamics, dt, selected_inds, factors.to(device), mins.to(device), iterations, p_bar)
            p_bar.set_postfix_str("train_loss {0:.5f}, val_loss {1:.5f}".format(train_loss, val_loss))
            p_bar.update(ep)
            wandb.log({"train_loss": train_loss, "val_loss": val_loss})

        early_stopping(val_loss, net)
        if early_stopping.early_stop:
            print("Early stopping")
            break

    net_test = simNet(state_dim, horizon, ref_dim, action_dim)
    net_test.load(folder, checkpoint_path)
    states_pred, action_pred, nrmse, com_t = test_recurrent_model(net_test, states_test, ref_test, dynamics, dt,
                                                            selected_inds, factors, mins)
    np.save(os.path.join(folder, 'check_states_pred.npy'), states_pred.detach().numpy())
    np.save(os.path.join(folder, 'check_action_pred.npy'), action_pred.detach().numpy())

    n_test = 500

    nrmse = np.mean(np.sqrt(((states_pred[..., selected_inds] - ref_test.detach().numpy()) /
                                factors.detach().numpy()[selected_inds].reshape(1, 1, -1)) ** 2))
    states_pred = reconstruct_data(states_pred, n_test)
    action_pred = reconstruct_data(action_pred, n_test)
    ref_test = reconstruct_data(ref_test, n_test)
    action_test = reconstruct_data(action_test, n_test)

    t = np.linspace(0 + dt, tfinal, int(tfinal / dt))
    if dynamics_name == 'single_track':
        labels = ["sx", "sy", "velocity [m/s]", "yaw angle [rad]",
                    "steering angle", "long. acceleration"]
    else:
        labels = ["sx", "sy", "steering angle [rad]", "velocity [m/s]", "yaw angle [rad]", "yaw rate [rad/s]",
                    "steering angle velocity", "long. acceleration"]

    for index in list(np.arange(0, n_test, 5)):
        plot_signals(t, ref_test[index][:, selected_inds], states_pred[index][:, selected_inds],
                     action_test[index], action_pred[index], labels, n_columns=3,
                     n_rows=2 if dynamics_name == 'single_track' else 3,
                     filepath=os.path.join(folder, 'check_test_{index}.png'))
        plt.close()