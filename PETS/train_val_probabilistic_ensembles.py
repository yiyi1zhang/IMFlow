import sys
import torch
import torch.nn as nn
import numpy as np
import os
from tqdm import tqdm
import torch.optim as op
import wandb
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


class PENN(nn.Module):
    """
    (P)robabilistic (E)nsemble of (N)eural (N)etworks
    """

    def __init__(self, num_nets, state_dim, action_dim, hidden_dim=64, learning_rate=1e-3):
        super(PENN, self).__init__()
        self.num_nets = num_nets
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.learning_rate = learning_rate

        # Log variance bounds
        self.max_logvar = nn.Parameter(torch.ones(1, state_dim) * -2)
        self.min_logvar = nn.Parameter(torch.ones(1, state_dim) * -4)

        self.models = nn.ModuleList([self._create_network(hidden_dim) for _ in range(num_nets)])
        self.trainable_parameters = []
        for i in range(self.num_nets):
            trainable_parameters = [self.max_logvar, self.min_logvar]
            for p in self.models[i].parameters():
                if p.requires_grad:
                    trainable_parameters.append(p)
            self.trainable_parameters.append(trainable_parameters)
        self.optimizers = [op.Adam(param, lr=learning_rate) for param in self.trainable_parameters]

    def _create_network(self, hidden_dim):
        return nn.Sequential(
            nn.Linear(self.state_dim + self.action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2 * self.state_dim)
        )

    def forward(self, states, action):
        """
        Forward pass through all networks in the ensemble.
        """
        x = torch.cat([states, action], dim=-1)
        means, logvars = [], []
        for model in self.models:
            output = model(x)
            mean, logvar = self.get_output(output)
            means.append(mean)
            logvars.append(logvar)
        return torch.stack(means, dim=1), torch.stack(logvars, dim=1)

    def get_output(self, output):
        """
        Splits the model output into mean and log variance, and bounds the log variance.
        """
        mean = output[:, :self.state_dim]
        raw_v = output[:, self.state_dim:]
        logvar = self.max_logvar - nn.functional.softplus(self.max_logvar - raw_v)
        logvar = self.min_logvar + torch.nn.functional.softplus(logvar - self.min_logvar)
        return mean, logvar

    def train_per_step(self, states, action, targets, noise=0):
        total_loss = 0
        states += noise * torch.randn_like(states)
        x = torch.cat([states, action], dim=-1)
        means, logvars = [], []
        for i in range(self.num_nets):
            output = self.models[i](x)
            mean, logvar = self.get_output(output)
            means.append(mean)
            logvars.append(logvar)
            self.optimizers[i].zero_grad()
            loss = self.loss(mean, logvar, targets)
            loss.backward()
            self.optimizers[i].step()
            total_loss += loss.item()
        return total_loss

    def eval_per_step(self, states, action, targets, noise=0):
        total_loss = 0
        states += noise * torch.randn_like(states)
        means, logvars = self(states, action)
        for i in range(self.num_nets):
            loss = self.loss(means[:, i], logvars[:, i], targets)
            total_loss += loss.item()
        return total_loss

    def predict(self, states, actions, idxs=None):
        """
        Predicts the next states from the ensemble given states and actions
        """
        means, logvars = self(states, actions)
        if idxs is None:
            means = torch.mean(means, dim=1)
            logvars = torch.mean(logvars, dim=1)
        else:
            means = means[torch.arange(idxs.shape[0]), idxs]
            logvars = logvars[torch.arange(idxs.shape[0]), idxs]
        sigmas = torch.exp(0.5 * logvars)
        next_states = torch.normal(means, sigmas)

        return next_states

    def loss(self, mean, logvar, targets):
        inv_var = torch.exp(-logvar)
        loss = torch.mean((mean - targets) ** 2 * inv_var, dim=1) + torch.mean(logvar, dim=1)
        loss = loss.mean()
        return loss

    def save(self, name):
        torch.save(self.state_dict(), name)
        print(f"save model to {name}")

    def load(self, name):
        print(f"load model from {name}")
        state_dicts = torch.load(name, map_location=torch.device('cpu'))
        self.load_state_dict(state_dicts)


def train_ensemble_model(net, train_loader, val_loader, iterations, pbar, noise=0):
    net.train(True)
    for it in range(iterations[0]):
        current_state, action, next_state = next(iter(train_loader))
        current_state, action, next_state = current_state.to(device), action.to(device), next_state.to(device)
        train_loss = net.train_per_step(current_state, action, next_state, noise=noise)
        pbar.update(1)
    net.train(False)
    for it in range(iterations[1]):
        current_state_val, action_val, next_state_val = next(iter(val_loader))
        current_state_val, action_val, next_state_val = current_state_val.to(device), action_val.to(device), next_state_val.to(device)
        val_loss = net.eval_per_step(current_state_val, action_val, next_state_val, noise=noise)
    return train_loss, val_loss

def test_ensemble_model(net, current_state, actions, noise=0):
    pred_states = []
    current_state += noise * torch.randn_like(current_state)
    for i in range(actions.shape[1]):
        _state = net.predict(current_state, actions[:, i], torch.randint(net.num_nets, (current_state.shape[0],)))
        current_state = _state + noise * torch.randn_like(_state)
        pred_states.append(current_state)
    pred_states = torch.stack(pred_states, dim=1)
    return pred_states


def get_factor(data):
    min_data = data.copy()
    max_data = data.copy()
    for i in range(len(data.shape) - 1):
        min_data = np.min(min_data, axis=0)
        max_data = np.max(max_data, axis=0)
    factor = max_data - min_data

    return factor, min_data


def reconstruct_data(data, n_test):
    data = np.concatenate([data[i * n_test:(i + 1) * n_test, :, :] for i in range(data.shape[0] // n_test)], axis=1)
    return data

def load_data(file_x, file_u, factory=None, miny=None, split=False, horizon=5):

    Y = np.load(file_x)
    if factory is None:
        factory, miny = get_factor(Y)
    else:
        if isinstance(factory, torch.Tensor):
            factory = factory.detach().numpy()
            miny = miny.detach().numpy()
    U = np.load(file_u)
    Y = (Y - miny.reshape(1, 1, -1)) / factory.reshape(1, 1, -1)
    if split:
        length = Y.shape[1]
        current_states = np.array_split(Y[:, :-(length % horizon)], length // horizon, axis=1)
        current_states = np.concatenate(current_states, axis=0)[:, 0]
        next_states = np.array_split(Y[:, (length % horizon):], length // horizon, axis=1)
        next_states = np.concatenate(next_states, axis=0)
        U = np.array_split(U[:, :-(length % horizon)], length // horizon, axis=1)
        U = np.concatenate(U, axis=0)
    else:
        current_states = Y[:, :-1].reshape(-1, Y.shape[-1])
        next_states = Y[:, 1:].reshape(-1, Y.shape[-1])
        U = U[:, :-1].reshape(-1, U.shape[-1])
    state_tensor = torch.tensor(current_states, dtype=torch.float32)
    ref_tensor = torch.tensor(next_states, dtype=torch.float32)
    U_tensor = torch.tensor(U, dtype=torch.float32)
    factory = torch.tensor(factory, dtype=torch.float32)
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


if __name__ == "__main__":
    dynamics_name = sys.argv[1] if len(sys.argv)> 1 else 'multi_body'
    noise = float(sys.argv[2]) if len(sys.argv)> 2 else 0.0
    if dynamics_name == 'single_track':
        state_dim = 4
    else:
        state_dim = 29
    action_dim = 2
    dt = 0.01
    tfinal = 3
    if noise > 0:
        checkpoint_path = f'PETS_probabilistic_ensembles_{dynamics_name}_noise_{noise}.pt'
    else:
        checkpoint_path = f'PETS_probabilistic_ensembles_{dynamics_name}.pt'
    data_path = '../data'
    states_train, ref_train, action_train, factors, mins = load_data(file_x=os.path.join(data_path,
                                                                                         f"{dynamics_name}_4000_x_train.npy"),
                                                    file_u=os.path.join(data_path, f"{dynamics_name}_4000_u_train.npy"))
    num_nets = 2
    start_lr = 1e-3

    net = PENN(num_nets, state_dim, action_dim, learning_rate=start_lr)

    wandb.init(
        # Set the project where this run will be logged
        project="model_predictive_control",
        # We pass a run name (otherwise it’ll be randomly assigned, like sunshine-lollypop-10)
        name=f"experiment_train_probabilistic_ensembles_{dynamics_name}_noise_{noise}")
    batch_size = 10000
    epochs = 5000

    train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(states_train, action_train, ref_train),
                                                batch_size=batch_size, shuffle=True)
    states_val, ref_val, action_val, _, _ = load_data(
        file_x=os.path.join(data_path, f"{dynamics_name}_500_x_val.npy"),
        file_u=os.path.join(data_path, f"{dynamics_name}_500_u_val.npy"),
        factory=factors, miny=mins)

    val_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(states_val, action_val, ref_val),
                                                batch_size=batch_size, shuffle=True)

    iterations = [int(states_train.shape[0] / batch_size),
                    int(states_val.shape[0] / min(batch_size, states_val.shape[0]))]

    early_stopping = EarlyStopping(patience=100, verbose=True, delta=0.0001,
                                    path=os.path.join(os.getcwd(), 'results', checkpoint_path))

    net.to(device)

    for ep in range(1, epochs + 1):
        with tqdm(total=iterations[0], desc='Training epoch {}'.format(ep), unit="epoch") as p_bar:
            train_loss, val_loss = train_ensemble_model(net, train_loader, val_loader, iterations, p_bar, noise=noise)
            p_bar.set_postfix_str("train_loss {0:.5f}, val_loss {1:.5f}".format(train_loss, val_loss))
            p_bar.update(ep)
            wandb.log({"train_loss": train_loss, "val_loss": val_loss})

        early_stopping(val_loss, net)
        if early_stopping.early_stop:
            print("Early stopping")
            break

    horizon = 5
    Y = np.load(os.path.join(data_path, f"{dynamics_name}_500_x_test.npy"))
    factors_test, mins_test = get_factor(Y)
    U = np.load(os.path.join(data_path, f"{dynamics_name}_500_u_test.npy"))
    states_test = (torch.tensor(Y[:, 0, :], dtype=torch.float32) - mins.reshape(1, -1)) / factors.reshape(1, -1)
    ref_test = Y[:, 1:, :]
    action_test = torch.tensor(U[:, :-1, :], dtype=torch.float32)

    n_test = 500
    folder = os.path.join(os.getcwd(), 'results', checkpoint_path.replace('.pt', ''))

    net_test = PENN(num_nets, state_dim, action_dim, learning_rate=start_lr)
    net_test.load(os.path.join(folder, checkpoint_path))
    states_pred = test_ensemble_model(net_test, states_test, action_test, noise=noise)
    states_pred = states_pred * factors.reshape(1, 1, -1) + mins.reshape(1, 1, -1)
    states_pred = states_pred.detach().numpy()
    np.save(os.path.join(folder, checkpoint_path.replace('.pt', f'states_pred.npy')), states_pred)

    t = np.linspace(0 + dt, tfinal, int(tfinal / dt))
    if dynamics_name == 'single_track':
        selected_inds = [0, 1, 2, 3]
        labels = ["sx", "sy", "velocity [m/s]", "yaw angle [rad]",
                    "steering angle", "long. acceleration"]
    else:
        selected_inds = [0, 1, 2, 3, 4, 5]
        labels = ["sx", "sy", "steering angle [rad]", "velocity [m/s]", "yaw angle [rad]", "yaw rate [rad/s]",
                    "steering angle velocity", "long. acceleration"]

    err = ((states_pred[:, :, selected_inds] - ref_test[:, :, selected_inds]) / factors_test[selected_inds].reshape(1, 1, -1)) ** 2
    nrmse = np.sqrt(np.mean(err))

    for index in list(np.arange(0, 5, 5)):
        plot_signals(t, ref_test[index][:, selected_inds], states_pred[index][:, selected_inds],
                     action_test[index], labels=labels, n_columns=3, n_rows=3 if dynamics_name == 'multi_body' else 2,
                     filepath=os.path.join(folder, checkpoint_path.replace('.pt', f'_test_{index}.png')))
        plt.close()
