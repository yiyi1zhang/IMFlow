import os
import sys
import torch
from tqdm import tqdm
import numpy as np
import time
from train_val_probabilistic_ensembles import PENN
device = torch.device("cpu")
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt


# class RandomPolicy:
#     def __init__(self, action_dim):
#         self.action_dim = action_dim
#
#     def reset(self):
#         pass
#
#     def act(self, r1=-1, r2=1, **kwargs):
#         return (r2 - r1) * torch.rand(self.action_dim) + r1


class MPC:
    def __init__(self,
                 dynamics,
                 state_dim,
                 ref_dim,
                 act_dim,
                 horizon,
                 model,
                 popsize,
                 num_elites,
                 max_iters,
                 num_particles=5,
                 use_gt_dynamics=True,
                 use_random_optimizer=False):

        self.dynamics = dynamics
        self.state_dim = state_dim
        self.ref_dim = ref_dim
        self.act_dim = act_dim
        self.horizon = horizon
        self.model = model

        self.popsize = popsize
        self.max_iters = max_iters
        self.num_elites = num_elites
        self.num_particles = num_particles

        self.use_gt_dynamics = use_gt_dynamics
        self.use_random_optimizer = use_random_optimizer

        if use_gt_dynamics:
            self.predict_next_state = self.predict_next_state_gt
            assert num_particles == 1
        else:
            self.predict_next_state = self.predict_next_state_model

        # Initialize your planner with the relevant arguments.
        # Write different optimizers for cem and random actions respectively
        if self.use_random_optimizer:
            self.opt = self.random_optimizer
        else:
            self.opt = self.cem_optimizer
        self.reset()

    def obs_cost_fn(self, state, action=None, ref=None, dims=None, factors=None):
        if action is None:
            return self.dynamics.loss(state, action, ref, dims, factors, eval_act=False, mean=False)
        else:
            return self.dynamics.loss(state, action, ref, dims, factors, mean=False)

    def predict_next_state_model(self, states, actions, dims, factors, mins, ref=None, noise=0):
        """Given a list of state action pairs, use the learned model to
        predict the next state.

        Returns:
            cost: cost of the given action sequence.
        """
        rows = actions.shape[0]  # pop_size * max_iters
        cost = self.obs_cost_fn(states, ref=ref, dims=dims, factors=factors)
        sampler = self.inds_sampling(rows)
        states = (states - mins.reshape(1, -1)) / factors.reshape(1, -1)
        states += noise * torch.randn_like(states)
        for i in range(self.horizon):
            action = actions[:, i * self.act_dim:(i + 1) * self.act_dim]
            idxs = sampler[:, i]
            states = self.model.predict(states, action, idxs)
            states += noise * torch.randn_like(states)
            cost += self.obs_cost_fn(states * factors.reshape(1, -1) + mins.reshape(1, -1), action=action, ref=ref, dims=dims, factors=factors)
        return cost

    def predict_next_state_gt(self, states, actions, dims, factors, ref=None, noise=0, dt=0.01):
        """Given a list of state action pairs, use the ground truth dynamics
        to predict the next state.
        """
        states += noise * torch.randn_like(states) * factors.reshape(1, -1)
        cost = self.obs_cost_fn(states, ref=ref, dims=dims, factors=factors)
        for i in range(self.horizon):
            action = actions[:, i * self.act_dim:(i + 1) * self.act_dim]
            states = self.dynamics(states, action, dt=dt) + noise * torch.randn_like(states) * factors.reshape(1, -1)
            cost += self.obs_cost_fn(states, action=action, ref=ref, dims=dims, factors=factors)
        return cost

    def inds_sampling(self, n):
        return torch.randint(self.model.num_nets, (n, self.horizon))

    def reset(self):
        """Initializes variables mu and sigma.
        """
        self.mu = torch.zeros(self.horizon * self.act_dim)
        self.sigma = torch.ones(self.horizon * self.act_dim)

    def random_optimizer(self, state, dims, factors, mins, bounds=None, ref=None, noise=0, dt=0.01):
        """Implements the random optimizer. It gives the best action sequence
        for a certain initial state.
        """
        # Generate M*I action sequences of length T according to N(0, 0.5I)
        if bounds is None:
            bounds = [-1, 1]
        r2 = torch.tensor(bounds[1], dtype=torch.float32).reshape(1, 1, -1)
        r1 = torch.tensor(bounds[0], dtype=torch.float32).reshape(1, 1, -1)
        for i in range(self.max_iters):
            actions = (r2 - r1) * torch.rand(self.popsize, self.horizon, self.act_dim) + r1
            actions = actions.reshape(self.popsize, -1)
            repeated_actions = torch.tile(actions, (self.num_particles, 1))
            states = torch.tile(state, (repeated_actions.shape[0], 1))
            if not self.use_gt_dynamics:
                costs = self.predict_next_state_model(states, repeated_actions, dims, factors, mins, ref=ref, noise=noise)
            else:
                costs = self.predict_next_state_gt(states, repeated_actions, dims, factors, ref=ref, noise=noise, dt=dt)

            costs = torch.mean(costs.reshape(self.num_particles, -1), dim=0)
            elite_actions = actions[torch.argsort(costs, descending=False)[:self.num_elites], :]
            elite_actions = elite_actions.reshape(-1, self.horizon, self.act_dim).reshape(-1, self.act_dim)
            r1 = torch.min(elite_actions, dim=0, keepdim=True).values
            r2 = torch.max(elite_actions, dim=0, keepdim=True).values
        min_cost_idx = torch.argmin(costs)
        return actions[min_cost_idx]

    def cem_optimizer(self, state, dims, factors, mins, bounds=None, ref=None, noise=0, dt=0.01):
        """Implements the Cross Entropy Method optimizer. It gives the action
        sequence for a certain initial state by choosing elite sequences and
        using their mean.
        """
        mu = self.mu
        sigma = self.sigma
        if bounds is None:
            bounds = [-1, 1]
        for i in range(self.max_iters):
            # Generate M action sequences of length T according to N(mu, std)
            shape = (self.popsize, self.horizon * self.act_dim)
            actions = mu + sigma * torch.randn(shape)
            actions = torch.clip(actions, min=bounds[0], max=bounds[1])
            repeated_actions = torch.tile(actions, (self.num_particles, 1))
            states = torch.tile(state, (repeated_actions.shape[0], 1))
            if not self.use_gt_dynamics:
                costs = self.predict_next_state_model(states, repeated_actions, dims, factors, mins, ref=ref, noise=noise)
            else:
                costs = self.predict_next_state_gt(states, repeated_actions, dims, factors, ref=ref, noise=noise, dt=dt)
            # Calculate mean and std using the elite action sequences
            costs = costs.reshape(self.num_particles, -1).mean(0)
            elite_actions = actions[torch.argsort(costs, descending=True)[:self.num_elites], :]
            mu = torch.mean(elite_actions, dim=0)
            sigma = torch.std(elite_actions, dim=0)
        return mu

    def train(self, obs_trajs, acs_trajs, noise=0, epochs=5):
        """
        Take the input obs, acs, rews and append to existing transitions the
        train model.

        Args:
          obs_trajs: states
          acs_trajs: actions
          rews_trajs: rewards (NOTE: this may not be used)
          epochs: number of epochs to train for
        """
        assert len(obs_trajs) == len(acs_trajs)
        input_states = torch.cat([traj[:-1] for traj in obs_trajs])
        targets = torch.cat([traj[1:] for traj in obs_trajs])
        actions = torch.cat([acs[:-1] for acs in acs_trajs])
        assert actions.shape[0] == input_states.shape[0]
        self.model.to(device)
        input_states, actions, targets = input_states.to(device), actions.to(device), targets.to(device)
        for ep in range(1, epochs + 1):
            with tqdm(total=epochs, desc='Training epoch {}'.format(ep), unit="epoch") as p_bar:
                loss = self.model.train_per_step(input_states, actions, targets, noise=noise)
                p_bar.set_postfix_str("loss {0:.5f}".format(loss))
                p_bar.update(ep)

    def act(self, state, t, dims, factors, mins, bounds=None, ref=None, noise=0, dt=0.01):
        """
        Find the action for current state.

        Arguments:
          state: current state
          t: current timestep
        """
        mu = self.opt(state, dims, factors, mins, bounds=bounds, ref=ref, noise=noise, dt=dt)
        action = mu[:self.act_dim]  # Get the first action
        self.mu[:-self.act_dim] = mu[self.act_dim:]
        self.mu[-self.act_dim:] = 0
        return action


class PETS(object):
    def __init__(self, dynamics_name='single_track', horizon=5, num_nets=2, popsize=10, num_elites=10, max_iters=10,
                 num_particles=5, use_random_optimizer=False, use_gt_dynamics=True, checkpoint_path=None):
        self.action_dim = 2
        self.horizon = horizon
        self.num_nets = num_nets
        self.popsize = popsize
        self.num_elites = num_elites
        self.max_iters = max_iters
        self.num_paraticles = num_particles
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
        self.model = PENN(num_nets, self.state_dim, self.action_dim)
        if checkpoint_path is not None:
            self.model.load(checkpoint_path)
        self.mpc_policy = MPC(self.dynamics, self.state_dim, self.ref_dim, self.action_dim, self.horizon, self.model,
                              popsize, num_elites, max_iters, num_particles=1 if use_gt_dynamics else num_particles,
                              use_gt_dynamics=use_gt_dynamics, use_random_optimizer=use_random_optimizer)
        # self.random_policy = RandomPolicy(self.action_dim)

    def __call__(self, trajectory, state, dims, factors, mins, bounds=None, noise=0, dt=0.01, retrain=False, threshold=0.01):
        states = []
        actions = []
        com_ts = []
        for k in range(trajectory.shape[0]):
            print('time step: {}/{}'.format(k + 1, trajectory.shape[0]), end='\r', flush=True)

            if retrain:
                if len(states) > 0:
                    if torch.mean((state[dims] - trajectory[k]))**2 > threshold:
                        stime = time.time()
                        self.mpc_policy.train(states, actions, noise=noise, epochs=5)
                        re_t = time.time() - stime
                        print(re_t/5)
            state, action, com_t = self.predict_per_step(self.horizon, state[-1], self.mpc_policy,
                                                  dims, factors, mins, bounds=bounds, noise=noise, ref=trajectory[k], dt=dt)

            states.append(state)
            actions.append(action)
            com_ts.append(com_t)
        states_seq = torch.cat(states)
        actions_seq = torch.cat(actions)
        com_ts = np.array(com_ts)
        nrmse = torch.mean(torch.sqrt(((states_seq[::self.horizon][:, dims] - trajectory) / factors[dims].reshape(1, -1)) ** 2))
        return states_seq, actions_seq, com_ts, nrmse

    def predict_per_step(self, horizon, state, policy, dims, factors, mins, bounds=None, noise=0, ref=None, dt=0.01):
        actions = []
        states = []
        com_t = 0
        for t in range(horizon):
            _stime = time.time()
            action = policy.act(state, t, dims, factors, mins, bounds=bounds, ref=ref, noise=noise, dt=dt) \
                if isinstance(policy, MPC) else policy.act()
            com_t += time.time() - _stime
            actions.append(action)
            state = self.dynamics(state.reshape(-1, self.state_dim), action.reshape(-1, self.action_dim), dt=dt)
            states.append(state)
        return torch.cat(states), torch.stack(actions), com_t


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
    state_tensor = torch.tensor(state_tensor, dtype=torch.float32)
    ref_tensor = torch.tensor(ref_tensor, dtype=torch.float32)
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

if dynamics_name == 'multi_body':
    selected_inds = [0, 1, 2, 3, 4, 5]
else:
    selected_inds = [0, 1, 2, 3]

if noise > 0:
    checkpoint_path = f'PETS_probabilistic_ensembles_{dynamics_name}_noise_{noise}.pt'
else:
    checkpoint_path = f'PETS_probabilistic_ensembles_{dynamics_name}.pt'
if dynamics_name == 'single_track':
    selected_inds = [0, 1, 2, 3]
    labels = ["sx", "sy", "velocity [m/s]", "yaw angle [rad]",
              "steering angle", "long. acceleration"]
else:
    selected_inds = [0, 1, 2, 3, 4, 5]
    labels = ["sx", "sy", "steering angle [rad]", "velocity [m/s]", "yaw angle [rad]", "yaw rate [rad/s]",
              "steering angle velocity", "long. acceleration"]

folder = os.path.join('..', 'results', checkpoint_path.replace('.pt', ''))
data_path = '../data'
factors, mins = get_factor(np.load(os.path.join(data_path, f"{dynamics_name}_4000_x_train.npy")))
states_test, ref_test, action_test, factors, mins = load_data(file_x=os.path.join(data_path, f"{dynamics_name}_500_x_test.npy"),
                                                  file_u=os.path.join(data_path, f"{dynamics_name}_500_u_test.npy"),
                                                  selected_inds=selected_inds, factory=factors, miny=mins)

exp_gt = PETS(dynamics_name=dynamics_name, horizon=horizon, num_nets=2, popsize=500, num_elites=50, max_iters=5,
           num_particles=5, use_gt_dynamics=True, use_random_optimizer=True,
           checkpoint_path=os.path.join(folder, checkpoint_path))

exp_pe = PETS(dynamics_name=dynamics_name, horizon=horizon, num_nets=2, popsize=10, num_elites=10, max_iters=5,
           num_particles=5, use_gt_dynamics=False, use_random_optimizer=True,
           checkpoint_path=os.path.join(folder, checkpoint_path))

inds = 0
_ref = ref_test[inds]
_state0 = states_test[inds]
_action = action_test[inds].detach().numpy()

states_gt, actions_gt, com_t_gt, nrmse_gt = exp_gt(_ref[::horizon], _state0.reshape(1, -1), selected_inds, factors, mins, 
                                                   bounds=np.array([[-0.5, -1], [0.5, 1]]), noise=noise, dt=dt)
states_gt = states_gt.detach().numpy()[:_ref.shape[0]]
actions_gt = actions_gt.detach().numpy()[:_ref.shape[0]]

states_pe, actions_pe, com_t_pe, nrmse_pe = exp_pe(_ref[::horizon], _state0.reshape(1, -1), selected_inds, factors, mins, 
                                                   bounds=np.array([[-0.5, -1], [0.5, 1]]), noise=noise, dt=dt, retrain=False)
states_pe = torch.cat(states_pe).detach().numpy()
actions_pe = torch.cat(actions_pe).detach().numpy()

t = np.linspace(0, (_ref.shape[0]-1)*0.01, _ref.shape[0])
plot_signals(t, _ref[:, selected_inds].detach().numpy(), states_gt[:, selected_inds], _action,
             u_pred=actions_gt, labels=labels, n_columns=3, n_rows=3 if dynamics_name == 'multi_body' else 2,
             filepath=os.path.join(folder, f'PETS_gt_test_{inds}.png'))
plot_signals(t, _ref[:, selected_inds], states_pe[:, selected_inds], _action,
             u_pred=actions_pe, labels=labels, n_columns=3, n_rows=3 if dynamics_name == 'multi_body' else 2,
             filepath=os.path.join(folder, f'PETS_pe_test_{inds}.png'))
