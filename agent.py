import numpy as np
import copy
import torch
import torch.nn as nn
import torch.optim as optim

from net import Actor, Critic


DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def soft_update(local_model, target_model, tau):
    """
    Soft update model parameters.
    θ_target = τ*θ_local + (1 - τ)*θ_target
    :param local_model: (nn.module) weights will be copied from
    :param target_model: (nn.module) weights will be copied to
    :param tau: (float) interpolation parameter
    :return: 
    """
    for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
        target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)


class MADDPGAgent(nn.Module):
    def __init__(self, state_size, action_size, actor_lr=5e-4, critic_lr=5e-4, tau=1e-3, gamma=0.99, n_agents=2):
        super(MADDPGAgent, self).__init__()
        """
        Interacts with and learns from the environment.
        :param state_size: (int)
        :param action_size: (int)
        :param actor_lr: (float)
        :param critic_lr: (float)
        :param tau: (float) soft update rate of target parameters
        :param gamma: (float) discount factor
        :param update_freq: (int) update local & target network every n steps
        :param buffer_size: (int)
        :param batch_size: (int) how many samples to use when doing single update
        """
        self.state_size = state_size
        self.action_size = action_size
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        self.gamma = gamma
        self.tau = tau
        self.n_agents = n_agents

        # actor
        self.actor_locals = nn.ModuleList([Actor(self.state_size, self.action_size).to(DEVICE) for _ in range(self.n_agents)])
        self.actor_targets = copy.deepcopy(self.actor_locals)
        for actor_target in self.actor_targets:
            for p in actor_target.parameters():
                p.requires_grad = False

        # critic
        self.critic_locals = nn.ModuleList([Critic(self.state_size * self.n_agents, self.action_size * self.n_agents).to(DEVICE) for _ in range(self.n_agents)])
        self.critic_targets = copy.deepcopy(self.critic_locals)
        for critic_target in self.critic_targets:
            for p in critic_target.parameters():
                p.requires_grad = False

        # optimizers
        self.actor_optimizers = [optim.Adam(self.actor_locals[i].parameters(), lr=self.actor_lr) for i in range(self.n_agents)]
        self.critic_optimizers = [optim.Adam(self.critic_locals[i].parameters(), lr=self.critic_lr) for i in range(self.n_agents)]

        # action noise process for each agent
        # self.noises = [OUNoise(action_size) for _ in range(self.n_agents)]

        # initialize time step
        self.t_step = 0

    def update(self, e):

        for k in e.keys():
            e[k] = e[k].to(DEVICE)

        # update critic
        next_action = torch.cat([self.actor_targets[i](e['next_state'][i]) for i in range(self.n_agents)], dim=-1)
        for i in range(self.n_agents):
            next_q = self.critic_targets[i](torch.cat([x for x in e['next_state']], dim=-1), next_action)
            target_q = e['reward'][i].unsqueeze(-1) + self.gamma * next_q * (1 - e['done'][i].unsqueeze(-1))  # IMPORTANT: unsqueeze to match dimension
            q = self.critic_locals[i](torch.cat([x for x in e['state']], dim=-1), torch.cat([x for x in e['action']], dim=-1))
            critic_loss = ((target_q - q) ** 2).mean()
            self.critic_optimizers[i].zero_grad()
            critic_loss.backward()
            self.critic_optimizers[i].step()

        # update actor
        for i in range(self.n_agents):
            action = torch.cat([self.actor_locals[j](e['state'][j]) if j == i else self.actor_locals[j](e['state'][j]).detach()
                                for j in range(self.n_agents)], dim=-1)
            actor_loss = -self.critic_locals[i](torch.cat([x for x in e['state']], dim=-1), action).mean()
            self.actor_optimizers[i].zero_grad()
            actor_loss.backward()
            self.actor_optimizers[i].step()

        # update target network towards local network
        for i in range(self.n_agents):
            soft_update(self.actor_locals[i], self.actor_targets[i], self.tau)
            soft_update(self.critic_locals[i], self.critic_targets[i], self.tau)

    def act(self, state, noise=0):
        """
        produce action from state using actor_local
        :param state: (np.array) [n_agents, state_size]
        :param noise: (float) noise scale
        :return: action (np.array) [n_agents, action_size]
        """
        state = torch.from_numpy(state).to(DEVICE)
        res = []
        for i in range(self.n_agents):
            self.actor_locals[i].eval()
            with torch.no_grad():
                action = self.actor_locals[i](state[i].unsqueeze(0)).squeeze(0).cpu().numpy()
            self.actor_locals[i].train()
            action += np.random.randn() * noise
            # if add_noise:
            #     action += self.noises[i].sample()
            action = np.clip(action, -1, 1)
            res.append(action)
        return np.stack(res, axis=0)

    def reset(self):
        """
        reset agent before each episode of training
        """
        # for n in self.noises:
        #     n.reset()
        pass


class OUNoise:
    """Ornstein-Uhlenbeck process."""

    def __init__(self, size, mu=0., theta=0.15, sigma=0.2):
        """Initialize parameters and noise process."""
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.state = self.mu.copy()

    def reset(self):
        """Reset the internal state (= noise) to mean (mu)."""
        self.state = self.mu.copy()

    def sample(self):
        """Update internal state and return it as a noise sample."""
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(len(x))
        self.state = x + dx
        return self.state
