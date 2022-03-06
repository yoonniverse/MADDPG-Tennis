import random
import time

import numpy as np
import torch
from argparse import ArgumentParser
import matplotlib.pyplot as plt

from agent import MADDPGAgent
from environment import Environment
from buffer import ReplayBuffer


def seed_all(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def train(env, agent, buffer, noise, noise_decay, min_noise, n_episodes=2000, print_freq=100):
    """
    :param agent: agent.Agent class instance
    :param env: environment class instance compatible with OpenAI gym
    :param n_episodes: (int) maximum number of training episodes
    :param print_freq: (int) print frequency of episodic score
    :return: scores: (list[float]) scores of last 100 episodes
    """
    scores = []
    for i_episode in range(1, n_episodes + 1):
        t0 = time.time()
        state = env.reset(options={'train_mode': True})
        agent.reset()
        score = np.zeros(agent.n_agents)
        while True:
            action = agent.act(state, noise=noise)
            next_state, reward, done, _ = env.step(action)
            experience_dict = {'state': state, 'action': action, 'reward': reward, 'next_state': next_state, 'done': done}
            buffer.add(experience_dict)
            if len(buffer) > buffer.batch_size:
                e = buffer.sample()
                agent.update(e)
            state = next_state
            score += reward
            if any(done):
                break
        scores.append(np.max(score))
        print(f'\rEpisode {i_episode:5d} | LAST: {scores[-1]:3.3f} | Mean: {np.mean(scores[-100:]):3.3f}'
              f' | Max: {np.max(scores[-100:]):3.3f} | Min: {np.min(scores[-100:]):3.3f} | Time: {int(time.time()-t0)}'
              f' | Noise: {noise:.3f}'
              , end="\n" if i_episode % print_freq == 0 else "")

        if np.mean(scores[-100:]) > 1:
            env.reset(options={'train_mode': True})
            return scores

        noise = max(min_noise, noise * noise_decay)

    env.reset(options={'train_mode': True})
    return scores


if __name__ == '__main__':

    parser = ArgumentParser()
    parser.add_argument('--env_path', type=str, default='Tennis_Linux/Tennis.x86_64', help='path for unity environment')
    parser.add_argument('--save_path', type=str, default='agent.ckpt', help='save path for trained agent weights')
    parser.add_argument('--actor_lr', type=float, default=5e-4, help='learning rate for updating actor')  # 1e-4
    parser.add_argument('--critic_lr', type=float, default=1e-3, help='learning rate for updating critic')  # 1e-3
    parser.add_argument('--tau', type=float, default=1e-3, help='learning rate for updating target network')
    parser.add_argument('--gamma', type=float, default=0.99, help='discount factor when calculating return')
    parser.add_argument('--buffer_size', type=int, default=int(1e6), help='maximum number of experiences to save in replay buffer')
    parser.add_argument('--batch_size', type=int, default=1024, help='number of experiences to do one step of update')
    parser.add_argument('--n_episodes', type=int, default=100000, help='total number of episodes to play')
    parser.add_argument('--print_freq', type=int, default=100, help='print training status every ~ steps')
    parser.add_argument('--noise', type=float, default=1, help='initial noise added to action for exploration')
    parser.add_argument('--noise_decay', type=float, default=0.999, help='noise decay rate per step')
    parser.add_argument('--min_noise', type=float, default=0.01, help='minimum noise')
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    args = parser.parse_args()
    print(args)

    seed_all(args.seed)

    env = Environment(args.env_path)

    agent = MADDPGAgent(
        state_size=env.observation_size[0],
        action_size=env.action_size,
        actor_lr=args.actor_lr,
        critic_lr=args.critic_lr,
        tau=args.tau,
        gamma=args.gamma,
        n_agents=env.n_agents
    )

    buffer = ReplayBuffer(
        n_agents=env.n_agents,
        state_size=env.observation_size[0],
        action_size=env.action_size,
        buffer_size=args.buffer_size,
        batch_size=args.batch_size,
    )

    scores = train(
        env=env,
        agent=agent,
        buffer=buffer,
        noise=args.noise,
        noise_decay=args.noise_decay,
        min_noise=args.min_noise,
        n_episodes=args.n_episodes,
        print_freq=args.print_freq
    )

    torch.save(agent.state_dict(), args.save_path)

    plt.figure(figsize=(20, 20))
    plt.plot(scores)
    plt.title(f'MAX{np.max(scores):3.2f} | LAST{scores[-1]:3.2f}')
    plt.ylabel('Score')
    plt.xlabel('Episode #')
    plt.show()
