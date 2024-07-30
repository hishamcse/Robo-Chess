import argparse

import numpy as np
import torch

from degree_freedom_queen import *
from degree_freedom_king1 import *
from degree_freedom_king2 import *
from generate_game import *
from Chess_env_gym import *

import tianshou as ts

from tianshou.data import Collector, ReplayBuffer
from tianshou.policy import DQNPolicy
from tianshou.utils.net.common import Net

import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--eps-test', type=float, default=0.05)
    parser.add_argument('--eps-train', type=float, default=0.1)
    parser.add_argument('--buffer-size', type=int, default=20000)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--gamma', type=float, default=0.95)
    parser.add_argument('--n-step', type=int, default=1)
    parser.add_argument('--target-update-freq', type=int, default=320)
    parser.add_argument('--epoch', type=int, default=10)
    parser.add_argument('--step-per-epoch', type=int, default=5000)
    parser.add_argument('--update-per-step', type=float, default=1 / 15)
    parser.add_argument('--step-per-collect', type=int, default=200)  # 15 for Double DQN
    parser.add_argument('--batch-size', type=int, default=128)        # 30 for Double DQN
    parser.add_argument('--render', type=float, default=0.)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--hidden-sizes', type=int, nargs='*', default=[128, 128])
    # args = parser.parse_known_args()[0]
    args = parser.parse_args()
    return args


def test_dqn(args=get_args()):
    print(args)
    # env
    size_board = 4
    env = Chess_Env_Gym(size_board)
    args.state_shape = env.state_shape
    args.action_shape = env.action_shape
    args.action_space = env.action_space

    # seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # model
    net = Net(
        args.state_shape,
        args.action_shape,
        hidden_sizes=args.hidden_sizes,
        device=args.device,
    ).to(args.device)
    optim = torch.optim.Adam(net.parameters(), lr=args.lr)
    policy = ts.policy.DQNPolicy(
        model=net,
        optim=optim,
        action_space=env.action_space,
        discount_factor=args.gamma,
        estimation_step=args.n_step,
        target_update_freq=args.target_update_freq,
        is_double=False,
        clip_loss_grad=True
    )

    # buffer
    buffer_train = ReplayBuffer(
        args.buffer_size,
        # ignore_obs_next=True
    )
    buffer_test = ReplayBuffer(
        args.buffer_size,
        # ignore_obs_next=True
    )

    # collector
    train_collector = Collector(policy, env, buffer_train, exploration_noise=True)
    test_collector = Collector(policy, env, buffer_test, exploration_noise=True)
    # pre-collect transitions using initialized policy before training
    train_collector.reset()
    train_collector.collect(n_step=args.batch_size)

    # training
    epochs = []
    losses = []
    mean_rets = []
    reward_threshold = 0.96
    eps_train_0 = args.eps_train
    for i in range(args.epoch):
        policy.set_eps(args.eps_train)
        args.eps_train = eps_train_0 / (1 + 10 * i)
        print(args.eps_train)
        step_acc = 0
        while step_acc < args.step_per_epoch:
            collect_result = train_collector.collect(n_step=args.step_per_collect)
            step_acc += collect_result['n/st']
            rew = collect_result['rew']
            if rew >= reward_threshold:
                print(f'Finished training! Mean returns: {rew}')
                break
            mean_rets.append(rew)
            for j in range(round(args.update_per_step * collect_result['n/st'])):
                # train policy with a sampled batch data from buffer
                loss = policy.update(args.batch_size, train_collector.buffer)
                loss = loss['loss']
        if rew >= reward_threshold:
            break
        epochs.append(i)
        losses.append(loss)
        print(f'Epoch: {i}, loss: {loss}')


    # testing
    policy.eval()
    policy.set_eps(args.eps_test)
    test_collector.reset()
    result = test_collector.collect(n_episode=1000)
    rews, lens = result["rews"], result["lens"]
    print(f'Mean testing return: {rews.mean()}, episodic length: {lens.mean()}')

    # plotting
    plt.plot(epochs, losses)
    plt.xlabel('Epochs')
    plt.ylabel('Training Loss')
    plt.title('Training Loss vs Epochs')
    plt.show()

    plt.plot(mean_rets)
    plt.show()

    """
    Result for vanilla DQN using early-stop
    0.1
    Epoch: 0, loss: 0.03749380260705948
    0.009090909090909092
    Epoch: 1, loss: 0.014625297859311104
    0.004761904761904762
    Finished training! Mean returns: 0.9722222222222222
    Mean testing return: 0.968, episodic length: 2.938
    """



if __name__ == '__main__':
    test_dqn(get_args())
