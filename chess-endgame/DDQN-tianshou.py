import argparse

import numpy as np
import torch

from degree_freedom_queen import *
from degree_freedom_king1 import *
from degree_freedom_king2 import *
from generate_game import *
from Chess_env_gym import *

from tianshou.data import Collector, ReplayBuffer
from tianshou.policy import DQNPolicy
from tianshou.utils.net.common import Net

import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')


def get_args():
    parser = argparse.ArgumentParser("DDQN using tianshou for Chess EndGame")
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
    parser.add_argument('--step-per-collect', type=int, default=15)
    parser.add_argument('--batch-size', type=int, default=30)
    parser.add_argument('--render', type=float, default=0.)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--hidden-sizes', type=int, nargs='*', default=[128, 128])
    args = parser.parse_known_args()[0]
    return args


def test_dqn(args=get_args()):
    # env
    size_board = 4
    env = Chess_Env_Gym(size_board)
    args.state_shape = env.state_shape
    args.action_shape = env.action_shape

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
    policy = DQNPolicy(
        net,
        optim,
        args.gamma,
        args.n_step,
        target_update_freq=args.target_update_freq,
        is_double=True
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
    for i in range(args.epoch):
        policy.set_eps(args.eps_train)
        step_acc = 0
        while step_acc < args.step_per_epoch:
            collect_result = train_collector.collect(n_step=args.step_per_collect)
            step_acc += collect_result['n/st']
            rew = collect_result['rew']
            mean_rets.append(rew)
            for j in range(round(args.update_per_step * collect_result['n/st'])):
                # train policy with a sampled batch data from buffer
                loss = policy.update(args.batch_size, train_collector.buffer)
                loss = loss['loss']
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
    Result for Double DQN
    Epoch: 0, loss: 0.03563058376312256
    Epoch: 1, loss: 0.019260264933109283
    Epoch: 2, loss: 0.0427643284201622
    Epoch: 3, loss: 0.021699130535125732
    Epoch: 4, loss: 0.013731436803936958
    Epoch: 5, loss: 0.008583026006817818
    Epoch: 6, loss: 0.005183699075132608
    Epoch: 7, loss: 0.008335601538419724
    Epoch: 8, loss: 0.0020940720569342375
    Epoch: 9, loss: 0.003840742167085409
    Mean testing return: 0.978, episodic length: 2.786
    """



if __name__ == '__main__':
    test_dqn(get_args())
