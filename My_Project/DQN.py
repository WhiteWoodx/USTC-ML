import random, math

import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd

# Create and wrap the environment
from atari_wrappers import make_atari, wrap_deepmind, LazyFrames
from IPython.display import clear_output
from tensorboardX import SummaryWriter

USE_CUDA = torch.cuda.is_available()
dtype = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
Variable = lambda *args, **kwargs: autograd.Variable(*args, **kwargs).cuda() if USE_CUDA else autograd.Variable(*args,
                                                                                                                **kwargs)


# env = make_atari('PongNoFrameskip-v4')  # only use in no frameskip environment
# env = wrap_deepmind(env, scale=False, frame_stack=True)
# n_actions = env.action_space.n
# state_dim = env.observation_space.shape
#
# env.render()
# test = env.reset()
# for i in range(100):
#     test = env.step(env.action_space.sample())[0]
#
# plt.imshow(test._force()[..., 0])


# plt.imshow(env.render("rgb_array"))

"""
    deep Q-learning network: in_channels: number of channel of input; 
                            num_actions: number of action-value to output, one-to-one correspondence to action in game.
"""


class DQN(nn.Module):
    def __init__(self, in_channels=4, num_actions=5):

        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.fc4 = nn.Linear(7 * 7 * 64, 512)
        self.fc5 = nn.Linear(512, num_actions)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.fc4(x.reshape(x.size(0), -1)))
        return self.fc5(x)


class Memory_Buffer(object):
    def __init__(self, memory_size=100000):
        self.buffer = []
        self.memory_size = memory_size
        self.next_idx = 0

    def push(self, state, action, reward, next_state, done):
        data = (state, action, reward, next_state, done)
        if len(self.buffer) <= self.memory_size:  # buffer not full
            self.buffer.append(data)
        else:
            self.buffer[self.next_idx] = data
        self.next_idx = (self.next_idx + 1) % self.memory_size

    def sample(self, batch_size):
        states, actions, rewards, next_states, dones = [], [], [], [], []
        for i in range(batch_size):
            idx = random.randint(0, self.size() - 1)
            data = self.buffer[idx]
            state, action, reward, next_state, done = data
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            next_states.append(next_state)
            dones.append(done)

        return np.concatenate(states), actions, rewards, np.concatenate(next_states), dones

    def size(self):
        return len(self.buffer)


class DQNAgent:
    def __init__(self, in_channels=1, action_space=[], USE_CUDA=False, memory_size=10000, epsilon=1, lr=1e-4):
        self.epsilon = epsilon
        self.action_space = action_space
        self.memory_buffer = Memory_Buffer(memory_size)
        self.DQN = DQN(in_channels=in_channels, num_actions=action_space.n)
        self.DQN_target = DQN(in_channels=in_channels, num_actions=action_space.n)
        self.DQN_target.load_state_dict(self.DQN.state_dict())

        self.USE_CUDA = USE_CUDA
        if USE_CUDA:
            self.DQN = self.DQN.cuda()
            self.DQN_target = self.DQN_target.cuda()
        self.optimizer = optim.RMSprop(self.DQN.parameters(), lr=lr, eps=0.001, alpha=0.95)

    def observe(self, lazyframe):# from Lazy frame to tensor

        state = torch.from_numpy(lazyframe._force().transpose(2, 0, 1)[None] / 255).float()
        if self.USE_CUDA:
            state = state.cuda()
        return state

    def value(self, state):
        q_values = self.DQN(state)
        return q_values

    def act(self, state, epsilon=None): # sample actions with epsilon-greedy policy

        if epsilon is None: epsilon = self.epsilon

        q_values = self.value(state).cpu().detach().numpy()
        if random.random() < epsilon:
            aciton = random.randrange(self.action_space.n)
        else:
            aciton = q_values.argmax(1)[0]
        return aciton

    def compute_td_loss(self, states, actions, rewards, next_states, is_done, gamma=0.99):
        # compute TD loss
        actions = torch.tensor(actions).long()
        rewards = torch.tensor(rewards, dtype=torch.float)
        is_done = torch.tensor(is_done).bool()

        if self.USE_CUDA:
            actions = actions.cuda()
            rewards = rewards.cuda()
            is_done = is_done.cuda()

        # q-values
        predicted_qvalues = self.DQN(states)

        # select q-values
        predicted_qvalues_for_actions = predicted_qvalues[range(states.shape[0]), actions]

        # compute q-values
        predicted_next_qvalues = self.DQN_target(next_states)

        # compute V*(next_states) using predicted next q-values
        next_state_values = predicted_next_qvalues.max(-1)[0]

        # "target q-values" for loss
        target_qvalues_for_actions = rewards + gamma * next_state_values

        # Q(s,a) = r(s,a) since s' doesn't exist
        target_qvalues_for_actions = torch.where(
            is_done, rewards, target_qvalues_for_actions)

        # mean squared error loss to minimize
        loss = F.smooth_l1_loss(predicted_qvalues_for_actions, target_qvalues_for_actions.detach())

        return loss

    def sample_from_buffer(self, batch_size):
        states, actions, rewards, next_states, dones = [], [], [], [], []
        for i in range(batch_size):
            idx = random.randint(0, self.memory_buffer.size() - 1)
            data = self.memory_buffer.buffer[idx]
            frame, action, reward, next_frame, done = data
            states.append(self.observe(frame))
            actions.append(action)
            rewards.append(reward)
            next_states.append(self.observe(next_frame))
            dones.append(done)
        return torch.cat(states), actions, rewards, torch.cat(next_states), dones

    def learn_from_experience(self, batch_size):
        if self.memory_buffer.size() > batch_size:
            states, actions, rewards, next_states, dones = self.sample_from_buffer(batch_size)
            td_loss = self.compute_td_loss(states, actions, rewards, next_states, dones)
            self.optimizer.zero_grad()
            td_loss.backward()
            for param in self.DQN.parameters():
                param.grad.data.clamp_(-1, 1)

            self.optimizer.step()
            return (td_loss.item())
        else:
            return (0)


if __name__ == '__main__':

    # Training DQN in PongNoFrameskip-v4
    env = make_atari('PongNoFrameskip-v4')
    env = wrap_deepmind(env, scale=False, frame_stack=True)

    gamma = 0.99
    epsilon_max = 1
    epsilon_min = 0.05
    eps_decay = 30000
    frames = 2000000
    USE_CUDA = True
    learning_rate = 2e-4
    max_buff = 100000
    update_tar_interval = 1000
    batch_size = 32
    print_interval = 1000
    log_interval = 1000
    learning_start = 10000
    win_reward = 18
    win_break = True

    action_space = env.action_space
    action_dim = env.action_space.n
    state_dim = env.observation_space.shape[0]
    state_channel = env.observation_space.shape[2]
    agent = DQNAgent(in_channels=state_channel, action_space=action_space, USE_CUDA=USE_CUDA, lr=learning_rate, memory_size=max_buff)

    frame = env.reset()

    episode_reward = 0
    all_rewards = []
    losses = []
    episode_num = 0
    is_win = False
    summary_writer = SummaryWriter(log_dir="DQN_stackframe", comment="good_makeatari")

    # epsilon-greedy decay
    epsilon_by_frame = lambda frame_idx: epsilon_min + (epsilon_max - epsilon_min) * math.exp(
        -1. * frame_idx / eps_decay)
    # plt.plot([epsilon_by_frame(i) for i in range(10000)])

    for i in range(frames):
        epsilon = epsilon_by_frame(i)
        state_tensor = agent.observe(frame)
        action = agent.act(state_tensor, epsilon)

        next_frame, reward, done, _ = env.step(action)

        episode_reward += reward
        agent.memory_buffer.push(frame, action, reward, next_frame, done)
        frame = next_frame

        loss = 0
        if agent.memory_buffer.size() >= learning_start:
            loss = agent.learn_from_experience(batch_size)
            losses.append(loss)

        if i % print_interval == 0:
            print("frames: %5d, reward: %5f, loss: %4f, epsilon: %5f, episode: %4d" % (
                i, np.mean(all_rewards[-10:]), loss, epsilon, episode_num))
            summary_writer.add_scalar("Temporal Difference Loss", loss, i)
            summary_writer.add_scalar("Mean Reward", np.mean(all_rewards[-10:]), i)
            summary_writer.add_scalar("Epsilon", epsilon, i)

        if i % update_tar_interval == 0:
            agent.DQN_target.load_state_dict(agent.DQN.state_dict())

        if done:
            frame = env.reset()

            all_rewards.append(episode_reward)
            episode_reward = 0
            episode_num += 1
            avg_reward = float(np.mean(all_rewards[-100:]))

    summary_writer.close()
    torch.save(agent.DQN.state_dict(), "trained model/DQN_dict.pth.tar")


    def moving_average(a, n=3):
        ret = np.cumsum(a, dtype=float)
        ret[n:] = ret[n:] - ret[:-n]
        return ret[n - 1:] / n


    def plot_training(frame_idx, rewards, losses):
        clear_output(True)
        plt.figure(figsize=(20, 5))
        plt.subplot(131)
        plt.title('frame %s. reward: %s' % (frame_idx, np.mean(rewards[-10:])))
        plt.plot(rewards)
        plt.subplot(132)
        plt.title('loss')
        plt.plot(losses)
        plt.show()


    plot_training(i, all_rewards, losses)
