from DDQN import *
import time

def get_state(obs):
    state = np.array(obs)
    state = state.transpose((2, 0, 1))
    state = torch.from_numpy(state)
    return state.unsqueeze(0)/255.0


def test():
    env = make_atari('PongNoFrameskip-v4')
    seed = random.randint(0, 10000)
    print(seed)
    env.seed(seed)
    env = wrap_deepmind(env, scale=False, frame_stack=True)
    dqn = DQN(in_channels=env.observation_space.shape[2], num_actions=env.action_space.n)
    dqn.load_state_dict(torch.load('trained model/DDQN_dict.pth.tar'))

    obs = env.reset()
    state = get_state(obs)
    total_reward = 0.0
    while True:

        action = dqn(state.to(torch.float32)).max(1)[1].view(1,1)
        if True:
            env.render()
            time.sleep(0.02)
        # obs, reward, done, info = env.step(action)
        obs, reward, done, info = env.step(action)
        total_reward += reward
        if not done:
            next_state = get_state(obs)
        else:
            next_state = None
        state = next_state
        if done:
            print('finished with reward {}'.format(total_reward))
            break

    env.close()
    return


if __name__ == '__main__':
    test()