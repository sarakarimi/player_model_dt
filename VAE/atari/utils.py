import ale_py.env.gym
import gym
from matplotlib import pyplot as plt
from skill_network import *
from configs import *
import numpy


def get_model(env: object, env_name: str) -> object:
    model = CnnLMP(latent_dim=config['latent_dim'], state_dim=(1, 84, 84),
                   action_dim=6, hidden_dims=config['hidden_dims'],
                   tanh=config['tanh'], latent_reg=config['latent_reg'], ar=False)
    return model


def load_ae_model(env, path, env_name=None):
    model = get_model(env, env_name=env_name)
    checkpoint = torch.load(path, map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint['gp_aa_model'])
    return model


def get_trained_ae_model(action_dim, state_dim, path, env_name=None):
    ae_model = LMP(latent_dim=config['latent_dim'], state_dim=state_dim, action_dim=action_dim,
                   hidden_dims=config['hidden_dims'], goal_idxs=config['goal_idxs'], tanh=config['tanh'],
                   latent_reg=config['latent_reg'], ar=False)
    checkpoint = torch.load(path, map_location=torch.device('cuda'))
    ae_model.load_state_dict(checkpoint['gp_aa_model'])
    return ae_model


def make_env(costume_map=False):
    env = gym.make(config['env_name'], reward_type='dense')
    return env


def play_policy(env, model, num_eval, traj_length, tanh, render=False, encode_state=False):
    model.eval()
    rewards = []
    for i in range(num_eval):
        print(i)
        state = env.reset()
        state = torch.FloatTensor(state[0]).unsqueeze(0).unsqueeze(0)
        done = False
        reward = 0
        num_steps = 0
        while not done:
            latent, _ = model.prior.act(latent=None, state=state, encode_state=True)
            print(latent)
            if tanh:
                latent = torch.tanh(latent)
            for t in range(traj_length):
                action, _ = model.decoder.act(latent, state, encode_state=True)
                action = action.cpu().numpy().flatten()
                print(action)
                s, r, terminated, truncated, info = env.step(action[0])
                done = terminated or truncated
                reward += r
                num_steps += 1
                state = torch.FloatTensor(s).unsqueeze(0).unsqueeze(0)
                # done = done and 'TimeLimit.truncated' not in info
                # if done:
                #     print(reward, info)
                #     rewards.append(reward)
                #     break
        print(reward)
        rewards.append(reward)
    print(numpy.mean(rewards))
    return rewards


def evaluate(env, model):
    rewards = play_policy(env, model, config['num_eval'], config['traj_length'], config['tanh'], render=True)
    plt.plot(rewards, label='AE')
    plt.xlabel("Steps")
    plt.ylabel("reward")
    plt.legend(loc='lower right', frameon=True)
    plt.title(config['env_name'])
    plt.show()


if __name__ == '__main__':
    import gym
    from gym.wrappers import AtariPreprocessing
    env = gym.make("ALE/SpaceInvaders-v5", frameskip=1, render_mode='human')
    print(env.action_space, env.observation_space)
    env = AtariPreprocessing(env)
    print(env.action_space, env.observation_space)

    model = load_ae_model(env, "/home/sara/repositories/player_model_dt/VAE/atari/models/incentives_dataset_model/atari-100.pt")
    play_policy(env, model, config['num_eval'], config['traj_length'], config['tanh'])