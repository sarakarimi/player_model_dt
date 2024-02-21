import random
import numpy as np
import torch

from decision_transformers.vector_obs.evaluation.bimodal_evalute import prompt_evaluate_episode_rtg, eval_and_gif
from envs.double_goal_minigrid import DoubleGoalEnv
from utils.minigrid_wrappers import FullyObsFeatureWrapper


def make_dataset(path):
    trajectories = np.load(path)
    trajectories = dict(trajectories)
    trajectories['obs'] = torch.flatten(torch.Tensor(trajectories['obs']), start_dim=1)
    trajectories['next_obs'] = torch.flatten(torch.Tensor(trajectories['next_obs']), start_dim=1)
    trajectories['action'] = torch.flatten(torch.Tensor(trajectories['action']), start_dim=0)
    trajectories['timestep'] = torch.flatten(torch.Tensor(trajectories['timestep']), start_dim=0)
    trajectories['reward'] = torch.flatten(torch.Tensor(trajectories['reward']), start_dim=0)
    trajectories['reward_to_go'] = torch.flatten(torch.Tensor(trajectories['reward_to_go']), start_dim=0)
    trajectories['dones'] = torch.flatten(torch.Tensor(trajectories['dones']), start_dim=0)

    indexes = np.where(np.array(trajectories['dones']) == 1)[0]
    previous = 0
    trajectories_list = []
    for i in indexes:
        trajectories_list.append({'timestep': np.array(trajectories['timestep'][previous: i + 1]),
                                  'obs': np.array(trajectories['obs'][previous: i + 1]),
                                  'action': np.array(trajectories['action'][previous: i + 1]),
                                  'reward': np.array(trajectories['reward'][previous: i + 1]),
                                  'reward_to_go': np.array(trajectories['reward_to_go'][previous: i + 1]),
                                  'next_obs': np.array(trajectories['next_obs'][previous: i + 1]),
                                  'dones': np.array(trajectories['dones'][previous: i + 1])})

        previous = i + 1
    return trajectories_list


def make_bimodal_dataset(mode_list):
    trajectories_list = []
    for mode in mode_list:
        dataset_path = f"/home/sara/repositories/player_model_dt/data/restricted_double_goal_mode_{mode}_image_ft_0_nsteps_100000_ppo_num_episodes100000_eps_0_gamma_0.99_dr_False.npz"
        trajectories = make_dataset(dataset_path)

        trajectories_list.append(trajectories)
    return trajectories_list


def process_dataset(trajectories, mode, env_name, pct_traj):
    # save all path information into separate lists
    states, traj_lens, returns = [], [], []
    for path in trajectories:
        if mode == 'delayed':  # delayed: all rewards moved to end of trajectory
            path['reward'][-1] = path['reward'].sum()
            path['reward'][:-1] = 0.
        states.append(path['obs'])
        traj_lens.append(len(path['obs']))
        returns.append(path['reward'].sum())
    traj_lens, returns = np.array(traj_lens), np.array(returns)

    # used for input normalization
    states = np.concatenate(states, axis=0)
    state_mean, state_std = np.mean(states, axis=0), np.std(states, axis=0) + 1e-6

    num_timesteps = sum(traj_lens)

    print('=' * 50)
    print(f'Starting new experiment: {env_name}')
    print(f'{len(traj_lens)} trajectories, {num_timesteps} timesteps found')
    print(f'Average return: {np.mean(returns):.2f}, std: {np.std(returns):.2f}')
    print(f'Max return: {np.max(returns):.2f}, min: {np.min(returns):.2f}')
    print('=' * 50)

    # only train on top pct_traj trajectories (for %BC experiment)
    num_timesteps = max(int(pct_traj * num_timesteps), 1)
    sorted_inds = np.argsort(returns)  # lowest to highest
    num_trajectories = 1
    timesteps = traj_lens[sorted_inds[-1]]
    ind = len(trajectories) - 2
    while ind >= 0 and timesteps + traj_lens[sorted_inds[ind]] < num_timesteps:
        timesteps += traj_lens[sorted_inds[ind]]
        num_trajectories += 1
        ind -= 1
    sorted_inds = sorted_inds[-num_trajectories:]

    # used to reweight sampling so we sample according to timesteps instead of trajectories
    p_sample = traj_lens[sorted_inds] / sum(traj_lens[sorted_inds])
    reward_info = [np.mean(returns), np.std(returns), np.max(returns), np.min(returns)]

    return trajectories, num_trajectories, sorted_inds, p_sample, state_mean, state_std, reward_info


def gen_env(env_mode):
    env = FullyObsFeatureWrapper(
        DoubleGoalEnv(
            mode=env_mode, agent_start_pos=None, max_steps=50, agent_pov=9
        )
    )
    return env


def get_env_list(env_name_list, max_ep_len, env_targets, scale, state_dim, act_dim, device):
    info = {}
    env_list = []

    for env_name in env_name_list:
        env_name = str(env_name)
        info[env_name] = {}
        env = gen_env(env_mode=env_name)
        info[env_name]['max_ep_len'] = max_ep_len
        info[env_name]['env_targets'] = env_targets
        info[env_name]['scale'] = scale
        info[env_name]['state_dim'] = state_dim
        info[env_name]['act_dim'] = act_dim
        info[env_name]['device'] = device
        env_list.append(env)
    return info, env_list


def process_info(env_name_list, trajectories_list, info, mode, pct_traj, variant):
    for i, env_name in enumerate(env_name_list):
        trajectories, num_trajectories, sorted_inds, p_sample, state_mean, state_std, reward_info = process_dataset(
            trajectories=trajectories_list[i], mode=mode, env_name=env_name_list[i], pct_traj=pct_traj)
        env_name = str(env_name)
        info[env_name]['num_trajectories'] = num_trajectories
        info[env_name]['sorted_inds'] = sorted_inds
        info[env_name]['p_sample'] = p_sample
        info[env_name]['state_mean'] = state_mean
        info[env_name]['state_std'] = state_std

    return info


def flatten_mode(prompt, batch_size):
    m, m_mask = prompt
    m = m.reshape((batch_size, -1, m.shape[-1]))
    m_mask = m_mask.reshape((batch_size, -1))
    return m, m_mask


def get_batch(trajectories, info, variant, env_id, mode_trajectories=True):
    num_trajectories, p_sample, sorted_inds = info['num_trajectories'], info['p_sample'], info['sorted_inds']
    max_ep_len, state_mean, state_std, scale = info['max_ep_len'], info['state_mean'], info['state_std'], info['scale']
    state_dim, act_dim, device = info['state_dim'], info['act_dim'], info['device']
    batch_size, K, prompt_len = variant['per_env_batch_size'], variant['K'], variant['prompt_length']

    def fn(batch_size=batch_size, max_len=K):
        batch_inds = np.random.choice(
            np.arange(num_trajectories),
            size=batch_size,
            replace=True,
            p=p_sample,  # reweights so we sample according to timesteps
        )

        s, a, r, d, rtg, timesteps, mask, m, mode_mask = [], [], [], [], [], [], [], [], []
        for i in range(batch_size):
            traj = trajectories[int(sorted_inds[batch_inds[i]])]

            si = random.randint(0, traj['reward'].shape[0] - 1)
            # print(si, max_len)

            # get sequences from dataset
            s.append(traj['obs'][si:si + max_len].reshape(1, -1, state_dim))
            a.append(traj['action'][si:si + max_len].reshape(1, -1, act_dim))
            r.append(traj['reward'][si:si + max_len].reshape(1, -1, 1))
            # rtg.append(traj['reward_to_go'][si:si + max_len].reshape(1, -1, 1))
            # timesteps.append(traj['timestep'][si:si + si + s[-1].shape[1]].reshape(1, -1))

            # if mode_trajectories is not None:
            #     # mode traj
            #     mode_traj = mode_trajectories[int(sorted_inds[batch_inds[i]])]
            #     si = random.randint(0, mode_traj['m'].shape[0] - 1)
            #     m.append(mode_traj['m'][si:si + max_len].reshape(1, -1, 1))

            if 'terminals' in traj:
                d.append(traj['terminals'][si:si + max_len].reshape(1, -1))
            else:
                d.append(traj['dones'][si:si + max_len].reshape(1, -1))
            timesteps.append(np.arange(si, si + s[-1].shape[1]).reshape(1, -1))
            timesteps[-1][timesteps[-1] >= max_ep_len] = max_ep_len - 1  # padding cutoff

            rtg.append(discount_cumsum(traj['reward'][si:], gamma=1.)[:s[-1].shape[1] + 1].reshape(1, -1, 1))
            if rtg[-1].shape[1] <= s[-1].shape[1]:
                rtg[-1] = np.concatenate([rtg[-1], np.zeros((1, 1, 1))], axis=1)

            # padding and state + reward normalization
            tlen = s[-1].shape[1]
            s[-1] = np.concatenate([np.zeros((1, max_len - tlen, state_dim)), s[-1]], axis=1)
            s[-1] = (s[-1] - state_mean) / state_std
            a[-1] = np.concatenate([np.zeros((1, max_len - tlen, act_dim)), a[-1]], axis=1)
            r[-1] = np.concatenate([np.zeros((1, max_len - tlen, 1)), r[-1]], axis=1)
            d[-1] = np.concatenate([np.ones((1, max_len - tlen)) * 2, d[-1]], axis=1)
            rtg[-1] = np.concatenate([np.zeros((1, max_len - tlen, 1)), rtg[-1]], axis=1) / scale
            timesteps[-1] = np.concatenate([np.zeros((1, max_len - tlen)), timesteps[-1]], axis=1)
            mask.append(np.concatenate([np.zeros((1, max_len - tlen)), np.ones((1, tlen))], axis=1))

            if mode_trajectories:
                # pad mode
                m.append(np.ones((1, prompt_len, 1)) * env_id)
                mode_mask.append(np.ones((1, prompt_len)))

        s = torch.from_numpy(np.concatenate(s, axis=0)).to(dtype=torch.float32, device=device)
        a = torch.from_numpy(np.concatenate(a, axis=0)).to(dtype=torch.float32, device=device)
        r = torch.from_numpy(np.concatenate(r, axis=0)).to(dtype=torch.float32, device=device)
        d = torch.from_numpy(np.concatenate(d, axis=0)).to(dtype=torch.long, device=device)
        rtg = torch.from_numpy(np.concatenate(rtg, axis=0)).to(dtype=torch.float32, device=device)
        # print(np.concatenate(timesteps, axis=0))
        timesteps = torch.from_numpy(np.concatenate(timesteps, axis=0)).to(dtype=torch.long, device=device)
        mask = torch.from_numpy(np.concatenate(mask, axis=0)).to(device=device)

        if mode_trajectories:
            m = torch.from_numpy(np.concatenate(m, axis=0)).to(dtype=torch.float32, device=device)
            mode_mask = torch.from_numpy(np.concatenate(mode_mask, axis=0)).to(device=device)
            return s, a, r, d, rtg, timesteps, mask, m, mode_mask

        return s, a, r, d, rtg, timesteps, mask

    return fn


def get_mode(info, env_id, variant):
    device = info['device']
    num_episodes, prompt_len = variant['prompt_episode'], variant['prompt_length']

    def fn(sample_size=1):

        m, mode_mask = [], []
        for i in range(int(num_episodes*sample_size)):
            # pad mode
            m.append(np.ones((1, prompt_len, 1)) * env_id)
            mode_mask.append(np.ones((1, prompt_len)))

        m = torch.from_numpy(np.concatenate(m, axis=0)).to(dtype=torch.float32, device=device)
        mode_mask = torch.from_numpy(np.concatenate(mode_mask, axis=0)).to(device=device)
        return m, mode_mask

    return fn


def get_mode_batch(trajectories_list, mode_trajectories, info, variant, train_env_list):
    per_env_batch_size = variant['per_env_batch_size']

    def fn(batch_size=per_env_batch_size):
        m_list, m_mask_list = [], []
        s_list, a_list, r_list, d_list, rtg_list, timesteps_list, mask_list = [], [], [], [], [], [], []

        # iterate over different modalities in the datasets and select batches with corresponding modes
        for env_id, env_name in enumerate(train_env_list):
            env_name = str(env_name)
            get_batch_fn = get_batch(trajectories_list[env_id], info[env_name], variant, env_id + 1, mode_trajectories)

            batch = get_batch_fn(batch_size=batch_size)
            s, a, r, d, rtg, timesteps, mask, m, m_mask = batch
            # m, m_mask = flatten_mode((m, m_mask), batch_size)

            if variant['no_r']:
                r = torch.zeros_like(r)
            if variant['no_rtg']:
                rtg = torch.zeros_like(rtg)
            s_list.append(s)
            a_list.append(a)
            r_list.append(r)
            d_list.append(d)
            rtg_list.append(rtg)
            timesteps_list.append(timesteps)
            mask_list.append(mask)
            m_mask_list.append(m_mask)
            m_list.append(m)

        m = torch.cat(m_list, dim=0)
        m_mask = torch.cat(m_mask_list, dim=0)
        s, a, r, d = torch.cat(s_list, dim=0), torch.cat(a_list, dim=0), torch.cat(r_list, dim=0), torch.cat(d_list,
                                                                                                             dim=0)
        rtg, timesteps, mask = torch.cat(rtg_list, dim=0), torch.cat(timesteps_list, dim=0), torch.cat(mask_list, dim=0)
        mode = m, m_mask
        batch = s, a, r, d, rtg, timesteps, mask
        return mode, batch

    return fn


def eval_episodes(target_rew, info, variant, env, env_name):
    max_ep_len, state_mean, state_std, scale = info['max_ep_len'], info['state_mean'], info['state_std'], info['scale']
    state_dim, act_dim, device = info['state_dim'], info['act_dim'], info['device']
    num_eval_episodes = variant['num_eval_episodes']
    mode = variant.get('mode', 'normal')

    def fn(model, prompt=None):
        returns = []
        for _ in range(num_eval_episodes):
            with torch.no_grad():
                ret = prompt_evaluate_episode_rtg(
                    env,
                    state_dim,
                    act_dim,
                    model,
                    max_ep_len=max_ep_len,
                    scale=scale,
                    target_return=target_rew / scale,
                    mode=mode,
                    state_mean=state_mean,
                    state_std=state_std,
                    device=device,
                    prompt=prompt,
                    no_r=variant['no_r'],
                    no_rtg=variant['no_rtg'],
                    no_state_normalize=variant['no_state_normalize']
                    )
            returns.append(ret)
        return {
            f'{env_name}_target_{target_rew}_return_mean': np.mean(returns),
            # f'{env_name}_target_{target_rew}_return_std': np.std(returns),
            }
    return fn


def eval_and_gif_multi_env(model, eval_iter, model_name, max_ep_len, get_mode, env_name_list,
                           info, variant, env, target_return, no_prompt=False):
    for env_id, env_name in enumerate(env_name_list):
        # need to sample eval_fn and prompt together
        state_mean = info[str(env_name)]['state_mean']
        state_std = info[str(env_name)]['state_std']
        get_mode_fn = get_mode(info[str(env_name)], env_id, variant)
        if not no_prompt:
            prompt = flatten_mode(get_mode_fn(), batch_size=1)
        else:
            prompt = None
        eval_and_gif(env, model, eval_iter, model_name, target_return, state_mean, state_std, max_ep_len,
                     env_name, prompt)


def discount_cumsum(x, gamma):
    discount_cumsum = np.zeros_like(x)
    discount_cumsum[-1] = x[-1]
    for t in reversed(range(x.shape[0] - 1)):
        discount_cumsum[t] = x[t] + gamma * discount_cumsum[t + 1]
    return discount_cumsum


def load_checkpoint(model, ckpt_path):
    checkpoint = torch.load(ckpt_path)
    raw_model = model.module if hasattr(model, "module") else model
    raw_model.load_state_dict(checkpoint)
    return raw_model
