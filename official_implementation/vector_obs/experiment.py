import numpy as np
import torch
import wandb

import argparse
import random
from torch.nn import functional as F

from decision_transformers.vector_obs.util_functions import load_checkpoint, make_dataset
from envs.double_goal_minigrid import DoubleGoalEnv
from evaluation.evaluate import evaluate_episode_rtg, eval_and_gif
from models.decision_transformer import DecisionTransformer
from training.seq_trainer import SequenceTrainer
from utils.minigrid_wrappers import FullyObsFeatureWrapper


def discount_cumsum(x, gamma):
    discount_cumsum = np.zeros_like(x)
    discount_cumsum[-1] = x[-1]
    for t in reversed(range(x.shape[0] - 1)):
        discount_cumsum[t] = x[t] + gamma * discount_cumsum[t + 1]
    return discount_cumsum


def experiment(variant, train):
    device = variant.get('device', 'cuda')
    log_to_wandb = variant.get('log_to_wandb', True)

    env_name = variant['env']
    env_mode = variant["env_mode"]
    env = FullyObsFeatureWrapper(
        DoubleGoalEnv(
            mode=env_mode, agent_start_pos=None, max_steps=50, agent_pov=9
        )
    )
    max_ep_len = 28  # 50 # 100
    env_targets = [1, 0.87, 0]
    scale = 1.  # 1000.  # normalization for rewards/returns

    state_dim = 300  # env.observation_space.shape[0]
    act_dim = 1
    K = variant['K']

    # save trained model and wandb metrics
    dataset_name = "uni_modal_restricted"
    group_name = f'{env_name}-{dataset_name}-dataset'
    exp_prefix = f'mode-{env_mode}-vocab_size-{K}-max_episode_len-{max_ep_len}-exp_id-{random.randint(int(1e5), int(1e6) - 1)}'
    model_name = f'dt_double_goal-mode-{env_mode}-vocab_size-{K}-max_episode_len-{max_ep_len}-env-{env_name}-dataset-{dataset_name}'
    checkpoint_path = f"/home/sara/repositories/player_model_dt/trained_models/{model_name}.pth"

    # load dataset
    dataset_path = "/home/sara/repositories/player_model_dt/data/restricted_double_goal_mode_" + str(
        env_mode) + "_image_ft_0_nsteps_100000_ppo_num_episodes100000_eps_0.1_gamma_0.99_dr_False.npz"
    trajectories = make_dataset(dataset_path)

    # Filtering the trajectories based on episode length
    # trajectories = [traj for traj in trajectories if len(traj['timestep']) >= K]
    print(max([len(traj['timestep']) for traj in trajectories]), min([len(traj['timestep']) for traj in trajectories]))
    # save all path information into separate lists
    mode = variant.get('mode', 'normal')
    # states = np.array(trajectories['obs'])
    # returns = np.array(trajectories['reward_to_go'])
    # traj_lens = np.array([len(trajectories['timestep']) for _ in trajectories['timestep']])

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

    batch_size = variant['batch_size']
    num_eval_episodes = variant['num_eval_episodes']
    pct_traj = variant.get('pct_traj', 1.)

    # only train on top pct_traj trajectories (for %BC experiment)
    num_timesteps = max(int(pct_traj * num_timesteps), 1)
    sorted_inds = np.argsort(returns)  # lowest to highest
    num_trajectories = 1
    timesteps = traj_lens[sorted_inds[-1]]
    ind = len(trajectories) - 2
    while ind >= 0 and timesteps + traj_lens[sorted_inds[ind]] <= num_timesteps:
        timesteps += traj_lens[sorted_inds[ind]]
        num_trajectories += 1
        ind -= 1
    sorted_inds = sorted_inds[-num_trajectories:]

    # used to re-weight sampling, so we sample according to timesteps instead of trajectories
    p_sample = traj_lens[sorted_inds] / sum(traj_lens[sorted_inds])

    def get_batch(batch_size=128, max_len=K):
        batch_inds = np.random.choice(
            np.arange(num_trajectories),
            size=batch_size,
            replace=True,
            p=p_sample,  # reweights so we sample according to timesteps
        )

        s, a, r, d, rtg, timesteps, mask = [], [], [], [], [], [], []
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

        s = torch.from_numpy(np.concatenate(s, axis=0)).to(dtype=torch.float32, device=device)
        a = torch.from_numpy(np.concatenate(a, axis=0)).to(dtype=torch.float32, device=device)
        r = torch.from_numpy(np.concatenate(r, axis=0)).to(dtype=torch.float32, device=device)
        d = torch.from_numpy(np.concatenate(d, axis=0)).to(dtype=torch.long, device=device)
        rtg = torch.from_numpy(np.concatenate(rtg, axis=0)).to(dtype=torch.float32, device=device)
        # print(np.concatenate(timesteps, axis=0))
        timesteps = torch.from_numpy(np.concatenate(timesteps, axis=0)).to(dtype=torch.long, device=device)
        mask = torch.from_numpy(np.concatenate(mask, axis=0)).to(device=device)

        return s, a, r, d, rtg, timesteps, mask

    def eval_episodes(target_rew):
        def fn(model):
            returns, lengths = [], []
            for _ in range(num_eval_episodes):
                with torch.no_grad():
                    ret, length = evaluate_episode_rtg(
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
                    )
                returns.append(ret)
                lengths.append(length)
            return {
                f'target_{target_rew}_return_mean': np.mean(returns),
                # f'target_{target_rew}_return_std': np.std(returns),
                # f'target_{target_rew}_length_mean': np.mean(lengths),
                # f'target_{target_rew}_length_std': np.std(lengths),
            }

        return fn

    model = DecisionTransformer(
        state_dim=state_dim,
        act_dim=4,
        max_length=K,
        max_ep_len=max_ep_len,
        hidden_size=variant['embed_dim'],
        n_layer=variant['n_layer'],
        n_head=variant['n_head'],
        n_inner=4 * variant['embed_dim'],
        activation_function=variant['activation_function'],
        n_positions=1024,
        resid_pdrop=variant['dropout'],
        attn_pdrop=variant['dropout'],
    )

    model = model.to(device=device)

    warmup_steps = variant['warmup_steps']
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=variant['learning_rate'],
        weight_decay=variant['weight_decay'],
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda steps: min((steps + 1) / warmup_steps, 1)
    )

    trainer = SequenceTrainer(
        model=model,
        optimizer=optimizer,
        batch_size=batch_size,
        get_batch=get_batch,
        checkpoint_path=checkpoint_path,
        scheduler=scheduler,
        loss_fn=lambda s_hat, a_hat, r_hat, s, a, r: F.cross_entropy(a_hat, a),
        eval_fns=[eval_episodes(tar) for tar in env_targets],
    )

    # set up wandb

    if log_to_wandb:
        wandb.init(
            name=exp_prefix,
            group=group_name,
            project='decision-transformer',
            config=variant
        )
    if train:
        for iter in range(variant['max_iters']):
            outputs = trainer.train_iteration(num_steps=variant['num_steps_per_iter'], iter_num=iter + 1,
                                              print_logs=True)
            if log_to_wandb:
                wandb.log(outputs)
    else:
        env = FullyObsFeatureWrapper(
            DoubleGoalEnv(
                mode=env_mode,
                render_mode="rgb_array",
                agent_start_pos=None,
                max_steps=50,
            )
        )
        model = load_checkpoint(model, checkpoint_path)
        eval_and_gif(env, model, variant['max_iters'], model_name, target_ret=[0.98], state_mean=state_mean,
                     state_std=state_std, max_ep_len=max_ep_len)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='miniGrid')
    parser.add_argument('--env_mode', type=int, default=0)
    parser.add_argument('--dataset_path', type=str)
    parser.add_argument('--mode', type=str, default='normal')  # normal for standard setting, delayed for sparse
    parser.add_argument('--K', type=int, default=5)
    parser.add_argument('--pct_traj', type=float, default=1.)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--embed_dim', type=int, default=128)
    parser.add_argument('--n_layer', type=int, default=2)
    parser.add_argument('--n_head', type=int, default=1)
    parser.add_argument('--activation_function', type=str, default='relu')
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--learning_rate', '-lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', '-wd', type=float, default=1e-3)
    parser.add_argument('--warmup_steps', type=int, default=10000)
    parser.add_argument('--num_eval_episodes', type=int, default=100)
    parser.add_argument('--max_iters', type=int, default=10)
    parser.add_argument('--num_steps_per_iter', type=int, default=5000)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--log_to_wandb', '-w', type=bool, default=True)
    args = parser.parse_args()

    experiment(variant=vars(args), train=True)
