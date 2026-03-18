import random
from typing import Callable
import numpy as np
import torch
import plotly.express as px
from functorch.einops import rearrange
from torch.utils.data import Dataset
from minigrid.core.constants import COLOR_TO_IDX, OBJECT_TO_IDX
from collections import Counter
from dataset_utils.utils import TrajectoryReader


def controls_from_episode_summary(
    episode_summary,
    max_enemy_distance: float = 12.0,
) -> np.ndarray:
    """
    Build a float32 control vector from an episode_summary dict produced by
    MiniGridThreeStyles.  Returns zeros for any missing fields.

    Dimensions: [risk_tolerance, resource_pref, commitment]
    """
    if not isinstance(episode_summary, dict):
        episode_summary = {}

    min_dist        = float(episode_summary.get("min_enemy_distance", 0.0))
    avg_dist        = float(episode_summary.get("avg_enemy_distance", 0.0))
    path_efficiency = float(episode_summary.get("path_efficiency",    0.0))
    items_picked    = int(  episode_summary.get("items_picked",       0))
    picked_weapon      = float(bool(episode_summary.get("picked_weapon",      False)))
    picked_camouflage  = float(bool(episode_summary.get("picked_camouflage",  False)))

    norm_min = np.clip(min_dist / max_enemy_distance, 0.0, 1.0)
    norm_avg = np.clip(avg_dist / max_enemy_distance, 0.0, 1.0)

    risk_tolerance = 1.0 - norm_min
    resource_pref  = np.clip(items_picked / 2.0, 0.0, 1.0)
    stealth_pref   = np.clip(
        norm_avg * (1.0 - picked_weapon) + picked_camouflage * 0.9,
        0.0, 1.0,
    )
    safety_pref = np.clip(norm_avg + picked_camouflage * 0.3, 0.0, 1.0)
    commitment  = path_efficiency

    return np.array(
        [risk_tolerance, resource_pref, commitment],
        dtype=np.float32,
    )


class TrajectoryDataset(Dataset):
    def __init__(
            self,
            trajectory_paths,
            sampling=False,
            index_channel_only=False,
            state_normalization_factor=1,
            action_normalization_factor=1,
            max_len=1,
            prob_go_from_end=0,
            pct_traj=1.0,
            rtg_scale=1,
            normalize_state=False,
            preprocess_observations: Callable = None,
            device="cpu",
    ):
        self.trajectory_paths = trajectory_paths
        self.max_len = max_len
        self.prob_go_from_end = prob_go_from_end
        self.pct_traj = pct_traj
        self.device = device
        self.normalize_state = normalize_state
        self.rtg_scale = rtg_scale
        self.preprocess_observations = preprocess_observations

        # used for input processing of the trajectory embedding model
        self.state_normalization_factor = state_normalization_factor
        self.action_normalization_factor = action_normalization_factor
        self.index_channel_only = index_channel_only
        self.sampling = sampling

        self.load_trajectories()



    def load_trajectories(self) -> None:
        (merge_observations, merge_actions, merge_rewards, merge_returns, merge_dones,
         merge_truncated, merge_infos, merge_modes, merge_timesteps, merge_tasks) = [], [], [], [], [], [], [], [], [], []

        # used only for DEC-VAE experiments
        obs, acts, tasks = [], [], []

        # Iterating over many dataset with different environment modes or play styles
        for i, path in enumerate(self.trajectory_paths):

            traj_reader = TrajectoryReader(path)
            data = traj_reader.read()
            observations = data["data"].get("observations")
            actions = data["data"].get("actions")
            rewards = data["data"].get("rewards")
            dones = data["data"].get("dones")
            truncated = data["data"].get("truncated")
            infos = data["data"].get("infos")
            # mode = np.zeros((int(len(dones) / (i + 1)), 8, len(self.trajectory_paths)))
            # mode[:, :, i] = 1
            # modes = mode

            observations = np.array(observations)
            T_steps, B_envs = observations.shape[0], observations.shape[1]
            actions = np.array(actions)
            rewards = np.array(rewards)
            dones = np.array(dones)
            # modes = np.array(modes)

            # check whether observations are flat or an image
            if observations.shape[-1] == 3:
                # use state space that includes  object IDX in each grid position
                if self.index_channel_only:
                    observations = observations[:, :, :, :, 0] # we use thi in the VAE model
                else:
                    observations = observations[:, :, :, :, :] # We use this in the DT model
                self.observation_type = "index"
            elif observations.shape[-1] == 20:
                self.observation_type = "one_hot"
            else:
                raise ValueError(
                    "Observations are not flat or images, check the shape of the observations: ",
                    observations.shape,
                )
            if self.observation_type != "flat":
                if self.index_channel_only:
                    # use state space that includes  object IDX in each grid position
                    t_observations = rearrange(
                        torch.tensor(observations), "t b h w  -> (b t) (h w)"
                    )
                else:
                    t_observations = rearrange(
                        torch.tensor(observations), "t b h w c -> (b t) h w c"
                        # "t b h w c -> (b t) (h w c)"  --> format used by traj embedding model
                    )
            else:
                t_observations = rearrange(
                    torch.tensor(observations), "t b f -> (b t) f"
                )

            t_actions = rearrange(torch.tensor(actions), "t b -> (b t)")
            t_rewards = rearrange(torch.tensor(rewards), "t b -> (b t)")
            t_dones = rearrange(torch.tensor(dones), "t b -> (b t)")
            t_truncated = rearrange(torch.tensor(truncated), "t b -> (b t)")
            # t_modes = rearrange(torch.tensor(modes), "t b f -> (b t) f")
            t_done_or_truncated = torch.logical_or(t_dones, t_truncated)
            done_indices = torch.where(t_done_or_truncated)[0]

            # Extract the terminal info dict for each trajectory.
            # With gymnasium SyncVectorEnv, terminal infos live under info["final_info"][b].
            raw_terminal_infos = []
            if infos is not None:
                for flat_idx in done_indices.numpy():
                    t_step = int(flat_idx) % T_steps
                    b_env = int(flat_idx) // T_steps
                    ep_info = {}
                    step_info = infos[t_step]
                    if isinstance(step_info, dict) and "final_info" in step_info:
                        fi = step_info["final_info"]
                        if fi is not None and b_env < len(fi) and fi[b_env] is not None:
                            ep_info = fi[b_env] if isinstance(fi[b_env], dict) else {}
                    raw_terminal_infos.append(ep_info)
                raw_terminal_infos.append({})  # sentinel for the trailing empty segment
            else:
                raw_terminal_infos = [{} for _ in range(len(done_indices) + 1)]

            actions = torch.tensor_split(t_actions, done_indices + 1)
            rewards = torch.tensor_split(t_rewards, done_indices + 1)
            dones = torch.tensor_split(t_dones, done_indices + 1)
            truncated = torch.tensor_split(t_truncated, done_indices + 1)
            states = torch.tensor_split(t_observations, done_indices + 1)
            # self.modes = torch.tensor_split(t_modes, done_indices + 1)
            returns = [r.sum() for r in rewards]
            returns = ['%.2f' % elem for elem in returns]
            timesteps = [torch.arange(len(i)) for i in states]
            # modes = torch.zeros(len(actions), len(self.trajectory_paths))
            # modes[:, i] = 1

            # Sampling trajectories based on their lengths and returns
            top_seq_lengths = self.get_top_trajectory_lengths(states, returns, top_k=15)
            # print(top_seq_lengths)
            seq_lens = [seq_len[0] for seq_len in top_seq_lengths]

            if self.sampling:  # Use random sampled trajectories
                indexes = []
                index_lists = []
                # TODO remove hard-coded values
                num_samples = 2000
                for seq_len in seq_lens:
                    index_list = [index for index, (state, ret) in enumerate(zip(states, returns)) if
                                  len(state) == seq_len]
                    index_lists.extend(index_list)
                index_list_sample = random.sample(index_lists, num_samples)
                indexes.extend(index_list_sample)

            else:  # Use non-random trajectories
                num_samples = 2000
                indexes = [index for index, (state, ret) in enumerate(zip(states, returns)) if len(state) in seq_lens][-num_samples:]

            print(len(indexes))
            tasks = np.ones(len([actions[i] for i in indexes]), dtype=np.int64) * i
            states = [states[i] for i in indexes]
            actions = [actions[i] for i in indexes]
            rewards = [rewards[i] for i in indexes]
            dones = [dones[i] for i in indexes]
            truncated = [truncated[i] for i in indexes]
            returns = [returns[i] for i in indexes]
            timesteps = [timesteps[i] for i in indexes]
            terminal_infos = [raw_terminal_infos[i] for i in indexes]
            # modes = [modes[i] for i in indexes]

            top_seq_lengths = self.get_top_trajectory_lengths(states, returns, top_k=15)
            print(top_seq_lengths)

            # merge datasets
            merge_actions.extend(actions)
            merge_rewards.extend(rewards)
            merge_dones.extend(dones)
            merge_truncated.extend(truncated)
            merge_observations.extend(states)
            # merge_modes.extend(modes)
            merge_returns.extend(returns)
            merge_timesteps.extend(timesteps)
            merge_tasks.extend(tasks)
            merge_infos.extend(terminal_infos)

        self.actions = merge_actions
        self.rewards = merge_rewards
        self.dones = merge_dones
        self.truncated = merge_truncated
        self.states = merge_observations
        self.returns = merge_returns
        self.timesteps = merge_timesteps
        self.tasks = merge_tasks
        self.infos = merge_infos

        # ==================================
        # remove trajectories with length 0

        self.traj_lens = np.array([len(i) for i in self.states])
        traj_len_mask = self.traj_lens > 0
        self.actions = [i / self.action_normalization_factor for i, m in zip(self.actions, traj_len_mask) if m]
        self.rewards = [i for i, m in zip(self.rewards, traj_len_mask) if m]
        self.dones = [i for i, m in zip(self.dones, traj_len_mask) if m]
        self.truncated = [i for i, m in zip(self.truncated, traj_len_mask) if m]
        self.states = [i / self.state_normalization_factor for i, m in zip(self.states, traj_len_mask) if m]
        self.returns = [i for i, m in zip(self.returns, traj_len_mask) if m]
        self.timesteps = [i for i, m in zip(self.timesteps, traj_len_mask) if m]
        self.tasks = [i for i, m in zip(self.tasks, traj_len_mask) if m]
        self.infos = [i for i, m in zip(self.infos, traj_len_mask) if m]

        # Build per-trajectory control vectors from episode_summary
        self.controls = np.stack(
            [
                controls_from_episode_summary(
                    ep_info.get("episode_summary") if isinstance(ep_info, dict) else None
                )
                for ep_info in self.infos
            ],
            axis=0,
        )  # [N, control_dim]

        self.traj_lens = self.traj_lens[traj_len_mask]
        self.num_timesteps = sum(self.traj_lens)
        self.num_trajectories = len(self.states)

        self.state_dim = list(self.states[0][0].shape)
        self.act_dim = list(self.actions[0][0].shape)
        self.max_ep_len = max([len(i) for i in self.states])
        self.metadata = data["metadata"]

        self.indices = self.get_indices_of_top_p_trajectories(self.pct_traj)
        self.sampling_probabilities = self.get_sampling_probabilities()

        if self.normalize_state:
            self.state_mean, self.state_std = self.get_state_mean_std()
        else:
            self.state_mean = 0
            self.state_std = 1

        # TODO Make this way less hacky
        if self.preprocess_observations == one_hot_encode_observation:
            self.observation_type = "one_hot"

        # top_seq_lengths = self.get_top_trajectory_lengths(self.states, self.returns, top_k=6)
        # print(top_seq_lengths)

    @staticmethod
    def get_top_trajectory_lengths(states, returns, top_k=5):
        lengths = [len(s) for s in states]
        returns = [float(r) for r in returns]
        freq = Counter(zip(lengths, returns))
        sorted_items = sorted(freq.items(), key=lambda x: (x[0][1], x[1]), reverse=True)
        top_seq_lengths = []
        seen_lengths = set()
        for (length, ret), count in sorted_items:
            if length not in seen_lengths:
                top_seq_lengths.append((length, ret, count))
                seen_lengths.add(length)
            if len(top_seq_lengths) == top_k:
                break
        return top_seq_lengths

    def get_indices_of_top_p_trajectories(self, pct_traj):
        num_timesteps = max(int(pct_traj * self.num_timesteps), 1)
        sorted_inds = np.argsort(self.returns)

        num_trajectories = 1
        timesteps = self.traj_lens[sorted_inds[-1]]
        ind = self.num_trajectories - 1

        while (
                ind >= 0
                and timesteps + self.traj_lens[sorted_inds[ind]] < num_timesteps
        ):
            timesteps += self.traj_lens[sorted_inds[ind]]
            ind -= 1
            num_trajectories += 1

        sorted_inds = sorted_inds[-num_trajectories:]
        return sorted_inds

    def get_sampling_probabilities(self):
        p_sample = self.traj_lens[self.indices] / sum(
            self.traj_lens[self.indices]
        )
        return p_sample

    def discount_cumsum(self, x, gamma):
        discount_cumsum = np.zeros_like(x)
        discount_cumsum[-1] = x[-1]
        for time in reversed(range(x.shape[0] - 1)):
            discount_cumsum[time] = x[time] + gamma * discount_cumsum[time + 1]
        return discount_cumsum

    def get_state_mean_std(self):
        # used for input normalization
        all_states = np.concatenate(self.states, axis=0)
        state_mean, state_std = (
            np.mean(all_states, axis=0),
            np.std(all_states, axis=0) + 1e-6,
        )
        return state_mean, state_std

    def get_batch(self, batch_size=256, max_len=100, prob_go_from_end=None):
        sorted_inds = self.indices

        batch_inds = np.random.choice(
            np.arange(len(sorted_inds)),
            size=batch_size,
            replace=True,
            p=self.sampling_probabilities,  # reweights so we sample according to timesteps
        )

        # initialize np arrays not lists
        states, actions, rewards, dones, rewards_to_gos, timesteps, mask = (
            [],
            [],
            [],
            [],
            [],
            [],
            [],
        )
        for i in range(batch_size):
            # get the trajectory
            traj_index = sorted_inds[batch_inds[i]]

            s, a, r, d, rtg, ti, m = self.get_traj(
                traj_index, max_len, prob_go_from_end=prob_go_from_end
            )

            rewards.append(r)
            actions.append(a)
            states.append(s)
            dones.append(d)
            rewards_to_gos.append(rtg)
            mask.append(m)
            timesteps.append(ti)

        return self.return_tensors(states, actions, rewards, rewards_to_gos, dones, timesteps, mask)

    def get_traj(self, traj_index, max_len=100, prob_go_from_end=None):
        traj_rewards = self.rewards[traj_index]
        traj_states = self.states[traj_index]
        traj_actions = self.actions[traj_index]
        traj_dones = self.dones[traj_index]

        # TODO: configure this so non-sparse tasks are dealt with correctly!
        # This line is very slow if we use the "correct method"
        traj_rtg = np.ones(traj_rewards.shape) * traj_rewards[-1].item()

        # "Correct method"
        # traj_rtg = self.discount_cumsum(traj_rewards, gamma=1.0)

        # start index
        si = random.randint(0, traj_rewards.shape[0] - 1)
        if prob_go_from_end is not None:
            if random.random() < prob_go_from_end:
                si = traj_rewards.shape[0] - max_len
                si = max(0, si)  # make sure it's not negative

        # get sequences from dataset
        s = traj_states[si: si + max_len].reshape(1, -1, *self.state_dim)
        a = traj_actions[si: si + max_len].reshape(1, -1, *self.act_dim)
        r = traj_rewards[si: si + max_len].reshape(1, -1, 1)
        rtg = traj_rtg[si: si + max_len].reshape(1, -1, 1)
        d = traj_dones[si: si + max_len].reshape(1, -1)
        ti = np.arange(si, si + s.shape[1]).reshape(1, -1)

        # sometime the trajectory is shorter than max_len (due to random start index or end of episode)
        tlen = s.shape[1]

        # sanity check
        assert tlen <= max_len, f"tlen: {tlen} max_len: {max_len}"

        padding_required = max_len - tlen
        s = self.add_padding(s, 0, padding_required)
        a = self.add_padding(a, -10, padding_required)
        r = self.add_padding(r, 0, padding_required)
        rtg = self.add_padding(rtg, rtg[0, -1], padding_required)
        d = self.add_padding(d, 2, padding_required)
        ti = self.add_padding(ti, 0, padding_required)
        m = self.add_padding(np.ones((1, tlen)), 0, padding_required)

        # padding and state + reward normalization
        s = (s - self.state_mean) / self.state_std
        rtg = rtg / self.rtg_scale

        return self.return_tensors(s, a, r, rtg, d, ti, m)

    def add_padding(self, tokens, padding_token, padding_required):
        if padding_required > 0:
            return np.concatenate(
                [
                    np.ones((1, padding_required, *tokens.shape[2:]))
                    * padding_token,
                    tokens,
                ],
                axis=1,
            )
        return tokens

    def return_tensors(self, s, a, r, rtg, d, timesteps, mask):
        if isinstance(s, torch.Tensor):
            s = s.to(dtype=torch.float32, device=self.device)
        else:
            s = torch.from_numpy(s).to(dtype=torch.float32, device=self.device)

        if isinstance(a, torch.Tensor):
            a = a.to(dtype=torch.long, device=self.device)
        else:
            a = torch.from_numpy(a).to(dtype=torch.long, device=self.device)

        if isinstance(r, torch.Tensor):
            r = r.to(dtype=torch.float32, device=self.device)
        else:
            r = torch.from_numpy(r).to(dtype=torch.float32, device=self.device)

        if isinstance(rtg, torch.Tensor):
            rtg = rtg.to(dtype=torch.float32, device=self.device)
        else:
            rtg = torch.from_numpy(rtg).to(
                dtype=torch.float32, device=self.device
            )

        if isinstance(d, torch.Tensor):
            d = d.to(dtype=torch.bool, device=self.device)
        else:
            d = torch.from_numpy(d).to(dtype=torch.bool, device=self.device)
        timesteps = torch.from_numpy(timesteps).to(
            dtype=torch.long, device=self.device
        )
        mask = torch.from_numpy(mask).to(dtype=torch.bool, device=self.device)

        # squeeze out the batch dimension
        s = s.squeeze(0)
        a = a.squeeze(0)
        r = r.squeeze(0)
        rtg = rtg.squeeze(0)
        d = d.squeeze(0)
        timesteps = timesteps.squeeze(0)
        mask = mask.squeeze(0)

        # TODO fix the order of d, rtg here.
        return s, a, r, d, rtg, timesteps, mask

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        traj_index = self.indices[idx]

        s, a, r, d, rtg, ti, m = self.get_traj(traj_index, max_len=self.max_len,
                                                             prob_go_from_end=self.prob_go_from_end)
        if self.preprocess_observations is not None:
            s = self.preprocess_observations(s)
        return s, a, r, d, rtg, ti, m


class TrajectoryVisualizer:
    def __init__(self, trajectory_dataset: TrajectoryDataset):
        self.trajectory_loader = trajectory_dataset

    def plot_reward_over_time(self):
        reward = [i[-1] for i in self.trajectory_loader.rewards if len(i) > 0]
        timesteps = [
            i.max() for i in self.trajectory_loader.timesteps if len(i) > 0
        ]

        # create a categorical color array for reward <0, 0, >0
        colors = np.zeros(len(reward))
        colors[np.array(reward) < 0] = -1
        colors[np.array(reward) > 0] = 1

        color_map = {-1: "Negative", 0: "Zero", 1: "Positive"}

        fig = px.scatter(
            y=reward,
            x=timesteps,
            color=[color_map[i] for i in colors],
            title="Reward vs Timesteps",
            template="plotly_white",
            labels={
                "x": "Timesteps",
                "y": "Reward",
            },
            marginal_x="histogram",
            marginal_y="histogram",
        )

        return fig

    def plot_base_action_frequencies(self):
        fig = px.bar(
            y=torch.concat(self.trajectory_loader.actions).bincount()
            # x=[IDX_TO_ACTION[i] for i in range(7)],
            # color=[IDX_TO_ACTION[i] for i in range(7)],
        )

        fig.update_layout(
            title="Base Action Frequencies",
            xaxis_title="Action",
            yaxis_title="Frequency",
        )

        return fig


def one_hot_encode_observation(img: torch.Tensor) -> torch.Tensor:
    """Converts a batch of observations into one-hot encoded numpy arrays."""

    img = img.to(int)  # .numpy()
    batch_size, height, width, num_channels = img.shape
    num_bits = 20
    new_observation_space = (batch_size, height, width, num_bits)

    out = np.zeros(new_observation_space, dtype="uint8")

    for b in range(batch_size):
        for i in range(height):
            for j in range(width):
                value = img[b, i, j, 0]
                color = img[b, i, j, 1]
                state = img[b, i, j, 2]

                out[b, i, j, value] = 1
                out[b, i, j, len(OBJECT_TO_IDX) + color] = 1
                out[
                    b, i, j, len(OBJECT_TO_IDX) + len(COLOR_TO_IDX) + state
                ] = 1

    return torch.from_numpy(out).float()


if __name__ == '__main__':
    paths = [
        # "/home/sara/repositories/player_model_dt/trajectory_embedding/datasets/minigrid/PPO_trajectories_multigoal-goal-0.gz",
        # "/home/sara/repositories/player_model_dt/trajectory_embedding/datasets/minigrid/PPO_trajectories_multigoal-goal-1.gz",
        # "/home/sara/repositories/player_model_dt/trajectory_embedding/datasets/minigrid/PPO_trajectories_goal2.gz",
        # "/home/sara/repositories/player_model_dt/trajectory_embedding/datasets/minigrid/PPO_trajectories_goal3.gz",

        "/home/sara/repositories/player_model_dt/datasets/minigrid/three_style_env/PPO_trajectories_PPO_trajectories_three_style_env_bypass.gz",
        "/home/sara/repositories/player_model_dt/datasets/minigrid/three_style_env/PPO_trajectories_PPO_trajectories_three_style_env_weapon.gz",
        "/home/sara/repositories/player_model_dt/datasets/minigrid/three_style_env/PPO_trajectories_PPO_trajectories_three_style_env_backstab.gz",

    ]
    trajectory_data_set = TrajectoryDataset(trajectory_paths=paths, sampling=True)
    print(trajectory_data_set.states[-1].shape)
