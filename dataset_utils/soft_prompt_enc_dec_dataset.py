from typing import Callable, Optional, Tuple
import numpy as np
import random
import torch

from dataset_utils.minigrid_trajectory_dataset import TrajectoryDataset


def create_padding_mask(
        x: torch.Tensor,
        pad_token: float = 0.0,
        causal: bool = False,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """
    x: (B, S, D) padded with pad_token across all D dims (your collate_fn uses 0.0)
    returns:
      pad_mask: (B, S) bool, True=PAD
      tgt_mask: (S, S) bool, True=MASKED (only if causal=True else None)
    """
    with torch.no_grad():
        # a position is PAD if *all* features equal pad_token
        pad_mask = (x == pad_token).all(dim=-1)  # True=PAD

    tgt_mask = None
    if causal:
        S = x.size(1)
        # upper triangular (exclude diagonal): True means "cannot attend"
        tgt_mask = torch.triu(torch.ones(S, S, device=x.device, dtype=torch.bool), diagonal=1)

    return pad_mask, tgt_mask

class SoftPromptEncDecDataset(TrajectoryDataset):
    def __init__(
            self,
            trajectory_paths,
            vae_model_type,
            vae_model_path,
            vae_model_parameters,
            max_len=1,
            prob_go_from_end=0,
            pct_traj=1.0,
            rtg_scale=1,
            normalize_state=False,
            preprocess_observations: Callable = None,
            sampling=False,
            index_channel_only=False,
            state_normalization_factor=1,
            action_normalization_factor=1,
            device="cpu",
    ):
        super().__init__(trajectory_paths=trajectory_paths, max_len=max_len, prob_go_from_end=prob_go_from_end,
                         pct_traj=pct_traj, rtg_scale=rtg_scale, normalize_state=normalize_state,
                         preprocess_observations=preprocess_observations, sampling=sampling,
                         index_channel_only=index_channel_only,
                         state_normalization_factor=state_normalization_factor,
                         action_normalization_factor=action_normalization_factor,
                         device=device)

        self.vae_model_path = vae_model_path
        self.vae_model_parameters = vae_model_parameters
        self.vae_model_type = vae_model_type
        self.seq_lens = [len(seq) for seq in self.states]
        self.max_seq_len = max(self.seq_lens)

    def sample_random_prompts(self, task_label, random_state=None, num_samples=1):
        """
        Sample random trajectories from a given task/cluster.
        Returns the same format as training data: full trajectory components.

        Returns:
            full_states: [num_samples, max_seq_len, *state_dim]
            full_actions: [num_samples, max_seq_len, *act_dim]
            full_timesteps: [num_samples, max_seq_len]
            full_attention_mask: [num_samples, max_seq_len]
        """
        rng = np.random.default_rng(random_state)
        indices = np.where(np.asarray(self.tasks) == task_label)[0]
        if len(indices) == 0:
            raise ValueError("No samples in the specified cluster.")

        # Sample num_samples trajectories
        sampled_indices = rng.choice(indices, size=num_samples, replace=True)

        full_states_list = []
        full_actions_list = []
        full_timesteps_list = []
        full_masks_list = []

        for idx in sampled_indices:
            traj_states = self.states[idx]
            traj_actions = self.actions[idx]

            # Get full trajectory
            s = traj_states.reshape(1, -1, *self.state_dim)
            a = traj_actions.reshape(1, -1, *self.act_dim)
            ti = np.arange(0, s.shape[1]).reshape(1, -1)

            # Pad to max_seq_len
            tlen = s.shape[1]
            padding = self.max_seq_len - tlen
            s = self.add_padding(s, 0, padding)
            a = self.add_padding(a, -10, padding)
            ti = self.add_padding(ti, 0, padding)
            m = self.add_padding(np.ones((1, tlen)), 0, padding)

            # Normalize
            s = (s - self.state_mean) / self.state_std

            # Convert to tensors and remove batch dimension
            s_tensor, a_tensor, _, _, _, ti_tensor, m_tensor = self.return_tensors(
                s, a, np.zeros((1, self.max_seq_len, 1)),
                np.zeros((1, self.max_seq_len, 1)),
                np.zeros((1, self.max_seq_len)),
                ti, m
            )

            full_states_list.append(s_tensor)
            full_actions_list.append(a_tensor)
            full_timesteps_list.append(ti_tensor)
            full_masks_list.append(m_tensor)

        # Stack into batch
        full_states = torch.stack(full_states_list, dim=0)  # [num_samples, max_seq_len, *state_dim]
        full_actions = torch.stack(full_actions_list, dim=0)  # [num_samples, max_seq_len, *act_dim]
        full_timesteps = torch.stack(full_timesteps_list, dim=0)  # [num_samples, max_seq_len]
        full_attention_mask = torch.stack(full_masks_list, dim=0)  # [num_samples, max_seq_len]

        return full_states, full_actions, full_timesteps, full_attention_mask


    def get_traj(self, traj_index, max_len=100, prob_go_from_end=None):
        """Sample a random context window from the trajectory."""
        traj_rewards = self.rewards[traj_index]
        traj_states = self.states[traj_index]
        traj_actions = self.actions[traj_index]
        traj_dones = self.dones[traj_index]
        traj_rtg = np.ones(traj_rewards.shape) * traj_rewards[-1].item()

        # Choose start index
        si = random.randint(0, traj_rewards.shape[0] - 1)
        if prob_go_from_end is not None and random.random() < prob_go_from_end:
            si = max(0, traj_rewards.shape[0] - max_len)

        # Slice trajectory
        s = traj_states[si:si + max_len].reshape(1, -1, *self.state_dim)
        a = traj_actions[si:si + max_len].reshape(1, -1, *self.act_dim)
        r = traj_rewards[si:si + max_len].reshape(1, -1, 1)
        rtg = traj_rtg[si:si + max_len].reshape(1, -1, 1)
        d = traj_dones[si:si + max_len].reshape(1, -1)
        ti = np.arange(si, si + s.shape[1]).reshape(1, -1)

        # Pad to max_len
        tlen = s.shape[1]
        padding = max_len - tlen
        s = self.add_padding(s, 0, padding)
        a = self.add_padding(a, -10, padding)  # -10 for invalid actions
        r = self.add_padding(r, 0, padding)
        rtg = self.add_padding(rtg, rtg[0, -1], padding)
        d = self.add_padding(d, 2, padding)
        ti = self.add_padding(ti, 0, padding)
        m = self.add_padding(np.ones((1, tlen)), 0, padding)

        # Normalize
        s = (s - self.state_mean) / self.state_std
        rtg = rtg / self.rtg_scale

        return self.return_tensors(s, a, r, rtg, d, ti, m)

    def get_full_traj(self, traj_index):
        """Get the full trajectory padded to max_seq_len."""
        traj_rewards = self.rewards[traj_index]
        traj_states = self.states[traj_index]
        traj_actions = self.actions[traj_index]
        traj_dones = self.dones[traj_index]
        traj_rtg = np.ones(traj_rewards.shape) * traj_rewards[-1].item()

        # Full sequence from t=0
        s = traj_states.reshape(1, -1, *self.state_dim)
        a = traj_actions.reshape(1, -1, *self.act_dim)
        r = traj_rewards.reshape(1, -1, 1)
        rtg = traj_rtg.reshape(1, -1, 1)
        d = traj_dones.reshape(1, -1)
        ti = np.arange(0, s.shape[1]).reshape(1, -1)

        # Pad to max_seq_len
        tlen = s.shape[1]
        padding = self.max_seq_len - tlen
        s = self.add_padding(s, 0, padding)
        a = self.add_padding(a, -10, padding)
        r = self.add_padding(r, 0, padding)
        rtg = self.add_padding(rtg, rtg[0, -1], padding)
        d = self.add_padding(d, 2, padding)
        ti = self.add_padding(ti, 0, padding)
        m = self.add_padding(np.ones((1, tlen)), 0, padding)

        # Normalize
        s = (s - self.state_mean) / self.state_std
        rtg = rtg / self.rtg_scale

        return self.return_tensors(s, a, r, rtg, d, ti, m)

    def __getitem__(self, idx):
        traj_index = self.indices[idx]

        # DT context window
        s, a, r, d, rtg, ti, m = self.get_traj(
            traj_index,
            max_len=self.max_len,
            prob_go_from_end=self.prob_go_from_end,
        )

        # Full episode for encoder
        full_s, full_a, full_r, full_rtg, full_d, full_ti, full_m = self.get_full_traj(traj_index)

        task_label = self.tasks[traj_index]

        if self.preprocess_observations is not None:
            s = self.preprocess_observations(s)
            full_s = self.preprocess_observations(full_s)

        return (
            s, a, r, d, rtg, ti, m,
            full_s, full_a, full_r, full_rtg, full_d, full_ti, full_m,
            task_label,
        )

    @staticmethod
    def collate_fn(batch):
        """Collate batch into dict format."""
        (
            states, actions, rewards, dones, rtgs, timesteps, masks,
            full_states, full_actions, full_rewards, full_rtgs, full_dones, full_timesteps, full_masks,
            task_labels,
        ) = zip(*batch)

        return {
            # Decoder (context window)
            "states": torch.stack(states, dim=0),
            "actions": torch.stack(actions, dim=0),
            "rewards": torch.stack(rewards, dim=0),
            "returns_to_go": torch.stack(rtgs, dim=0),
            "timesteps": torch.stack(timesteps, dim=0),
            "attention_mask": torch.stack(masks, dim=0),
            "dones": torch.stack(dones, dim=0),
            # Encoder (full trajectory)
            "full_states": torch.stack(full_states, dim=0),
            "full_actions": torch.stack(full_actions, dim=0),
            "full_rewards": torch.stack(full_rewards, dim=0),
            "full_returns_to_go": torch.stack(full_rtgs, dim=0),
            "full_timesteps": torch.stack(full_timesteps, dim=0),
            "full_attention_mask": torch.stack(full_masks, dim=0),
            "full_dones": torch.stack(full_dones, dim=0),
            # Labels
            "task_labels": torch.tensor(task_labels, dtype=torch.long),
        }


if __name__ == '__main__':
    paths = [
        "/home/sara/repositories/player_model_dt/datasets/minigrid/three_style_env/PPO_trajectories_PPO_trajectories_three_style_env_backstab.gz",
        "/home/sara/repositories/player_model_dt/datasets/minigrid/three_style_env/PPO_trajectories_PPO_trajectories_three_style_env_bypass.gz",
        "/home/sara/repositories/player_model_dt/datasets/minigrid/three_style_env/PPO_trajectories_PPO_trajectories_three_style_env_weapon.gz",

    ]
    trajectory_data_set = SoftPromptEncDecDataset(trajectory_paths=paths, vae_model_path=vae_model_save_path,
                                            vae_model_parameters=vae_model_params)
    print(list(trajectory_data_set.actions))
