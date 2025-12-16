from typing import Callable
import numpy as np
from dataset_utils.minigrid_trajectory_dataset import TrajectoryDataset
from trajectory_embedding.style_dec_vae.lstm.eval import predict_clusters_vae
import random
import torch

class SoftPromptDataset(TrajectoryDataset):
    def __init__(
            self,
            trajectory_paths,
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
        self.cluster_predictions, self.style_vectors, self.cluster_centroids = self.predict_clusters_using_saved_model()

    def predict_clusters_using_saved_model(self):
        dataset_paths = self.trajectory_paths
        dataset_parameters = {
            'sampling': self.sampling,
            'index_channel_only': True,
            'state_normalization_factor': 9,
            'action_normalization_factor': 6,
        }
        model_path = self.vae_model_path
        model_parameters = self.vae_model_parameters
        cluster_predictions, style_vectors, cluster_centroids = predict_clusters_vae(model_path, model_parameters,
                                                                                     dataset_paths,
                                                                                     dataset_parameters, batch_size=128)
        return cluster_predictions, style_vectors, cluster_centroids

    def sample_random_prompts(self, task_label, random_state=None, num_samples=1):
        rng = np.random.default_rng(random_state)
        indices = np.where(self.cluster_predictions == task_label)[0]
        if len(indices) == 0:
            raise ValueError("No samples in the specified cluster.")
        idxs = rng.choice(indices)
        style_vectors = self.style_vectors[idxs]
        if isinstance(style_vectors, torch.Tensor):
            style_vectors = style_vectors.to(dtype=torch.float32, device=self.device)
        else:
            style_vectors = torch.from_numpy(style_vectors).to(dtype=torch.float32, device=self.device)
        return style_vectors

    def get_batch(self, batch_size=256, max_len=100, prob_go_from_end=None):
        sorted_inds = self.indices

        batch_inds = np.random.choice(
            np.arange(len(sorted_inds)),
            size=batch_size,
            replace=True,
            p=self.sampling_probabilities,  # reweights so we sample according to timesteps
        )

        # initialize np arrays not lists
        states, actions, rewards, dones, rewards_to_gos, timesteps, mask, style_vectors = (
            [],
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

            s, a, r, d, rtg, ti, m, style_vector = self.get_traj(
                traj_index, max_len, prob_go_from_end=prob_go_from_end
            )

            rewards.append(r)
            actions.append(a)
            states.append(s)
            dones.append(d)
            rewards_to_gos.append(rtg)
            mask.append(m)
            timesteps.append(ti)
            style_vectors.append(style_vector)

        return *self.return_tensors(states, actions, rewards, rewards_to_gos, dones, timesteps, mask), style_vectors

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

    def get_prompt_traj(self, prompt_index):
        traj_style_vectors = self.style_vectors[prompt_index]
        traj_style_vectors = traj_style_vectors.reshape(1, -1)
        if isinstance(traj_style_vectors, torch.Tensor):
            traj_style_vectors = traj_style_vectors.to(dtype=torch.long, device=self.device)
        else:
            traj_style_vectors = torch.from_numpy(traj_style_vectors).to(dtype=torch.float32, device=self.device)
        return traj_style_vectors

    def __getitem__(self, idx):
        traj_index = self.indices[idx]

        # find a random prompt trajectory that matches the env ID of the selected trajectory
        env_id = self.tasks[traj_index]
        matching_indexes = [i for i, val in enumerate(self.cluster_predictions) if val == env_id]
        prompt_index = random.choice(matching_indexes)

        s, a, r, d, rtg, ti, m = self.get_traj(traj_index, max_len=self.max_len,
                                                             prob_go_from_end=self.prob_go_from_end)

        style_vector = self.get_prompt_traj(prompt_index)


        if self.preprocess_observations is not None:
            s = self.preprocess_observations(s)
        return s, a, r, d, rtg, ti, m, style_vector




if __name__ == '__main__':
    from trajectory_embedding.style_dec_vae.configs.config_minigrid import *

    paths = [
        "/home/sara/repositories/player_model_dt/datasets/minigrid/three_style_env/PPO_trajectories_PPO_trajectories_three_style_env_backstab.gz",
        "/home/sara/repositories/player_model_dt/datasets/minigrid/three_style_env/PPO_trajectories_PPO_trajectories_three_style_env_bypass.gz",
        "/home/sara/repositories/player_model_dt/datasets/minigrid/three_style_env/PPO_trajectories_PPO_trajectories_three_style_env_weapon.gz",

    ]
    trajectory_data_set = SoftPromptDataset(trajectory_paths=paths, vae_model_path=vae_model_save_path,
                                            vae_model_parameters=vae_model_params)
    # print(trajectory_data_set.style_vectors[0])
    # print(trajectory_data_set.style_vectors[1999])
    # print(trajectory_data_set.style_vectors[2999])
    print(list(trajectory_data_set.actions))
