from typing import Callable
from dataset_utils.minigrid_trajectory_dataset import TrajectoryDataset
from dataset_utils.utils import TrajectoryReader
import numpy as np
import torch
from einops import rearrange
import random

class HardPromptDataset(TrajectoryDataset):
    def __init__(
            self,
            trajectory_paths,
            max_len=1,
            max_prompt_len=2,
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
        self.max_prompt_len = max_prompt_len
        self.load_prompts_trajectories()


    def load_prompts_trajectories(self) -> None:
        (merge_observations, merge_actions, merge_rewards, merge_returns, merge_dones,
         merge_truncated, merge_infos, merge_modes, merge_timesteps,
         merge_tasks) = [], [], [], [], [], [], [], [], [], []


        # Iterating over many dataset with different environment modes or play styles
        for i, path in enumerate(self.trajectory_paths):

            traj_reader = TrajectoryReader(path)
            data = traj_reader.read()
            observations = data["data"].get("observations")
            actions = data["data"].get("actions")
            rewards = data["data"].get("rewards")
            dones = data["data"].get("dones")
            truncated = data["data"].get("truncated")


            observations = np.array(observations)
            actions = np.array(actions)
            rewards = np.array(rewards)
            dones = np.array(dones)

            # check whether observations are flat or an image
            if observations.shape[-1] == 3:
                # use state space that includes  object IDX in each grid position
                if self.index_channel_only:
                    observations = observations[:, :, :, :, 0]  # we use this in the VAE model
                else:
                    observations = observations[:, :, :, :, :]  # We use this in the DT model
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
            t_done_or_truncated = torch.logical_or(t_dones, t_truncated)
            done_indices = torch.where(t_done_or_truncated)[0]

            actions = torch.tensor_split(t_actions, done_indices + 1)
            rewards = torch.tensor_split(t_rewards, done_indices + 1)
            dones = torch.tensor_split(t_dones, done_indices + 1)
            truncated = torch.tensor_split(t_truncated, done_indices + 1)
            states = torch.tensor_split(t_observations, done_indices + 1)
            returns = [r.sum() for r in rewards]
            returns = ['%.2f' % elem for elem in returns]
            timesteps = [torch.arange(len(i)) for i in states]


            # Sampling trajectories based on their lengths and returns
            top_seq_lengths = self.get_top_trajectory_lengths(states, returns, top_k=2)
            seq_lens = [seq_len[0] for seq_len in top_seq_lengths]

            indexes = []
            # TODO remove hard-coded values
            num_samples = 3
            for seq_len in seq_lens:
                index_list = [index for index, (state, ret) in enumerate(zip(states, returns)) if
                              len(state) == seq_len]
                if len(index_list) < num_samples:
                    num_samples = len(index_list)
                index_list_sample = random.sample(index_list, num_samples)
                indexes.extend(index_list_sample)


            tasks = np.ones(len([actions[i] for i in indexes]), dtype=np.int64) * i
            states = [states[i] for i in indexes]
            actions = [actions[i] for i in indexes]
            rewards = [rewards[i] for i in indexes]
            dones = [dones[i] for i in indexes]
            truncated = [truncated[i] for i in indexes]
            returns = [returns[i] for i in indexes]
            timesteps = [timesteps[i] for i in indexes]

            # merge datasets
            merge_actions.extend(actions)
            merge_rewards.extend(rewards)
            merge_dones.extend(dones)
            merge_truncated.extend(truncated)
            merge_observations.extend(states)
            merge_returns.extend(returns)
            merge_timesteps.extend(timesteps)
            merge_tasks.extend(tasks)

        self.action_prompts = merge_actions
        self.reward_prompts = merge_rewards
        self.done_prompts = merge_dones
        self.truncated_prompts = merge_truncated
        self.state_prompts = merge_observations
        self.return_prompts = merge_returns
        self.timestep_prompts = merge_timesteps
        self.task_prompts = merge_tasks

        # ==================================
        # remove trajectories with length 0
        self.traj_lens = np.array([len(i) for i in self.states])
        traj_len_mask = self.traj_lens > 0
        self.action_prompts = [i / self.action_normalization_factor for i, m in zip(self.action_prompts, traj_len_mask) if m]
        self.reward_prompts = [i for i, m in zip(self.reward_prompts, traj_len_mask) if m]
        self.done_prompts = [i for i, m in zip(self.done_prompts, traj_len_mask) if m]
        self.truncated_prompts = [i for i, m in zip(self.truncated_prompts, traj_len_mask) if m]
        self.state_prompts = [i / self.state_normalization_factor for i, m in zip(self.state_prompts, traj_len_mask) if m]
        self.return_prompts = [i for i, m in zip(self.return_prompts, traj_len_mask) if m]
        self.timestep_prompts = [i for i, m in zip(self.timestep_prompts, traj_len_mask) if m]
        self.task_prompts = [i for i, m in zip(self.task_prompts, traj_len_mask) if m]

        top_seq_lengths = self.get_top_trajectory_lengths(self.states, self.returns, top_k=6)
        print(top_seq_lengths)



    def return_tensors_prompt(self, s, a, rtg, timesteps):
        if isinstance(s, torch.Tensor):
            s = s.to(dtype=torch.float32, device=self.device)
        else:
            s = torch.from_numpy(s).to(dtype=torch.float32, device=self.device)

        if isinstance(a, torch.Tensor):
            a = a.to(dtype=torch.long, device=self.device)
        else:
            a = torch.from_numpy(a).to(dtype=torch.long, device=self.device)

        if isinstance(rtg, torch.Tensor):
            rtg = rtg.to(dtype=torch.float32, device=self.device)
        else:
            rtg = torch.from_numpy(rtg).to(
                dtype=torch.float32, device=self.device
            )

        timesteps = torch.from_numpy(timesteps).to(
            dtype=torch.long, device=self.device
        )

        # squeeze out the batch dimension
        s = s.squeeze(0)
        a = a.squeeze(0)
        rtg = rtg.squeeze(0)
        timesteps = timesteps.squeeze(0)

        # TODO fix the order of d, rtg here.
        return s, a, rtg, timesteps


    def get_prompt_traj(self, prompt_traj_index, max_prompt_len=2):
        traj_prompt_rewards = self.reward_prompts[prompt_traj_index]
        traj_prompt_states = self.state_prompts[prompt_traj_index]
        traj_prompt_actions = self.action_prompts[prompt_traj_index]
        traj_prompt_dones = self.done_prompts[prompt_traj_index]

        # TODO: configure this so non-sparse tasks are dealt with correctly!
        # This line is very slow if we use the "correct method"
        traj_prompt_rtg = np.ones(traj_prompt_rewards.shape) * traj_prompt_rewards[-1].item()


        # start index
        si = max(0, traj_prompt_rewards.shape[0] - max_prompt_len)  # select the last traj with length max_len
        si = np.random.choice(si)  # randomly sample starting time step from prompt 0 - si, instead of always using si...

        # get sequences from dataset
        s_prompt = traj_prompt_states[si: si + max_prompt_len].reshape(1, -1, *self.state_dim)
        a_prompt = traj_prompt_actions[si: si + max_prompt_len].reshape(1, -1, *self.act_dim)
        r_prompt = traj_prompt_rewards[si: si + max_prompt_len].reshape(1, -1, 1)
        rtg_prompt = traj_prompt_rtg[si: si + max_prompt_len].reshape(1, -1, 1)
        d_prompt = traj_prompt_dones[si: si + max_prompt_len].reshape(1, -1)
        ti_prompt = np.arange(si, si + s_prompt.shape[1]).reshape(1, -1)
        # sometime the trajectory is shorter than max_len (due to random start index or end of episode)
        tlen = s_prompt.shape[1]

        # sanity check
        assert tlen <= max_prompt_len, f"tlen: {tlen} max_len: {max_prompt_len}"

        padding_required = max_prompt_len - tlen
        s_prompt = self.add_padding(s_prompt, 0, padding_required)
        a_prompt = self.add_padding(a_prompt, -10, padding_required)
        r_prompt = self.add_padding(r_prompt, 0, padding_required)
        rtg_prompt = self.add_padding(rtg_prompt, rtg_prompt[0, -1], padding_required)
        d_prompt = self.add_padding(d_prompt, 2, padding_required)
        ti_prompt = self.add_padding(ti_prompt, 0, padding_required)
        m_prompt = self.add_padding(np.ones((1, tlen)), 0, padding_required)


        # padding and state + reward normalization
        s_prompt = (s_prompt - self.state_mean) / self.state_std
        rtg_prompt = rtg_prompt / self.rtg_scale


        s_prompt, a_prompt, r_prompt,  d_prompt, rtg_prompt, ti_prompt, m_prompt = self.return_tensors(s_prompt, a_prompt, r_prompt, rtg_prompt, d_prompt, ti_prompt, m_prompt)

        # rtg_prompt = rtg_prompt.unsqueeze(-1)
        a_prompt = a_prompt.unsqueeze(-1)
        ti_prompt = ti_prompt.unsqueeze(-1)

        return s_prompt, a_prompt, rtg_prompt, ti_prompt

    def __getitem__(self, idx):
        traj_index = self.indices[idx]

        # find a random prompt trajectory that matches the env ID of the selected trajectory
        env_id = self.tasks[traj_index]
        matching_indexes = [i for i, val in enumerate(self.task_prompts) if val == env_id]
        prompt_idx = random.choice(matching_indexes)

        s, a, r, d, rtg, ti, m = self.get_traj(traj_index, max_len=self.max_len,
                                                             prob_go_from_end=self.prob_go_from_end)
        prompt = self.get_prompt_traj(prompt_idx, max_prompt_len=self.max_prompt_len)


        if self.preprocess_observations is not None:
            s = self.preprocess_observations(s)
        return s, a, r, d, rtg, ti, m, prompt


    def sample_random_prompts(self, task_label, random_state=None, num_samples=1):
        rng = np.random.default_rng(random_state)
        indices = [i for i, val in enumerate(self.task_prompts) if val == task_label]
        if len(indices) == 0:
            raise ValueError("No samples in the specified cluster.")

        idxs = rng.choice(indices, size=num_samples)
        prompt_states, prompt_actions, prompt_rtg, prompt_timesteps= (
            [],
            [],
            [],
            [],
        )
        for i in range(num_samples):
            prompt_traj_index = int(idxs[i])
            s_prompt, a_prompt, rtg_prompt, ti_prompt = self.get_prompt_traj(
                prompt_traj_index, max_prompt_len=self.max_prompt_len
            )

            prompt_states.append(s_prompt)
            prompt_actions.append(a_prompt)
            prompt_rtg.append(rtg_prompt)
            prompt_timesteps.append(ti_prompt)
        return self.return_tensors_prompt(np.asarray(prompt_states), np.asarray(prompt_actions), np.asarray(prompt_rtg), np.asarray(prompt_timesteps))


if __name__ == '__main__':

    paths = [
        "/home/sara/repositories/player_model_dt/datasets/minigrid/three_style_env/PPO_trajectories_PPO_trajectories_three_style_env_backstab.gz",
        "/home/sara/repositories/player_model_dt/datasets/minigrid/three_style_env/PPO_trajectories_PPO_trajectories_three_style_env_bypass.gz",
        "/home/sara/repositories/player_model_dt/datasets/minigrid/three_style_env/PPO_trajectories_PPO_trajectories_three_style_env_weapon.gz",

    ]
    trajectory_data_set = HardPromptDataset(trajectory_paths=paths, sampling=True)
    # print(trajectory_data_set.style_vectors[0])
    # print(trajectory_data_set.style_vectors[1999])
    # print(trajectory_data_set.style_vectors[2999])
    print(trajectory_data_set.states[0])
