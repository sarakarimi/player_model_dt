import gzip
import pickle
from random import random, sample
from typing import Callable

import torch
import torch.nn as nn
from einops import rearrange
from torch.utils.data import Dataset
import numpy as np
from triton.language import dtype


class TrajectoryReader:
    """
    The trajectory reader is responsible for reading trajectories from a file.
    """

    def __init__(self, path):
        self.path = path.strip()

    def read(self):
        # if path ends in .pkl, read as pickle
        if self.path.endswith(".pkl"):
            with open(self.path, "rb") as f:
                data = pickle.load(f)
        # if path ends in .xz, read as lzma
        elif self.path.endswith(".npz"):
            data = np.load(self.path)
        elif self.path.endswith(".gz"):
            with gzip.open(self.path, "rb") as f:
                data = pickle.load(f)
        else:
            raise ValueError(
                f"Path {self.path} is not a valid trajectory file"
            )

        return data



class TrajectoryDataset(Dataset):
    def __init__(
            self,
            trajectory_paths,
            normalize_state=False,
            device="cpu",
    ):
        self.trajectory_paths = trajectory_paths
        self.device = device
        self.normalize_state = normalize_state
        self.load_trajectories()

    def load_trajectories(self) -> None:
        merge_observations, merge_actions, merge_rewards, merge_dones, merge_truncated, merge_infos, merge_modes, merge_timesteps = [], [], [], [], [], [], [], []

        # used only for DEC-VAE experiments
        obs, acts, tasks = [], [], []

        # Iterating over many dataset with different environment modes or play styles
        for i, path in enumerate(self.trajectory_paths):
            traj_reader = TrajectoryReader(path)
            data = traj_reader.read()
            data = {k: [dic[k] for dic in data] for k in data[0]}
            # print(data.keys())

            self.states = torch.tensor(data["observations"])
            self.actions = torch.tensor(data["actions"])
            self.rewards = torch.tensor(data["rewards"])
            self.dones = torch.tensor(data["terminals"])
            # self.truncated = data["terminals"]


            # self.observations = np.array(observations)
            # self.actions = np.array(actions)
            # self.rewards = np.array(rewards)
            # self.dones = np.array(dones)

            self.returns = [r.sum() for r in self.rewards]
            self.returns = ['%.2f' % elem for elem in self.returns]
            self.timesteps = [torch.arange(len(i)) for i in self.states]
            obs.extend(self.states[:, -20:, :])
            acts.extend(self.actions[:, -20:, :])
            tasks.extend(np.ones(len(self.actions[:, -20:, :]), dtype=np.int64) * i)



        self.obs = obs
        self.acts = acts
        self.tasks = tasks

        self.traj_lens = np.array([len(i) for i in self.obs])
        # remove trajs with length 0
        traj_len_mask = self.traj_lens > 0
        self.acts = [i for i, m in zip(self.acts, traj_len_mask) if m]

        # print(self.obs[0][0])
        # state normalization
        all_states = np.concatenate(obs, axis=0)
        self.state_mean, self.state_std = (
            np.mean(all_states, axis=0),
            np.std(all_states, axis=0) + 1e-6,
        )
        self.obs = [(i - self.state_mean) / self.state_std for i, m in zip(self.obs, traj_len_mask) if m]
        # self.modes = [i for i, m in zip(self.modes, traj_len_mask) if m]
        self.tasks = [i for i, m in zip(self.tasks, traj_len_mask) if m]

        self.traj_lens = self.traj_lens[traj_len_mask]
        # print(self.obs[0][0])
        # exit(0)


class MujocoDataset(TrajectoryDataset):
    def __init__(self, trajectory_paths):
        super().__init__(trajectory_paths)

        self.sequences = self.obs
        self.labels = self.tasks
        self.seq_lens = [len(seq) for seq in self.sequences]
        # self.timesteps = [torch.arange(i) for i in self.seq_lens]

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        sequence = self.sequences[idx]
        # sequence = (sequence - self.state_mean) / self.state_std
        labels = self.labels[idx]
        length = len(sequence)
        # timesteps = self.timesteps[idx]

        return sequence, labels, length#, timesteps

def collate_fn(batch):
    # Separate sequences and their lengths
    sequences, labels, lengths = zip(*batch)
    sequences_padded = nn.utils.rnn.pad_sequence(sequences, batch_first=True, padding_value=0.0)
    sequences_padded = sequences_padded.to(torch.float32)
    return sequences_padded, torch.tensor(labels), torch.tensor(lengths)



if __name__ == '__main__':
    paths = ["/home/sara/repositories/player_model_dt/trajectory_embedding/datasets/mujoco/cheetah_vel/cheetah_vel-5-expert.pkl",
             "/home/sara/repositories/player_model_dt/trajectory_embedding/datasets/mujoco/cheetah_vel/cheetah_vel-10-expert.pkl",
             "/home/sara/repositories/player_model_dt/trajectory_embedding/datasets/mujoco/cheetah_vel/cheetah_vel-15-expert.pkl",
             "/home/sara/repositories/player_model_dt/trajectory_embedding/datasets/mujoco/cheetah_vel/cheetah_vel-30-expert.pkl"]
    dataset = TrajectoryDataset(paths)
    print(dataset.obs[200])