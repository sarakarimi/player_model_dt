import torch
import numpy as np
import torch.nn as nn
from einops import rearrange
from torch.utils.data import Dataset
from triton.language import dtype

from new_implementation.decision_transformer.dataset import TrajectoryDataset


# # Define a dataset class that returns sequences of varying lengths
# class VariedLengthDataset(Dataset):
#     def __init__(self, sequences, labels):
#         self.sequences = sequences
#         self.labels = labels
#         self.seq_lens = [len(seq) for seq in sequences]
#
#     def __len__(self):
#         return len(self.sequences)
#
#     def __getitem__(self, idx):
#         sequence = self.sequences[idx]
#         labels = self.labels[idx]
#         length = len(sequence)
#         return torch.tensor(sequence, dtype=torch.float32), torch.tensor(labels, dtype=torch.int32), length
#
#
# def generate_varied_length_data(num_samples, max_len, feature_size):
#     sequences = [np.random.rand(np.random.randint(1, max_len), feature_size) for _ in range(num_samples)]
#     labels = [np.random.randint(0, 2) for _ in range(num_samples)]
#     return sequences, labels
#
#
# def collate_fn(batch):
#     # Separate sequences and their lengths
#     sequences, labels, lengths = zip(*batch)
#     sequences_padded = nn.utils.rnn.pad_sequence(sequences, batch_first=True, padding_value=0)
#     return sequences_padded, torch.tensor(labels), torch.tensor(lengths)


# Define a dataset class that returns sequences of varying lengths
class VariedLengthDataset(Dataset):
    def __init__(self, sequences, labels):
        self.sequences = sequences
        self.labels = labels
        self.seq_lens = [len(seq) for seq in sequences]

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        sequence = self.sequences[idx]
        labels = self.labels[idx]
        length = len(sequence)
        return torch.tensor(sequence, dtype=torch.float32), torch.tensor(labels, dtype=torch.int32), length


def generate_varied_length_data(num_samples, max_len, feature_size, cluster_params):
    sequences = []
    labels = []
    for _ in range(num_samples):
        # Randomly choose a cluster (0 or 1)
        cluster_label = np.random.randint(0, 2)
        if cluster_label == 0:
            # Generate sequences from distribution of cluster 0
            mean, std = cluster_params[0]
        else:
            # Generate sequences from distribution of cluster 1
            mean, std = cluster_params[1]

        # Generate a random sequence with a length between 1 and max_len
        seq_len = np.random.randint(1, max_len)
        sequence = np.random.normal(mean, std, (seq_len, feature_size))
        sequences.append(sequence)
        labels.append(cluster_label)

    return sequences, labels


# def collate_fn(batch):
#     # Separate sequences and their lengths
#     sequences, labels, lengths = zip(*batch)
#     sequences_padded = nn.utils.rnn.pad_sequence(sequences, batch_first=True, padding_value=0)
#     return sequences_padded, torch.tensor(labels), torch.tensor(lengths)


class MiniGridDataset(TrajectoryDataset):
    def __init__(self, trajectory_paths):
        super().__init__(trajectory_paths)

        self.sequences = self.obs
        self.labels = self.tasks
        self.seq_lens = [len(seq) for seq in self.sequences]

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        sequence = self.sequences[idx]
        labels = self.labels[idx]
        length = len(sequence)
        return sequence, labels, length

def collate_fn(batch):
    # Separate sequences and their lengths
    sequences, labels, lengths = zip(*batch)
    sequences_padded = nn.utils.rnn.pad_sequence(sequences, batch_first=True, padding_value=0)
    return sequences_padded, torch.tensor(labels), torch.tensor(lengths)
