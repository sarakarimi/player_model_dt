import torch
import torch.nn as nn
from dataset_utils.minigrid_trajectory_dataset import TrajectoryDataset


class MiniGridDataset(TrajectoryDataset):
    def __init__(self, trajectory_paths, index_channel_only=True, sampling=True, state_normalization_factor=9,
                 action_normalization_factor=6):
        super().__init__(trajectory_paths=trajectory_paths, index_channel_only=index_channel_only, sampling=sampling,
                         state_normalization_factor=state_normalization_factor,
                         action_normalization_factor=action_normalization_factor)

        self.state_sequences = self.states
        self.actions_sequences = self.actions
        self.labels = self.tasks
        self.seq_lens = [len(seq) for seq in self.state_sequences]

    def __len__(self):
        return len(self.state_sequences)

    def __getitem__(self, idx):
        state_seq = self.state_sequences[idx]
        # sequence = (sequence - self.state_mean) / self.state_std
        action_seq = self.actions_sequences[idx]
        labels = self.labels[idx]
        length = len(state_seq)

        return state_seq, action_seq, labels, length


def collate_fn(batch):
    # Separate sequences and their lengths
    state_seq, action_seq, labels, lengths = zip(*batch)
    state_seq_padded = nn.utils.rnn.pad_sequence(state_seq, batch_first=True, padding_value=0.0)
    state_seq_padded = state_seq_padded.to(torch.float32)
    action_seq_padded = nn.utils.rnn.pad_sequence(action_seq, batch_first=True, padding_value=0.0)
    action_seq_padded = action_seq_padded.to(torch.float32)
    return state_seq_padded, action_seq_padded, torch.tensor(labels), torch.tensor(lengths)


if __name__ == '__main__':
    from trajectory_embedding.style_dec_vae.configs.config_minigrid import paths
    import pandas
    import seaborn as sns
    import matplotlib.pyplot as plt
    from sklearn.manifold import TSNE

    dataset = MiniGridDataset(paths, sampling=True, index_channel_only=True)
    tasks = dataset.tasks
    print(tasks)
    exit(0)

    obs = torch.cat(obs, 0).cpu().numpy()

    gtruth = dataset.tasks
    embedding_data = []
    for ob, gtruth_task in zip(obs, gtruth):
        embedding_data.append({'embeddings': obs, 'task_ground_truth': gtruth_task})
    df = pandas.DataFrame(embedding_data)
    df = df.fillna(0)

    df_fig = df.copy()
    tsne = TSNE(init='pca', perplexity=20)
    tsne_results = tsne.fit_transform(obs)
    print(tsne_results.shape)
    df_fig['tsne-2d-one'] = tsne_results[:, 0]
    df_fig['tsne-2d-two'] = tsne_results[:, 1]

    sns.scatterplot(
        x="tsne-2d-one", y="tsne-2d-two",
        hue="task_ground_truth",
        data=df_fig,
    )
    # plt.savefig('tsne2.png')
    plt.show()
