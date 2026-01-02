import os

from sklearn.cluster import KMeans
from tqdm import tqdm
import torch
import torch.nn as nn
from trajectory_embedding.style_dec_vae.configs.config_minigrid import *
from dataset_utils.minigrid_vae_dataset import MiniGridDataset, collate_fn
# from trajectory_embedding.style_dec_vae.utils.dataset_mujoco import MujocoDataset, collate_fn
from trajectory_embedding.style_dec_vae.utils.loss import MiniGridLoss
from trajectory_embedding.style_dec_vae.utils.utils import convert_to_one_hot
from torch.autograd import Variable
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas
from sklearn.manifold import TSNE

##########################
# Parts of the code is taken from https://github.com/CUN-bjy/lstm-vae-torch
##########################

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv0 = nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=3, padding=1)
        self.conv1 = nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=3, padding=1)

    def forward(self, x):
        inputs = x
        x = nn.functional.relu(x)
        x = self.conv0(x)
        x = nn.functional.relu(x)
        x = self.conv1(x)
        return x + inputs


class ConvSequence(nn.Module):
    def __init__(self, input_shape, out_channels):
        super().__init__()
        self._input_shape = input_shape
        self._out_channels = out_channels
        self.conv = nn.Conv2d(in_channels=self._input_shape[0], out_channels=self._out_channels, kernel_size=3,
                              padding=1)
        self.res_block0 = ResidualBlock(self._out_channels)
        self.res_block1 = ResidualBlock(self._out_channels)

    def forward(self, x):
        x = self.conv(x)
        x = nn.functional.max_pool2d(x, kernel_size=3, stride=2, padding=1)
        x = self.res_block0(x)
        x = self.res_block1(x)
        assert x.shape[1:] == self.get_output_shape()
        return x

    def get_output_shape(self):
        _c, h, w = self._input_shape
        return (self._out_channels, (h + 1) // 2, (w + 1) // 2)


class ConvEncoder(nn.Module):
    def __init__(self, inp_dim, out_dim=256):
        super(ConvEncoder, self).__init__()
        self.state_dim = inp_dim
        self.out_dim = out_dim
        c, h, w = self.state_dim
        shape = (c, h, w)
        conv_seqs = []
        for out_channels in [16, 32, 32]:
            conv_seq = ConvSequence(shape, out_channels)
            shape = conv_seq.get_output_shape()
            conv_seqs.append(conv_seq)
        conv_seqs += [
            nn.Flatten(),
            nn.ReLU(),
            nn.Linear(in_features=shape[0] * shape[1] * shape[2], out_features=self.out_dim),
            nn.ReLU(),
        ]
        self.network = nn.Sequential(*conv_seqs)

    def forward(self, inp):
        return self.network(inp)

# ----------------------------------------------------
# State CNN encoder
# ----------------------------------------------------

class PerStepCNN(nn.Module):
    """Encodes a 1x3x3 image into a small feature vector via a tiny CNN."""
    def __init__(self, out_dim: int = 8):
        super().__init__()
        # For a 3x3 input, convs are almost overkill, but you asked for CNN.
        self.conv = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=2, stride=1, padding=0),  # -> 8x2x2
            nn.ReLU(inplace=True),
            nn.Conv2d(8, 16, kernel_size=2, stride=1, padding=0), # -> 16x1x1
            nn.ReLU(inplace=True),
        )
        self.proj = nn.Linear(16, out_dim)

    def forward(self, imgs):  # imgs: [B, T, 1, 3, 3]
        B, T, C, H, W = imgs.shape
        x = imgs.reshape(B*T, C, H, W)
        x = self.conv(x)                   # [B*T, 16, 1, 1]
        x = x.view(B*T, 16)                # [B*T, 16]
        x = self.proj(x)                   # [B*T, out_dim]
        x = x.view(B, T, -1)               # [B, T, out_dim]
        return x

class PerStepFeaturizer(nn.Module):
    """CNN(img) + concat scalar -> per-step feature for LSTM."""
    def __init__(self, img_feat_dim: int = 8):
        super().__init__()
        self.cnn = PerStepCNN(out_dim=img_feat_dim)

    def forward(self, imgs, scalars):
        """
        imgs:    [B, T, 1, 3, 3]
        scalars: [B, T, 1] (or [B, T])
        returns: [B, T, img_feat_dim + 1]
        """
        img_feat = self.cnn(imgs)                # [B, T, img_feat_dim]
        s = scalars if scalars.dim() == 3 else scalars.unsqueeze(-1)  # [B, T, 1]
        feat = torch.cat([img_feat, s], dim=-1)  # [B, T, img_feat_dim+1]
        return feat

# ----------------------------------------------------
# Encoder and decoder models with state featurizer
# ----------------------------------------------------

# TODO: implement new Encoder  and Decoder that have the featurizer



# ----------------------------------------------------

class Stochastic(nn.Module):
    def reparameterize(self, mu, log_var):
        epsilon = Variable(torch.randn(mu.size()), requires_grad=False)
        if mu.is_cuda:
            epsilon = epsilon.cuda()

        z = mu + torch.exp(log_var / 2) * epsilon
        return z


class GaussianSampling(Stochastic):
    def __init__(self, in_features, out_features):
        super(GaussianSampling, self).__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.mu = nn.Linear(self.in_features, self.out_features, 'mu_sampl')
        self.log_var = nn.Linear(self.in_features, self.out_features, 'log_var_sampl')

    def forward(self, x):
        mu = self.mu(x)
        log_var = self.log_var(x)

        z = self.reparameterize(mu, log_var)

        return z, mu, log_var


class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size, latent_size, device, num_layers=1):
        super(Encoder, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.latent_size = latent_size
        self.device = device
        self.lstm = nn.LSTM(
            input_size,
            hidden_size,
            num_layers,
            batch_first=True,
            bidirectional=False,
        )

        self.sampling = GaussianSampling(self.hidden_size, self.latent_size)

    def reparametize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        noise = torch.randn_like(std).to(self.device)

        z = mu + noise * std
        return z, mu, logvar

    def forward(self, x, seq_lens, hidden_enc=None):
        # Pack the padded sequence
        packed_input = nn.utils.rnn.pack_padded_sequence(x, seq_lens, batch_first=True, enforce_sorted=False)
        outputs, hidden_enc = self.lstm(packed_input) #, hidden_enc)
        enc_h = hidden_enc[0].view(-1, self.hidden_size).to(self.device) # [-1]  # .

        # extract latent variable z(hidden space to latent space)
        z, mean, logvar = self.sampling(enc_h)

        return z, mean, logvar, hidden_enc


class Decoder(nn.Module):
    def __init__(
            self, input_size, latent_size, hidden_size, device, num_layers=1):
        super(Decoder, self).__init__()
        self.hidden_size = hidden_size
        self.latent_size = latent_size
        self.num_layers = num_layers
        self.device = device
        self.lstm = nn.LSTM(
            latent_size,
            hidden_size,
            num_layers,
            batch_first=True,
            bidirectional=False,
        )
        self.fc1 = nn.Linear(self.latent_size, self.hidden_size)
        self.fc2 = nn.Linear(hidden_size, input_size) # used to be input_size
        self.final = nn.Sigmoid()

    def forward(self, z, lengths, total_padding_length=None):
        max_seq_len = max(lengths)
        z_repeat = z.repeat(1, max_seq_len, 1)
        z_repeat = z_repeat.view(-1, max_seq_len, self.latent_size).to(self.device)

        # TODO replace with the z as hidden
        # initialize hidden state as inputs
        h_ = self.fc1(z).unsqueeze(0)
        hidden = (h_.contiguous(), h_.contiguous())
        # h_0 = torch.zeros(1, z.size(0), self.hidden_size).to(z.device)
        # c_0 = torch.zeros(1, z.size(0), self.hidden_size).to(z.device)
        # hidden = (h_0, c_0)

        x = torch.nn.utils.rnn.pack_padded_sequence(z_repeat, lengths, batch_first=True, enforce_sorted=False)

        output, hidden = self.lstm(x, hidden)
        output, _ = torch.nn.utils.rnn.pad_packed_sequence(output, batch_first=True, total_length=max_seq_len)

        prediction = self.fc2(output)
        # prediction = self.final(prediction)
        return prediction, hidden


class LSTMVAE(nn.Module):
    """LSTM-based Variational Auto Encoder"""

    def __init__(
            self, input_size, hidden_size, latent_size
    ):
        """
        input_size: int, batch_size x sequence_length x input_dim
        hidden_size: int, output size of LSTM VAE
        latent_size: int, latent z-layer size
        num_lstm_layer: int, number of layers in LSTM
        """
        super(LSTMVAE, self).__init__()

        # dimensions
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.latent_size = latent_size
        self.num_layers = 1
        self.device = torch.device('cuda' if (torch.cuda.is_available()) else 'cpu')

        # lstm vae
        self.lstm_enc = Encoder(
            input_size=input_size,
            hidden_size=hidden_size,
            latent_size=latent_size,
            device=self.device,
            num_layers=self.num_layers
        )
        self.lstm_dec = Decoder(
            input_size=input_size,
            latent_size=latent_size,
            hidden_size=hidden_size,
            device=self.device,
            num_layers=self.num_layers,
        )

    def forward(self, x, seq_lens, hidden_enc):
        # encode input space to hidden space
        z, mean, logvar, hidden_enc = self.lstm_enc(x, seq_lens) #, hidden_enc)

        # decode latent space to input space
        reconstruct_output, hidden = self.lstm_dec(z, seq_lens)
        x_hat = reconstruct_output

        # calculate vae loss
        losses = self.loss_function(x_hat, x, mean, logvar, seq_lens)
        m_loss, recon_loss, kld_loss = (
            losses["loss"],
            losses["Reconstruction_Loss"],
            losses["KLD"],
        )

        return m_loss, x_hat, (recon_loss, kld_loss, z)

    def loss_function(self, *args, **kwargs) -> dict:
        recons = args[0]
        og_input = args[1]
        mu = args[2]
        log_var = args[3]
        lengths = args[4]
        loss_fn = MiniGridLoss()

        kld_weight = 0.00025  # Account for the minibatch samples from the dataset

        # MSE loss
        recons_loss = 0
        for i, length in enumerate(lengths):
            recons_loss += nn.functional.mse_loss(recons[i, :length], og_input[i, :length], reduction='sum')
        recons_loss /= len(lengths)

        # binary cross entropy loss
        # recons_loss = 0
        # for i, length in enumerate(lengths):
        #     recons_loss += nn.functional.binary_cross_entropy(recons[i, :length], og_input[i, :length], reduction='sum')
        # recons_loss /= len(lengths)

        # per on-hot category costume loss
        # recons_loss = 0
        # for i, length in enumerate(lengths):
        #     recons_loss += loss_fn(recons[i, :length], og_input[i, :length], reduction='mean')
        # recons_loss /= len(lengths)
        # recons_loss = nn.functional.mse_loss(recons, og_input)

        kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim=-1), dim=0)

        loss = recons_loss + kld_weight * kld_loss
        return {
            "loss": loss,
            "Reconstruction_Loss": recons_loss.detach(),
            "KLD": -kld_loss.detach(),
        }


def train(model, train_loader, test_loader, epochs, save_path=None):
    # optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    # optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    steplr = torch.optim.lr_scheduler.StepLR(optimizer, 20, 0.9)

    ## training
    count = 0
    for epoch in range(epochs):
        model.train()
        # optimizer.zero_grad()
        train_iterator = tqdm(
            train_loader, total=len(train_loader), desc="training"
        )

        for state_data, action_data, labels, lengths in train_iterator:
            state_data = state_data.to(torch.float32).to(model.device)

            action_data = action_data.to(torch.float32).to(model.device).unsqueeze(-1)
            batch_data = torch.concat([state_data, action_data], -1)

            lengths = lengths.to(torch.int64)  # .to(model.device)

            optimizer.zero_grad()
            h_0 = torch.zeros(1, train_loader.batch_size, model.hidden_size).to(model.device)
            c_0 = torch.zeros(1, train_loader.batch_size, model.hidden_size).to(model.device)
            hidden_enc = (h_0, c_0)
            mloss, recon_x, info = model(batch_data, lengths, hidden_enc)

            # Backward and optimize
            mloss.mean().backward()
            optimizer.step()

            train_iterator.set_postfix({"train_loss": float(mloss.mean())})

    validate(model, test_loader)

    # Save the trained model
    assert save_path is not None, "Please provide a valid path to save the model."
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {os.path.abspath(save_path)}")

    return model


def validate(model, test_loader, load_model=False, model_path=None):
    if load_model:
        model.load_state_dict(torch.load(model_path, map_location='cpu'))
        model.eval()
        if torch.cuda.is_available():
            model = model.cuda()

    model.eval()
    eval_loss = 0
    total_rec_loss = 0
    test_iterator = tqdm(
        test_loader, total=len(test_loader), desc="testing"
    )

    with torch.no_grad():
        Z, labels_list = [], []

        for state_data, action_data, labels, lengths in test_iterator:
            state_data = state_data.to(torch.float32).to(model.device)
            action_data = action_data.to(torch.float32).to(model.device).unsqueeze(-1)
            batch_data = torch.concat([state_data, action_data], -1)
            lengths = lengths.to(torch.int64)

            h_0 = torch.zeros(1, test_loader.batch_size, model.hidden_size).to(model.device)
            c_0 = torch.zeros(1, test_loader.batch_size, model.hidden_size).to(model.device)
            hidden_enc = (h_0, c_0)
            mloss, recon_x, (_, _, z) = model(batch_data, lengths, hidden_enc)
            Z.append(z.cpu().detach().numpy())
            labels_list.append(labels)
            rec_loss = 0
            rec_xs = []
            for i, length in enumerate(lengths):
                rec_x = recon_x[i, :length]
                rec_xs.append(rec_x)
                rec_loss += nn.MSELoss(reduction='sum')(rec_x, batch_data[i, :length])
            rec_loss /= len(lengths)
            total_rec_loss += rec_loss.item()
            eval_loss += mloss.mean().item()

            test_iterator.set_postfix({"eval_loss": mloss.mean().item()})
            test_iterator.set_postfix({"eval_rec_loss": rec_loss.item()})
        torch.set_printoptions(linewidth=1000)

    eval_loss = eval_loss / len(test_loader)
    total_rec_loss = total_rec_loss / len(test_loader)

    print("Evaluation Score : [{}]".format(eval_loss))
    print("Evaluation reconstruction bit-wise loss : [{}]".format(total_rec_loss))

    # Plot trained embeddings
    true_labels = np.concatenate(labels_list, 0)
    Z = np.concatenate(Z, 0)
    predicted_labels, cluster_centroids = cluster_latents(Z, n_clusters)
    # plot_embeddings(gtruth=predicted_labels, Z=Z, label_name='task_predicted')
    # plot_embeddings(gtruth=true_labels, Z=Z, label_name='task_ground_truth')

    return predicted_labels, Z, cluster_centroids


def cluster_latents(Z, n_clusters):
    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(Z)

    predicted_labels, cluster_centroids = kmeans.labels_, kmeans.cluster_centers_
    def order_kmeans_labels(labels, centroids):
        # Order by centroid norm (or any other property)
        order = np.argsort(np.linalg.norm(centroids, axis=1))
        remap = np.zeros_like(order)
        remap[order] = np.arange(len(order))

        new_labels = remap[labels]

        new_centroids = centroids[order]
        return new_labels, new_centroids

    predicted_labels, cluster_centroids = order_kmeans_labels(predicted_labels, cluster_centroids)
    return predicted_labels, cluster_centroids


def plot_embeddings(gtruth, Z, label_name='task_ground_truth'):
    embedding_data = []
    for embedding, gtruth_task in zip(Z, gtruth):
        embedding_data.append({'embeddings': embedding, label_name: gtruth_task})
    df = pandas.DataFrame(embedding_data)
    df = df.fillna(0)
    # print(df)

    df_fig = df.copy()
    tsne = TSNE(init='pca', perplexity=20)
    tsne_results = tsne.fit_transform(Z)
    df_fig['tsne-2d-one'] = tsne_results[:,0]
    df_fig['tsne-2d-two'] = tsne_results[:,1]
    sns.scatterplot(
        x="tsne-2d-one", y="tsne-2d-two",
        hue=label_name,
        data=df_fig,
    )
    plt.show()



if __name__ == "__main__":

    # trajectory_data_set = MujocoDataset(trajectory_paths=paths)

    trajectory_data_set = MiniGridDataset(trajectory_paths=paths, **dataset_params)
    train_loader = torch.utils.data.DataLoader(
        dataset=trajectory_data_set, batch_size=vae_batch_size, shuffle=True, collate_fn=collate_fn
    )
    test_loader = train_loader

    # define LSTM-based VAE model
    model = LSTMVAE(**vae_model_params)

    # training
    if torch.cuda.is_available():
        print('Cuda is available')
        dec_cluster = model.cuda()
    else:
        print('No GPU')
    train(model, train_loader, test_loader, num_epochs, save_path=vae_model_save_path)
