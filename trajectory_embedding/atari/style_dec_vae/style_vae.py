import itertools

import torch
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import DataLoader

from trajectory_embedding.atari.style_dec_vae.utils.loss import MiniGridLoss
from trajectory_embedding.atari.style_dec_vae.utils.utils import convert_to_one_hot
from utils.dataset import *
from torch.autograd import Variable


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
        # self.fc21 = nn.Linear(self.hidden_size, self.latent_size)
        # self.fc22 = nn.Linear(self.hidden_size, self.latent_size)

        self.sampling = GaussianSampling(self.hidden_size, self.latent_size)

    def reparametize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        noise = torch.randn_like(std).to(self.device)

        z = mu + noise * std
        return z, mu, logvar

    def forward(self, x, seq_lens):
        # Pack the padded sequence
        packed_input = nn.utils.rnn.pack_padded_sequence(x, seq_lens, batch_first=True, enforce_sorted=False)
        outputs, (hidden, cell) = self.lstm(packed_input)
        enc_h = hidden[-1]  # .view(batch_size, self.hidden_size).to(self.device)

        # extract latent variable z(hidden space to latent space)
        # mean = self.fc21(enc_h)
        # logvar = self.fc22(enc_h)
        # z, _, _ = self.reparametize(mean, logvar)  # batch_size x latent_size
        z, mean, logvar = self.sampling(enc_h)

        return z, mean, logvar, (hidden, cell)


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
        # self.fc1 = nn.Linear(self.latent_size, self.hidden_size)
        self.fc2 = nn.Linear(hidden_size, input_size)
        # self.final = nn.Sigmoid()

    def forward(self, z, lengths, total_padding_length=None):
        max_seq_len = max(lengths)
        z_repeat = z.repeat(1, max_seq_len, 1)
        z_repeat = z_repeat.view(-1, max_seq_len, self.latent_size).to(self.device)

        # TODO replace with the z as hidden
        # initialize hidden state as inputs
        # h_ = self.fc1(z)
        # hidden = (h_.contiguous(), h_.contiguous())
        h_0 = torch.zeros(1, z.size(0), self.hidden_size).to(z.device)
        c_0 = torch.zeros(1, z.size(0), self.hidden_size).to(z.device)
        hidden = (h_0, c_0)

        x = torch.nn.utils.rnn.pack_padded_sequence(z_repeat, lengths, batch_first=True, enforce_sorted=False)

        output, (hidden, cell) = self.lstm(x, hidden)
        output, _ = torch.nn.utils.rnn.pad_packed_sequence(output, batch_first=True)

        prediction = self.fc2(output)
        # prediction = self.final(prediction)
        return prediction, (hidden, cell)


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

    def forward(self, x, seq_lens):
        # encode input space to hidden space
        z, mean, logvar, _ = self.lstm_enc(x, seq_lens)

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

        return m_loss, x_hat, (recon_loss, kld_loss)

    def loss_function(self, *args, **kwargs) -> dict:
        recons = args[0]
        og_input = args[1]
        mu = args[2]
        log_var = args[3]
        lengths = args[4]
        loss_fn = MiniGridLoss()

        kld_weight = 0.00025  # Account for the minibatch samples from the dataset

        # MSE loss
        # recons_loss = 0
        # for i, length in enumerate(lengths):
        #     recons_loss += nn.functional.mse_loss(recons[i, :length], og_input[i, :length], reduction='sum')
        # recons_loss /= len(lengths)

        # binary cross entropy loss
        # recons_loss = 0
        # for i, length in enumerate(lengths):
        #     recons_loss += nn.functional.binary_cross_entropy(recons[i, :length], og_input[i, :length], reduction='sum')
        # recons_loss /= len(lengths)

        # per on-hot category nll loss
        recons_loss = 0
        for i, length in enumerate(lengths):
            recons_loss += loss_fn(recons[i, :length], og_input[i, :length])
        recons_loss /= len(lengths)

        kld_loss = -0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim=1) / len(lengths)

        loss = recons_loss + kld_weight * kld_loss
        return {
            "loss": loss,
            "Reconstruction_Loss": recons_loss.detach(),
            "KLD": -kld_loss.detach(),
        }


def train(model, train_loader, test_loader, epochs):
    # optimizer
    # optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    # optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    optimizer = torch.optim.Adam(itertools.chain(model.encoder.parameters(),
                                                 model.decoder.parameters()), lr=0.002)
    steplr = torch.optim.lr_scheduler.StepLR(optimizer, 10, 0.9)


    ## training
    count = 0
    for epoch in range(epochs):
        model.train()
        # optimizer.zero_grad()
        train_iterator = tqdm(
            train_loader, total=len(train_loader), desc="training"
        )

        for batch_data, labels, lengths in train_iterator:
            batch_data = batch_data.to(torch.float32).to(model.device)
            lengths = lengths.to(torch.int64)#.to(model.device)

            optimizer.zero_grad()
            mloss, recon_x, info = model(batch_data, lengths)

            # Backward and optimize
            mloss.mean().backward()
            optimizer.step()

            train_iterator.set_postfix({"train_loss": float(mloss.mean())})

    model.eval()
    eval_loss = 0
    test_iterator = tqdm(
        test_loader, total=len(test_loader), desc="testing"
    )

    with torch.no_grad():
        for batch_data, labels, lengths in test_iterator:
            batch_data = batch_data.to(torch.float32).to(model.device)
            lengths = lengths.to(torch.int64)  # .to(device)


            mloss, recon_x, info = model(batch_data, lengths)
            torch.set_printoptions(linewidth=200)
            print("x", batch_data.to(torch.int32))
            print("x_hat", convert_to_one_hot(recon_x))
            eval_loss += mloss.mean().item()

            test_iterator.set_postfix({"eval_loss": float(mloss.mean())})

    eval_loss = eval_loss / len(test_loader)
    print("Evaluation Score : [{}]".format(eval_loss))

    return model


if __name__ == "__main__":
    # Hyperparameters
    input_size = 500  # Number of features in each timestep
    hidden_size = 128  # 20
    latent_size = 10  # 8
    num_epochs = 1000  # 10000
    batch_size = 32
    # max_seq_len = 4
    # feature_size = 2
    # num_samples = 10

    # # Fake data
    # sequences = generate_varied_length_data(num_samples, max_seq_len, feature_size)
    #
    # # Create dataset and data loaders
    # train_set = VariedLengthDataset(sequences)
    # test_set = VariedLengthDataset(sequences)

    paths = ["/home/sara/repositories/player_model_dt/data/new_implementation_datasets/PPO_trajectories_mode1.gz",
             "/home/sara/repositories/player_model_dt/data/new_implementation_datasets/PPO_trajectories_mode2.gz"
             ]
    trajectory_data_set = MiniGridDataset(trajectory_paths=paths)
    train_loader = torch.utils.data.DataLoader(
        dataset=trajectory_data_set, batch_size=batch_size, shuffle=True, collate_fn=collate_fn
    )
    test_loader = train_loader

    # define LSTM-based VAE model
    model = LSTMVAE(input_size, hidden_size, latent_size)

    # convert to format of data loader
    # train_loader = torch.utils.data.DataLoader(
    #     dataset=train_set, batch_size=batch_size, shuffle=True, collate_fn=collate_fn
    # )
    # test_loader = torch.utils.data.DataLoader(
    #     dataset=test_set, batch_size=batch_size, shuffle=False, collate_fn=collate_fn
    # )

    # training
    if torch.cuda.is_available():
        print('Cuda is available')
        dec_cluster = model.cuda()
    else:
        print('No GPU')
    train(model, train_loader, test_loader, num_epochs)
