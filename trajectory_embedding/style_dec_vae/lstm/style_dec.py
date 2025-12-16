import torch
import torch.nn as nn
import numpy as np

from trajectory_embedding.style_dec_vae.lstm.style_vae import Encoder, Decoder
from trajectory_embedding.style_dec_vae.utils.loss import MiniGridLoss


class ClusteringBasedVAE(nn.Module):
    def __init__(self, n_clusters, dimensions, alpha, **kwargs):
        super(ClusteringBasedVAE, self).__init__()
        self.input_size = dimensions[0]
        self.hidden_size = dimensions[1]
        self.latent_size = dimensions[2]
        self.device = torch.device('cuda' if (torch.cuda.is_available()) else 'cpu')

        self.encoder = Encoder(input_size=self.input_size,
                               hidden_size=self.hidden_size,
                               latent_size=self.latent_size,
                               device=self.device,
                               num_layers=1)
        self.decoder = Decoder(input_size=self.input_size,
                               latent_size=self.latent_size,
                               hidden_size=self.hidden_size,
                               device=self.device,
                               num_layers=1)

        self.is_logits = kwargs.get('logits', False)
        self.alpha = alpha

        self.n_centroids = n_clusters

        # learnable GMM parameters initialization
        self.pi = nn.Parameter(torch.ones(self.n_centroids, dtype=torch.float32) / self.n_centroids)
        self.mu_c = nn.Parameter(torch.zeros((self.n_centroids, self.latent_size), dtype=torch.float32))
        self.log_sigma_c = nn.Parameter(torch.ones((self.n_centroids, self.latent_size), dtype=torch.float32))

        self.custom_loss_fn = MiniGridLoss()


    def forward(self, x, seq_lens, hidden_enc):
        latent, z_mean, z_log_var, hidden_enc = self.encoder(x, seq_lens) #, hidden_enc)

        pi = self.pi
        log_sigmac_c = self.log_sigma_c
        mu_c = self.mu_c

        pzc = torch.exp(torch.log(pi.unsqueeze(0)) + self.gaussian_pdfs_log(latent, mu_c, log_sigmac_c))
        return pzc, latent, hidden_enc

    def elbo_loss(self, x, seq_lengths, epoch, L=1):
        det = 1e-5 # change back to 1e-10
        res_loss = 0.0
        z, mu, logvar, hidden_enc = self.encoder(x, seq_lengths)

        for l in range(L):
            z = torch.randn_like(mu) * torch.exp(logvar / 2) + mu

            x_decoded, _ = self.decoder(z, seq_lengths)

            # Since the sequence length are different we want to calculate the loss for each un-masked sequence separately
            # if self.is_logits:
            #     res_loss = 0
            #     for i, length in enumerate(seq_lengths):
            #         res_loss += nn.functional.mse_loss(x_decoded[i, :length], x[i, :length], reduction='sum')
            #     res_loss /= len(seq_lengths)
            # else:
            #     res_loss = 0
            #     for i, length in enumerate(seq_lengths):
            #         res_loss += self.custom_loss_fn(x_decoded[i, :length], x[i, :length], reduction='mean') #/ length
            #     res_loss /= len(seq_lengths) #* x.size(-1)
            res_loss = nn.functional.mse_loss(x_decoded, x)

        res_loss /= L
        res_loss = self.alpha * res_loss  * x.size(-1)  #* 20

        pi = self.pi #+ 1e-5
        log_sigma2_c = self.log_sigma_c
        mu_c = self.mu_c
        z = torch.randn_like(mu) * torch.exp(logvar / 2) + mu

        # calculate the p(z|c) or gamma
        p = torch.log(pi.unsqueeze(0))
        if torch.isnan(p).any():
            print(p)
        pdf = self.gaussian_pdfs_log(z, mu_c, log_sigma2_c)
        if torch.isnan(pdf).any() :
            print(pdf)
        su = p + pdf
        if torch.isnan(torch.exp(su)).any():
            print(su)
        torch.exp(su)
        pcz = torch.exp(su)
        pcz = pcz + det
        pcz = pcz / (pcz.sum(1).view(-1, 1))  # batch_size*clusters

        # calculating the KL losses
        kl_loss_1 =  0.5 * torch.sum(pcz * torch.sum(log_sigma2_c.unsqueeze(0) +
                                                           torch.exp(logvar.unsqueeze(1) - log_sigma2_c.unsqueeze(0)) +
                                                           (mu.unsqueeze(1) - mu_c.unsqueeze(0)) ** 2 / torch.exp(log_sigma2_c.unsqueeze(0)), 2), 1)

        kl_loss_2 = -torch.sum(pcz * torch.log(pi.unsqueeze(0) / (pcz)), 1)

        kl_loss_3 = -0.5 * torch.sum(1 + logvar, 1)

        kl_loss_1 = kl_loss_1.mean()
        kl_loss_2 = kl_loss_2.mean()
        kl_loss_3 = kl_loss_3.mean()
        kl_weight = 1 #0.0016 # self.get_beta(epoch)
        # kl_weight = lambda epoch: 0.0016 if epoch < 6 else (
        #     0.0016 + (epoch - 6) / 1000 * (0.1 - 0.0016) if epoch < 1006 else 1.0)

        # kl_terms = kl_weight(epoch) * (kl_loss_1 + kl_loss_2 + kl_loss_3)
        kl_terms = kl_weight * (kl_loss_1 + kl_loss_2 + kl_loss_3)

        loss = res_loss + kl_terms
        # loss /= x.size(0)
        return loss, res_loss, kl_terms, kl_weight

    def get_beta(self, epoch):
        beta_min = 5e-3  # Minimum beta value
        beta_max = 1.0  # Maximum beta value


        # Adjust epoch count to start cycles after 500 epochs.
        new_epoch = epoch

        # Define cycle parameters.
        ramp_steps = 90  # Number of epochs for the ramp-up phase.
        plateau_steps = 10  # Number of epochs where beta remains at beta_max.
        cycle_length = ramp_steps + plateau_steps
        total_cycles = 5  # Total number of cycles.
        p = 2.0  # Polynomial degree (quadratic ramp).

        # If training goes beyond the total cycles, beta remains at beta_max.
        if new_epoch >= cycle_length * total_cycles:
            return beta_max

        cycle_epoch = new_epoch % cycle_length
        if cycle_epoch < ramp_steps:
            # Ramp phase: increase beta from beta_min to beta_max following a polynomial schedule.
            beta = beta_min + (beta_max - beta_min) * ((cycle_epoch / ramp_steps) ** p)
        else:
            # Plateau phase: beta remains constant at beta_max.
            beta = beta_max
        return beta

    def gaussian_pdfs_log(self, x, mus, log_sigma2s):
        G = []
        for c in range(self.n_centroids):
            G.append(self.gaussian_pdf_log(x, mus[c:c + 1, :], log_sigma2s[c:c + 1, :]).view(-1, 1))
        return torch.cat(G, 1)

    @staticmethod
    def gaussian_pdf_log(x, mu, log_sigma2):
        if torch.isnan(torch.exp(log_sigma2)).any():
            print(log_sigma2)
        return -0.5 * (torch.sum(np.log(np.pi * 2) + log_sigma2 + (x - mu).pow(2) / torch.exp(log_sigma2), 1))
