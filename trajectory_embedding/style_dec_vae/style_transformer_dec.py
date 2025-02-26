import torch
from torch import nn
from torch.nn.parameter import Parameter
from trajectory_embedding.style_dec_vae.style_transformer_vae import Encoder, Decoder


class TransformerVaDE(nn.Module):
    def __init__(self, input_dim=25, output_dim=25,
                 e_dim=128, z_dim=32, nheads=4, ff_dim=256,
                 nTElayers=4, nTDlayers=4, n_classes=4):
        super(TransformerVaDE, self).__init__()

        self.pi_prior = Parameter(torch.ones(n_classes)/n_classes)
        self.mu_prior = Parameter(torch.zeros(n_classes, z_dim))
        self.log_var_prior = Parameter(torch.randn(n_classes, z_dim))

        self.encoder = Encoder(input_dim, e_dim, z_dim, nheads, nTElayers, ff_dim)
        self.decoder = Decoder(output_dim, e_dim, z_dim, nheads, nTDlayers, ff_dim)

    def encode(self, x, pad_mask=None):
        return self.encoder(x, pad_mask=pad_mask)

    def decode(self, z, x, pad_mask=None, tgt_mask=None):
        return self.decoder(z, x, pad_mask=pad_mask, tgt_mask=tgt_mask)

    def reparameterize(self, mu, log_var):
        std = torch.exp(log_var / 2)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x, pad_mask=None, tgt_mask=None):
        mu, logvar = self.encoder(x, pad_mask)
        z = self.reparameterize(mu, logvar)
        recon = self.decoder(z, x, tgt_mask=tgt_mask, tgt_pad_mask=pad_mask)
        return recon.transpose(0,1), mu, logvar, z


class TransformerAE(nn.Module):
    def __init__(self, input_dim=25, output_dim=25,
                 e_dim=128, z_dim=32, nheads=4, ff_dim=256,
                 nTElayers=4, nTDlayers=4):
        super(TransformerAE, self).__init__()
        self.z_dim = z_dim
        self.encoder = Encoder(input_dim, e_dim, z_dim, nheads, nTElayers, ff_dim)
        self.decoder = Decoder(output_dim, e_dim, z_dim, nheads, nTDlayers, ff_dim)

    def encode(self, x, pad_mask=None):
        mu, _ = self.encoder(x, pad_mask=pad_mask)
        return mu

    def decode(self, z, x, pad_mask=None, tgt_mask=None):
        return self.decoder(z, x, pad_mask=pad_mask, tgt_mask=tgt_mask)


    def forward(self, x, pad_mask=None, tgt_mask=None):
        mu, logvar = self.encoder(x, pad_mask)
        z = mu
        recon = self.decoder(z, x, tgt_mask=tgt_mask, tgt_pad_mask=pad_mask)
        return mu, logvar, recon.transpose(0,1)
