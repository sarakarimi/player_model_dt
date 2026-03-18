"""
Transformer VAE for PADDED variable-length sequences of float vectors (B, S_max, 9).

Key points:
- mask: (B, S_max) where True=valid token, False=PAD
- Encoder uses src_key_padding_mask (True=PAD) and masked-mean pooling
- Decoder generates outputs at all S_max positions using learned query tokens
- Output is masked in the loss so PAD positions don't contribute

Works well as a baseline for continuous sequence reconstruction.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Tuple
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset_utils.minigrid_vae_dataset import MiniGridDataset, collate_fn
from style_decision_transformer import paths
from style_decision_transformer.lstm.style_vae import cluster_latents, plot_embeddings


def beta_cyclical(
    epoch: int,
    beta_min: float,
    beta_max: float,
    beta_warmup: int,
    beta_period: int,
) -> float:
    """
    Simple cyclical warmup schedule per epoch:
      - first beta_warmup epochs: linearly ramp beta_min -> beta_max
      - afterwards: repeat cycles of length beta_period, ramping beta_min -> beta_max each cycle
    """
    if beta_period <= 0:
        return beta_max

    if epoch < beta_warmup:
        t = epoch / max(1, beta_warmup)
        return beta_min + (beta_max - beta_min) * t

    # cyclical ramp within each period
    t = (epoch - beta_warmup) % beta_period
    frac = t / max(1, beta_period - 1)
    return beta_min + (beta_max - beta_min) * frac



def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # make deterministic-ish (can slow down)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


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


def reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    return mu + eps * std


def masked_mse(x_hat: torch.Tensor, x: torch.Tensor, pad_mask: torch.Tensor) -> torch.Tensor:
    """
    x_hat, x: (B, S, D)
    pad_mask: (B, S) True=PAD
    """
    valid = (~pad_mask).unsqueeze(-1).to(x.dtype)  # (B,S,1)
    denom = (valid.sum() * x.size(-1)).clamp_min(1e-8)
    return (((x_hat - x) ** 2) * valid).sum() #/ denom


def kl_divergence(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    # mean over batch
    return -0.5 * torch.mean(torch.sum(1.0 + logvar - mu.pow(2) - logvar.exp(), dim=-1), 0)


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding (batch-first)."""

    def __init__(self, d_model: int, max_len: int = 1000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float) * (-torch.log(torch.tensor(10000.0)) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div)
        pe[:, 1::2] = torch.cos(position * div)
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)  # (1, max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        s = x.size(1)
        return x + self.pe[:, :s, :]


@dataclass
class TransformerVAEConfig:
    max_len: int  # S_max (the padded length you batch to)
    in_dim: int = 9 + 1

    d_model: int = 128
    nhead: int = 8
    num_enc_layers: int = 2 #4
    num_dec_layers: int = 2 #4
    dim_ff: int = 256 #512
    dropout: float = 0.1

    z_dim: int = 32
    mem_tokens: int = 4  # how many latent "memory" tokens to give the decoder
    pos_max_len: int = 100  # must be >= max_len


class TransformerVAE(nn.Module):
    """
    Variable-length (padded) Transformer VAE.

    Inputs:
      x:    (B, S_max, 9)
      mask: (B, S_max) True=valid, False=PAD
    Outputs:
      x_hat: (B, S_max, 9) (you'll usually ignore PAD positions via mask)
    """

    def __init__(self, cfg: TransformerVAEConfig):
        super().__init__()
        if cfg.pos_max_len < cfg.max_len:
            print(cfg.pos_max_len, cfg.max_len)
            raise ValueError("pos_max_len must be >= max_len")

        self.cfg = cfg

        self.in_proj = nn.Linear(cfg.in_dim, cfg.d_model)
        self.pos_enc = PositionalEncoding(cfg.d_model, max_len=cfg.pos_max_len)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=cfg.d_model,
            nhead=cfg.nhead,
            dim_feedforward=cfg.dim_ff,
            dropout=cfg.dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=cfg.num_enc_layers)

        # Pooling head -> latent params
        self.pool_norm = nn.LayerNorm(cfg.d_model)
        self.to_mu = nn.Linear(cfg.d_model, cfg.z_dim)
        self.to_logvar = nn.Linear(cfg.d_model, cfg.z_dim)

        # Latent -> decoder memory tokens
        self.z_to_mem = nn.Linear(cfg.z_dim, cfg.mem_tokens * cfg.d_model)

        dec_layer = nn.TransformerDecoderLayer(
            d_model=cfg.d_model,
            nhead=cfg.nhead,
            dim_feedforward=cfg.dim_ff,
            dropout=cfg.dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(dec_layer, num_layers=cfg.num_dec_layers)

        # Learned queries for all S_max positions
        # self.query_tokens = nn.Parameter(torch.randn(1, cfg.max_len, cfg.d_model) * 0.02)
        # self.query_pos = PositionalEncoding(cfg.d_model, max_len=cfg.pos_max_len)
        self.query_pos = PositionalEncoding(cfg.d_model, max_len=cfg.pos_max_len)

        self.out_proj = nn.Linear(cfg.d_model, cfg.in_dim)

    def encode(self, x: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        x: (B, S, 9), mask: (B, S) True=valid
        """
        h = self.in_proj(x)
        h = self.pos_enc(h)

        # Transformer expects True for PAD positions:
        src_key_padding_mask = ~mask  # (B,S) True=PAD
        h = self.encoder(h, src_key_padding_mask=src_key_padding_mask)  # (B,S,D)

        # masked mean pooling over valid tokens
        h = self.pool_norm(h)
        m = mask.unsqueeze(-1).to(h.dtype)  # (B,S,1)
        denom = m.sum(dim=1).clamp_min(1.0)
        pooled = (h * m).sum(dim=1) / denom  # (B,D)

        mu = self.to_mu(pooled)
        logvar = self.to_logvar(pooled)
        return mu, logvar

    def decode(self, z: torch.Tensor, tgt_key_padding_mask: Optional[torch.Tensor]) -> torch.Tensor:
        B = z.size(0)
        S = tgt_key_padding_mask.size(1)  # current batch length
        mem = self.z_to_mem(z).view(B, self.cfg.mem_tokens, self.cfg.d_model)

        # dynamic query tokens
        q = torch.zeros(B, S, self.cfg.d_model, device=z.device, dtype=mem.dtype)
        q = self.query_pos(q)

        dec = self.decoder(
            tgt=q,
            memory=mem,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=None,
        )
        return self.out_proj(dec)


    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        x: (B, S_max, 9)
        mask: (B, S_max) True=valid, False=PAD
        """
        # if x.size(1) != self.cfg.max_len:
        #     raise ValueError(f"Expected padded length S_max={self.cfg.max_len}, got {x.size(1)}")
        if mask.shape[:2] != x.shape[:2]:
            raise ValueError("mask must have shape (B, S_max) matching x")

        mu, logvar = self.encode(x, mask)
        z = reparameterize(mu, logvar)

        # For decoder tgt_key_padding_mask: True=PAD
        tgt_key_padding_mask = ~mask
        x_hat = self.decode(z, tgt_key_padding_mask=tgt_key_padding_mask)
        return x_hat, mu, logvar, z


# ----------------------------
# Training loop
# ----------------------------
def train_TransformerVAE(
        model,
        loader,
        num_epochs: int,
        lr: float,
        beta_min: float,
        beta_max: float,
        beta_warmup: int,
        beta_period: int,
        device: str,
        seed: int = 42,
        causal_decoder: bool = False,  # set True if you want autoregressive-style decoding
        grad_clip: float = 1.0,
) -> TransformerVAE:
    set_seed(seed)

    opt = torch.optim.AdamW(model.parameters(), lr=lr)

    for epoch in range(num_epochs):
        model.train()
        beta = 0.0085 #beta_cyclical(epoch, beta_min, beta_max, beta_warmup, beta_period) #0.00025  #

        running_loss = 0.0
        running_recon = 0.0
        running_kl = 0.0
        n_batches = 0

        for batch_data, _action, _labels, _lengths in loader:
            batch_data = batch_data.to(device)

            action_data = _action.to(torch.float32).to(device).unsqueeze(-1)
            batch_data = torch.concat([batch_data, action_data], -1)

            pad_mask, tgt_mask = create_padding_mask(batch_data, pad_token=0.0, causal=causal_decoder)
            # ensure masks are on device
            pad_mask = pad_mask.to(device)

            x_hat, mu, logvar, _z = model(batch_data, ~pad_mask)

            recon = masked_mse(x_hat, batch_data, pad_mask)
            kl = kl_divergence(mu, logvar)
            loss = recon + beta * kl
            loss = loss.mean()

            opt.zero_grad(set_to_none=True)
            loss.backward()
            if grad_clip is not None and grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            opt.step()

            running_loss += loss.item()
            running_recon += recon.item()
            running_kl += kl.item()
            n_batches += 1

        print(
            f"Epoch {epoch + 1:03d}/{num_epochs} | beta={beta:.5f} "
            f"| loss={running_loss / n_batches:.6f} recon={running_recon / n_batches:.6f} kl={running_kl / n_batches:.6f}"
        )

    # save model
    print('Saving model ...\n')
    path = '/home/sara/repositories/player_model_dt/trained_models/minigrid_model/style_vae/three_style_env_hard_transformer_model.pth'
    torch.save(model.state_dict(), path)
    return model

def transformer_validate(model, loader, load_model=False, model_path=None, device='cuda'):
    if load_model:
        model.load_state_dict(torch.load(model_path, map_location='cpu'))
        model.eval()
        if torch.cuda.is_available():
            model = model.cuda()

    model.eval()

    with torch.no_grad():
        Z, labels_list = [], []
        for batch_data, _action, _labels, _lengths in loader:
            batch_data = batch_data.to(device)

            action_data = _action.to(torch.float32).to(device).unsqueeze(-1)
            batch_data = torch.concat([batch_data, action_data], -1)

            pad_mask, tgt_mask = create_padding_mask(batch_data, pad_token=0.0)
            # ensure masks are on device
            pad_mask = pad_mask.to(device)

            recon_x, _, _, z = model(batch_data, ~pad_mask)
            Z.append(z.cpu().detach().numpy())
            labels_list.append(_labels)

    # Plot trained embeddings
    true_labels = np.concatenate(labels_list, 0)
    Z = np.concatenate(Z, 0)
    predicted_labels, cluster_centroids = cluster_latents(Z, 3)
    # plot_embeddings(gtruth=predicted_labels, Z=Z, label_name='task_predicted')
    # plot_embeddings(gtruth=true_labels, Z=Z, label_name='task_ground_truth')

    return predicted_labels, Z, cluster_centroids


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # Load dataset
    dataset_params = {
        'sampling': False,
        'index_channel_only': True,
        'state_normalization_factor': 9,
        'action_normalization_factor': 6
    }
    dataset = MiniGridDataset(trajectory_paths=paths, **dataset_params)
    max_len = max(dataset.seq_lens)
    loader = DataLoader(dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)

    # model
    cfg = TransformerVAEConfig(
        max_len=max_len,  # pad/truncate all sequences to 128
        in_dim=9 + 1,  # state_dim + action_dim
        d_model=128,
        nhead=8,
        num_enc_layers=4,
        num_dec_layers=4,
        dim_ff=512,
        z_dim=16,
        mem_tokens=4,
        pos_max_len=max_len,  # or a bit larger
    )
    model = TransformerVAE(cfg).to(device)

    # Train
    model = train_TransformerVAE(
        model=model,
        loader=loader,
        num_epochs=3,  # 200
        lr=1e-3,
        beta_min=0.006,
        beta_max=0.04,
        beta_warmup=2,
        beta_period=8,
        device=device,
        seed=42
    )

    # Evaluate

    path = '/home/sara/repositories/player_model_dt/trained_models/minigrid_model/style_vae/three_style_env_hard_transformer_model.pth'
    model.load_state_dict(torch.load(path, map_location=torch.device(device)))
    model.eval()

    Z, y_true = [], []
    with torch.no_grad():
        for batch_idx, (batch_data, action, labels, lengths) in enumerate(loader):
            batch_data = batch_data.to(device)

            action_data = action.to(torch.float32).to(device).unsqueeze(-1)
            batch_data = torch.concat([batch_data, action_data], -1)

            pad_mask, tgt_mask = create_padding_mask(batch_data, pad_token=0.0, causal=False)
            pad_mask = pad_mask.to(device)
            _, _, _, z = model(batch_data, ~pad_mask)

            Z.append(z)
            y_true.extend(labels.numpy())

        Z = torch.cat(Z, 0).cpu().numpy()
        predicted_labels, cluster_centroids = cluster_latents(Z, 3)

        plot_embeddings(gtruth=predicted_labels, Z=Z, label_name='task_predicted')
        plot_embeddings(gtruth=y_true, Z=Z, label_name='task_ground_truth')


