import torch
from torch import nn
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader
import math
import numpy as np
import time
import os
import matplotlib.pyplot as plt
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from trajectory_embedding.style_dec_vae.config import paths
from trajectory_embedding.style_dec_vae.utils.dataset import MiniGridDataset, collate_fn
from torch.nn import TransformerDecoder, TransformerDecoderLayer



def create_padding_mask(batch, pad_val=0.0):
    """
    For a batch of shape [B, seq_len, 25],
    we consider a position padded if all 25 dims = pad_val (0.0).
    Returns a mask [B, seq_len] with True at padded positions.
    """
    with torch.no_grad():
        # check if entire 25-d vector is zero
        pad_positions = torch.all(torch.eq(batch, pad_val), dim=-1)  # [B, seq_len]
    return pad_positions

def calc_kl(mu, logvar, reduction='mean'):
    kl = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).sum(dim=1)
    if reduction == 'sum':
        return kl.sum()
    elif reduction == 'mean':
        return kl.mean()
    else:
        return kl  # no reduction


def cyc_beta_scheduler(epochs=20, warmup_epochs=4, beta_min=0.0, beta_max=0.03, period=8, ratio=0.75):
    """
    Same as original cyc beta. Creates an array of length `epochs` with
    cyclical or warm-up + hold scheduling for beta.
    """
    beta_warmup = np.ones(warmup_epochs) * beta_min
    beta_cyc = np.ones(epochs - warmup_epochs) * beta_max
    n_cycle = int(np.floor((epochs - warmup_epochs)/period))
    step = (beta_max - beta_min)/(period * ratio)
    for c in range(n_cycle):
        curr_beta, i = beta_min, 0
        while curr_beta <= beta_max and (int(i + c*period) < epochs - warmup_epochs):
              beta_cyc[int(i + c*period)] = curr_beta
              curr_beta += step
              i += 1
    beta = np.concatenate((beta_warmup, beta_cyc), axis=0)
    return beta



class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=1000, dropout=0.1):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(1)  # shape [max_len, 1, d_model]
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        x is expected to have shape [seq_len, batch_size, d_model]
        """
        seq_len = x.size(0)
        x = x + self.pe[:seq_len, :]
        return self.dropout(x)

class Word2SentenceEmbedding(nn.Module):
    def __init__(self, hdim):
        super(Word2SentenceEmbedding, self).__init__()
        self.dense = nn.Linear(hdim, hdim)
        self.activation = nn.Tanh()

    def forward(self, hidden_states):
        # hidden_states shape [seq_len, batch_size, hdim]
        # we take the hidden state at t=0 (or an average, etc.)
        first_token_tensor = hidden_states[0]  # [batch_size, hdim]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output


class Encoder(nn.Module):
    def __init__(self, input_dim=25, e_dim=128, z_dim=32, nheads=4, nTlayers=4, ff_dim=256):
        """
        Args:
          input_dim: dimension of each time step (25 in your case).
          e_dim:     model dimension in the Transformer.
          z_dim:     latent dimension in VAE.
          nheads:    number of attention heads.
          nTlayers:  number of encoder layers.
          ff_dim:    feed-forward dimension in Transformer.
        """
        super(Encoder, self).__init__()
        self.input_proj = nn.Linear(input_dim, e_dim)   # project 25 -> e_dim
        self.pos_encoding = PositionalEncoding(e_dim)
        encoder_layer = TransformerEncoderLayer(d_model=e_dim,
                                                nhead=nheads,
                                                dim_feedforward=ff_dim,
                                                dropout=0.1)
        self.transformer_encoder = TransformerEncoder(encoder_layer, num_layers=nTlayers)
        self.word2sen_hidden = Word2SentenceEmbedding(hdim=e_dim)
        self.hid2latparams = nn.Linear(e_dim, 2 * z_dim)

    def forward(self, x, pad_mask=None):
        """
        x: shape [B, seq_len, 25]
        pad_mask: shape [B, seq_len] with True at pad positions

        We'll transpose to [seq_len, B, 25], then apply input_proj.
        """
        # [B, seq_len, input_dim] -> [seq_len, B, input_dim]
        x = x.transpose(0, 1)
        x = self.input_proj(x)                # [seq_len, B, e_dim]
        x = self.pos_encoding(x)              # add positional embeddings
        # The Transformer expects a src_key_padding_mask of shape [B, seq_len]
        hidden = self.transformer_encoder(x, src_key_padding_mask=pad_mask)   # [seq_len, B, e_dim]
        hidden = self.word2sen_hidden(hidden) # [B, e_dim]
        y = self.hid2latparams(hidden)        # [B, 2*z_dim]
        mu, logvar = y.chunk(2, dim=1)        # each [B, z_dim]
        return mu, logvar



class Decoder(nn.Module):
    def __init__(self, output_dim=25, e_dim=128, z_dim=32, nheads=4, nTlayers=4, ff_dim=256):
        super(Decoder, self).__init__()
        self.lat2hid = nn.Linear(z_dim, e_dim)
        self.pos_encoding = PositionalEncoding(e_dim)
        # NEW: projection from continuous input dimension (25) to model dimension (e_dim)
        self.target_proj = nn.Linear(output_dim, e_dim)
        decoder_layer = TransformerDecoderLayer(d_model=e_dim,
                                                nhead=nheads,
                                                dim_feedforward=ff_dim,
                                                dropout=0.1)
        self.transformer_decoder = TransformerDecoder(decoder_layer, num_layers=nTlayers)
        self.hid2recon = nn.Linear(e_dim, output_dim)  # project back to 25-dim for reconstruction

    def forward(self, z, target_seq, tgt_mask=None, tgt_pad_mask=None):
        # Project latent vector to initial hidden state and treat as memory
        memory = self.lat2hid(z).unsqueeze(0)  # shape: [1, B, e_dim]

        # Transpose target_seq from [B, seq_len, 25] to [seq_len, B, 25]
        target_seq = target_seq.transpose(0, 1)
        # Project target sequence from 25 to e_dim
        target_seq = self.target_proj(target_seq)
        # Apply positional encoding (now expecting a tensor of shape [seq_len, B, e_dim])
        target_seq = self.pos_encoding(target_seq)

        # Decode with Transformer decoder
        hidden = self.transformer_decoder(target_seq,
                                          memory,
                                          tgt_mask=tgt_mask,
                                          tgt_key_padding_mask=tgt_pad_mask)
        # Project hidden states back to the original dimension (25)
        recon = self.hid2recon(hidden)  # shape: [seq_len, B, 25]
        return recon


class TransformerVAE(nn.Module):
    def __init__(self, input_dim=25, output_dim=25,
                 e_dim=128, z_dim=32, nheads=4, ff_dim=256,
                 nTElayers=4, nTDlayers=4):
        super(TransformerVAE, self).__init__()
        self.z_dim = z_dim
        self.encoder = Encoder(input_dim, e_dim, z_dim, nheads, nTElayers, ff_dim)
        self.decoder = Decoder(output_dim, e_dim, z_dim, nheads, nTDlayers, ff_dim)

    def reparameterize(self, mu, logvar):
        """
        z = mu + std * eps,  eps ~ N(0, I)
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x, pad_mask=None, tgt_mask=None):
        """
        x: [B, seq_len, 25]
        """
        mu, logvar = self.encoder(x, pad_mask)
        z = self.reparameterize(mu, logvar)
        recon = self.decoder(z, x, tgt_mask=tgt_mask, tgt_pad_mask=pad_mask)  # [seq_len, B, 25]
        return mu, logvar, recon.transpose(0,1)  # return as [B, seq_len, 25]





def train_TransformerVAE(
    input_dim=25,
    e_dim=128,
    z_dim=32,
    nheads=4,
    nTElayers=4,
    nTDlayers=4,
    ff_dim=256,
    num_epochs=20,
    batch_size=16,
    lr=1e-3,
    beta_min=0.0,
    beta_max=0.03,
    beta_warmup=2,
    beta_period=5,
    device="cpu",
    seed=42
):
    # Reproducibility
    if seed is not None:
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # ====== Dataset / DataLoader ======
    trajectory_data_set = MiniGridDataset(trajectory_paths=paths)
    loader = torch.utils.data.DataLoader(
        dataset=trajectory_data_set, batch_size=batch_size, shuffle=True, collate_fn=collate_fn
    )

    # train_iterator = tqdm(
    #     loader, total=len(loader), desc="training"
    # )
    # ====== Model ======
    model = TransformerVAE(input_dim=input_dim,
                           output_dim=input_dim,
                           e_dim=e_dim,
                           z_dim=z_dim,
                           nheads=nheads,
                           ff_dim=ff_dim,
                           nTElayers=nTElayers,
                           nTDlayers=nTDlayers).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[20, ], gamma=0.1)

    # Beta scheduling
    beta_scheduler = cyc_beta_scheduler(epochs=num_epochs,
                                        warmup_epochs=beta_warmup,
                                        beta_min=beta_min,
                                        beta_max=beta_max,
                                        period=beta_period,
                                        ratio=0.75)
    beta = beta_scheduler[0]

    # For reconstruction (MSE)
    mse_loss_fn = nn.MSELoss(reduction='none')  # we will mask out padded positions

    # ====== Train ======
    rec_loss_log = []
    kl_loss_log = []
    total_loss_log = []

    model.train()
    for epoch in range(1, num_epochs + 1):
        # beta = beta_scheduler[epoch]
        if epoch < num_epochs:
            beta = beta_scheduler[epoch]
        epoch_rec_loss = 0.0
        epoch_kl_loss = 0.0
        epoch_total = 0.0
        start_time = time.time()

        for batch_data, labels, lengths in loader:
            # batch_data: shape [B, max_seq_len, 25]
            batch_data = batch_data.to(device)
            # Create padding mask
            pad_mask = create_padding_mask(batch_data, pad_val=0.0).to(device)  # [B, max_seq_len]
            # Optionally create a causal mask if you want auto-regression
            # For standard reconstruction, no need for a causal mask:
            tgt_mask = None

            # Forward
            mu, logvar, recon = model(batch_data, pad_mask, tgt_mask)
            # recon is [B, max_seq_len, 25]

            # MSE loss ignoring padded positions
            mse_per_step = mse_loss_fn(recon, batch_data).sum(dim=2)  # shape [B, max_seq_len]
            # zero out padded positions
            mse_per_step = mse_per_step.masked_fill(pad_mask, 0.0)
            # average over actual (non-pad) positions
            rec_loss = mse_per_step.sum() / (~pad_mask).sum()  # or 'mean over entire batch'

            kl = calc_kl(mu, logvar, reduction='mean')
            loss = rec_loss + beta * kl

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

            epoch_rec_loss += rec_loss.item()
            epoch_kl_loss += kl.item()
            epoch_total += loss.item()

        scheduler.step()
        num_batches = len(loader)
        epoch_rec_loss /= num_batches
        epoch_kl_loss  /= num_batches
        epoch_total    /= num_batches

        rec_loss_log.append(epoch_rec_loss)
        kl_loss_log.append(epoch_kl_loss)
        total_loss_log.append(epoch_total)

        elapsed = time.time() - start_time
        print(
          f"Epoch [{epoch}/{num_epochs}] | Time: {elapsed:.2f}s | "
          f"Rec: {epoch_rec_loss:5.4f} | KL: {epoch_kl_loss:5.4f} | "
          f"Beta: {beta:.4f} | Total: {epoch_total:.4f}"
        )

        # save model
        print("\n")
        if epoch % 10 == 0 or epoch == num_epochs:
            print('Saving model ...\n')
            if not os.path.isdir('checkpoints'):
                os.mkdir('checkpoints')
            path = f'./checkpoints/TransformerVAE_epoch_{epoch}.pth'

            torch.save(model.state_dict(), path)

    # Plot losses
    plt.figure(figsize=(12, 4))
    plt.subplot(1,3,1)
    plt.plot(rec_loss_log, label="MSE")
    plt.title("Reconstruction Loss")

    plt.subplot(1,3,2)
    plt.plot(kl_loss_log, label="KL")
    plt.title("KL Loss")

    plt.subplot(1,3,3)
    plt.plot(total_loss_log, label="Total")
    plt.title("Total VAE Loss")

    plt.tight_layout()
    plt.show()

    return model

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)


    # Train
    model = train_TransformerVAE(
        input_dim=25,
        e_dim=128,
        z_dim=10,
        nheads=2,
        nTElayers=2,
        nTDlayers=2,
        ff_dim=256,
        num_epochs=100,
        batch_size=32,
        lr=1e-3,
        beta_min=0.006,
        beta_max=0.04,
        beta_warmup=2,
        beta_period=8,
        device=device,
        seed=42
    )


    # Evaluate
    dataset = MiniGridDataset(trajectory_paths=paths)
    loader = DataLoader(dataset, batch_size=32, shuffle=False, collate_fn=collate_fn)

    trained_model = TransformerVAE(input_dim=25,
                           output_dim=25,
                           e_dim=128,
                           z_dim=10,
                           nheads=4,
                           ff_dim=256,
                           nTElayers=2,
                           nTDlayers=2).to(device)
    path = './checkpoints/TransformerVAE_epoch_50.pth'
    trained_model.load_state_dict(torch.load(path, map_location=torch.device(device)))
    trained_model.eval()

    with torch.no_grad():
        for batch_idx, (batch_data, labels, lengths) in enumerate(loader):
            batch_data = batch_data.float().to(device)
            pad_mask = create_padding_mask(batch_data, pad_val=0.0).to(device)
            tgt_mask = None
            _, _, recon_batch = trained_model(batch_data, pad_mask, tgt_mask)

            for i in range(batch_data.size(0)):
                true_length = lengths[i].item()
                # Trim the padded sequences to the true length
                input_seq = batch_data[i, :true_length, :].cpu().numpy()
                recon_seq = recon_batch[i, :true_length, :].cpu().numpy()


                print(f"=== Sample {batch_idx * loader.batch_size + i} ===")
                print("Input sequence:")
                print(input_seq)
                print("Reconstructed sequence:")
                print(recon_seq)
                print("\n")
                exit(0)