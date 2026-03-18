import torch
import numpy as np
import torch.nn.functional as F
from sklearn.mixture import GaussianMixture
from style_transformer_dec import TransformerVaDE, TransformerAE
# from trajectory_embedding.style_dec_vae.config import paths
from style_decision_transformer.transformer.style_transformer_vae import create_padding_mask
# from trajectory_embedding.style_dec_vae.utils.dataset import MiniGridDataset, collate_fn
from style_decision_transformer.utils import MujocoDataset, collate_fn

from style_decision_transformer.utils import cluster_accuracy, plot_embeddings
import argparse
import torch.utils.data
from torch import nn


def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Linear") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)

def weights_init_xavier(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)

class TrainerVaDE:
    def __init__(self, args, device, dataloader):
        self.autoencoder = TransformerAE(input_dim=20, output_dim=20,
                                         e_dim=128, z_dim=10, nheads=2, ff_dim=256,
                                         nTElayers=2, nTDlayers=2).to(device)
        self.VaDE = TransformerVaDE(input_dim=20, output_dim=20,
                                    e_dim=128, z_dim=10, nheads=2, ff_dim=256,
                                    nTElayers=2, nTDlayers=2, n_classes=2).to(device)
        self.dataloader = dataloader
        self.device = device
        self.args = args

    def pretrain(self):
        mse_loss_fn = nn.MSELoss(reduction='none') #nn.BCELoss(reduction='none')
        # optimizer = optim.Adam(self.autoencoder.parameters(), lr=0.001)
        optimizer = torch.optim.SGD(self.autoencoder.parameters(), lr=0.1, momentum=0.9)  # torch.optim.Adam(model.parameters(), lr=lr)
        # scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[20, ], gamma=0.1)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 20, 0.1)


        # self.autoencoder.apply(weights_init_xavier)  # intializing weights using normal distribution.
        self.autoencoder.train()
        print('Training the autoencoder...')
        for epoch in range(self.args.pretrain_epochs):
            total_loss = 0

            for batch_data, labels, lengths in self.dataloader:
                batch_data = batch_data.to(self.device)
                # Create padding mask
                pad_mask, _ = create_padding_mask(batch_data, pad_token=0.0)#.to(self.device)
                tgt_mask = None

                # Forward
                mu, logvar, recon = self.autoencoder(batch_data, pad_mask, tgt_mask)
                mse_per_step = mse_loss_fn(recon, batch_data).sum(dim=2)
                mse_per_step = mse_per_step.masked_fill(pad_mask, 0.0)
                loss = mse_per_step.sum() / (~pad_mask).sum()


                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.autoencoder.parameters(), 5.0)
                optimizer.step()
                total_loss += loss.item()
            scheduler.step()
            print(f'Pretrain Epoch: {epoch}, Loss: {total_loss/len(self.dataloader):.6f}')
        self.train_GMM()  # training a GMM for initialize the VaDE
        self.save_weights_for_VaDE() # saving weights for the VaDE

    def test_AE(self):
        # self.autoencoder.eval()
        self.VaDE.eval()
        with torch.no_grad():
            for batch_data, labels, lengths in self.dataloader:
                batch_data = batch_data.to(self.device)
                pad_mask, _ = create_padding_mask(batch_data, pad_token=0.0)#.to(self.device)
                tgt_mask = None
                # _, _, recon_batch = self.autoencoder(batch_data, pad_mask, tgt_mask)
                recon_batch, _, _, _ = self.VaDE(batch_data, pad_mask, tgt_mask)
                for i in range(batch_data.size(0)):
                    true_length = lengths[i].item()
                    input_seq = batch_data[i, :true_length, :].cpu().numpy()
                    recon_seq = recon_batch[i, :true_length, :].cpu().numpy()

                    print("Input sequence:")
                    print(input_seq)
                    print("Reconstructed sequence:")
                    print(np.around(np.array(recon_seq), 2)
                          )
                    print("\n")
                    break

    def train_GMM(self):
        self.autoencoder.eval()
        # self.VaDE.eval()
        Z, y_true = [], []
        print('Fiting Gaussian Mixture Model...')
        with torch.no_grad():
            for batch_data, labels, lengths in self.dataloader:
                batch_data = batch_data.to(self.device)
                pad_mask, _ = create_padding_mask(batch_data, pad_token=0.0)#.to(self.device)
                z = self.autoencoder.encode(batch_data, pad_mask)
                # z = self.VaDE.encode(batch_data, pad_mask)
                Z.append(z)

                y_true.extend(labels.numpy())
                # Z.append(z.cpu().detach().numpy())

        Z = torch.cat(Z, 0).cpu().numpy()
        self.gmm = GaussianMixture(n_components=2, covariance_type='diag', max_iter=1000, random_state=100)

        preds = self.gmm.fit_predict(Z)
        acc = cluster_accuracy(np.array(preds), np.array(y_true))
        print('Testing AE... Acc: {}'.format( acc[0]*100))
        plot_embeddings(y_true, Z, preds)

    def save_weights_for_VaDE(self):
        print('Saving weights.')
        # torch.save(self.autoencoder.state_dict(), self.args.trained_ae_path)
        state_dict = self.autoencoder.state_dict()

        self.VaDE.load_state_dict(state_dict, strict=False)
        self.VaDE.pi_prior.data = torch.from_numpy(self.gmm.weights_).float().to(self.device)
        self.VaDE.mu_prior.data = torch.from_numpy(self.gmm.means_).float().to(self.device)
        self.VaDE.log_var_prior.data = torch.log(torch.from_numpy(self.gmm.covariances_)).float().to(self.device)
        torch.save(self.VaDE.state_dict(), self.args.trained_vade_path)

    def train(self):
        if self.args.pretrain == True:
            self.VaDE.load_state_dict(torch.load(self.args.trained_vade_path,
                                                 map_location=self.device))
        else:
            self.VaDE.apply(weights_init_xavier)
        self.optimizer = torch.optim.Adam(self.VaDE.parameters(), lr=2e-3)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, 10, 0.9)
        # self.optimizer = torch.optim.SGD(self.VaDE.parameters(), lr=0.1, momentum=0.9)
        # lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=[20, ], gamma=0.1)

        print('Training VaDE...')
        for epoch in range(self.args.epochs):
            self.train_VaDE(epoch)
            # if epoch == args.pretrain_epochs:
            #     self.train_GMM()
            #     self.VaDE.pi_prior.data = torch.from_numpy(self.gmm.weights_).float().to(self.device)
            #     self.VaDE.mu_prior.data = torch.from_numpy(self.gmm.means_).float().to(self.device)
            #     self.VaDE.log_var_prior.data = torch.log(torch.from_numpy(self.gmm.covariances_)).float().to(
            #         self.device)
            #     print(self.VaDE.pi_prior.data, self.VaDE.mu_prior.data, self.VaDE.log_var_prior.data)
            #     self.test_AE()
            #
            #     self.optimizer = torch.optim.SGD(self.VaDE.parameters(), lr=args.lr, momentum=0.9)
            #     lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=[300, ], gamma=0.1)
            self.test_VaDE(epoch)
            lr_scheduler.step()

    def train_VaDE(self, epoch):
        self.VaDE.train()

        total_loss, total_recon_loss, total_kl1_loss, total_kl2_loss, total_kl3_loss, total_kl4_loss, total_kl_terms = 0, 0, 0, 0, 0, 0, 0
        len_dataloader = len(self.dataloader)
        for batch_data, labels, lengths in self.dataloader:
            self.optimizer.zero_grad()

            batch_data = batch_data.to(self.device)
            pad_mask, _ = create_padding_mask(batch_data, pad_token=0.0)#.to(self.device)
            tgt_mask = None

            x_hat, mu, log_var, z = self.VaDE(batch_data, pad_mask, tgt_mask)

            # print('Before backward: {}'.format(self.VaDE.pi_prior))
            loss, recon_loss, kl_terms, beta = self.compute_loss(batch_data, x_hat, mu, log_var, z, pad_mask, epoch)

            loss.backward()
            # nn.utils.clip_grad_norm_(self.VaDE.parameters(), 5.0)
            self.optimizer.step()

            total_loss += loss.item()
            total_recon_loss += recon_loss.item()
            total_kl_terms += kl_terms.item()
            # total_kl1_loss += kl1.item()
            # total_kl2_loss += kl2.item()
            # total_kl3_loss += kl3.item()
            # total_kl4_loss += kl4.item()

            # print('After backward: {}'.format(self.VaDE.pi_prior))
        print(
            'Training VaDE... Epoch: {}, Loss: {}, recon loss: {}, beta: {}, kl: {}'.format(
                epoch,
                total_loss / len_dataloader,
                total_recon_loss / len_dataloader,
                beta,
                total_kl_terms / len_dataloader))


        # print('Training VaDE... Epoch: {}, Loss: {}, recon loss: {}, beta: {}, kl1: {}, kl2: {}, kl3: {}, kl4: {}'.format(epoch,
        #                                                                                                         total_loss / len_dataloader,
        #                                                                                                         total_recon_loss / len_dataloader,
        #                                                                                                         beta,
        #                                                                                                         total_kl1_loss / len_dataloader,
        #                                                                                                         total_kl2_loss / len_dataloader,
        #                                                                                                         total_kl3_loss / len_dataloader,
        #                                                                                                         total_kl4_loss / len_dataloader))
        # if epoch % 20 == 0:
        #     torch.save(self.VaDE.state_dict(), self.args.trained_vade_path)

    def test_VaDE(self, epoch):
        self.VaDE.eval()
        with torch.no_grad():
            total_loss =  0
            y_true, y_pred, Z = [], [], []
            for batch_data, labels, lengths in self.dataloader:
                # self.optimizer.zero_grad()
                batch_data = batch_data.to(self.device)
                pad_mask, _ = create_padding_mask(batch_data, pad_token=0.0)#.to(self.device)
                tgt_mask = None
                x_hat, mu, log_var, z = self.VaDE(batch_data, pad_mask, tgt_mask)

                gamma = self.compute_gamma(z, self.VaDE.pi_prior)
                pred = torch.argmax(gamma, dim=1)
                y_true.extend(labels.numpy())
                y_pred.extend(pred.cpu().detach().numpy())
                Z.append(z.cpu().detach().numpy())

            acc = cluster_accuracy(np.array(y_pred), np.array(y_true))
            print(
                'Testing VaDE... Epoch: {}, Acc: {}'.format(epoch, acc[0]))
            if epoch % 5 == 0:
                Z = np.concatenate(Z, 0)
                plot_embeddings(y_true, Z, y_pred)

    def compute_loss(self, x, x_hat, mu, log_var, z, pad_mask, epoch):
        p_c = self.VaDE.pi_prior
        gamma = self.compute_gamma(z, p_c)

        # log_p_x_given_z = F.binary_cross_entropy(x_hat, x, reduction='none').sum(dim=2)
        log_p_x_given_z = F.mse_loss(x_hat, x, reduction='none').sum(dim=2)

        log_p_x_given_z = log_p_x_given_z.masked_fill(pad_mask, 0.0).sum() / (~pad_mask).sum()

        h = log_var.exp().unsqueeze(1) + (mu.unsqueeze(1) - self.VaDE.mu_prior).pow(2)
        h = torch.sum(self.VaDE.log_var_prior + h / self.VaDE.log_var_prior.exp(), dim=2)
        log_p_z_given_c = 0.5 * torch.sum(gamma * h, dim=1)
        log_p_c = torch.sum(gamma * torch.log(p_c + 1e-9), dim=1)
        log_q_c_given_x = torch.sum(gamma * torch.log(gamma + 1e-9), dim=1)
        log_q_z_given_x = 0.5 * torch.sum(1 + log_var, dim=1)

        kl_weight = 1 #self.get_beta(epoch) #min(1.0, (epoch + 1) / 1000000)
        kl_terms = log_p_z_given_c.mean() - log_p_c.mean() + log_q_c_given_x.mean() - log_q_z_given_x.mean()
        loss = log_p_x_given_z + kl_weight * kl_terms

        # loss = log_p_x_given_z + log_p_z_given_c - log_p_c + log_q_c_given_x - log_q_z_given_x
        # loss /= x.size(0)
        return loss, log_p_x_given_z, kl_weight * kl_terms , kl_weight

    def compute_gamma(self, z, p_c):
        h = (z.unsqueeze(1) - self.VaDE.mu_prior).pow(2) / self.VaDE.log_var_prior.exp()
        h += self.VaDE.log_var_prior
        h += torch.tensor(np.log(2 * np.pi)).to(self.device)
        log_p_z_c = torch.log(p_c + 1e-9).unsqueeze(0) - 0.5 * torch.sum(h, dim=2)
        log_sum = torch.logsumexp(log_p_z_c, dim=1, keepdim=True)
        gamma = torch.exp(log_p_z_c - log_sum)
        return gamma

    def get_beta(self, epoch):
        beta_min = 5e-4  # Minimum beta value
        beta_max = 0.04#1.0  # Maximum beta value

        # For the first 500 epochs, return constant beta equal to beta_min.
        if epoch < args.pretrain_epochs:
            return beta_min

        # Adjust epoch count to start cycles after 500 epochs.
        new_epoch = epoch - args.pretrain_epochs

        # Define cycle parameters.
        ramp_steps = 90  # Number of epochs for the ramp-up phase.
        plateau_steps = 10  # Number of epochs where beta remains at beta_max.
        cycle_length = ramp_steps + plateau_steps
        total_cycles = 10  # Total number of cycles.
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



def make_sequence_cluster_dataset(
        seq_len=50,
        dim=17,
        n_clusters=4,
        n_per_cluster=100,
        seed=42,
        separation=6.0,
        process_rho=0.7,
        noise_sigma=0.30,
):
        """
        Generate sequences from n_clusters distinct Gaussian AR(1) processes with well-separated means.
        Returns:
            X: (N, seq_len, dim) float32 array of sequences
            y: (N,) int labels in [0, n_clusters-1]
            mus: (n_clusters, dim) cluster means used
        """
        rng = np.random.default_rng(seed)

        # Choose orthogonal-ish mean directions to ensure separability.
        # Spread the cluster centers over distant coordinates.
        axes = np.linspace(0, dim - 1, n_clusters, dtype=int)
        mus = np.zeros((n_clusters, dim), dtype=np.float32)
        for k, ax in enumerate(axes):
            mus[k, ax] = separation
            # Small extra offsets on neighboring dims to avoid exact orthogonality (helps PCA)
            if ax + 1 < dim:
                mus[k, ax + 1] = separation * 0.25

        N = n_clusters * n_per_cluster
        X = np.zeros((N, seq_len, dim), dtype=np.float32)
        y = np.zeros(N, dtype=np.int64)

        # Slightly different per-cluster dynamics (still very separable because of centers)
        # e.g., small cluster-specific AR coefficients and diagonal noise scaling
        rhos = np.clip(process_rho + np.linspace(-0.15, 0.15, n_clusters), 0.3, 0.95)
        cov_scales = 1.0 + np.linspace(-0.3, 0.3, n_clusters)

        idx = 0
        for k in range(n_clusters):
            mu = mus[k]
            rho = rhos[k]
            sigma = noise_sigma * cov_scales[k]

            for _ in range(n_per_cluster):
                # Start near the cluster mean
                x_t = rng.normal(loc=mu, scale=sigma, size=(dim,))
                seq = [x_t.astype(np.float32)]
                for t in range(1, seq_len):
                    innovation = rng.normal(loc=mu, scale=sigma, size=(dim,))
                    x_t = rho * x_t + (1 - rho) * innovation
                    seq.append(x_t.astype(np.float32))
                X[idx] = np.stack(seq, axis=0)
                y[idx] = k
                idx += 1

        return X, y, mus


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=300,
                        help="number of iterations")
    parser.add_argument("--pretrain_epochs", type=int, default=1,
                        help="number of iterations")
    parser.add_argument("--patience", type=int, default=30,
                        help="Patience for Early Stopping")
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='learning rate')
    parser.add_argument("--batch_size", type=int, default=100,
                        help="Batch size")
    parser.add_argument('--pretrain', type=bool, default=True,
                        help='pretrain')
    parser.add_argument('--trained_ae_path', type=str, default='mujoco_model/transformer_ae.pth',
                        help='Output path')
    parser.add_argument('--trained_vade_path', type=str, default='mujoco_model/transformer_vade.pth',
                        help='Output path')
    args = parser.parse_args()
    #
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # device = torch.device('cpu')

    # trajectory_data_set = MiniGridDataset(trajectory_paths=paths)
    trajectory_data_set = MujocoDataset(trajectory_paths=paths) # MiniGridDataset(trajectory_paths=paths)
    X, y, mus = make_sequence_cluster_dataset(
        seq_len=50, dim=17, n_clusters=2, n_per_cluster=1000, seed=7,
        separation=1.0, process_rho=0.65, noise_sigma=0.48
    )
    trajectory_data_set.obs = list(X)
    trajectory_data_set.labels = list(y)

    #
    dataloader = torch.utils.data.DataLoader(
        dataset=trajectory_data_set, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn
    )
    print(len(dataloader))
    vade = TrainerVaDE(args, device, dataloader)
    if args.pretrain == True:
        vade.pretrain()
    vade.train()

    # import numpy as np

    # def flatten_sequences(X):
    #     """Flatten (N, T, D) -> (N, T*D)."""
    #     N, T, D = X.shape
    #     return X.reshape(N, T * D)


    # if __name__ == "__main__":
    #     # ----- Generate data -----
    #     X, y, mus = make_sequence_cluster_dataset(
    #         seq_len=50, dim=17, n_clusters=4, n_per_cluster=1000, seed=7,
    #         separation=1.0, process_rho=0.65, noise_sigma=0.48
    #     )
    #
    #     # X = np.asarray(trajectory_data_set.obs)
    #     # y = np.asarray(trajectory_data_set.labels)
    #     print("Data shape:", X.shape, "Labels shape:", y.shape)  # (400, 50, 17)
    #
    #     # ----- Dimensionality reduction (PCA) for viz -----
    #     X_flat = flatten_sequences(X)
    #     pca = PCA(n_components=2, random_state=0)
    #     X_2d = pca.fit_transform(X_flat)
    #
    #     # tsne = TSNE(init='pca', perplexity=30)
    #     # X_2d = tsne.fit_transform(X_flat)
    #
    #     # Plot colored by true labels (default matplotlib colors)
    #     plt.figure(figsize=(6, 5))
    #     for k in np.unique(y):
    #         sel = y == k
    #         plt.scatter(X_2d[sel, 0], X_2d[sel, 1], s=10, label=f"Cluster {k}", alpha=0.7)
    #     plt.title("PCA of Flattened Sequences (True Clusters)")
    #     plt.xlabel("PC1")
    #     plt.ylabel("PC2")
    #     plt.legend()
    #     plt.tight_layout()
    #     plt.show()
    #
    #     # ----- Clustering sanity check -----
    #     km = KMeans(n_clusters=4, n_init=20, random_state=0)
    #     pred = km.fit_predict(X_flat)
    #
    #     ari = adjusted_rand_score(y, pred)
    #     sil = silhouette_score(X_flat, pred, sample_size=min(5000, X_flat.shape[0]), random_state=0)
    #     print(f"Adjusted Rand Index (KMeans on flattened): {ari:.3f}")
    #     print(f"Silhouette score: {sil:.3f}")
    #
    #     # Also show how separable the KMeans clusters look in PCA space
    #     plt.figure(figsize=(6, 5))
    #     for k in np.unique(pred):
    #         sel = pred == k
    #         plt.scatter(X_2d[sel, 0], X_2d[sel, 1], s=10, label=f"KMeans {k}", alpha=0.7)
    #     plt.title("PCA of Flattened Sequences (KMeans Labels)")
    #     plt.xlabel("PC1")
    #     plt.ylabel("PC2")
    #     plt.legend()
    #     plt.tight_layout()
    #     plt.show()
