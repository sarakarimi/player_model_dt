import itertools
import os
from time import strftime, gmtime
import torch.nn
from sklearn.mixture import GaussianMixture
from torch.utils.data import DataLoader
from style_vae import *
from style_dec import *
from trajectory_embedding.style_dec_vae.utils.dataset import MiniGridDataset, collate_fn
from trajectory_embedding.style_dec_vae.utils.utils import plot_embeddings, cluster_accuracy
import torch.nn.functional as F
torch.autograd.set_detect_anomaly(True)
pretrained_save_path = 'model/pretrained/model.pt'


def pretrain(model: ClusteringBasedVAE, train_dataloader, val_dataloader, **params):
    if os.path.exists(pretrained_save_path):
        model.load_state_dict(torch.load(pretrained_save_path))
        return
    else:
        os.makedirs(os.path.dirname(pretrained_save_path))

    num_pretrained_epoch = params.get('pretrained_epochs', 10)
    save_path = params.get('save_path', 'output/model')
    device = torch.device('cuda' if (torch.cuda.is_available()) else 'cpu')
    model = model.to(device)

    # res_loss = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(itertools.chain(model.encoder.parameters(),
                                                 model.decoder.parameters()), lr=0.001)
    # steplr = torch.optim.lr_scheduler.StepLR(optimizer, 5, 0.9)

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    loss_fn = MiniGridLoss()

    print('Pretrains VAE using only reconstruction loss...')
    for pre_epoch in range(num_pretrained_epoch):
        total_loss = 0.0
        iters = 0
        train_iterator = tqdm(
            train_dataloader, total=len(train_dataloader), desc="training"
        )

        # h_0 = torch.zeros(1, train_dataloader.batch_size, model.hidden_size).to(device)
        # c_0 = torch.zeros(1, train_dataloader.batch_size, model.hidden_size).to(device)
        # hidden_enc = (h_0, c_0)

        for batch_data, labels, lengths in train_iterator:
            h_0 = torch.zeros(1, train_dataloader.batch_size, model.hidden_size).to(device)
            c_0 = torch.zeros(1, train_dataloader.batch_size, model.hidden_size).to(device)
            hidden_enc = (h_0, c_0)
            # print(batch_data.shape, labels, lengths)
            x = batch_data.to(torch.float32).to(device)
            lengths = lengths.to(torch.int64) #.to(device)
            # Forward pass
            _, z_mu, _, hidden_enc = model.encoder(x, lengths, hidden_enc)
            x_decoded, _ = model.decoder(z_mu, lengths)
            # hidden_enc = hidden_enc[0].detach(), hidden_enc[1].detach()
            # loss = res_loss(x_decoded, x)

            # TODO they  use MSE loss in pretrain and crossEntropy in train steps! why???
            # calculate  reconstruction loss
            recons_loss = 0
            for i, length in enumerate(lengths):
                # recons_loss += nn.functional.mse_loss(x_decoded[i, :length], x[i, :length], reduction='sum')
                recons_loss += loss_fn(x_decoded[i, :length], x[i, :length])

            recons_loss /= len(lengths)

            total_loss += recons_loss.detach().cpu().numpy()

            # Calculate gradient and optimize
            optimizer.zero_grad()
            recons_loss.backward()
            optimizer.step()

            iters += 1

        print('VAE resconstruction loss: ', total_loss / iters)
        # steplr.step()

    model.encoder.sampling.log_var.load_state_dict(model.encoder.sampling.mu.state_dict())

    Z = []
    Y = []
    with torch.no_grad():
        # h_0 = torch.zeros(1, train_dataloader.batch_size, model.hidden_size).to(device)
        # c_0 = torch.zeros(1, train_dataloader.batch_size, model.hidden_size).to(device)
        # hidden_enc = (h_0, c_0)
        for x, labels, lengths in val_dataloader:
            h_0 = torch.zeros(1, train_dataloader.batch_size, model.hidden_size).to(device)
            c_0 = torch.zeros(1, train_dataloader.batch_size, model.hidden_size).to(device)
            hidden_enc = (h_0, c_0)
            x = x.to(torch.float32).to(device)
            labels = labels.to(device)
            lengths = lengths.to(torch.int64) #.to(device)


            z, mu, log_var, _ = model.encoder(x, lengths, hidden_enc)
            # hidden_enc = hidden_enc[0].detach(), hidden_enc[1].detach()

            assert F.mse_loss(mu, log_var) == 0
            Z.append(mu)
            Y.append(labels)

    Z = torch.cat(Z, 0).detach().cpu().numpy()
    Y = torch.cat(Y, 0).to(torch.int32).detach().cpu().numpy()
    gmm = GaussianMixture(n_components=model.n_centroids, n_init=100, reg_covar=1e-5, covariance_type='diag')
    predict = gmm.fit_predict(Z)
    # print(Z)
    # print(predict, Y)

    print('Accuracy = {:.4f}%'.format(cluster_accuracy(predict, Y)[0] * 100))

    model.mu_c.data = torch.from_numpy(gmm.means_).to(device).float()
    model.log_sigma_c.data = torch.log(torch.from_numpy(gmm.covariances_).to(device).float())
    model.pi.data = torch.from_numpy(gmm.weights_).to(device).float()
    print("pi", model.pi)
    print("mean", model.mu_c)
    print("sigma", model.log_sigma_c)

    torch.save(model.state_dict(), pretrained_save_path)
    # return  model

def train(model, train_dataloader, val_dataloader, **params):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    steplr = torch.optim.lr_scheduler.StepLR(optimizer, 10, 0.95)
    num_epochs = params.get('epochs', 10)
    save_path = params.get('save_path', 'output/model')

    if not os.path.exists(save_path):
        os.makedirs(save_path)


    for epoch in range(num_epochs):
        train_iters = 0
        total_loss = 0.0

        for i, data in enumerate(train_dataloader):
            h_0 = torch.zeros(1, train_dataloader.batch_size, model.hidden_size).to(model.device)
            c_0 = torch.zeros(1, train_dataloader.batch_size, model.hidden_size).to(model.device)
            hidden_enc = (h_0, c_0)
            steplr.step()
            model.zero_grad()

            x = data[0]
            lengths = data[2]

            x = x.to(torch.float32).to(model.device)
            lengths = lengths.to(torch.int64)#.to(model.device)

            # Acquire the loss
            loss, hidden_enc = model.elbo_loss(x, lengths, hidden_enc, 1)

            # Calculate gradients
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1)

            # Update models
            optimizer.step()

            train_iters += 1

            total_loss += loss.detach().cpu().numpy()

        print('Training loss: ', total_loss / train_iters)


        # evaluate the model
        eval(model, val_dataloader)

    torch.save(model.state_dict(), os.path.join(save_path, 'vae-dec-model-{}'
                                                       .format(strftime("%Y-%m-%d-%H-%M", gmtime())
                                                               )))


def eval(model, val_dataloader):
    gtruth = []
    predicted = []
    Z = []

    # For each epoch, log the p_c_z accuracy
    with torch.no_grad():
        mean_accuracy = 0.0
        iters = 0

        for i, data in enumerate(val_dataloader):
            h_0 = torch.zeros(1, val_dataloader.batch_size, model.hidden_size).to(model.device)
            c_0 = torch.zeros(1, val_dataloader.batch_size, model.hidden_size).to(model.device)
            hidden_enc = (h_0, c_0)

            # Get z value
            x = data[0].to(torch.float32).to(model.device)
            lengths = data[2].to(torch.int64)  # .to(model.device)
            labels = data[1].to(torch.int32).cpu().detach().numpy()

            gamma, z, _ = model(x, lengths, hidden_enc)

            # Cluster the latent space
            sample = np.argmax(gamma.cpu().detach().numpy(), axis=1)
            predicted.append(sample)
            Z.append(z.cpu().detach().numpy())
            gtruth.append(labels)
            # print(model.mu_c, sample)
            # mean_accuracy += cluster_accuracy(sample, labels)[0]
            iters += 1

        gtruth = np.concatenate(gtruth, 0)
        predicted = np.concatenate(predicted, 0)
        Z = np.concatenate(Z, 0)

        print('accuracy p(c|z): {:0.4f}'.format(cluster_accuracy(predicted, gtruth)[0] * 100))

        # plot the clusters during training
        plot_embeddings(gtruth, Z, predicted)


if __name__ == '__main__':
    # Hyperparameters
    dimensions = [500, 256, 16] # it worked with 64!
    model_params = {
        'decoder_final_activation': 'relu',
        'pretrained_epochs': 1,
        'epochs': 1,
        'save_path': 'output/model'
    }
    batch_size = 64

    dec_cluster = ClusteringBasedVAE(4, dimensions, 1, **model_params)

    if torch.cuda.is_available():
        print('Cuda is available')
        dec_cluster = dec_cluster.cuda()
    else:
        print('No GPU')

    paths = [
        "../datasets/minigrid/PPO_trajectories_goal0.gz",
        "../datasets/minigrid/PPO_trajectories_goal3.gz",
        "../datasets/minigrid/PPO_trajectories_goal5.gz",
        "../datasets/minigrid/PPO_trajectories_goal6.gz",
             ]
    trajectory_data_set = MiniGridDataset(trajectory_paths=paths)
    gen_dataloader = torch.utils.data.DataLoader(
        dataset=trajectory_data_set, batch_size=batch_size, shuffle=True, collate_fn=collate_fn
    )

    # first pretrain the VAE with reconstruction loss
    pretrain(dec_cluster, gen_dataloader, gen_dataloader, **model_params)
    train(dec_cluster, gen_dataloader, gen_dataloader, **model_params)
