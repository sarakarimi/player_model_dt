import itertools
import os
from time import strftime, gmtime
import torch.nn
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans
from torch.utils.data import DataLoader
from style_vae import *
from style_dec import *
from trajectory_embedding.style_dec_vae.eval import evaluate
from trajectory_embedding.style_dec_vae.utils.dataset import MiniGridDataset, collate_fn
from trajectory_embedding.style_dec_vae.utils.utils import cluster_accuracy
import torch.nn.functional as F
from config import *
torch.autograd.set_detect_anomaly(True)
pretrained_save_path = 'model/pretrained/model.pt'


def pretrain(model: ClusteringBasedVAE, train_dataloader, val_dataloader, **params):
    if os.path.exists(pretrained_save_path):
        print("here")
        model.load_state_dict(torch.load(pretrained_save_path))

        model.eval()
        loss_fn = MiniGridLoss()
        total_loss = 0.0
        iters = 0
        train_iterator = tqdm(
            train_dataloader, total=len(train_dataloader)
        )

        for batch_data, labels, lengths in train_iterator:
            x = batch_data.to(torch.float32).to(model.device)
            lengths = lengths.to(torch.int64)  # .to(device)

            h_0 = torch.zeros(1, train_dataloader.batch_size, model.hidden_size).to(model.device)
            c_0 = torch.zeros(1, train_dataloader.batch_size, model.hidden_size).to(model.device)
            hidden_enc = (h_0, c_0)

            # Forward pass
            z, z_mu, _, hidden_enc = model.encoder(x, lengths, hidden_enc)
            x_decoded, _ = model.decoder(z, lengths)

            # calculate  reconstruction loss
            recons_loss = 0
            for i, length in enumerate(lengths):
                recons_loss += loss_fn(x_decoded[i, :length], x[i, :length])
            recons_loss /= len(lengths)

            total_loss += recons_loss.detach().cpu().numpy()
            iters += 1

        print('VAE resconstruction loss: ', total_loss / iters)

        # print("pi", model.pi.data)
        # print("mean", model.mu_c.data)
        # print("sigma", model.log_sigma_c.data)

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
        model.train()

        total_loss = 0.0
        iters = 0
        train_iterator = tqdm(
            train_dataloader, total=len(train_dataloader), desc="training"
        )

        for batch_data, labels, lengths in train_iterator:
            x = batch_data.to(torch.float32).to(device)
            lengths = lengths.to(torch.int64)  # .to(device)


            h_0 = torch.zeros(1, train_dataloader.batch_size, model.hidden_size).to(device)
            c_0 = torch.zeros(1, train_dataloader.batch_size, model.hidden_size).to(device)
            hidden_enc = (h_0, c_0)

            # Forward pass
            z, z_mu, _, hidden_enc = model.encoder(x, lengths, hidden_enc)
            x_decoded, _ = model.decoder(z_mu, lengths)
            # hidden_enc = hidden_enc[0].detach(), hidden_enc[1].detach()

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

    model.eval()
    Z = []
    Y = []
    with torch.no_grad():
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

    # kmean = KMeans(model.n_centroids, random_state=0)
    # kmean.fit_predict(Z)
    # model.mu_c.data = torch.from_numpy(kmean.cluster_centers_).to(device).float()

    gmm = GaussianMixture(n_components=model.n_centroids, random_state=100, max_iter=1000, covariance_type='diag')
    predict = gmm.fit_predict(Z)

    print('Accuracy = {:.4f}%'.format(cluster_accuracy(predict, Y)[0] * 100))

    model.mu_c.data = torch.from_numpy(gmm.means_).to(device).float()
    model.log_sigma_c.data = torch.log(torch.from_numpy(gmm.covariances_ + 1e-10).to(device).float())
    # model.pi.data = torch.from_numpy(gmm.weights_).to(device).float()
    print("pi", model.pi)
    print("mean", model.mu_c)
    print("sigma", model.log_sigma_c)

    torch.save(model.state_dict(), pretrained_save_path)


def train(model, train_dataloader, val_dataloader, **params):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)
    steplr = torch.optim.lr_scheduler.StepLR(optimizer, 10, 0.95)
    num_epochs = params.get('epochs', 10)
    save_path = params.get('save_path', 'output/model')

    if not os.path.exists(save_path):
        os.makedirs(save_path)


    for epoch in range(num_epochs):
        model.train()
        train_iters = 0
        total_loss, total_rec, total_kl1, total_kl2 = 0.0, 0.0, 0.0, 0.0

        train_iterator = tqdm(
            train_dataloader, total=len(train_dataloader), desc="training"
        )

        # for i, data in enumerate(train_dataloader):
        for data in train_iterator:
            h_0 = torch.zeros(1, train_dataloader.batch_size, model.hidden_size).to(model.device)
            c_0 = torch.zeros(1, train_dataloader.batch_size, model.hidden_size).to(model.device)
            hidden_enc = (h_0, c_0)

            x = data[0]
            lengths = data[2]

            x = x.to(torch.float32).to(model.device)
            lengths = lengths.to(torch.int64)#.to(model.device)

            # Acquire the loss
            loss, rec_loss, kl_1, kl_2, hidden_enc = model.elbo_loss(x, lengths, hidden_enc, 1)

            # Calculate gradients
            optimizer.zero_grad()
            loss.backward()
            # torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)

            optimizer.step()
            steplr.step()



            total_loss += loss.detach().cpu().numpy()
            total_rec += rec_loss.detach().cpu().numpy()
            total_kl1 += kl_1.detach().cpu().numpy()
            total_kl2 += kl_2.detach().cpu().numpy()
            train_iters += 1


        print('Training loss: ', total_loss / train_iters)
        print('Training rec loss: ', total_rec / train_iters)
        print('Training kl1 loss: ', total_kl1 / train_iters)
        print('Training kl2 loss: ', total_kl2 / train_iters)

        # evaluate the model
        evaluate(model, val_dataloader)
        torch.save(model.state_dict(), os.path.join(save_path, 'vae-dec-model-{}'
                                                    .format(strftime("%Y-%m-%d-%H-%M", gmtime())
                                                            )))
    torch.save(model.state_dict(), os.path.join(save_path, 'vae-dec-model-{}'
                                                       .format(strftime("%Y-%m-%d-%H-%M", gmtime())
                                                               )))



if __name__ == '__main__':

    dec_cluster = ClusteringBasedVAE(n_clusters, dimensions, alpha, **model_params)

    if torch.cuda.is_available():
        print('Cuda is available')
        dec_cluster = dec_cluster.cuda()
    else:
        print('No GPU')


    trajectory_data_set = MiniGridDataset(trajectory_paths=paths)
    gen_dataloader = torch.utils.data.DataLoader(
        dataset=trajectory_data_set, batch_size=batch_size, shuffle=True, collate_fn=collate_fn
    )

    # first pretrain the VAE with reconstruction loss
    # pretrain(dec_cluster, gen_dataloader, gen_dataloader, **model_params)
    # train(dec_cluster, gen_dataloader, gen_dataloader, **model_params)

    # Evaluate trained model
    dec_cluster.load_state_dict(torch.load("output/model/vae-dec-model-2025-02-08-16-00"))
    predicted, Z = evaluate(dec_cluster, gen_dataloader)



