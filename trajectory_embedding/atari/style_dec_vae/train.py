import os
from time import strftime, gmtime

import numpy as np
import torch.nn
from torch.utils.data import DataLoader
import itertools
from style_vae import *
from style_dec import *
from utils.utils import *


pretrained_save_path = 'model/pretrained/model.pt'


def pretrain(model: ClusteringBasedVAE, train_dataloader, val_dataloader, **params):
    if os.path.exists(pretrained_save_path):
        model.load_state_dict(torch.load(pretrained_save_path))
        return
    else:
        os.makedirs(os.path.dirname(pretrained_save_path))

    num_pretrained_epoch = params.get('pretrained_epochs', 10)
    save_path = params.get('save_path', 'output/model')
    dataset_name = params.get('dataset_name', '')
    device = torch.device('cuda' if (torch.cuda.is_available()) else 'cpu')
    model = model.to(device)

    # res_loss = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(itertools.chain(model.encoder.parameters(),
                                                 model.decoder.parameters()), lr=0.002)
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
        for batch_data, labels, lengths in train_iterator:
            # print(batch_data.shape, labels.shape, lengths.shape)
            x = batch_data.to(torch.float32).to(device)
            lengths = lengths.to(torch.int64) #.to(device)
            # Forward pass
            _, z_mu, _, _ = model.encoder(x, lengths)
            x_decoded, _ = model.decoder(z_mu, lengths)
            # loss = res_loss(x_decoded, x)

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
        for x, labels, lengths in val_dataloader:
            x = x.to(torch.float32).to(device)
            labels = labels.to(device)
            lengths = lengths.to(torch.int64) #.to(device)
            # print("x", x.shape, lengths.shape, labels.shape)


            z, mu, log_var, _ = model.encoder(x, lengths)
            assert F.mse_loss(mu, log_var) == 0
            Z.append(mu)
            Y.append(labels)
    # print(mu.shape)
    # print(len(Z))
    Z = torch.cat(Z, 0).detach().cpu().numpy()
    Y = torch.cat(Y, 0).to(torch.int32).detach().cpu().numpy()
    # print(Z.shape)
    # print(Z)
    gmm = GaussianMixture(n_components=model.n_centroids, max_iter=10000, n_init=100, reg_covar=1e-5, covariance_type='diag')
    predict = gmm.fit_predict(Z)

    print('Accuracy = {:.4f}%'.format(cluster_accuracy(predict, Y)[0] * 100))

    model.mu_c.data = torch.from_numpy(gmm.means_).to(device).float()
    model.log_sigma_c.data = torch.log(torch.from_numpy(gmm.covariances_).to(device).float())
    model.pi.data = torch.from_numpy(gmm.weights_).to(device).float()
    print("pi", model.pi)
    print("mean", model.mu_c)
    print("sigma", model.log_sigma_c)

    torch.save(model.state_dict(), pretrained_save_path)


def train(model, train_dataloader, val_dataloader, **params):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.002, eps=1e-4)
    # optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    steplr = torch.optim.lr_scheduler.StepLR(optimizer, 10, 0.95)
    num_epochs = params.get('epochs', 10)
    num_pretrained_epoch = params.get('pretrained_epochs', 10)
    save_path = params.get('save_path', 'output/model')
    dataset_name = params.get('dataset_name', '')

    if not os.path.exists(save_path):
        os.makedirs(save_path)


    for epoch in range(num_epochs):
        train_iters = 0
        total_loss = 0.0

        for i, data in enumerate(train_dataloader):
            steplr.step()
            # model.zero_grad()

            x = data[0]
            lengths = data[2]

            x = x.to(torch.float32).to(model.device)
            lengths = lengths.to(torch.int64)#.to(model.device)

            # Acquire the loss
            loss = model.elbo_loss(x, lengths, 1)

            # Calculate gradients
            optimizer.zero_grad()
            loss.backward()

            # torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)

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
            # Get z value
            x = data[0].to(torch.float32).to(model.device)
            lengths = data[2].to(torch.int64)  # .to(model.device)
            labels = data[1].to(torch.int32).cpu().detach().numpy()

            # x_decoded, latent, z_mean, z_log_var, gamma = model(x)
            gamma, z = model(x, lengths)
            # z, mu, log_var, _ = model.encoder(x, lengths)
            # Z.append(z.cpu().detach().numpy())
            gtruth.append(labels)

            # Cluster the latent space
            sample = np.argmax(gamma.cpu().detach().numpy(), axis=1)
            predicted.append(sample)
            # print(sample)
            # print(model.mu_c, sample)
            # print(model.mu_c[sample])
            # mean_accuracy += cluster_accuracy(sample, labels)[0]
            iters += 1

        gtruth = np.concatenate(gtruth, 0)
        predicted = np.concatenate(predicted, 0)
        # Z = np.concatenate(Z, 0)

        print('accuracy p(c|z): {:0.4f}'.format(cluster_accuracy(predicted, gtruth)[0] * 100))

        # plot the clusters during training
        # plot_embeddings(model, gtruth, Z, predicted)


if __name__ == '__main__':
    dimensions = [500, 128, 10]
    model_params = {
        'decoder_final_activation': 'relu',
        'pretrained_epochs': 50,
        'epochs': 100,
        'save_path': 'output/model'
    }

    # Hyperparameters
    # num_epochs = 1000  # 10000
    batch_size = 512
    # max_seq_len = 4
    # feature_size = 2
    # num_samples = 200 #30
    # cluster_params = [(0, 1), (5, 1)]  # Cluster 0 has mean 0, std 1, Cluster 1 has mean 5, std 1

    dec_cluster = ClusteringBasedVAE(2, dimensions, 1, **model_params)

    if torch.cuda.is_available():
        print('Cuda is available')
        dec_cluster = dec_cluster.cuda()
    else:
        print('No GPU')
    # Fake data
    # sequences, labels = generate_varied_length_data(num_samples, max_seq_len, feature_size, cluster_params)
    # print(sequences, labels)

    # Create dataset and data loaders
    # train_set = VariedLengthDataset(sequences, labels)

    # convert to format of data loader
    # gen_dataloader = torch.utils.data.DataLoader(
    #     dataset=train_set, batch_size=batch_size, shuffle=True, collate_fn=collate_fn
    # )

    paths = ["/home/sara/repositories/player_model_dt/data/new_implementation_datasets/PPO_trajectories_mode1.gz",
             "/home/sara/repositories/player_model_dt/data/new_implementation_datasets/PPO_trajectories_mode2.gz"
             ]
    trajectory_data_set = MiniGridDataset(trajectory_paths=paths)
    gen_dataloader = torch.utils.data.DataLoader(
        dataset=trajectory_data_set, batch_size=batch_size, shuffle=True, collate_fn=collate_fn
    )


    pretrain(dec_cluster, gen_dataloader, gen_dataloader, **model_params)
    train(dec_cluster, gen_dataloader, gen_dataloader, **model_params)
