import torch.nn

from dataset_utils.minigrid_vae_dataset import MiniGridDataset, collate_fn
from trajectory_embedding.style_dec_vae.lstm.style_vae import LSTMVAE, lstm_validate
from trajectory_embedding.style_dec_vae.transformer.style_transformer_vae import TransformerVAE, TransformerVAEConfig, \
    transformer_validate
from trajectory_embedding.style_dec_vae.utils.utils import cluster_accuracy
import numpy as np
torch.autograd.set_detect_anomaly(True)


def evaluate(model, val_dataloader, epoch=None):
    gtruth = []
    predicted = []
    Z = []
    model.eval()
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
            iters += 1

        gtruth = np.concatenate(gtruth, 0)
        predicted = np.concatenate(predicted, 0)
        Z = np.concatenate(Z, 0)

        print('accuracy p(c|z): {:0.4f}'.format(cluster_accuracy(predicted, gtruth)[0] * 100))
        # print("pi", model.pi.data)


        # plot the clusters during training
        # if epoch is not None and epoch % 50 == 0:
        #     plot_embeddings(gtruth, Z, predicted)
        return predicted, Z

def predict_clusters_vae(vae_model_type, model_path, model_parameters, dataset_paths, dataset_parameters, batch_size=None):
    if vae_model_type == 'lstm':
        return predict_clusters_lstm_vae(model_path, model_parameters, dataset_paths, dataset_parameters, batch_size)
    elif vae_model_type == 'transformer':
        return predict_clusters_transformer_vae(model_path, model_parameters, dataset_paths, dataset_parameters, batch_size)
    else:
        raise ValueError(f"Unsupported model type: {vae_model_type}")


def predict_clusters_lstm_vae(model_path, model_parameters, dataset_paths, dataset_parameters, batch_size=None):

    trajectory_data_set = MiniGridDataset(trajectory_paths=dataset_paths, **dataset_parameters)
    data_loader = torch.utils.data.DataLoader(
        dataset=trajectory_data_set, batch_size=batch_size, shuffle=False, collate_fn=collate_fn
    )
    # define LSTM-based VAE model
    model = LSTMVAE(**model_parameters)
    predicted_labels, Z, cluster_centroids = lstm_validate(model, data_loader, load_model=True, model_path=model_path)


    # TODO remove! just testing a one-hot prompts instead of soft prompts
    # num_clusters = 3
    # Z = np.eye(num_clusters)[predicted_labels]
    # cluster_centroids = [Z[0], Z[1999], Z[2999]]
    # print(cluster_centroids)

    return predicted_labels, Z, cluster_centroids

def predict_clusters_transformer_vae(model_path, model_parameters, dataset_paths, dataset_parameters, batch_size=None):

    trajectory_data_set = MiniGridDataset(trajectory_paths=dataset_paths, **dataset_parameters)
    data_loader = torch.utils.data.DataLoader(
        dataset=trajectory_data_set, batch_size=batch_size, shuffle=False, collate_fn=collate_fn
    )
    max_len = max(trajectory_data_set.seq_lens)

    # model configuration
    cfg = TransformerVAEConfig(
        max_len=max_len,
        in_dim=9 + 1,
        d_model=128,
        nhead=8,
        num_enc_layers=4,
        num_dec_layers=4,
        dim_ff=512,
        z_dim=16,
        mem_tokens=4,
        pos_max_len=max_len,
    )
    # define LSTM-based VAE model
    model = TransformerVAE(cfg)
    predicted_labels, Z, cluster_centroids = transformer_validate(model, data_loader, load_model=True, model_path=model_path)
    return predicted_labels, Z, cluster_centroids

