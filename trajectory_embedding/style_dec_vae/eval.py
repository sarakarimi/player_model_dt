import torch.nn
from torch.utils.data import DataLoader

from trajectory_embedding.style_dec_vae.style_dec import ClusteringBasedVAE
from trajectory_embedding.style_dec_vae.utils.dataset import MiniGridDataset, collate_fn
from trajectory_embedding.style_dec_vae.utils.utils import plot_embeddings, cluster_accuracy
from trajectory_embedding.style_dec_vae.config import *
import numpy as np
torch.autograd.set_detect_anomaly(True)


def evaluate(model, val_dataloader):
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
        # plot_embeddings(gtruth, Z, predicted)
        return predicted, Z

def predict_clusters():
    dec_cluster = ClusteringBasedVAE(n_clusters, dimensions, alpha, **model_params)
    if torch.cuda.is_available():
        print('Cuda is available')
        dec_cluster = dec_cluster.cuda()
    else:
        print('No GPU')

    trajectory_data_set = MiniGridDataset(trajectory_paths=paths)
    gen_dataloader = torch.utils.data.DataLoader(
        dataset=trajectory_data_set, batch_size=batch_size, shuffle=False, collate_fn=collate_fn
    )

    dec_cluster.load_state_dict(torch.load(path_to_model))
    predicted, Z = evaluate(dec_cluster, gen_dataloader)
    return predicted, Z
