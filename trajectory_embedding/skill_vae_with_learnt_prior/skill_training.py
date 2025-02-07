import os
import uuid

import numpy as np

from env_wrapper import get_trajectory, get_dataset
import torch.utils.data as data_utils
# from torch.utils.tensorboard import SummaryWriter
from utils import *


def train():
    # DEFAULT PARAMS
    if 'lr' not in config:
        config['lr'] = 3e-4
    if 'weight_decay' not in config:
        config['weight_decay'] = 0
    # END DEFAULT PARAMS

    torch.manual_seed(config['seed'])
    torch.cuda.manual_seed_all(config['seed'])
    np.random.seed(config['seed'])

    exp_id = str(uuid.uuid4())
    print("the experiment id is:", exp_id)

    # summary_writer = SummaryWriter(log_dir="LOG_DIR")
    is_cuda = torch.cuda.is_available()
    # env = make_env(costume_map=True)

    # load model and resume training
    # gp_aa_model = load_ae_model(env, PATH_TO_AE_MODEL)

    # start training from scratch
    gp_aa_model = get_model(None, config['env_name'])

    if is_cuda:
        gp_aa_model = gp_aa_model.cuda()

    optimizer = torch.optim.Adam(gp_aa_model.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])

    data = get_dataset(config['dataset_path'])

    state_traj, action_traj, _ = get_trajectory(config['env_name'], config['traj_length'], dataset=data, random_start=True)
    print(len(action_traj))
    state_traj, action_traj = torch.FloatTensor(state_traj), torch.FloatTensor(action_traj)
    print("done!")

    # Clustering trajectories
    dataset = data_utils.TensorDataset(state_traj, action_traj)
    print("here")
    loader = data_utils.DataLoader(dataset, config['batch_size'])
    print("here")

    gp_aa_model.train()
    # play_policy(env, gp_aa_model, config['num_eval'], config['traj_length'], config['tanh'], is_cuda)
    i = 0
    for epoch_num in range(config['train_epochs'] + 1):
        i = i + 1
        total_loss = 0.
        total_kl_loss = 0.
        total_nll_loss = 0.
        for (idx, (state_b, action_b)) in enumerate(loader):
            if is_cuda:
                action_b = action_b.cuda()
                state_b = state_b.cuda()
            kl_loss, nll_loss = gp_aa_model.calc_loss(state_b, action_b, is_cuda)

            # Assumes config['reg'] >= 0
            if config['reg'] > 0:
                loss = nll_loss + config['reg'] * kl_loss
            else:
                loss = nll_loss
            optimizer.zero_grad()
            loss.mean().backward()
            # added gradient clipping
            torch.nn.utils.clip_grad_norm_(gp_aa_model.parameters(), 0.5)

            optimizer.step()
            total_loss += loss.mean().item()
            total_nll_loss += nll_loss.item()
            total_kl_loss += kl_loss.item()
        avg_loss = total_loss / (1 + idx)
        avg_kl_loss = total_kl_loss / (1 + idx)
        avg_nll_loss = total_nll_loss / (1 + idx)

        print(f"Avg loss for epoch {epoch_num} is {avg_loss}")
        print(f"Avg KL loss for epoch {epoch_num} is {avg_kl_loss}")
        print(f"Avg NLL loss for epoch {epoch_num} is {avg_nll_loss}")
        # summary_writer.add_scalar("avg loss", avg_loss, epoch_num)
        # summary_writer.add_scalar("avg KL loss", avg_kl_loss, epoch_num)
        # summary_writer.add_scalar("avg NLL loss", avg_nll_loss, epoch_num)

        path = os.path.join("models/", config["env_name"] + f'-{epoch_num}.pt')
        if epoch_num % 50 == 0:
            save_dict = {'gp_aa_model': gp_aa_model.state_dict(), 'opt': optimizer.state_dict()}
            torch.save(save_dict, path)
            print(f"Model save at epoch {epoch_num}")


if __name__ == '__main__':
    train()
