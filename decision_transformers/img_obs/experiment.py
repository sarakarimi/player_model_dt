import logging

# make deterministic
from utils.utils import set_seed, StateActionReturnDataset
import numpy as np

from models.model import GPT, GPTConfig
from training.trainer import Trainer, TrainerConfig
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=123)
parser.add_argument('--is_train', type=bool, default=True)
parser.add_argument('--vector_obs', type=bool, default=True, help="Set only when feature observations are used")
parser.add_argument('--context_length', type=int, default=5)
parser.add_argument('--epochs', type=int, default=50)
parser.add_argument('--model_type', type=str, default='reward_conditioned')
parser.add_argument('--batch_size', type=int, default=256)
parser.add_argument('--dataset_path', type=str,
                    default="../../data/double_goal_mode_1_image_ft_0_nsteps_100000_ppo_num_episodes100000_eps_0.1_gamma_0.99_dr_False.npz")
parser.add_argument('--checkpoint_path', type=str,
                    default="../../trained_models/dt_double_goal_mode_1_model.pth")
args = parser.parse_args()
set_seed(args.seed)

if __name__ == '__main__':

    # loading the dataset
    data = np.load(args.dataset_path)
    obss, actions, returns, done_idxs, rtgs, timesteps = data['obs'], data['action'].flatten(), data[
        'reward'].flatten(), data['dones'].flatten(), data['reward_to_go'].flatten(), data['timestep'].flatten()
    done_idxs = np.nonzero(done_idxs)[0][1:]

    # Flatten out observations
    if args.vector_obs:
        obss = obss.reshape((-1, 300))

    # set up logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )

    train_dataset = StateActionReturnDataset(obss, args.context_length * 3, actions, done_idxs, rtgs, timesteps)

    mconf = GPTConfig(train_dataset.vocab_size, train_dataset.block_size,
                      n_layer=6, n_head=8, n_embd=128, model_type=args.model_type, max_timestep=max(timesteps),
                      vector_obs=args.vector_obs)
    model = GPT(mconf)

    # initialize a trainer instance and kick off training
    epochs = args.epochs
    tconf = TrainerConfig(max_epochs=epochs, batch_size=args.batch_size, learning_rate=6e-4,
                          lr_decay=True, warmup_tokens=512 * 20,
                          final_tokens=2 * len(train_dataset) * args.context_length * 3,
                          num_workers=4, seed=args.seed, model_type=args.model_type,
                          max_timestep=max(timesteps), ckpt_path=args.checkpoint_path, vector_obs=args.vector_obs)
    trainer = Trainer(model, train_dataset, None, tconf)

    if args.is_train:
        trainer.train()
    else:
        trainer.evaluate(0.98)
