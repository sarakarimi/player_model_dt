config = {
    'env_name': "atari",
    'seed': 0,
    'reg': 0.1,
    'tanh': True,
    'traj_length': 10,  # for Procgen 5
    'batch_size': 50,
    'latent_dim': 8,
    'latent_reg': 0.1,
    'hidden_dims': [200, 200],
    'num_eval': 500,
    'eval_interval': 10,
    'train_epochs': 600,
    'goal_idxs': None,
    'render': True,
}

PATH_TO_AE_MODEL = "../models/AE_models/" + config['env_name'] + "-opal/600.pt"
