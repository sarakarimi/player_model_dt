# DEC VAE Model Hyperparameters
dimensions = [275, 128, 10] # [input_dim, hidden_dim, latent_dim]
dec_vae_model_params = {
    'decoder_final_activation': 'relu',
    'pretrained_epochs': 500,
    'epochs': 200,
    'save_path': 'output/model',
}

vae_dec_batch_size = 200  # 64
n_clusters = 3
alpha = 1

# Dataset Config
dataset_params = {
    'sampling': False,
    'index_channel_only': True,
    'state_normalization_factor': 9,
    'action_normalization_factor': 6
}

# VAE LSTM Model config
vae_model_params = {
    'input_size': 9 + 1, # 20 #500  # Number of features in each timestep
    'hidden_size': 256, # 20
    'latent_size': 32, # 10  # 64 looks ok for mujoco
}
num_epochs = 15 # 7 for easy env, 15 for hard env with kl_weight of 0.00025, with 100 epochs kl of 0.03 works
vae_batch_size = 128

vae_model_save_path = '/home/sara/repositories/player_model_dt/trained_models/minigrid_model/style_vae/three_style_env_hard_lstm_model.pth'


paths = [

    # Three style env
    # "/home/sara/repositories/player_model_dt/datasets/minigrid/three_style_env/PPO_trajectories_PPO_trajectories_three_style_env_bypass.gz",
    # "/home/sara/repositories/player_model_dt/datasets/minigrid/three_style_env/PPO_trajectories_PPO_trajectories_three_style_env_weapon.gz",
    # "/home/sara/repositories/player_model_dt/datasets/minigrid/three_style_env/PPO_trajectories_PPO_trajectories_three_style_env_backstab.gz",

    # Three style env hard
    "/home/sara/repositories/player_model_dt/datasets/minigrid/three_style_env_hard/PPO_trajectories_three_style_env_bypass.gz",
    "/home/sara/repositories/player_model_dt/datasets/minigrid/three_style_env_hard/PPO_trajectories_three_style_env_weapon.gz",
    "/home/sara/repositories/player_model_dt/datasets/minigrid/three_style_env_hard/PPO_trajectories_three_style_env_camouflage.gz",

]

# path_to_model = "/home/sara/repositories/player_model_dt/trajectory_embedding/style_dec_vae/output/model/vae-dec-model-2025-01-08-13-25"
