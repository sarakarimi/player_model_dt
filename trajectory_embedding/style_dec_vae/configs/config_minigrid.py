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

# VAE Model config
vae_model_params = {
    'input_size': 9 + 1, # 20 #500  # Number of features in each timestep
    'hidden_size': 256, # 20
    'latent_size': 64, # 10  # 64 looks ok for mujoco
}
num_epochs = 7 # 10000
vae_batch_size = 128

vae_model_save_path = '/home/sara/repositories/player_model_dt/trained_models/minigrid_model/style_vae/trained_model.pth'


paths = [
    # "/home/sara_karimi/player_model_dt/trajectory_embedding/datasets/minigrid/PPO_trajectories_multigoal-goal-0-no-wrapper.gz",
    # "/home/sara_karimi/player_model_dt/trajectory_embedding/datasets/minigrid/PPO_trajectories_multigoal-goal-1-no-wrapper.gz",
    # "/home/sara_karimi/player_model_dt/trajectory_embedding/datasets/minigrid/PPO_trajectories_multigoal-goal-2-no-wrapper.gz",
    # "/home/sara_karimi/player_model_dt/trajectory_embedding/datasets/minigrid/PPO_trajectories_multigoal-goal-3-no-wrapper.gz",
    # "/home/sara/repositories/player_model_dt/trajectory_embedding/datasets/minigrid/old/PPO_trajectories_goal0.gz",
    # "/home/sara/repositories/player_model_dt/trajectory_embedding/datasets/minigrid/old/PPO_trajectories_goal1.gz",
    # "/home/sara/repositories/player_model_dt/trajectory_embedding/datasets/minigrid/old/PPO_trajectories_goal2.gz",
    # "/home/sara/repositories/player_model_dt/trajectory_embedding/datasets/minigrid/old/PPO_trajectories_goal3.gz",
    # "/home/sara/repositories/player_model_dt/trajectory_embedding/datasets/minigrid/PPO_trajectories_multigoal-11x11-env-goal-0-no-wrapper.gz",
    # "/home/sara/repositories/player_model_dt/trajectory_embedding/datasets/minigrid/PPO_trajectories_multigoal-11x11-env-goal-1-no-wrapper.gz",
    # "/home/sara/repositories/player_model_dt/trajectory_embedding/datasets/minigrid/PPO_trajectories_multigoal-11x11-env-goal-2-no-wrapper.gz",
    # "/home/sara/repositories/player_model_dt/trajectory_embedding/datasets/minigrid/PPO_trajectories_multigoal-11x11-env-goal-3-no-wrapper.gz",

    # "/home/sara_karimi/player_model_dt/trajectory_embedding/datasets/minigrid/old/PPO_trajectories_goal0.gz",
    # "/home/sara_karimi/player_model_dt/trajectory_embedding/datasets/minigrid/old/PPO_trajectories_goal1.gz",
    # "/home/sara_karimi/player_model_dt/trajectory_embedding/datasets/minigrid/old/PPO_trajectories_goal2.gz",
    # "/home/sara_karimi/player_model_dt/trajectory_embedding/datasets/minigrid/old/PPO_trajectories_goal3.gz",


    # Three style env
    "/home/sara/repositories/player_model_dt/datasets/minigrid/three_style_env/PPO_trajectories_PPO_trajectories_three_style_env_bypass.gz",
    "/home/sara/repositories/player_model_dt/datasets/minigrid/three_style_env/PPO_trajectories_PPO_trajectories_three_style_env_weapon.gz",
    "/home/sara/repositories/player_model_dt/datasets/minigrid/three_style_env/PPO_trajectories_PPO_trajectories_three_style_env_backstab.gz",

]

# path_to_model = "/home/sara_karimi/player_model_dt/trajectory_embedding/style_dec_vae/output/model/vae-dec-model-2025-01-08-13-25"
path_to_model = "/home/sara/repositories/player_model_dt/trajectory_embedding/style_dec_vae/output/model/vae-dec-model-2025-01-08-13-25"
# path_to_model = "/home/sara_karimi/player_model_dt/trajectory_embedding/style_dec_vae/output/model/vae-dec-model-2025-02-08-16-00"
# path_to_model = "/home/sara/repositories/player_model_dt/trajectory_embedding/style_dec_vae/output/model/vae-dec-model-2025-02-20-14-45"
