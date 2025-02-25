# Hyperparameters
dimensions = [25, 128, 10]
model_params = {
    'decoder_final_activation': 'relu',
    'pretrained_epochs': 10, #50
    'epochs': 200,
    'save_path': 'output/model'
}
batch_size = 256  # 64
n_clusters = 4
alpha = 1

paths = [
        "/home/sara_karimi/player_model_dt/trajectory_embedding/datasets/minigrid/PPO_trajectories_multigoal-goal-0-no-wrapper.gz",
        "/home/sara_karimi/player_model_dt/trajectory_embedding/datasets/minigrid/PPO_trajectories_multigoal-goal-1-no-wrapper.gz",
        "/home/sara_karimi/player_model_dt/trajectory_embedding/datasets/minigrid/PPO_trajectories_multigoal-goal-2-no-wrapper.gz",
        "/home/sara_karimi/player_model_dt/trajectory_embedding/datasets/minigrid/PPO_trajectories_multigoal-goal-3-no-wrapper.gz",
    # "/home/sara/repositories/player_model_dt/trajectory_embedding/datasets/minigrid/PPO_trajectories_multigoal-goal-0-no-wrapper.gz",
    # "/home/sara/repositories/player_model_dt/trajectory_embedding/datasets/minigrid/PPO_trajectories_multigoal-goal-1-no-wrapper.gz",
    # "/home/sara/repositories/player_model_dt/trajectory_embedding/datasets/minigrid/PPO_trajectories_multigoal-goal-2-no-wrapper.gz",
    # "/home/sara/repositories/player_model_dt/trajectory_embedding/datasets/minigrid/PPO_trajectories_multigoal-goal-3-no-wrapper.gz",
    ]
# path_to_model = "/home/sara_karimi/player_model_dt/trajectory_embedding/style_dec_vae/output/model/vae-dec-model-2025-01-08-13-25"
# path_to_model = "/home/sara/repositories/player_model_dt/trajectory_embedding/style_dec_vae/output/model/vae-dec-model-2025-01-08-13-25"
# path_to_model = "/home/sara_karimi/player_model_dt/trajectory_embedding/style_dec_vae/output/model/vae-dec-model-2025-02-08-16-00"
path_to_model = "/home/sara/repositories/player_model_dt/trajectory_embedding/style_dec_vae/output/model/vae-dec-model-2025-02-20-14-45"