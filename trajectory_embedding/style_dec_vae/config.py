# Hyperparameters
dimensions = [500, 128, 10]
model_params = {
    'decoder_final_activation': 'relu',
    'pretrained_epochs': 200,
    'epochs': 500,
    'save_path': 'output/model'
}
batch_size = 200  # 64
n_clusters = 4
alpha = 1

paths = [
        "/home/sara/repositories/player_model_dt/trajectory_embedding/datasets/minigrid/PPO_trajectories_goal0.gz",
        "/home/sara/repositories/player_model_dt/trajectory_embedding/datasets/minigrid/PPO_trajectories_goal1.gz",
        "/home/sara/repositories/player_model_dt/trajectory_embedding/datasets/minigrid/PPO_trajectories_goal2.gz",
        "/home/sara/repositories/player_model_dt/trajectory_embedding/datasets/minigrid/PPO_trajectories_goal3.gz",
    ]
