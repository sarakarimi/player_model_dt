# Hyperparameters
dimensions = [20, 256, 64]
model_params = {
    'decoder_final_activation': 'relu',
    'pretrained_epochs': 10,
    'epochs': 200,
    'save_path': 'output/model'
}
batch_size = 128 #64 #200  # 64
n_clusters = 2 # 4
alpha = 1



paths = [
    "/home/sara/repositories/player_model_dt/trajectory_embedding/datasets/mujoco/cheetah_vel/cheetah_vel-5-expert.pkl",
    # "/home/sara/repositories/player_model_dt/trajectory_embedding/datasets/mujoco/cheetah_vel/cheetah_vel-10-expert.pkl",
    # "/home/sara/repositories/player_model_dt/trajectory_embedding/datasets/mujoco/cheetah_vel/cheetah_vel-15-expert.pkl",
    "/home/sara/repositories/player_model_dt/trajectory_embedding/datasets/mujoco/cheetah_vel/cheetah_vel-30-expert.pkl"
    ]

path_to_model = ""
