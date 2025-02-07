import numpy as np
from new_implementation.configs import RunConfig, OnlineTrainConfig, EnvironmentConfig
from new_implementation.decision_transformer.dataset import TrajectoryReader
from new_implementation.ppo.trajectory_writer import TrajectoryWriter

if __name__ == '__main__':
    paths = [
             "/home/sara/repositories/player_model_dt/data/new_implementation_datasets/PPO_lava_trajectories_mode2.gz",
             "/home/sara/repositories/player_model_dt/data/new_implementation_datasets/PPO_trajectories_mode2.gz",
             ]

    num_envs = 8
    trajectory_path = "/home/sara/repositories/player_model_dt/data/new_implementation_datasets/uniform_trajectories_mode2"
    trajectory_writer = TrajectoryWriter(
        path=trajectory_path,
        run_config=RunConfig(track=False),
        environment_config=EnvironmentConfig(),
        online_config=OnlineTrainConfig(num_envs=num_envs),
    )
    for path in paths:
        data = TrajectoryReader(path)
        data = data.read()["data"]
        print(len(data.get("rewards")))
        # returns = [r.sum() for r in data.get("rewards")]
        # index_list = []
        # returns = ['%.2f' % elem for elem in returns]
        # indexes = np.where(returns == 0.99)[0]
        # obs = [data.get("observations")[i] for i in index_list]
        # rewards = [data.get("rewards")[i] for i in index_list]
        # actions = [data.get("rewards")[i] for i in index_list]
        # dones = [data.get("rewards")[i] for i in index_list]
        # truncateds = [data.get("rewards")[i] for i in index_list]
        # infos = [data.get("rewards")[i] for i in index_list]
        # rets = [returns[i] for i in index_list]
        #
        # trajectory_writer.accumulate_trajectory(next_obs=obs,
        #                                         reward=rewards,
        #                                         actions=actions,
        #                                         dones=dones,
        #                                         truncated=truncateds,
        #                                         info=infos,
        #                                         rtg=rets)
