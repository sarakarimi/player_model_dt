import numpy as np
import robosuite as suite
import h5py
from tqdm import tqdm


def get_keys(h5file):
    keys = []

    def visitor(name, item):
        if isinstance(item, h5py.Dataset):
            keys.append(name)

    h5file.visititems(visitor)
    return keys


def get_dataset(h5path):
    data_dict = {}
    with h5py.File(h5path, 'r') as dataset_file:
        for k in tqdm(get_keys(dataset_file), desc="load datafile"):
            try:  # first try loading as an array
                data_dict[k] = dataset_file[k][:]
            except ValueError as e:  # try loading as a scalar
                data_dict[k] = dataset_file[k][()]
    return data_dict


if __name__ == '__main__':
    data = get_dataset("/home/sara/repositories/player_model_dt/VAE/robosuit/low_dim_v141.hdf5")
    print(data['data/demo_90/rewards'])
    # create environment instance
    env = suite.make(
        env_name="Lift", # try with other tasks like "Stack" and "Door"
        robots="Panda",  # try with other robots like "Sawyer" and "Jaco"
        has_renderer=True,
        has_offscreen_renderer=False,
        use_camera_obs=False,
    )


    # reset the environment
    # env.reset()
    # print(env.observation_spec())
    # for i in range(1000):
    #     action = np.random.randn(env.robots[0].dof) # sample random action
    #     obs, reward, done, info = env.step(action)
    #     print(action, obs)
    #     exit(0)
        # take action in the environment
    #     env.render()  #