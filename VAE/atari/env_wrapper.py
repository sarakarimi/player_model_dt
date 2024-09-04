import gym
import numpy as np
import h5py
from tqdm import tqdm
import random

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



def get_trajectory(env_name, traj_len, dataset=None, random_start=False):
    list_of_states, list_of_actions, list_of_tasks = [], [], []
    if dataset is None:
        env = gym.make(env_name)
        dataset = env.get_dataset()
    if random_start is False:
        dataset_len = len(dataset['observations'])
        print(dataset_len)
        list_of_states = [dataset['observations'][i:i + traj_len] for i in range(0, dataset_len, traj_len)]
        list_of_actions = [dataset['actions'][i:i + traj_len] for i in range(0, dataset_len, traj_len)]
        list_of_tasks = [dataset['tasks'][i:i + traj_len] for i in range(0, dataset_len, traj_len)]

    else:
        num_samples = len(dataset['observations'])
        print(num_samples)
        number_of_traj = int(num_samples / traj_len)
        start_indexes = np.random.randint(0, num_samples - traj_len - 1, size=number_of_traj)
        list_of_states = [dataset['observations'][i:i + traj_len] for i in start_indexes]
        list_of_actions = [dataset['actions'][i:i + traj_len] for i in start_indexes]
        list_of_tasks = [dataset['tasks'][i:i + traj_len] for i in start_indexes]


    c = list(zip(list_of_states, list_of_actions, list_of_tasks))
    random.shuffle(c)
    list_of_states, list_of_actions, list_of_tasks = zip(*c)
    return list_of_states, list_of_actions, list_of_tasks


if __name__ == '__main__':
    data = get_dataset("/home/sara/repositories/player_model_dt/VAE/atari/dataset.hdf5")
    print(len(data['actions']))
