import gzip
import pickle
import numpy as np


class TrajectoryReader:
    """
    The trajectory reader is responsible for reading trajectories from a file.
    """

    def __init__(self, path):
        self.path = path.strip()

    def read(self):
        # if path ends in .pkl, read as pickle
        if self.path.endswith(".pkl"):
            with open(self.path, "rb") as f:
                data = pickle.load(f)
        # if path ends in .xz, read as lzma
        elif self.path.endswith(".npz"):
            data = np.load(self.path)
        elif self.path.endswith(".gz"):
            with gzip.open(self.path, "rb") as f:
                data = pickle.load(f)
        else:
            raise ValueError(
                f"Path {self.path} is not a valid trajectory file"
            )

        return data

