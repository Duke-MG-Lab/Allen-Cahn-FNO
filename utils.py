
import os
import numpy as np
import skimage.transform
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader, Dataset

class DataModule():

    def __init__(self, data_dir, resize=(60, 60), max_data=None, n_test_simulation=12, batch_size=50,
                    skip_steps=10, store_steps_ahead=5, test_ratio=0.2):

        # Set class attributes
        self.data_dir = data_dir

        # Data constraints
        self.resize = resize
        self.max_data = max_data
        self.n_test_simulations = n_test_simulation
        self.batch_size = batch_size

        self.skip_steps = skip_steps
        self.store_steps_ahead = store_steps_ahead

        # Load all data
        self.names, self.data_arrays = self.load_data()
        self.n_train = int((1 - self.test_ratio) * len(self.data_arrays))


    def load_data(self):

        data_files = [name for name in os.listdir(self.data_dir) if "array" in name]

        if self.max_data:
            data_files = data_files[:self.max_data]

        data_arrays = [np.load(os.path.join(self.data_dir,file)) for file in tqdm(data_files)]

        if self.esize:
            for i,array in tqdm(enumerate(data_arrays)):
                data_arrays[i] = np.reshape(skimage.transform.resize(array,
                    (len(array), self.esize[0], self.resize[1])),
                    (len(array), 1, self.resize[0], self.resize[1]))

        return data_files, data_arrays

    def prepare_x_y(self, simulations , skip_steps = 10, store_steps_ahead = 5):

        X = []
        Y = []

        for simulation in tqdm(simulations):
            sim = simulation[2:] # Remove first spiky snapshots
            lsim = len(sim)

            for i in range(int(np.floor(lsim/skip_steps)) - store_steps_ahead):
                s = i * skip_steps
                _Y = []

                for j in range(1,store_steps_ahead):
                    sj = (i + j) * skip_steps
                    _Y.append(sim[sj])

                _Y = np.array(_Y)

                X.append(sim[s])
                Y.append(_Y)

        X = np.array(X)
        Y = np.array(Y)

        return X, Y