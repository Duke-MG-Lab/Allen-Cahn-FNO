
import os
import numpy as np
import skimage.transform
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader, Dataset

class PairDataset(Dataset):

    def __init__(self, x, y):
        super(PairDataset, self).__init__()
        self.x = torch.Tensor(x)
        self.y = torch.Tensor(y)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        sample = {"X": self.x[idx], "Y": self.y[idx]}
        return sample

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
        self.test_ratio = test_ratio

        # Load all data
        self.names, self.data_arrays = self.load_data()
        self.n_train = int((1 - self.test_ratio) * len(self.data_arrays))

        # Split training and test data
        self.arrays_train = self.data_arrays[:self.n_train]
        self.arrays_test = self.data_arrays[self.n_train:]

        self.x_train, self.y_train = self.prepare_x_y(self.arrays_train,
                                                        skip_steps = self.skip_steps,
                                                        store_steps_ahead = self.store_steps_ahead)

        self.train_dataset = PairDataset(self.x_train, self.y_train)


        self.x_test, self.y_test = self.prepare_x_y(self.arrays_test,
                                                        skip_steps = self.skip_steps,
                                                        store_steps_ahead = self.store_steps_ahead)

        self.test_dataset = PairDataset(self.x_test, self.y_test)
        self.test_simulations = self.select_test_simulations(self.arrays_test)

        self.train_dataloader = DataLoader(self.train_dataset, batch_size = self.batch_size, shuffle = True)
        self.val_dataloader = DataLoader(self.test_dataset, batch_size = self.batch_size, shuffle = True)


    def load_data(self):

        data_files = [name for name in os.listdir(self.data_dir) if "array" in name]

        if self.max_data:
            data_files = data_files[:self.max_data]

        data_arrays = [np.load(os.path.join(self.data_dir,file)) for file in tqdm(data_files)]

        if self.resize:
            for i,array in tqdm(enumerate(data_arrays)):
                data_arrays[i] = np.reshape(skimage.transform.resize(array,
                    (len(array), self.resize[0], self.resize[1])),
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

    def select_test_simulations(self, arrays_test):

        if len(arrays_test) < self.n_test_simulations:
            n_test_simulation = len(arrays_test)
        else:
            n_test_simulation = self.n_test_simulations

        test_simulations = [arrays_test[index][8:] for index in np.arange(0, n_test_simulation, 1)]

        extra_sims = [arrays_test[0],
                      arrays_test[1][5:],
                      arrays_test[2][15:]] #different u0s

        test_simulations.extend(extra_sims)

        return test_simulations