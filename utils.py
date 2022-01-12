
import os
import numpy as np
import skimage.transform
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader, Dataset

class DataModule():

    def __init__(self, data_dir, resize=(60, 60), max_data=None, n_test_simulation=12, batch_size=50):

        # Set class attributes
        self.data_dir = data_dir
        self.resize = resize
        self.max_data = max_data
        self.n_test_simulations = n_test_simulation
        self.batch_size = batch_size

        self.names, self.data_arrays = self.load_data()

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