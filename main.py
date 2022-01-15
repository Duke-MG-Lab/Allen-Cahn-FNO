
import os
import numpy as np

import torch

from utils import DataModule
from fourier_neural_operator import Fourier_Net2D

def train(input_config, output_config, model_config):

    ic, oc, mc = input_config, output_config, model_config
    results_root_dir, logs_dir, name_experiment = oc["results_root_dir"], oc["logs_dir"], oc["name_experiment"]

    if name_experiment == None:
        name_experiment = "experiment_" + mc["name_model"]

    results_dir = os.path.join(results_root_dir, name_experiment)
    models_checkpoints_dir = os.path.join(logs_dir, "models_checkpoints")

    # Prepare training data
    data_module = DataModule(ic["data_dir"], max_data = ic["max_data"])

    # Build model and configure training devices
    model = Fourier_Net2D(mc["modes_fourier"], mc["modes_fourier"], mc["width_fourier"])

    optimizer = torch.optim.Adam(model.parameters(), lr = 0.01)
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5, threshold = 1e-3 ,verbose = True, eps = 1e-6)

    
