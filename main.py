
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

    data_module = DataModule(ic["data_dir"], max_data = ic["max_data"])
    