
import os
import numpy as np

import torch

from utils import DataModule
from fourier_neural_operator import Fourier_Net2D

def train(input_config, output_config, model_config):

    ic, oc, mc = input_config, output_config, model_config
    results_root_dir, logs_dir, name_experiment = oc["results_root_dir"], oc["logs_dir"], oc["name_experiment"]
    