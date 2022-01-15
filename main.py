
import os
import numpy as np

import torch

from utils import DataModule
from fourier_neural_operator import Fourier_Net2D

def train(input_config, output_config, model_config):

    # Training constants
    step = 1
    batch_size = 20
    scheduler_step = 100
    scheduler_gamma = 0.5

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
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=scheduler_step, gamma=scheduler_gamma)
    l1_loss = torch.nn.L1Loss()

    for i in range(mc["max_epochs"]):

        train_l2_step = 0
        train_l2_full = 0

        for index, batch in enumerate(data_module.train_dataloader):

            Xb, Ystep1, Ystep2, Ystep3, Ystep4 = batch["X"],batch["Y"][:,0,:,:,:],batch["Y"][:,1,:,:,:],batch["Y"][:,2,:,:,:],batch["Y"][:,3,:,:,:]
            Ydata = [Ystep1, Ystep2, Ystep3, Ystep4]

            Ypred1 = model(Xb)

            loss1 = l1_loss(Ypred1, Ystep1)

            Ypred = Ypred1

            losses = []
            losses.append(loss1)

            for i in range(1, model.n_steps_ahead):

                Ypred = model(Ypred)
                losses.append(l1_loss(Ypred, Ydata[i]))

            losses = [l.view(1) for l in losses]
            loss = torch.mean( torch.cat(losses, 0) )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()