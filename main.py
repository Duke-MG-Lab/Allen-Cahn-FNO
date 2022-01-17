
import os
from tabnanny import check
import numpy as np

import torch

from utils import DataModule
from fourier_neural_operator import Fourier_Net2D

# Model params
output_config = {
    "results_root_dir": "./results",
    "save_every": 10,
    "logs_dir": "logs",
    "name_experiment": None,
}

input_config = {
    "skip_steps": 6,
    "max_data": None,
    "data_dir": "./data/AC",
    "gpus": -1,
}

model_config = {
    "max_epochs": 100,
    "early_stopping_patience": 10,
    "early_stopping_min_delta": 1e-4,
    "lr": 5*1e-4,
    "normalization": False,
    "n_blocks": 6,
    "layers_per_block": 3,
    "channels": 70,
    "name_model": "fourier",
    "skip_con_weight": 0.1,
    "modes_fourier": 16,
    "width_fourier": 60,
}

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

    for epoch in range(mc["max_epochs"]):

        epoch_training_loss = []
        epoch_validation_loss = []

        # Training
        for i, batch in enumerate(data_module.train_dataloader):

            Xb, Ystep1, Ystep2, Ystep3, Ystep4 = batch["X"], batch["Y"][:,0,:,:,:], batch["Y"][:,1,:,:,:], batch["Y"][:,2,:,:,:], batch["Y"][:,3,:,:,:]
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
            loss = torch.mean(torch.cat(losses, 0))

            epoch_training_loss.append(loss)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Update learning rate with scheduler and output epoch training loss
        scheduler.step()
        mean_training_loss = torch.mean(torch.Tensor(epoch_training_loss))

        print("epoch: ", epoch, "training loss: ", mean_training_loss)

        # Validation
        with torch.no_grad():

            for i, batch in enumerate(data_module.val_dataloader):

                Xb, Ystep1, Ystep2, Ystep3, Ystep4 = batch["X"], batch["Y"][:,0,:,:,:], batch["Y"][:,1,:,:,:], batch["Y"][:,2,:,:,:], batch["Y"][:,2,:,:,:]
                Ypred1 = model(Xb)
                Ypred2 = model(Ypred1)
                Ypred3 = model(Ypred2)
                Ypred4 = model(Ypred3)

                val_loss1 = l1_loss(Ypred1, Ystep1)
                val_loss2 = l1_loss(Ypred2, Ystep2)
                val_loss3 = l1_loss(Ypred3, Ystep3)
                val_loss4 = l1_loss(Ypred4, Ystep4)

                epoch_validation_loss.append([val_loss1, val_loss2, val_loss3, val_loss4])

                if (val_loss1 < model.tol_next_step) and model.n_steps_ahead <= 2:
                    model.n_steps_ahead += 1


        validation_step_outputs = np.array(torch.mean(torch.Tensor(epoch_validation_loss), axis =0))
        print("epoch: ", epoch, "validation loss: ", validation_step_outputs[0])

    torch.save({
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            }, "./models/model.pt")

def load_model(input_config, output_config, model_config):

    ic, oc, mc = input_config, output_config, model_config
    scheduler_step = 100
    scheduler_gamma = 0.5

    # Load Fourier_Net2D, optimizer, and scheduler
    model = Fourier_Net2D(mc["modes_fourier"], mc["modes_fourier"], mc["width_fourier"])
    optimizer = torch.optim.Adam(model.parameters(), lr = 0.01)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=scheduler_step, gamma=scheduler_gamma)

    # Load checkpoint file
    checkpoint = torch.load("./models/model.pt")

    # Load state dictionaries
    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    scheduler.load_state_dict(checkpoint["scheduler"])

    return model, optimizer, scheduler

if __name__ == "__main__":
    train(input_config, output_config, model_config)