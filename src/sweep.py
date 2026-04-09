import numpy as np
import torch

import wandb
from src.data import get_loader, split_data
from src.gnn import Model
from src.train import train_eval

sweep_config = {
    "method": "bayes",
    "metric": {
        "name": "val_loss",
        "goal": "minimize",  # Tell W&B we want the lowest score possible
    },
    "parameters": {
        "lr": {
            "values": [0.005, 0.001, 0.0005, 0.0001],
        },
        "hidden_channels": {
            "values": [64, 128, 256],
        },
        "out_channels": {
            "values": [32, 64],
        },
    },
}

# Load your data once globally so it doesn't reload every run
full_data = torch.load("data/graph.pt", weights_only=False)


# 2. Define the training function
def sweep_train():
    # Initialize the run. W&B will automatically assign the ID, name, and config!
    wandb.init()

    # Grab the parameters the Cloud Brain chose for this specific run
    lr = wandb.config.lr
    hidden_channels = wandb.config.hidden_channels
    out_channels = wandb.config.out_channels
    #
    # Do a standard split (e.g., 80% train, 20% val)
    train_data, val_data, _ = split_data(full_data, val_ratio=0.2, test_ratio=0.0)

    train_loader = get_loader(train_data, batch_size=128, shuffle=True)
    val_loader = get_loader(val_data, batch_size=128, shuffle=False)

    model = Model(
        hidden_channels=hidden_channels,
        out_channels=out_channels,
        data=train_data,
    )

    # Train the model.
    _, history = train_eval(model, train_loader, val_loader, lr=lr, show_progress=False)

    # The final validation loss W&B uses to decide what parameters to test next
    final_val_loss = min(history["valid_loss"])

    # Log the final metric matching the name in your sweep_config
    wandb.log({"val_loss": final_val_loss})


if __name__ == "__main__":
    sweep_id = wandb.sweep(sweep_config, project="drug-comb-gnn")
    wandb.agent(sweep_id, function=sweep_train, count=20)  # Run 20 different sets of hyperparameters
