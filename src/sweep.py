import wandb

from src.data import get_loader, split_data
from src.gnn import Model
from src.train import train_eval

# Global variables for the sweep agent (since it doesn't take arguments)
_SWEEP_DATA = None
_SWEEP_GRAPH_TYPE = None
_SWEEP_TRACKER = None


def sweep_train(offline: bool = False):
    # Use a dummy config first to get parameters from sweep agent
    mode = "offline" if offline else "online"
    run = wandb.init(mode=mode)
    
    # Get structured config and add metadata
    structured_config = _SWEEP_TRACKER.get_structured_config(_SWEEP_DATA, _SWEEP_GRAPH_TYPE, dict(wandb.config))
    structured_config["metadata"] = {
        "job_type": "sweep",
        "fold": None,
    }

    # Update wandb config with structured format and metadata
    wandb.config.update(structured_config, allow_val_change=True)

    # Train on full data for the sweep (Step 4)
    train_data, val_data, _ = split_data(_SWEEP_DATA, val_ratio=0.2, test_ratio=0.0)

    train_loader = get_loader(train_data, batch_size=128, shuffle=True)
    val_loader = get_loader(val_data, batch_size=128, shuffle=False)

    model = Model(
        hidden_channels=wandb.config.model["hidden_channels"],
        out_channels=wandb.config.model["out_channels"],
        data=train_data,
    )

    # Train and log using tracker
    _, history = train_eval(
        model, train_loader, val_loader, lr=wandb.config.model["lr"], show_progress=False, tracker=_SWEEP_TRACKER
    )

    final_roc_auc = max(history["roc_auc"])
    # Return the metric W&B sweep uses to optimize
    wandb.log({"roc_auc": final_roc_auc})


def run_sweep(data, graph_type, tracker, count=20, offline: bool = False):
    global _SWEEP_DATA, _SWEEP_GRAPH_TYPE, _SWEEP_TRACKER
    _SWEEP_DATA = data
    _SWEEP_GRAPH_TYPE = graph_type
    _SWEEP_TRACKER = tracker

    sweep_config = {
        "method": "bayes",
        "metric": {"name": "roc_auc", "goal": "maximize"},
        "parameters": {
            "lr": {"values": [0.005, 0.001, 0.0005, 0.0001]},
            "hidden_channels": {"values": [64, 128, 256]},
            "out_channels": {"values": [32, 64]},
        },
    }

    sweep_id = wandb.sweep(sweep_config, project="drug-comb-gnn")

    # We need to wrap sweep_train to pass the offline parameter
    from functools import partial
    train_func = partial(sweep_train, offline=offline)

    wandb.agent(sweep_id, function=train_func, count=count)

    # In a real scenario, you'd fetch the best params from W&B API here
    # and return them. For now, we'll just complete the sweep.
    return {"status": "sweep_completed"}
