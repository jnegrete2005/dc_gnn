import itertools

import numpy as np
from torch_geometric.data import HeteroData
from tqdm import tqdm

from src.data import get_loader, split_data
from src.eval import validation
from src.gnn import Model
from src.train import train_eval


def nested_cv(data: HeteroData, outer: int = 3, inner: int = 2) -> tuple[float, list[dict]]:
    param_grid = {
        "lr": [0.005, 0.001],
        "hidden_channels": [64, 128],
        "out_channels": [32, 64],
    }

    # Get all combinations of hyperparameters
    keys, values = zip(*param_grid.items())
    param_combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]
    outer_test_results = []
    best_params_per_outer: list[dict] = []

    for _ in tqdm(range(outer), desc="Outer CV Folds"):
        outer_train_data, _, outer_test_data = split_data(data, val_ratio=0.0, test_ratio=0.2)
        outer_test_loader = get_loader(outer_test_data, batch_size=128, shuffle=False)

        best_inner_val_loss = float("inf")
        best_params: dict = {}

        for params in tqdm(param_combinations, desc="Inner CV Hyperparameter Combinations", leave=False):
            inner_val_losses = []

            for _ in range(inner):
                # Clone the outer training data to ensure a clean split for each inner fold
                clean_outer_train_data = outer_train_data.clone()

                # Clean up edge_labels
                for edge_type in clean_outer_train_data.edge_types:
                    if "edge_label" in clean_outer_train_data[edge_type]:
                        del clean_outer_train_data[edge_type].edge_label
                    if "edge_label_index" in clean_outer_train_data[edge_type]:
                        del clean_outer_train_data[edge_type].edge_label_index

                # Use clean data
                inner_train_data, inner_val_data, _ = split_data(
                    clean_outer_train_data,
                    val_ratio=0.2,
                    test_ratio=0.0,
                )

                inner_train_loader = get_loader(inner_train_data, batch_size=128, shuffle=True)
                inner_val_loader = get_loader(inner_val_data, batch_size=128, shuffle=False)

                model = Model(
                    hidden_channels=params["hidden_channels"],
                    out_channels=params["out_channels"],
                    data=inner_train_data,
                )

                _, history = train_eval(
                    model,
                    inner_train_loader,
                    inner_val_loader,
                    lr=params["lr"],
                    show_progress=False,
                )
                inner_val_losses.append(min(history["valid_loss"]))

            avg_val_loss = np.mean(inner_val_losses)

            if avg_val_loss < best_inner_val_loss:
                best_inner_val_loss = avg_val_loss
                best_params = params

        best_params_per_outer.append(best_params.copy())

        final_model = Model(
            hidden_channels=best_params["hidden_channels"],
            out_channels=best_params["out_channels"],
            data=outer_train_data,
        )

        outer_train_loader = get_loader(outer_train_data, batch_size=128, shuffle=True)
        final_model, _ = train_eval(
            final_model,
            outer_train_loader,
            outer_test_loader,
            lr=best_params["lr"],
            show_progress=False,
        )

        test_loss, test_report = validation(final_model, outer_test_loader)
        print(test_report)
        outer_test_results.append(test_loss)

    final_generalization_loss = float(np.mean(outer_test_results))
    return final_generalization_loss, best_params_per_outer
