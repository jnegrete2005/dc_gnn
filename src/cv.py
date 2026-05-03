import itertools

import numpy as np
from torch_geometric.data import HeteroData
from tqdm import tqdm

from src.data import get_loader, split_data
from src.eval import validation
from src.gnn import Model
from src.train import train_eval


def run_nested_cv(
    data: HeteroData,
    graph_type: str,
    tracker,
    outer: int = 3,
    inner: int = 2,
    offline: bool = False,
) -> tuple[float, list[dict]]:
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

    for i in range(outer):
        print(f"\n--- Outer Fold {i + 1}/{outer} ---")
        outer_train_data, _, outer_test_data = split_data(data, val_ratio=0.0, test_ratio=0.2)
        outer_test_loader = get_loader(outer_test_data, batch_size=128, shuffle=False)

        best_inner_roc_auc = -float("inf")
        best_params: dict = {}

        for j, params in enumerate(tqdm(param_combinations, desc="Inner CV Hyperparameter Combinations", leave=False)):
            # Create structured config
            config = tracker.get_structured_config(data, graph_type, params)

            # Initialize Audit run
            tracker.init_run(
                name=f"Audit_{graph_type}_Fold_{i}_Combo_{j}",
                group=f"Audit_{graph_type}_Fold_{i}",
                config=config,
                job_type="audit",
                fold=i,
                offline=offline,
            )

            inner_roc_aucs = []

            for _ in range(inner):
                # Clone and clean as before
                clean_outer_train_data = outer_train_data.clone()
                for edge_type in clean_outer_train_data.edge_types:
                    if "edge_label" in clean_outer_train_data[edge_type]:
                        del clean_outer_train_data[edge_type].edge_label
                    if "edge_label_index" in clean_outer_train_data[edge_type]:
                        del clean_outer_train_data[edge_type].edge_label_index

                inner_train_data, inner_val_data, _ = split_data(clean_outer_train_data, val_ratio=0.2, test_ratio=0.0)

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
                    tracker=tracker,  # Training metrics logged per inner fold
                )
                inner_roc_aucs.append(max(history["roc_auc"]))

            avg_roc_auc = np.mean(inner_roc_aucs)
            tracker.finish()

            if avg_roc_auc > best_inner_roc_auc:
                best_inner_roc_auc = avg_roc_auc
                best_params = params

        best_params_per_outer.append(best_params.copy())

        # Final evaluation on this outer fold
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
            show_progress=True,
        )

        test_loss, metrics = validation(final_model, outer_test_loader)
        print(f"Test Loss: {test_loss:.4f} | ROC AUC: {metrics['roc_auc']:.4f}")
        outer_test_results.append(metrics['roc_auc'])

    final_generalization_auc = float(np.mean(outer_test_results))
    return final_generalization_auc, best_params_per_outer
