import itertools

import numpy as np
from torch_geometric.data import HeteroData
from tqdm import tqdm

from src.data import get_loader, split_data
from src.eval import validation
from src.gnn import Model
from src.train import train_eval


def compute_metrics_range(metrics_list):
    ranges = {}
    if not metrics_list:
        return ranges
    keys = [k for k in metrics_list[0].keys() if isinstance(metrics_list[0][k], (int, float))]
    for k in keys:
        values = [m[k] for m in metrics_list]
        ranges[k] = {
            "min": float(np.min(values)),
            "max": float(np.max(values)),
            "range": float(np.max(values) - np.min(values)),
        }
    return ranges


def run_nested_cv(
    data: HeteroData,
    graph_type: str,
    tracker,
    outer: int = 3,
    inner: int = 2,
    offline: bool = False,
    k: int = None,
    verbose: bool = True,
) -> tuple[float, float, list[dict]]:
    param_grid = {
        "lr": [0.005, 0.001],
        "hidden_channels": [64, 128],
        "out_channels": [32, 64],
    }

    # Get all combinations of hyperparameters
    keys, values = zip(*param_grid.items())
    param_combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]
    outer_test_results = []
    outer_train_results = []
    report = []

    for i in range(outer):
        if verbose:
            print(f"\n--- Outer Fold {i + 1}/{outer} ---")
        outer_train_data, _, outer_test_data = split_data(data, val_ratio=0.0, test_ratio=0.2)
        outer_test_loader = get_loader(outer_test_data, batch_size=128, shuffle=False)

        best_inner_roc_auc = -float("inf")
        best_params: dict = {}
        best_inner_metrics_list = []

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
            inner_metrics_list = []

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

                trained_model, _ = train_eval(
                    model,
                    inner_train_loader,
                    inner_val_loader,
                    lr=params["lr"],
                    show_progress=False,
                    tracker=tracker,  # Training metrics logged per inner fold
                    k=k,
                )

                # Evaluate the best inner fold model on the inner validation set to get all metrics
                _, inner_metrics = validation(trained_model, inner_val_loader, k=k)
                inner_roc_aucs.append(inner_metrics["roc_auc"])
                inner_metrics_list.append(inner_metrics)

            avg_roc_auc = np.mean(inner_roc_aucs)
            tracker.finish()

            if avg_roc_auc > best_inner_roc_auc:
                best_inner_roc_auc = avg_roc_auc
                best_params = params
                best_inner_metrics_list = inner_metrics_list

        # Compute range summary for the best inner fold metrics
        inner_metrics_range = compute_metrics_range(best_inner_metrics_list)

        # Final evaluation on this outer fold
        final_model = Model(
            hidden_channels=best_params["hidden_channels"],
            out_channels=best_params["out_channels"],
            data=outer_train_data,
        )

        outer_train_loader = get_loader(outer_train_data, batch_size=128, shuffle=True)
        final_model, history = train_eval(
            final_model,
            outer_train_loader,
            outer_test_loader,
            lr=best_params["lr"],
            show_progress=True,
            k=k,
        )

        # Evaluate on test set
        test_loss, metrics = validation(final_model, outer_test_loader, k=k)

        # Evaluate on training set
        outer_train_eval_loader = get_loader(outer_train_data, batch_size=128, shuffle=False, neg_sampling="binary")
        train_loss, train_metrics = validation(final_model, outer_train_eval_loader, k=k)

        print(f"Fold {i + 1} Metrics:")
        print(f"  Train -> Loss: {train_loss:.4f} | ROC AUC: {train_metrics['roc_auc']:.4f}")
        print(f"  Test  -> Loss: {test_loss:.4f} | ROC AUC: {metrics['roc_auc']:.4f}")

        outer_test_results.append(metrics["roc_auc"])
        outer_train_results.append(train_metrics["roc_auc"])

        fold_report = {
            "fold": i + 1,
            "best_params": best_params.copy(),
            "inner_metrics_range": inner_metrics_range,
            "outer_metrics": metrics,
            "train_metrics": train_metrics,
            "history": history,
        }
        report.append(fold_report)

    final_generalization_auc = float(np.mean(outer_test_results))
    final_train_auc = float(np.mean(outer_train_results))
    return final_generalization_auc, final_train_auc, report
