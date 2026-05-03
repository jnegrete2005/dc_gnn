import os
import wandb
from torch_geometric.data import HeteroData

# Silence W&B terminal output
os.environ["WANDB_SILENT"] = "true"


class WandbTracker:
    def __init__(self, project_name: str = "drug-comb-gnn", entity: str = "medal-upm"):
        self.project_name = project_name
        self.entity = entity

    def get_structured_config(self, data: HeteroData, graph_type: str, params: dict):
        # Gracefully handle missing node types depending on the graph
        def get_num_nodes(ntype):
            return data[ntype].num_nodes if ntype in data.node_types else 0

        def get_num_features(ntype):
            if ntype in data.node_types and hasattr(data[ntype], "x") and data[ntype].x is not None:
                return data[ntype].x.shape[1]
            return 0

        # Define what the features are based on node type (placeholder strings as requested)
        feature_descriptions = {
            "drug": "ones",
            "dc": "ones",
            "disease": "ones",
        }

        return {
            "input": {
                "graph_type": graph_type,
                "num_nodes": {
                    "drug": get_num_nodes("drug"),
                    "drug_combinations": get_num_nodes("dc"),
                    "disease": get_num_nodes("disease"),
                },
                "node_feature_description": {
                    "drug": feature_descriptions["drug"],
                    "drug_combinations": feature_descriptions["dc"],
                    "disease": feature_descriptions["disease"],
                },
                "node_feature_length": {
                    "drug": get_num_features("drug"),
                    "drug_combinations": get_num_features("dc"),
                    "disease": get_num_features("disease"),
                },
            },
            "model": {
                "gnn_layer_type": "SAGEConv",
                "gnn_num_layers": 2,
                "hidden_channels": params.get("hidden_channels"),
                "out_channels": params.get("out_channels"),
                "lr": params.get("lr"),
                "dropout": params.get("dropout", 0.0),
                "regularization": params.get("weight_decay", 0.0),
            },
        }

    def init_run(self, name: str, group: str, config: dict, job_type: str, fold: int | None = None, offline: bool = False):
        # Add metadata to config
        config["metadata"] = {
            "job_type": job_type,
            "fold": fold,
        }

        mode = "offline" if offline else "online"

        return wandb.init(
            entity=self.entity,
            project=self.project_name,
            name=name,
            group=group,
            config=config,
            job_type=job_type,
            reinit=True,
            mode=mode,
        )

    def log_metrics(self, metrics: dict, epoch: int = None):
        """
        Expects a dictionary with keys like 'val_loss', 'roc_auc', 'f1', 'hits_at_k', 'ap'
        """
        formatted_metrics = {
            "output/ROC_AUC": metrics.get("roc_auc"),
            "output/F1_score": metrics.get("f1"),
            "output/AP": metrics.get("ap"),
            "output/val_loss": metrics.get("val_loss"),
            "output/optimal_threshold": metrics.get("optimal_threshold"),
        }

        # Dynamically find hits_at_k keys (e.g., 'hits_at_5')
        for key in metrics:
            if key.startswith("hits_at_"):
                # Format for W&B: 'hits_at_5' -> 'output/Hits@5'
                k_val = key.split("_")[-1]
                formatted_metrics[f"output/Hits@{k_val}"] = metrics[key]

        if epoch is not None:
            formatted_metrics["epoch"] = epoch

        # Also log raw train/val loss if provided (for curves)
        if "train_loss" in metrics:
            formatted_metrics["train/loss"] = metrics["train_loss"]
        if "val_loss" in metrics:
            formatted_metrics["val/loss"] = metrics["val_loss"]

        wandb.log(formatted_metrics)

    def finish(self):
        if wandb.run:
            wandb.finish()
