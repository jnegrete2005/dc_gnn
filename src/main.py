import matplotlib.pyplot as plt
import torch
from torch_geometric import seed_everything
from torch_geometric.loader import LinkNeighborLoader

from src.cv import nested_cv
from src.gnn import Model
from src.train import train_eval


def run_nested_cv():
    data = torch.load("data/graph_ones.pt", weights_only=False)

    final_generalization_loss, best_params = nested_cv(data, outer=5, inner=2)

    print(f"Final Generalization Loss from Nested CV: {final_generalization_loss:.4f}")

    for i, param_set in enumerate(best_params):
        print(f"Outer Fold {i + 1}: Best Hyperparameters: {param_set}")


def train_model(model: Model, train_loader: LinkNeighborLoader, val_loader: LinkNeighborLoader, version: str):
    model, history = train_eval(model, train_loader, val_loader, version)

    plot_history(history, version)

    if history["saved_model"]:
        print(f"Best model saved to 'data/{version}/best_model.pth'")
    else:
        print("No model was saved during training.")

    return model


def plot_history(history: dict, version: str = "v1"):
    save_path = f"data/{version}/training_curve.png"
    epochs = range(1, len(history["train_loss"]) + 1)

    min_val_loss = min(history["valid_loss"])
    best_epoch = epochs[history["valid_loss"].index(min_val_loss)]

    plt.figure(figsize=(12, 5), dpi=180)
    plt.plot(epochs, history["train_loss"], label="Train Loss", color="green")
    plt.plot(epochs, history["valid_loss"], label="Validation Loss", color="blue")

    plt.axvline(x=best_epoch, color="red", linestyle="--", label="Best Epoch")
    plt.scatter(best_epoch, min_val_loss, color="orange", s=50, label=f"Min Loss: {min_val_loss:.4f}", zorder=5)

    plt.title("Train vs Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Training curves saved to '{save_path}'")


if __name__ == "__main__":
    seed_everything(42)
    run_nested_cv()
