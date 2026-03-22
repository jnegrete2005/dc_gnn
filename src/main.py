import os
from sys import argv

import matplotlib.pyplot as plt
import torch
from torch_geometric import seed_everything
from torch_geometric.loader import LinkNeighborLoader

from src.data import get_loader, split_data
from src.eval import validation
from src.gnn import Model
from src.train import train_eval


def main(version, train=True):
    data = torch.load("data/graph_ones.pt", weights_only=False)

    batch_size = 128
    hidden_channels = 64
    out_channels = 16

    train_data, val_data, _ = split_data(data, train_ratio=0.8)
    train_loader = get_loader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = get_loader(val_data, batch_size=batch_size, shuffle=False)

    base_model = Model(hidden_channels=hidden_channels, out_channels=out_channels, data=train_data)

    if train:
        model = train_model(base_model, train_loader, val_loader, version)
    else:
        model = get_best_model(base_model, f"data/{version}/best_model.pth")

    # Final evaluation on the validation set
    val_loss, val_report = validation(model, val_loader)
    print(f"Final Validation Loss: {val_loss:.4f}")
    print("Validation Classification Report:")
    print(val_report)


def train_model(model: Model, train_loader: LinkNeighborLoader, val_loader: LinkNeighborLoader, version: str):
    model, history = train_eval(model, train_loader, val_loader, version)

    plot_history(history, version)

    if history["saved_model"]:
        print(f"Best model saved to 'data/{version}/best_model.pth'")
    else:
        print("No model was saved during training.")

    return model


def get_best_model(model: Model, model_save_path: str) -> tuple[float, dict]:
    if not os.path.exists(model_save_path):
        raise FileNotFoundError(f"No saved model found at '{model_save_path}'")

    best_model_weights = torch.load(model_save_path, weights_only=True)
    model.load_state_dict(best_model_weights)
    return model


def plot_history(history: dict, version: str = "v1"):
    save_path = f"data/{version}/training_curve.png"
    epochs = range(1, len(history['train_loss']) + 1)

    min_val_loss = min(history['valid_loss'])
    best_epoch = epochs[history['valid_loss'].index(min_val_loss)]

    plt.figure(figsize=(12, 5), dpi=180)
    plt.plot(epochs, history['train_loss'], label='Train Loss', color='green')
    plt.plot(epochs, history['valid_loss'], label='Validation Loss', color='blue')

    plt.axvline(x=best_epoch, color='red', linestyle='--', label='Best Epoch')
    plt.scatter(best_epoch, min_val_loss, color='orange', s=50, label=f'Min Loss: {min_val_loss:.4f}', zorder=5)

    plt.title('Train vs Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Training curves saved to '{save_path}'")


if __name__ == "__main__":
    seed_everything(42)

    version = "v0.1.0"
    val_mode = "--val" in argv
    os.makedirs(f"data/{version}", exist_ok=True)
    main(version, train=not val_mode)
