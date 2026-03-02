import os

import torch

from torch_geometric import seed_everything

import matplotlib.pyplot as plt

from src.data import get_loader, split_data
from src.eval import validation
from src.gnn import Model
from src.train import train_eval


def main(version):
    data = torch.load("data/graph.pt", weights_only=False)
    train_data, val_data, _ = split_data(data)

    train_loader = get_loader(train_data, batch_size=128, shuffle=True)
    val_loader = get_loader(val_data, batch_size=128, shuffle=False)

    model = Model(hidden_channels=64, out_channels=32, data=train_data)
    model, history = train_eval(model, train_loader, val_loader, version)

    plot_history(history, version)

    # Final evaluation on the validation set
    val_loss, val_report = validation(model, val_loader)
    print(f"Final Validation Loss: {val_loss:.4f}")
    print("Validation Classification Report:")
    print(val_report)

    if history["saved_model"]:
        print(f"Best model saved to 'data/{version}/best_model.pth'")
    else:
        print("No model was saved during training.")


def plot_history(history: dict, version: str = "v1"):
    save_path = f"data/{version}/training_curve.png"
    epochs = range(1, len(history['train_loss']) + 1)

    plt.figure(figsize=(12, 5))
    plt.plot(epochs, history['train_loss'], label='Train Loss', color='green')
    plt.plot(epochs, history['valid_loss'], label='Validation Loss', color='blue')
    best_epoch = epochs[history['valid_loss'].index(min(history['valid_loss']))]
    plt.axvline(x=best_epoch, color='red', linestyle='--', label='Best Epoch')
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

    version = "v1"
    os.makedirs(f"data/{version}", exist_ok=True)
    main(version)
