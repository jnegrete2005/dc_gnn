import os
from copy import deepcopy

import torch
import torch.nn.functional as F
from torch_geometric.loader import LinkNeighborLoader
import tqdm

from src.eval import validation
from src.gnn import Model

EPOCHS = 100


def train_eval(model: Model, train_loader: LinkNeighborLoader, val_loader: LinkNeighborLoader, version) -> tuple[Model, dict]:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    model_save_path = f"data/{version}/best_model.pth"

    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)

    best_model_weights = deepcopy(model.state_dict())
    current_best_val_loss = float('inf')
    best_val_loss = get_best_loss(model, val_loader, model_save_path)

    history = {
        "train_loss": [None] * EPOCHS,
        "valid_loss": [None] * EPOCHS,
        "saved_model": False
    }

    for epoch in tqdm.tqdm(range(EPOCHS), desc="Training Epochs"):
        train_loss = train(model, train_loader, optimizer, device)

        model.eval()
        val_loss, _ = validation(model, val_loader)

        # Save metrics
        history["train_loss"][epoch] = train_loss
        history["valid_loss"][epoch] = val_loss

        scheduler.step()

        # Check for local best model based on validation loss
        if val_loss < current_best_val_loss:
            best_model_weights = deepcopy(model.state_dict())
            current_best_val_loss = val_loss

        # Check if we have a new best model
        best_val_loss = check_best_model(model, val_loss, best_val_loss, model_save_path)
        if not history["saved_model"] and best_val_loss == val_loss:
            history["saved_model"] = True

    model.load_state_dict(best_model_weights)
    return model, history


def check_best_model(model: Model, val_loss: float, best_val_loss: float, model_save_path: str):
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_model_weights = deepcopy(model.state_dict())
        torch.save(best_model_weights, model_save_path)

    return best_val_loss


def train(model: Model, train_loader: LinkNeighborLoader, optimizer: torch.optim.Optimizer, device: torch.device) -> float:
    model.train()
    total_loss = total_examples = 0

    for sampled_data in train_loader:
        optimizer.zero_grad()
        sampled_data = sampled_data.to(device)

        pred = model(sampled_data)
        ground_truth = sampled_data["dc", "treats", "disease"].edge_label.float()

        loss = F.binary_cross_entropy_with_logits(pred, ground_truth)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * pred.numel()
        total_examples += pred.numel()

    return total_loss / total_examples


def get_best_loss(model: Model, val_loader: LinkNeighborLoader, model_save_path: str) -> float:
    if not os.path.exists(model_save_path):
        return float('inf'), deepcopy(model.state_dict())

    temp_model = deepcopy(model)
    best_model_weights = torch.load(model_save_path, weights_only=True)
    temp_model.load_state_dict(best_model_weights)
    temp_model.eval()

    best_val_loss, _ = validation(temp_model, val_loader)
    del temp_model
    return best_val_loss
