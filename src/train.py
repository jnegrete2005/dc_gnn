from copy import deepcopy

import torch
import torch.nn.functional as F
from torch_geometric.loader import LinkNeighborLoader
import tqdm

from src.eval import validation
from src.gnn import Model

EPOCHS = 50


def train_eval(
    model: Model,
    train_loader: LinkNeighborLoader,
    val_loader: LinkNeighborLoader,
    lr: float,
    show_progress: bool = True,
) -> tuple[Model, dict]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    best_model_weights = deepcopy(model.state_dict())
    best_val_loss = float("inf")

    history = {"train_loss": [None] * EPOCHS, "valid_loss": [None] * EPOCHS, "saved_model": False}

    for epoch in tqdm.tqdm(range(EPOCHS), desc="Training Epochs", disable=not show_progress):
        train_loss = train(model, train_loader, optimizer, device)

        model.eval()
        val_loss, _ = validation(model, val_loader)

        # Save metrics
        history["train_loss"][epoch] = train_loss
        history["valid_loss"][epoch] = val_loss

        scheduler.step()

        # Check for local best model based on validation loss
        if val_loss < best_val_loss:
            best_model_weights = deepcopy(model.state_dict())
            best_val_loss = val_loss

    model.load_state_dict(best_model_weights)
    return model, history


def train(
    model: Model,
    train_loader: LinkNeighborLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
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
