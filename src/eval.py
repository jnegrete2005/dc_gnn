import torch
import torch.nn.functional as F
from torch_geometric.loader import LinkNeighborLoader

from sklearn.metrics import classification_report

from src.gnn import Model


def validation(model: Model, val_loader: LinkNeighborLoader):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = model.to(device)

    preds = []
    ground_truths = []

    for sampled_data in val_loader:
        with torch.no_grad():
            sampled_data = sampled_data.to(device)
            pred = model(sampled_data)
            ground_truth = sampled_data["dc", "treats", "disease"].edge_label.float()

            preds.append(pred.cpu())
            ground_truths.append(ground_truth.cpu())

    pred = torch.cat(preds, dim=0).cpu()
    ground_truth = torch.cat(ground_truths, dim=0).cpu()

    val_loss = F.binary_cross_entropy_with_logits(pred, ground_truth).item()

    pred_probs = torch.sigmoid(pred).numpy()
    preds_binary = pred_probs > 0.5
    ground_truth = ground_truth.numpy()

    class_report = classification_report(ground_truth, preds_binary, zero_division=0)

    return val_loss, class_report
