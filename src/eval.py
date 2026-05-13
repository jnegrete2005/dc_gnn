import torch
import torch.nn.functional as F
from torch_geometric.loader import LinkNeighborLoader
import numpy as np

from sklearn.metrics import classification_report, roc_auc_score, f1_score, precision_recall_curve, average_precision_score

from src.gnn import Model


def calculate_hits_at_k(y_true: np.ndarray, y_score: np.ndarray, query_ids: np.ndarray, k: int = 5) -> float:
    """
    Industry-standard Hits@K for link prediction.
    Groups predictions by source node (query_id) and checks if the true positive
    edge is within the top K highest-scored edges for that specific query.
    """
    hits = []
    unique_queries = np.unique(query_ids)

    for query in unique_queries:
        # 1. Isolate the predictions for this specific drug combination
        mask = query_ids == query
        query_y_true = y_true[mask]
        query_y_score = y_score[mask]

        # 2. Skip if there are no true positive edges for this query in the current batch/split
        if np.sum(query_y_true) == 0:
            continue

        # 3. Get indices of the top K scores (argsort sorts ascending, so we take the last K)
        # If there are fewer than K edges total for this query, we just take all of them.
        actual_k = min(k, len(query_y_score))
        top_k_indices = np.argsort(query_y_score)[-actual_k:]

        # 4. Did a true positive edge make it into the top K?
        hit = np.sum(query_y_true[top_k_indices]) > 0
        hits.append(float(hit))

    # Return the average Hits@K across all valid queries
    return float(np.mean(hits)) if len(hits) > 0 else 0.0


def validation(model: Model, val_loader: LinkNeighborLoader, k: int = 5):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = model.to(device)
    model.eval()

    preds = []
    ground_truths = []
    query_ids_list = []

    for sampled_data in val_loader:
        with torch.no_grad():
            sampled_data = sampled_data.to(device)
            pred = model(sampled_data)
            ground_truth = sampled_data["dc", "treats", "disease"].edge_label.float()

            # Get local source indices
            local_idx = sampled_data["dc", "treats", "disease"].edge_label_index[0]

            # Map them to global graph indices
            global_idx = sampled_data["dc"].n_id[local_idx]

            preds.append(pred.cpu())
            ground_truths.append(ground_truth.cpu())
            query_ids_list.append(global_idx.cpu())

    pred = torch.cat(preds, dim=0)
    ground_truth = torch.cat(ground_truths, dim=0)
    query_ids = torch.cat(query_ids_list, dim=0).numpy()

    val_loss = F.binary_cross_entropy_with_logits(pred, ground_truth).item()

    pred_probs = torch.sigmoid(pred).numpy()
    ground_truth_np = ground_truth.numpy()

    # Dynamic thresholding for F1
    precisions, recalls, thresholds = precision_recall_curve(ground_truth_np, pred_probs)
    f1_scores = (2 * precisions * recalls) / (precisions + recalls + 1e-8)
    best_idx = np.argmax(f1_scores)
    # thresholds list is 1 shorter than precisions/recalls
    optimal_threshold = thresholds[best_idx] if best_idx < len(thresholds) else 0.5

    preds_binary = pred_probs >= optimal_threshold

    # Calculate metrics for W&B
    if len(np.unique(ground_truth_np)) > 1:
        roc_auc = roc_auc_score(ground_truth_np, pred_probs)
    else:
        roc_auc = float("nan")
    f1 = f1_score(ground_truth_np, preds_binary, zero_division=0.0)
    ap = average_precision_score(ground_truth_np, pred_probs)

    # Calculate the new strict ranking Hits@K
    hits_at_k = calculate_hits_at_k(ground_truth_np, pred_probs, query_ids, k=k)

    class_report = classification_report(ground_truth_np, preds_binary, zero_division=0.0)

    metrics = {
        "val_loss": val_loss,
        "roc_auc": roc_auc,
        "f1": f1,
        "ap": ap,
        f"hits_at_{k}": hits_at_k,
        "class_report": class_report,
        "optimal_threshold": float(optimal_threshold),
    }

    return val_loss, metrics
