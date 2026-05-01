import torch_geometric.transforms as T
from torch_geometric.data import HeteroData
from torch_geometric.loader import LinkNeighborLoader


def split_data(
    data: HeteroData,
    val_ratio=0.2,
    test_ratio=0.0,
    val_neg_ratio: float = 19.0,
) -> tuple[HeteroData, HeteroData, HeteroData]:

    transform = T.RandomLinkSplit(
        num_val=val_ratio,
        num_test=test_ratio,
        is_undirected=True,
        disjoint_train_ratio=0.3,
        neg_sampling_ratio=val_neg_ratio,
        add_negative_train_samples=False,
        edge_types=("dc", "treats", "disease"),
        rev_edge_types=("disease", "rev_treats", "dc"),
    )
    return transform(data)


def get_loader(data: HeteroData, batch_size: int, shuffle: bool) -> LinkNeighborLoader:
    edge_label_index = data["dc", "treats", "disease"].edge_label_index
    edge_label = data["dc", "treats", "disease"].edge_label

    disease_max_neighbors = [10, 10]
    dc_max_neighbors = [10, 10]
    neighbor_config = {
        # The main treatment edges
        ("dc", "treats", "disease"): disease_max_neighbors,
        ("disease", "rev_treats", "dc"): disease_max_neighbors,
        # The side-information edges
        ("drug", "interacts", "dc"): dc_max_neighbors,
        ("dc", "rev_interacts", "drug"): dc_max_neighbors,
    }

    neg_sampling: str | None = "binary" if shuffle else None  # Only apply negative sampling during training

    return LinkNeighborLoader(
        data=data,
        num_neighbors=neighbor_config,
        edge_label_index=(("dc", "treats", "disease"), edge_label_index),
        edge_label=edge_label,
        batch_size=batch_size,
        neg_sampling=neg_sampling,
        shuffle=shuffle,
    )
