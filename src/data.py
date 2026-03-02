import torch_geometric.transforms as T
from torch_geometric.data import HeteroData
from torch_geometric.loader import LinkNeighborLoader
from torch_geometric import seed_everything


def split_data(data, train_ratio=0.8) -> tuple[HeteroData, HeteroData, HeteroData]:
    seed_everything(42)
    transform = T.RandomLinkSplit(
        num_val=1 - train_ratio,
        num_test=0,
        is_undirected=True,
        disjoint_train_ratio=0.3,
        neg_sampling_ratio=1.0,
        add_negative_train_samples=False,
        edge_types=("dc", "treats", "disease"),
        rev_edge_types=("disease", "rev_treats", "dc")
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

    seed_everything(42)

    return LinkNeighborLoader(
        data=data,
        num_neighbors=neighbor_config,
        edge_label_index=(("dc", "treats", "disease"), edge_label_index),
        edge_label=edge_label,
        batch_size=batch_size,
        neg_sampling="binary" if shuffle else None,  # Only apply negative sampling during training
        shuffle=shuffle
    )
