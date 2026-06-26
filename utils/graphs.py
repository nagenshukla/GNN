"""
Synthetic GRAPH generators with a KNOWN ground truth.

This mirrors the philosophy of the sibling GBDT series (utils/data.py there): we
*plant* the signal ourselves so that afterwards we can ask the sharp question -

    Did the GNN recover the structure we planted, and ignore the noise?

For graphs the "signal" can live in three different places, and different
generators below put it in different places on purpose:

  * in the node FEATURES        (a plain MLP can already use this)
  * in the local STRUCTURE      (only a model that reads edges can use this)
  * in long-range STRUCTURE     (only a deep / global model can use this)

Every generator returns a torch_geometric ``Data`` object (or a list of them for
graph-level tasks) plus a ``GraphGroundTruth`` describing exactly what we planted.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import networkx as nx
import torch
from torch_geometric.data import Data


# ===========================================================================
# Ground-truth bookkeeping
# ===========================================================================
@dataclass
class GraphGroundTruth:
    """Everything we deliberately planted in a synthetic graph dataset."""
    task: str                                   # node_regression / node_classification / graph_classification
    description: str = ""
    own_coefs: dict = field(default_factory=dict)        # node-regression: feature -> coef (own features)
    neighbor_coefs: dict = field(default_factory=dict)   # node-regression: feature -> coef (neighbor mean)
    structure_weight: float = 0.0                        # node-regression: the lambda knob
    n_features: int = 0
    n_informative: int = 0
    homophily: float | None = None                       # measured edge homophily, if a label exists
    motif: str | None = None
    extra: dict = field(default_factory=dict)

    def summary(self) -> pd.DataFrame:
        """A tidy one-column table of the planted facts, handy for display."""
        rows = {
            "task": self.task,
            "description": self.description,
            "n_features": self.n_features,
        }
        if self.own_coefs:
            rows["own_feature_effect"] = ", ".join(f"{k}:{v:+.2f}" for k, v in self.own_coefs.items())
        if self.neighbor_coefs:
            rows["neighbor_mean_effect"] = ", ".join(f"{k}:{v:+.2f}" for k, v in self.neighbor_coefs.items())
        if self.structure_weight:
            rows["structure_weight (lambda)"] = self.structure_weight
        if self.homophily is not None:
            rows["edge_homophily"] = round(self.homophily, 3)
        if self.motif:
            rows["planted_motif"] = self.motif
        for k, v in self.extra.items():
            rows[k] = v
        return pd.DataFrame({"value": rows})


# ===========================================================================
# Small graph helpers
# ===========================================================================
def _edge_index_from_nx(G: nx.Graph) -> torch.Tensor:
    """Undirected networkx graph -> a [2, 2E] edge_index with both directions."""
    edges = list(G.edges())
    if not edges:
        return torch.empty((2, 0), dtype=torch.long)
    e = np.array(edges, dtype=np.int64).T          # 2 x E
    e = np.concatenate([e, e[::-1]], axis=1)       # add the reverse of every edge
    return torch.tensor(e, dtype=torch.long)


def _neighbor_mean(X: np.ndarray, edge_index: torch.Tensor, n: int) -> np.ndarray:
    """Mean of each node's neighbour features (edge_index assumed symmetric)."""
    d = X.shape[1]
    src = edge_index[0].numpy()
    dst = edge_index[1].numpy()
    out = np.zeros((n, d), dtype=np.float64)
    deg = np.zeros(n, dtype=np.float64)
    np.add.at(out, dst, X[src])
    np.add.at(deg, dst, 1.0)
    deg[deg == 0] = 1.0
    return out / deg[:, None]


def homophily_ratio(edge_index: torch.Tensor, y: torch.Tensor) -> float:
    """Edge homophily: fraction of edges whose two endpoints share a label."""
    src, dst = edge_index
    return (y[src] == y[dst]).float().mean().item()


def add_node_splits(data: Data, train: float = 0.6, val: float = 0.2, seed: int = 0) -> Data:
    """Attach random train/val/test boolean masks for a transductive node task."""
    n = data.num_nodes
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    n_tr, n_va = int(train * n), int(val * n)
    for name, sl in [("train_mask", idx[:n_tr]),
                     ("val_mask", idx[n_tr:n_tr + n_va]),
                     ("test_mask", idx[n_tr + n_va:])]:
        m = torch.zeros(n, dtype=torch.bool)
        m[sl] = True
        setattr(data, name, m)
    return data


# ===========================================================================
# 1. Neighbour-aggregation regression  -  the "GNN vs. tabular model" probe
# ===========================================================================
def make_neighbor_aggregation_regression(
    n_nodes: int = 800,
    structure_weight: float = 1.0,
    n_features: int = 8,
    k: int = 6,
    rewire: float = 0.15,
    noise_std: float = 0.3,
    seed: int = 42,
):
    """
    y_i = (own linear effect on x_i)  +  lambda * (linear effect on MEAN of neighbour x)  +  noise

    The ``structure_weight`` (lambda) knob is the whole story of the series:
      * lambda = 0  -> the target is a function of a node's OWN features only, so an
        MLP / GBDT that ignores edges is already optimal.
      * lambda > 0  -> part of the target depends on the *average of the neighbours'*
        features, which a node cannot see by itself. Only a model that reads edges
        (a GNN) can recover that part. The MLP's error floor rises with lambda.

    Features x4..x{n-1} are pure NOISE (zero true effect).
    """
    rng = np.random.default_rng(seed)
    G = nx.connected_watts_strogatz_graph(n_nodes, k, rewire, seed=seed)
    edge_index = _edge_index_from_nx(G)

    X = rng.standard_normal((n_nodes, n_features)).astype(np.float32)
    own = {0: 1.5, 1: -1.0}            # effect of a node's OWN features
    nbr = {2: 1.2, 3: -0.8}            # effect of the NEIGHBOUR MEAN of these features
    nbr_mean = _neighbor_mean(X, edge_index, n_nodes)

    y = np.zeros(n_nodes, dtype=np.float64)
    for j, c in own.items():
        y += c * X[:, j]
    nbr_term = np.zeros(n_nodes, dtype=np.float64)
    for j, c in nbr.items():
        nbr_term += c * nbr_mean[:, j]
    y += structure_weight * nbr_term
    y += rng.normal(0.0, noise_std, n_nodes)

    data = Data(x=torch.tensor(X),
                edge_index=edge_index,
                y=torch.tensor(y, dtype=torch.float32).view(-1, 1))
    add_node_splits(data, seed=seed)

    gt = GraphGroundTruth(
        task="node_regression",
        description="y = own-feature effect + lambda * neighbour-mean effect + noise",
        own_coefs={f"x{j}": c for j, c in own.items()},
        neighbor_coefs={f"x{j}": c for j, c in nbr.items()},
        structure_weight=structure_weight,
        n_features=n_features,
        n_informative=4,
        extra={"n_nodes": n_nodes, "noise_features": [f"x{j}" for j in range(4, n_features)]},
    )
    return data, gt


# ===========================================================================
# 2. SBM with a tunable homophily knob  -  node classification
# ===========================================================================
def make_sbm_homophily(
    n_per_block: int = 150,
    n_blocks: int = 3,
    homophily: float = 0.8,
    avg_degree: float = 10.0,
    n_features: int = 10,
    feature_signal: float = 0.6,
    seed: int = 42,
):
    """
    Stochastic Block Model: the label is the community a node belongs to.

    ``homophily`` in (0, 1) sets the fraction of a node's edges that stay inside
    its own community (intra-block). High homophily -> neighbours share your label
    (the assumption vanilla GCN relies on). Low homophily -> heterophily.

    Node features are a class centroid (scaled by ``feature_signal``) plus gaussian
    noise, so part of the signal is in the features and part is in the structure.
    """
    sizes = [n_per_block] * n_blocks
    n = n_per_block * n_blocks
    deg = avg_degree
    p_in = min(1.0, homophily * deg / max(1, n_per_block - 1))
    cross_nodes = max(1, n_per_block * (n_blocks - 1))
    p_out = min(1.0, (1.0 - homophily) * deg / cross_nodes)
    probs = [[p_in if i == j else p_out for j in range(n_blocks)] for i in range(n_blocks)]

    G = nx.stochastic_block_model(sizes, probs, seed=seed)
    edge_index = _edge_index_from_nx(G)
    y = np.repeat(np.arange(n_blocks), n_per_block)

    rng = np.random.default_rng(seed)
    centroids = rng.standard_normal((n_blocks, n_features)).astype(np.float32)
    X = feature_signal * centroids[y] + rng.standard_normal((n, n_features)).astype(np.float32)

    data = Data(x=torch.tensor(X), edge_index=edge_index,
                y=torch.tensor(y, dtype=torch.long))
    add_node_splits(data, seed=seed)
    h = homophily_ratio(edge_index, data.y)

    gt = GraphGroundTruth(
        task="node_classification",
        description="SBM: label = community. Features carry partial class signal; structure carries the rest.",
        n_features=n_features,
        homophily=h,
        extra={"n_classes": n_blocks, "feature_signal": feature_signal,
               "requested_homophily": homophily},
    )
    return data, gt


def make_heterophilous_graph(
    n_per_block: int = 250,
    homophily: float = 0.2,
    avg_degree: float = 10.0,
    n_features: int = 6,
    feature_signal: float = 0.5,
    seed: int = 42,
):
    """
    A disassortative (heterophilous) 2-class graph: neighbours tend to be the
    OPPOSITE class. This deliberately breaks the homophily assumption baked into
    vanilla GCN, which averages neighbours and therefore smears the classes
    together. (Implemented as a low-homophily SBM.)
    """
    data, gt = make_sbm_homophily(
        n_per_block=n_per_block, n_blocks=2, homophily=homophily,
        avg_degree=avg_degree, n_features=n_features,
        feature_signal=feature_signal, seed=seed,
    )
    gt.description = "Disassortative SBM: a node's neighbours tend to be the OPPOSITE class (homophily < 0.5)."
    return data, gt


# ===========================================================================
# 3. Structural-role node classification  -  signal is purely in the topology
# ===========================================================================
def make_structural_role(n_gadgets: int = 160, n_features: int = 6, seed: int = 42):
    """
    Each node belongs to a small 3-node *gadget* that is either a TRIANGLE (a
    3-clique) or a PATH. The label is "am I in a triangle?". Node features are
    PURE NOISE, so the label is decided entirely by local structure - a clean
    probe of whether a model can read topology at all (motivates GIN and the
    structural encodings in notebook 08).
    """
    rng = np.random.default_rng(seed)
    G = nx.Graph()
    labels: list[int] = []
    reps: list[int] = []
    node = 0
    for g in range(n_gadgets):
        is_tri = (g % 2 == 0)
        a, b, c = node, node + 1, node + 2
        G.add_nodes_from([a, b, c])
        G.add_edges_from([(a, b), (b, c)])
        if is_tri:
            G.add_edge(a, c)                      # close the triangle
        labels += [int(is_tri)] * 3
        reps.append(a)
        node += 3
    # Stitch gadgets into one connected graph via a path through their reps.
    # (A path adds no triangles, so it does not contaminate the labels.)
    for u, v in zip(reps[:-1], reps[1:]):
        G.add_edge(u, v)

    n = node
    edge_index = _edge_index_from_nx(G)
    X = rng.standard_normal((n, n_features)).astype(np.float32)   # pure noise
    y = np.array(labels, dtype=np.int64)

    data = Data(x=torch.tensor(X), edge_index=edge_index,
                y=torch.tensor(y, dtype=torch.long))
    add_node_splits(data, seed=seed)

    gt = GraphGroundTruth(
        task="node_classification",
        description="Label = node sits in a triangle gadget vs a path gadget. Features are PURE NOISE; only local structure decides the label.",
        n_features=n_features, motif="triangle",
        extra={"n_classes": 2, "n_nodes": n},
    )
    return data, gt


# ===========================================================================
# 4. Graph-level expressivity dataset  -  why sum-aggregation (GIN) beats mean
# ===========================================================================
def make_expressivity_graphs(
    n_graphs: int = 600,
    n_nodes: int = 18,
    p_low: float = 0.15,
    p_high: float = 0.32,
    seed: int = 42,
):
    """
    Two classes of Erdos-Renyi graphs that differ only in EDGE DENSITY (hence in
    node degrees). Node features are CONSTANT (all ones).

    Why this separates GIN from a mean-aggregation GNN:
      * With constant features, the mean of a node's neighbours is always 1, no
        matter how many neighbours it has -> a mean-aggregation model is blind to
        degree and cannot tell the two classes apart.
      * SUM aggregation (GIN) encodes the number of neighbours, so it reads the
        degree/density difference and classifies correctly.
    This is the practical face of the Weisfeiler-Lehman expressivity argument.
    """
    rng = np.random.default_rng(seed)
    graphs = []
    for i in range(n_graphs):
        dense = (i % 2 == 0)
        p = p_high if dense else p_low
        G = nx.erdos_renyi_graph(n_nodes, p, seed=int(rng.integers(1_000_000_000)))
        edge_index = _edge_index_from_nx(G)
        x = torch.ones((n_nodes, 1), dtype=torch.float32)        # constant features
        graphs.append(Data(x=x, edge_index=edge_index,
                           y=torch.tensor([int(dense)], dtype=torch.long)))

    gt = GraphGroundTruth(
        task="graph_classification",
        description="Class = denser vs sparser graph (a degree/density difference). Constant node features mean the signal is purely structural - sum-aggregation reads it, mean-aggregation cannot.",
        n_features=1, motif="edge-density",
        extra={"n_classes": 2, "n_graphs": n_graphs, "p_low": p_low, "p_high": p_high},
    )
    return graphs, gt


# ===========================================================================
# 5. RingTransfer  -  long-range dependency / over-squashing probe
# ===========================================================================
def make_ring_transfer(
    num_graphs: int = 600,
    ring_size: int = 12,
    num_classes: int = 5,
    seed: int = 42,
):
    """
    A ring of ``ring_size`` nodes. A one-hot class label is placed on the *target*
    node, which sits ``ring_size // 2`` hops away from the *source* node. The graph
    label must be read out from the SOURCE node's embedding.

    To solve it, information has to travel ~half the ring through a narrow path -
    the textbook setup for **over-squashing**: a k-layer MPNN needs k >= distance,
    and even then the signal gets squashed through the bottleneck. A graph
    transformer (notebook 08) can read the target in a single global-attention hop.

    Each graph carries a boolean ``is_source`` node mask; the readout uses it.
    """
    rng = np.random.default_rng(seed)
    half = ring_size // 2
    fdim = num_classes + 1                       # +1 marker channel for the source node
    edges = [(j, (j + 1) % ring_size) for j in range(ring_size)]
    base = nx.Graph()
    base.add_edges_from(edges)
    edge_index = _edge_index_from_nx(base)

    graphs = []
    for _ in range(num_graphs):
        label = int(rng.integers(num_classes))
        x = np.zeros((ring_size, fdim), dtype=np.float32)
        x[half, label] = 1.0                     # target node carries the answer
        x[0, num_classes] = 1.0                  # source node marker
        is_source = torch.zeros(ring_size, dtype=torch.bool)
        is_source[0] = True
        d = Data(x=torch.tensor(x), edge_index=edge_index,
                 y=torch.tensor([label], dtype=torch.long))
        d.is_source = is_source
        graphs.append(d)

    gt = GraphGroundTruth(
        task="graph_classification",
        description=f"RingTransfer: the answer sits {half} hops from the source node, which must read it out. Needs >= {half} message-passing steps; a long-range / over-squashing probe.",
        n_features=fdim, motif="ring-transfer",
        extra={"n_classes": num_classes, "ring_size": ring_size, "distance": half,
               "n_graphs": num_graphs},
    )
    return graphs, gt


# ===========================================================================
# Real benchmarks (download-guarded; callers should wrap in try/except)
# ===========================================================================
def load_cora(root: str = "../data/Planetoid"):
    """Cora citation network (node classification). Requires a one-time download."""
    from torch_geometric.datasets import Planetoid
    ds = Planetoid(root=root, name="Cora")
    return ds[0], ds


def load_mutag(root: str = "../data/TUDataset"):
    """MUTAG molecules (graph classification). Requires a one-time download."""
    from torch_geometric.datasets import TUDataset
    ds = TUDataset(root=root, name="MUTAG")
    return ds
