"""
Shared plotting helpers so the notebooks stay focused on ideas, not matplotlib
boilerplate. Mirrors the GBDT series' plotting module (consistent per-model
colours, small composable functions).
"""
from __future__ import annotations

import numpy as np
import torch
import matplotlib.pyplot as plt
import networkx as nx
from torch_geometric.utils import to_networkx

# A consistent colour per model across all notebooks.
MODEL_COLORS = {
    "MLP": "#7f7f7f",          # the "tabular" control
    "Scratch": "#8c564b",
    "GCN": "#1f77b4",
    "GraphSAGE": "#2ca02c",
    "SAGE": "#2ca02c",
    "GAT": "#d62728",
    "GATv2": "#e377c2",
    "GIN": "#9467bd",
    "GPS": "#ff7f0e",          # graph transformer
    "H2GCN": "#17becf",
    "GPR-GNN": "#bcbd22",
}


def color_for(name: str) -> str:
    return MODEL_COLORS.get(name, "#333333")


# ---------------------------------------------------------------------------
# Graph drawing
# ---------------------------------------------------------------------------
def draw_graph(data, node_color=None, title: str = "", ax=None, layout: str = "spring",
               max_nodes: int = 400, node_size: int = 60, seed: int = 0,
               cmap: str = "tab10", with_labels: bool = False):
    """Draw a PyG graph (subsampled if large). ``node_color`` may be a label
    tensor/array; defaults to the node's ``y`` if present."""
    if ax is None:
        _, ax = plt.subplots(figsize=(5.5, 5))
    g_full = to_networkx(data, to_undirected=True)

    nodes = list(g_full.nodes())
    if len(nodes) > max_nodes:
        nodes = nodes[:max_nodes]
    G = g_full.subgraph(nodes)

    if node_color is None and getattr(data, "y", None) is not None and data.y.dim() == 1:
        node_color = data.y
    if node_color is not None:
        node_color = np.asarray(node_color)[nodes]

    pos = (nx.spring_layout(G, seed=seed) if layout == "spring"
           else nx.kamada_kawai_layout(G))
    nx.draw_networkx_edges(G, pos, ax=ax, alpha=0.25, width=0.7)
    nx.draw_networkx_nodes(G, pos, ax=ax, node_color=node_color, cmap=plt.get_cmap(cmap),
                           node_size=node_size, linewidths=0.4, edgecolors="white")
    if with_labels:
        nx.draw_networkx_labels(G, pos, ax=ax, font_size=7)
    ax.set_title(title)
    ax.axis("off")
    return ax


def draw_attention(data, edge_index, alpha, title: str = "", ax=None,
                   max_nodes: int = 200, seed: int = 0):
    """Draw a graph with edge widths/opacity proportional to a GAT attention
    weight vector ``alpha`` (one value per directed edge in ``edge_index``)."""
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 5.5))
    a = alpha.detach().cpu().numpy()
    if a.ndim > 1:
        a = a.mean(axis=1)                      # average attention heads
    ei = edge_index.cpu().numpy()

    G = nx.DiGraph()
    n = data.num_nodes
    keep = set(range(min(n, max_nodes)))
    for k in range(ei.shape[1]):
        u, v = int(ei[0, k]), int(ei[1, k])
        if u in keep and v in keep:
            G.add_edge(u, v, w=float(a[k]))
    pos = nx.spring_layout(G, seed=seed)
    ws = np.array([G[u][v]["w"] for u, v in G.edges()])
    ws = ws / (ws.max() + 1e-9)
    yc = None
    if getattr(data, "y", None) is not None and data.y.dim() == 1:
        yc = np.asarray(data.y)[list(G.nodes())]
    nx.draw_networkx_edges(G, pos, ax=ax, width=2.5 * ws, alpha=0.6,
                           edge_color="#444444", arrowsize=6)
    nx.draw_networkx_nodes(G, pos, ax=ax, node_color=yc, cmap=plt.cm.tab10,
                           node_size=80, edgecolors="white", linewidths=0.4)
    ax.set_title(title)
    ax.axis("off")
    return ax


# ---------------------------------------------------------------------------
# Training curves & model comparison
# ---------------------------------------------------------------------------
def plot_training_curves(history: dict, title: str = "Training", ax=None):
    """Plot a results['history'] dict. Handles loss histories and acc histories."""
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 4))
    for key, values in history.items():
        ax.plot(range(1, len(values) + 1), values, label=key)
    ax.set_xlabel("epoch")
    ax.set_ylabel("metric")
    ax.set_title(title)
    ax.legend()
    ax.grid(alpha=0.3)
    return ax


def bar_compare(results: dict, metric_key: str = "test_metric", title: str = "",
                ylabel: str = "test metric", ax=None):
    """Bar chart comparing models. ``results`` maps model-name -> results dict
    (or model-name -> float)."""
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 4))
    names = list(results.keys())
    vals = [r[metric_key] if isinstance(r, dict) else r for r in results.values()]
    colors = [color_for(n) for n in names]
    bars = ax.bar(names, vals, color=colors)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.3f}",
                ha="center", va="bottom", fontsize=9)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.3)
    return ax


def plot_sweep(x, series: dict, xlabel: str, ylabel: str, title: str = "", ax=None):
    """Line chart of one metric vs a swept knob (e.g. lambda, depth) for several
    models. ``series`` maps model-name -> list of y-values aligned with ``x``."""
    if ax is None:
        _, ax = plt.subplots(figsize=(6.5, 4.5))
    for name, ys in series.items():
        ax.plot(x, ys, marker="o", label=name, color=color_for(name))
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    ax.grid(alpha=0.3)
    return ax


# ---------------------------------------------------------------------------
# Embedding scatter (homophily / over-smoothing visual)
# ---------------------------------------------------------------------------
def scatter_embeddings(emb, labels=None, title: str = "", ax=None, method: str = "pca"):
    """Project node embeddings to 2-D and scatter, coloured by label. Good for
    *seeing* class separation (notebook 02) or collapse (notebook 06)."""
    if ax is None:
        _, ax = plt.subplots(figsize=(5, 4.5))
    h = emb.detach().cpu().numpy() if torch.is_tensor(emb) else np.asarray(emb)
    if h.shape[1] > 2:
        from sklearn.decomposition import PCA
        h = PCA(n_components=2).fit_transform(h)
    c = None if labels is None else (np.asarray(labels))
    ax.scatter(h[:, 0], h[:, 1], c=c, cmap="tab10", s=18, alpha=0.8,
               edgecolors="white", linewidths=0.2)
    ax.set_title(title)
    ax.set_xticks([]); ax.set_yticks([])
    return ax
