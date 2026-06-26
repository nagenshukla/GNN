"""
A small, reusable model zoo shared across the notebooks.

The point is that every notebook can pull a clean model from here and stay focused
on the *idea* it is teaching rather than on boilerplate. Everything is a thin
``nn.Module`` over PyTorch Geometric layers, except the deliberately hand-rolled
``ScratchGNN`` used in notebook 01 to show message passing with no magic.

Models come in two flavours:
  * node-level   (``MLP``, ``GNN``)              -> one prediction per node
  * graph-level  (``GraphGNN``, ``GPSModel``)    -> one prediction per graph (pooled)
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import (
    GCNConv, SAGEConv, GATConv, GATv2Conv, GINConv,
    GPSConv, global_mean_pool, global_add_pool,
)


# ---------------------------------------------------------------------------
# A from-scratch message-passing layer (notebook 01) - no PyG magic.
# ---------------------------------------------------------------------------
class ScratchGNN(nn.Module):
    """
    Message passing written out by hand so the mechanics are visible:

        for each layer:
            m_i = MEAN over neighbours j of h_j          (message + aggregate)
            h_i = ReLU( W_self h_i + W_neigh m_i )       (update)

    We build the (symmetric, self-loop-free) mean-aggregation matrix once from
    edge_index and reuse it. This is exactly what ``GCNConv`` etc. do for you,
    just spelled out.
    """
    def __init__(self, in_dim: int, hidden: int, out_dim: int, n_layers: int = 2):
        super().__init__()
        self.self_lins = nn.ModuleList()
        self.neigh_lins = nn.ModuleList()
        dims = [in_dim] + [hidden] * (n_layers - 1)
        for d in dims:
            self.self_lins.append(nn.Linear(d, hidden))
            self.neigh_lins.append(nn.Linear(d, hidden))
        self.head = nn.Linear(hidden, out_dim)

    @staticmethod
    def neighbor_mean(h: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """Mean of each node's neighbour embeddings (edge_index is symmetric)."""
        src, dst = edge_index
        n = h.size(0)
        agg = torch.zeros_like(h)
        agg.index_add_(0, dst, h[src])
        deg = torch.zeros(n, device=h.device).index_add_(
            0, dst, torch.ones(dst.size(0), device=h.device))
        deg = deg.clamp_min(1.0).unsqueeze(1)
        return agg / deg

    def forward(self, x, edge_index):
        h = x
        for self_lin, neigh_lin in zip(self.self_lins, self.neigh_lins):
            m = self.neighbor_mean(h, edge_index)
            h = F.relu(self_lin(h) + neigh_lin(m))
        return self.head(h)


# ---------------------------------------------------------------------------
# PairNorm - a cheap anti-over-smoothing normalization (notebook 06).
# ---------------------------------------------------------------------------
class PairNorm(nn.Module):
    """Centre node features, then rescale so total feature variance is preserved
    across layers - counteracts the collapse that causes over-smoothing."""
    def __init__(self, scale: float = 1.0):
        super().__init__()
        self.scale = scale

    def forward(self, x):
        x = x - x.mean(dim=0, keepdim=True)
        denom = x.pow(2).sum(dim=1, keepdim=True).mean().sqrt().clamp_min(1e-6)
        return self.scale * x / denom


# ---------------------------------------------------------------------------
# Node-level baseline that IGNORES the graph (the "tabular model" stand-in).
# ---------------------------------------------------------------------------
class MLP(nn.Module):
    """A plain MLP on node features. It never looks at ``edge_index`` - so it is
    the control that isolates *what reading the graph actually buys you*."""
    def __init__(self, in_dim, hidden, out_dim, n_layers: int = 2, dropout: float = 0.5):
        super().__init__()
        layers, d = [], in_dim
        for _ in range(n_layers - 1):
            layers += [nn.Linear(d, hidden), nn.ReLU(), nn.Dropout(dropout)]
            d = hidden
        layers += [nn.Linear(d, out_dim)]
        self.net = nn.Sequential(*layers)

    def forward(self, x, edge_index=None, *args, **kwargs):
        return self.net(x)


# ---------------------------------------------------------------------------
# Conv factory.
# ---------------------------------------------------------------------------
def make_conv(kind: str, in_dim: int, out_dim: int, heads: int = 4):
    kind = kind.lower()
    if kind == "gcn":
        return GCNConv(in_dim, out_dim)
    if kind == "sage":
        return SAGEConv(in_dim, out_dim)
    if kind == "gat":                       # heads averaged -> output dim == out_dim
        return GATConv(in_dim, out_dim, heads=heads, concat=False)
    if kind == "gatv2":
        return GATv2Conv(in_dim, out_dim, heads=heads, concat=False)
    if kind == "gin":
        mlp = nn.Sequential(nn.Linear(in_dim, out_dim), nn.ReLU(),
                            nn.Linear(out_dim, out_dim))
        return GINConv(mlp, train_eps=True)
    raise ValueError(f"unknown conv kind: {kind!r}")


# ---------------------------------------------------------------------------
# Configurable node-level GNN, used by most notebooks.
# ---------------------------------------------------------------------------
class GNN(nn.Module):
    """
    A stack of message-passing layers for node-level tasks, with the knobs the
    notebooks need: conv type, depth, dropout, normalization, residual
    connections, and Jumping-Knowledge (concat of every layer's output).
    """
    def __init__(self, in_dim, hidden, out_dim, conv: str = "gcn",
                 n_layers: int = 2, heads: int = 4, dropout: float = 0.5,
                 norm: str | None = None, residual: bool = False, jk: bool = False):
        super().__init__()
        self.conv_kind = conv
        self.dropout = dropout
        self.residual = residual
        self.jk = jk
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        for i in range(n_layers):
            cin = in_dim if i == 0 else hidden
            self.convs.append(make_conv(conv, cin, hidden, heads=heads))
            self.norms.append(self._make_norm(norm, hidden))
        head_in = hidden * n_layers if jk else hidden
        self.head = nn.Linear(head_in, out_dim)

    @staticmethod
    def _make_norm(norm, hidden):
        if norm is None:
            return nn.Identity()
        if norm == "pair":
            return PairNorm()
        if norm == "batch":
            return nn.BatchNorm1d(hidden)
        if norm == "layer":
            return nn.LayerNorm(hidden)
        raise ValueError(f"unknown norm: {norm!r}")

    def forward(self, x, edge_index, return_attention: bool = False):
        h = x
        layer_outs, attentions = [], []
        for conv, norm in zip(self.convs, self.norms):
            if return_attention and self.conv_kind in ("gat", "gatv2"):
                h_new, (ei, alpha) = conv(h, edge_index, return_attention_weights=True)
                attentions.append((ei, alpha))
            else:
                h_new = conv(h, edge_index)
            h_new = norm(h_new)
            h_new = F.relu(h_new)
            h_new = F.dropout(h_new, p=self.dropout, training=self.training)
            if self.residual and h_new.shape == h.shape:
                h_new = h_new + h
            h = h_new
            layer_outs.append(h)
        rep = torch.cat(layer_outs, dim=1) if self.jk else h
        out = self.head(rep)
        if return_attention:
            return out, attentions
        return out


# ---------------------------------------------------------------------------
# Graph-level GNN (pooling readout), used for graph classification.
# ---------------------------------------------------------------------------
class GraphGNN(nn.Module):
    """
    Node embeddings -> readout -> one prediction per graph.

    ``pool`` chooses the readout:
      * "mean" / "add" : global pooling over all nodes
      * "source"       : gather the single node flagged by ``is_source`` per graph
                         (used by the RingTransfer task)
    """
    def __init__(self, in_dim, hidden, out_dim, conv: str = "gin",
                 n_layers: int = 3, heads: int = 4, dropout: float = 0.5,
                 pool: str = "add"):
        super().__init__()
        self.dropout = dropout
        self.pool = pool
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        for i in range(n_layers):
            cin = in_dim if i == 0 else hidden
            self.convs.append(make_conv(conv, cin, hidden, heads=heads))
            self.norms.append(nn.BatchNorm1d(hidden))
        self.head = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(),
                                  nn.Dropout(dropout), nn.Linear(hidden, out_dim))

    def _readout(self, h, batch, is_source):
        if self.pool == "source":
            return h[is_source]                          # one row per graph, in graph order
        if self.pool == "mean":
            return global_mean_pool(h, batch)
        return global_add_pool(h, batch)

    def forward(self, x, edge_index, batch, is_source=None):
        h = x
        for conv, norm in zip(self.convs, self.norms):
            h = F.relu(norm(conv(h, edge_index)))
            h = F.dropout(h, p=self.dropout, training=self.training)
        return self.head(self._readout(h, batch, is_source))


# ---------------------------------------------------------------------------
# GraphGPS-style graph transformer (notebook 08): local MPNN + global attention.
# ---------------------------------------------------------------------------
class GPSModel(nn.Module):
    """
    A GraphGPS hybrid: every layer runs a local GIN message-passing step AND a
    global multi-head attention step, then fuses them. This lets a node exchange
    information with *any* other node in one layer - sidestepping the locality
    bottleneck that causes over-squashing on long-range tasks.

    Positional/structural encodings should be concatenated into ``x`` before the
    model (so ``in_dim`` already includes them).
    """
    def __init__(self, in_dim, hidden, out_dim, n_layers: int = 2, heads: int = 4,
                 dropout: float = 0.0, pool: str = "mean"):
        super().__init__()
        self.pool = pool
        self.lin_in = nn.Linear(in_dim, hidden)
        self.convs = nn.ModuleList()
        for _ in range(n_layers):
            local = GINConv(nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(),
                                          nn.Linear(hidden, hidden)))
            self.convs.append(GPSConv(hidden, conv=local, heads=heads, dropout=dropout))
        self.head = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(),
                                  nn.Linear(hidden, out_dim))

    def _readout(self, h, batch, is_source):
        if self.pool == "source":
            return h[is_source]
        if self.pool == "add":
            return global_add_pool(h, batch)
        return global_mean_pool(h, batch)

    def forward(self, x, edge_index, batch, is_source=None):
        h = self.lin_in(x)
        for conv in self.convs:
            h = conv(h, edge_index, batch)
        return self.head(self._readout(h, batch, is_source))
