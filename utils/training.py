"""
Generic training / evaluation loops so the notebooks don't repeat boilerplate.

Two families of tasks:
  * node-level  (transductive): one graph, train/val/test *masks* over its nodes.
  * graph-level: many small graphs, batched with a DataLoader.

Every trainer returns a tidy results dict (history + best-epoch test metric +
parameter count) so notebooks can tabulate and plot model comparisons directly.
"""
from __future__ import annotations

import copy
import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader


# ---------------------------------------------------------------------------
# Reproducibility & misc
# ---------------------------------------------------------------------------
def set_seed(seed: int = 0):
    np.random.seed(seed)
    torch.manual_seed(seed)


def count_params(model) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def _accuracy(logits, y):
    return (logits.argmax(dim=1) == y).float().mean().item()


def _r2(pred, y):
    ss_res = ((pred - y) ** 2).sum().item()
    ss_tot = ((y - y.mean()) ** 2).sum().item() + 1e-12
    return 1.0 - ss_res / ss_tot


def _rmse(pred, y):
    return ((pred - y) ** 2).mean().sqrt().item()


# ---------------------------------------------------------------------------
# Node-level training (transductive, mask-based)
# ---------------------------------------------------------------------------
def train_node(model, data, task: str = "classification", epochs: int = 200,
               lr: float = 0.01, weight_decay: float = 5e-4, patience: int = 60,
               verbose: bool = False):
    """
    Train a node-level model and select the epoch with the best validation loss.

    task : "classification" (cross-entropy, reports accuracy) or
           "regression"     (MSE, reports R2 and RMSE).
    Returns a results dict with the val-selected test metric and a loss history.
    """
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    tr, va, te = data.train_mask, data.val_mask, data.test_mask
    y = data.y

    def loss_on(out, mask):
        if task == "classification":
            return F.cross_entropy(out[mask], y[mask])
        return F.mse_loss(out[mask], y[mask])

    def metric_on(out, mask):
        if task == "classification":
            return _accuracy(out[mask], y[mask])
        return _r2(out[mask], y[mask])

    history = {"train_loss": [], "val_loss": []}
    best_val, best_state, best_epoch = float("inf"), None, 0
    bad = 0
    for epoch in range(epochs):
        model.train()
        opt.zero_grad()
        out = model(data.x, data.edge_index)
        loss = loss_on(out, tr)
        loss.backward()
        opt.step()

        model.eval()
        with torch.no_grad():
            out = model(data.x, data.edge_index)
            vloss = loss_on(out, va).item()
        history["train_loss"].append(loss.item())
        history["val_loss"].append(vloss)

        if vloss < best_val - 1e-5:
            best_val, best_state, best_epoch, bad = vloss, copy.deepcopy(model.state_dict()), epoch, 0
        else:
            bad += 1
            if bad >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        out = model(data.x, data.edge_index)
        res = {
            "history": history,
            "best_epoch": best_epoch,
            "n_params": count_params(model),
            "train_metric": metric_on(out, tr),
            "val_metric": metric_on(out, va),
            "test_metric": metric_on(out, te),
        }
        if task == "regression":
            res["test_rmse"] = _rmse(out[te], y[te])
    if verbose:
        print(f"best epoch {best_epoch}: test metric = {res['test_metric']:.4f}")
    return res


# ---------------------------------------------------------------------------
# Graph-level training (batched)
# ---------------------------------------------------------------------------
def make_graph_loaders(graphs, batch_size: int = 32,
                       splits=(0.7, 0.15, 0.15), seed: int = 0):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(graphs))
    n_tr = int(splits[0] * len(graphs))
    n_va = int(splits[1] * len(graphs))
    parts = {
        "train": [graphs[i] for i in idx[:n_tr]],
        "val":   [graphs[i] for i in idx[n_tr:n_tr + n_va]],
        "test":  [graphs[i] for i in idx[n_tr + n_va:]],
    }
    return {k: DataLoader(v, batch_size=batch_size, shuffle=(k == "train"))
            for k, v in parts.items()}


def _graph_forward(model, batch):
    """Call a graph-level model, passing is_source through when the task needs it."""
    is_source = getattr(batch, "is_source", None)
    return model(batch.x, batch.edge_index, batch.batch, is_source=is_source)


@torch.no_grad()
def eval_graph(model, loader):
    model.eval()
    correct = total = 0
    for batch in loader:
        logits = _graph_forward(model, batch)
        correct += (logits.argmax(dim=1) == batch.y).sum().item()
        total += batch.y.size(0)
    return correct / max(1, total)


def train_graph(model, loaders, epochs: int = 100, lr: float = 0.01,
                weight_decay: float = 0.0, patience: int = 40, verbose: bool = False):
    """Train a graph-classification model, select best epoch by val accuracy."""
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    history = {"train_acc": [], "val_acc": []}
    best_val, best_state, best_epoch, bad = -1.0, None, 0, 0

    for epoch in range(epochs):
        model.train()
        for batch in loaders["train"]:
            opt.zero_grad()
            logits = _graph_forward(model, batch)
            loss = F.cross_entropy(logits, batch.y)
            loss.backward()
            opt.step()

        tr_acc = eval_graph(model, loaders["train"])
        va_acc = eval_graph(model, loaders["val"])
        history["train_acc"].append(tr_acc)
        history["val_acc"].append(va_acc)

        if va_acc > best_val + 1e-5:
            best_val, best_state, best_epoch, bad = va_acc, copy.deepcopy(model.state_dict()), epoch, 0
        else:
            bad += 1
            if bad >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    res = {
        "history": history,
        "best_epoch": best_epoch,
        "n_params": count_params(model),
        "train_metric": eval_graph(model, loaders["train"]),
        "val_metric": eval_graph(model, loaders["val"]),
        "test_metric": eval_graph(model, loaders["test"]),
    }
    if verbose:
        print(f"best epoch {best_epoch}: test acc = {res['test_metric']:.4f}")
    return res


# ---------------------------------------------------------------------------
# Over-smoothing diagnostics (notebook 06)
# ---------------------------------------------------------------------------
def dirichlet_energy(h: torch.Tensor, edge_index: torch.Tensor) -> float:
    """
    Sum over edges of ||h_i - h_j||^2 (normalized by #edges). Falls toward zero as
    node embeddings collapse into each other - the hallmark of over-smoothing.
    """
    src, dst = edge_index
    diff = h[src] - h[dst]
    return (diff.pow(2).sum(dim=1).mean()).item()
