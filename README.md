# Understanding Graph Neural Networks (GNNs)

A hands-on notebook series that takes apart modern GNNs and shows **how they
actually work** - message, aggregate, update - not just how to call `.fit()`. It is
the graph-shaped sequel to the sibling **GBDTs** series, and shares its core trick.

## The trick that makes this series work

We **don't** use a messy real dataset. We *generate* graphs where **we** plant the
signal - and we deliberately put it in different places:

- in the **node features** (a plain MLP can already use this),
- in the **local structure** (only a model that reads edges can use this),
- in **long-range structure** (only a deep / global model can use this).

Because we know the ground truth, every notebook asks a sharper question than
"what's the accuracy?":

> **Did the GNN recover the structure we planted - and did it ignore the noise?**

That also makes the most important question - **when should you use a GNN at all,
vs. a GBDT or a plain MLP?** - something we can *measure* instead of hand-wave.

## What's covered

| Idea | Headline | Notebook |
|---|---|---|
| **Message passing** | the message->aggregate->update template, from scratch | 01 |
| **GCN** | spectral-inspired degree-normalised averaging | 02 |
| **GraphSAGE** | inductive, sampled, scalable; separate self/neighbour | 03 |
| **GAT / GATv2** | learned attention over neighbours | 04 |
| **GIN** | maximal expressivity (1-WL), sum aggregation | 05 |
| **Over-smoothing / over-squashing** | why deeper isn't better, and the fixes | 06 |
| **Heterophily** | when neighbours *disagree* and GCN breaks | 07 |
| **Graph transformers** | global attention + positional/structural encodings (GraphGPS, Exphormer, Graph-Mamba) | 08 |
| **Link prediction** | edge-level tasks, encoder/decoder, negative sampling | 09 |
| **Prescription** | which model when - and when *not* a GNN | 10 |

## Notebooks

Run them in order:

| # | Notebook | Focus |
|---|---|---|
| 00 | `00_setup_and_synthetic_graphs.ipynb` | Env check, graph vocabulary, generate & visualise the synthetic suite |
| 01 | `01_message_passing_from_scratch.ipynb` | Build an MPNN by hand; beat an MLP exactly when the answer is in the graph |
| 02 | `02_gcn.ipynb` | Build A_hat by hand; where GCN shines (homophily) and where it blends/fails |
| 03 | `03_graphsage.ipynb` | Inductive learning on unseen nodes; neighbour sampling for scale |
| 04 | `04_gat.ipynb` | Attention over neighbours; visualise it; GATv2's dynamic-attention fix |
| 05 | `05_gin_and_expressivity.ipynb` | Weisfeiler-Lehman, the 1-WL ceiling, sum vs mean; graph classification |
| 06 | `06_oversmoothing_and_oversquashing.ipynb` | Depth pathologies, Dirichlet energy, PairNorm/JK/residual; bottlenecks |
| 07 | `07_heterophily.ipynb` | Disassortative graphs; ego/neighbour separation + 2-hop (H2GCN-style) |
| 08 | `08_graph_transformers_and_encodings.ipynb` | Global attention, RWSE/Laplacian PE, GraphGPS beats MPNN on long range |
| 09 | `09_link_prediction.ipynb` | Edge prediction, encoder/decoder, negative sampling, ROC-AUC |
| 10 | `10_prescription_and_comparison.ipynb` | Head-to-head + a decision guide, incl. **when not to use a GNN** |

## Setup

```bash
pip install -r requirements.txt
jupyter lab
```

Tested on **Python 3.14**, **PyTorch 2.12 (CPU)**, **PyTorch Geometric 2.8**. The
synthetic graphs are small - **no GPU needed**. We deliberately use only PyG's
pure-Python core (no `pyg-lib` / `torch-sparse` compiled extensions), so everything
installs from wheels with no compiler.

## Layout

```
GNN/
+-- requirements.txt
+-- README.md
+-- utils/
|   +-- graphs.py     # synthetic graph generators with a known GraphGroundTruth
|   +-- models.py     # model zoo: MLP, ScratchGNN, GCN, SAGE, GAT/GATv2, GIN, GraphGPS
|   +-- training.py   # train/eval loops (node / graph), metrics, over-smoothing diagnostics
|   +-- plotting.py   # graph drawing, attention viz, training curves, comparisons
+-- data/             # cached real datasets (Cora, MUTAG) - created on first use
+-- notebooks/        # the series
```


## The one-line prescription

> If your rows are independent and the signal is in each row's own features, reach
> for a **GBDT** (see the sibling series). Reach for a **GNN** when the target genuinely
> depends on **relationships** - and then let notebook 10's decision guide pick which one.
