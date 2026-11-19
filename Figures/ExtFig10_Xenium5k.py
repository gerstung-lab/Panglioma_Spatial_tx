"""
Extended Data Figure 10: same-slide multimodal profiling of astrocytoma (Xenium 5K panel).

  A  UMAPs: all cells coloured by cell type; tumour cells coloured by tumour state
  B  z-scored mean expression of lineage marker genes per cell cluster
  C  composition of each neighbourhood: tumour states + TME (left),
     and the TME cell types within the TME (right)

Input:  data/xenium5k_astrocytoma_annotated.h5ad
        (annotated object; tumour-cell UMAP stored in obsm["X_umap_tumour"])
"""

# %% Imports and setup
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy import sparse

# %% Paths (relative to the repository root)
ADATA = Path("data/xenium5k_astrocytoma_annotated.h5ad")
OUT_DIR = Path("results/ExtDataFig10")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# keep text editable in Illustrator/Inkscape
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42

# %% Settings
CT_COL = "annotation_5k"            # detailed annotation (tumour states + TME cell types)
BROAD_COL = "annotation_5k_broad"   # tumour states collapsed into "Tumor"
NHOOD_COL = "neighbourhood"
LAYER = "lognorm"

# labels in the object -> labels in the figure
CT_NAMES = {
    "Tumor AC-like": "Tumour AC-like", "Tumor OPC-like": "Tumour OPC-like",
    "Tumor Proliferating": "Tumour Proliferating", "Tumor MES1": "Tumour MES1",
    "Tumor Stressed": "Tumour Stressed", "Tumor": "Tumour", "OPC/Oligodendrocytes": "OPC/OC",
}

TUMOUR_STATE_COLORS = {
    "Tumour AC-like": "#c97d63", "Tumour OPC-like": "#feb144", "Tumour Proliferating": "#e55934",
    "Tumour MES1": "#c97dba", "Tumour Stressed": "#3e1929",
}
TME_COLORS = {   # also the order of the TME bars in panel C
    "Endothelia": "#fad644", "Pericytes": "#d7b259", "Astrocytes": "#e04f39",
    "OPC/OC": "#fea022", "Excitatory Neurons": "#1976D2", "Inhibitory Neurons": "#64B5F6",
    "Myeloid CD163+": "#ff9cae", "Myeloid P2RY12+": "#eaccd1", "Neutrophils": "#cc87fa",
    "T cells": "#4CAF50",
}
CELL_TYPE_COLORS = {"Tumour": "#7fcdbb", **TME_COLORS}

NHOOD_ORDER = ["Vascular", "TBI", "OPC-like", "AC-like", "Gliosis", "Proliferating", "Stressed"]


def save_rasterized_umap(adata, color, basis, filename, size, title):
    """UMAP with rasterised points and vector text/legend."""
    fig = sc.pl.embedding(adata, basis=basis, color=color, size=size, frameon=False,
                          title=title, show=False, return_fig=True)
    for ax in fig.axes:
        for coll in ax.collections:
            coll.set_rasterized(True)
    fig.savefig(OUT_DIR / filename, format="pdf", bbox_inches="tight", dpi=300)
    plt.close(fig)
    print("Saved:", OUT_DIR / filename)


# %% Load data, figure labels and colours
adata = sc.read_h5ad(ADATA)
for col in [CT_COL, BROAD_COL]:
    adata.obs[col] = adata.obs[col].astype(str).replace(CT_NAMES).astype("category")

broad_cats = [c for c in CELL_TYPE_COLORS if c in set(adata.obs[BROAD_COL])]
adata.obs[BROAD_COL] = adata.obs[BROAD_COL].cat.reorder_categories(broad_cats)
adata.uns[f"{BROAD_COL}_colors"] = [CELL_TYPE_COLORS[c] for c in broad_cats]

tumour = adata[adata.obs[BROAD_COL] == "Tumour"].copy()
tumour.obsm["X_umap"] = tumour.obsm["X_umap_tumour"]
tumour.obs[CT_COL] = tumour.obs[CT_COL].cat.remove_unused_categories()
tumour.uns[f"{CT_COL}_colors"] = [TUMOUR_STATE_COLORS[c] for c in tumour.obs[CT_COL].cat.categories]

print(f"{adata.n_obs:,} cells, {tumour.n_obs:,} tumour cells")

# %% ED Fig. 10A: UMAPs
save_rasterized_umap(adata, BROAD_COL, "umap", "EDFig10A_umap_cell_types.pdf", 5, "Cell types")
save_rasterized_umap(tumour, CT_COL, "umap", "EDFig10A_umap_tumour_states.pdf", 7, "Tumour states")
del tumour

# %% ED Fig. 10B: lineage marker heatmap (z-score of mean expression per cluster)
group_order = ["Excitatory Neurons", "Inhibitory Neurons", "Tumour Proliferating",
               "Tumour OPC-like", "Tumour Stressed", "Tumour MES1", "Astrocytes",
               "Tumour AC-like", "OPC/OC", "Endothelia", "Pericytes", "Myeloid CD163+",
               "Myeloid P2RY12+", "Neutrophils", "T cells"]
group_order = [g for g in group_order if g in set(adata.obs[CT_COL])]

genes = ["SNAP25", "STMN2", "GAD1",               # neurons
         "TOP2A", "MKI67",                        # proliferation
         "EGFR", "OLIG1", "BCAN",                 # OPC-like
         "DNAJB5", "CD44",                        # stress / MES1
         "GFAP", "AQP4", "FGFR3",                 # astrocytes
         "TF", "MOG",                             # oligodendrocytes
         "VWF", "CLDN5", "MYO1B", "DCN",          # vasculature
         "C1QC", "CD163", "P2RY12", "CXCL8", "S100A9",   # myeloid
         "TRBC2", "KLRB1", "GZMA", "CD8A",        # lymphoid
         "NES"]                                   # progenitor
genes = [g for g in genes if g in adata.var_names]

X = adata[:, genes].layers[LAYER]
X = X.toarray() if sparse.issparse(X) else np.asarray(X)
mean_expr = (pd.DataFrame(X, index=adata.obs_names, columns=genes)
             .groupby(adata.obs[CT_COL].astype(str).values).mean()
             .loc[group_order])
z = ((mean_expr - mean_expr.mean(axis=0)) / mean_expr.std(axis=0)).replace(
    [np.inf, -np.inf], np.nan).fillna(0)

fig, ax = plt.subplots(figsize=(14, max(5, 0.35 * len(group_order))))
sns.heatmap(z, cmap="RdBu_r", center=0, vmax=3, linewidths=0.3, linecolor="white",
            cbar_kws={"label": "Mean expression (z-score)"}, ax=ax)
ax.set_xlabel("Marker genes", fontweight="bold")
ax.set_ylabel("Cell clusters", fontweight="bold")
plt.setp(ax.get_xticklabels(), rotation=90, fontstyle="italic")
plt.setp(ax.get_yticklabels(), rotation=0)
plt.tight_layout()
fig.savefig(OUT_DIR / "EDFig10B_marker_heatmap.pdf", bbox_inches="tight")
plt.close(fig)
z.to_csv(OUT_DIR / "EDFig10B_source_data_zscore.csv")

# %% ED Fig. 10C: neighbourhood composition
obs = adata.obs[[NHOOD_COL, CT_COL]].dropna().astype(str)
nhoods = [n for n in NHOOD_ORDER if n in set(obs[NHOOD_COL])]
tumour_states = [s for s in TUMOUR_STATE_COLORS if s in set(obs[CT_COL])]
obs["compartment"] = np.where(obs[CT_COL].isin(tumour_states), obs[CT_COL], "TME")

# left: tumour states + all TME, % of all cells in the neighbourhood
left = pd.crosstab(obs[NHOOD_COL], obs["compartment"]).reindex(
    index=nhoods, columns=tumour_states + ["TME"], fill_value=0)
left = left.div(left.sum(axis=1), axis=0) * 100

# right: TME cell types, % of the TME cells in the neighbourhood
tme = obs[obs["compartment"] == "TME"]
tme_types = [t for t in TME_COLORS if t in set(tme[CT_COL])]
right = pd.crosstab(tme[NHOOD_COL], tme[CT_COL]).reindex(index=nhoods, columns=tme_types, fill_value=0)
right = right.div(right.sum(axis=1), axis=0) * 100

left_colors = {**TUMOUR_STATE_COLORS, "TME": "#BDBDBD"}
y = np.arange(len(nhoods))
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, max(3.5, 0.42 * len(nhoods))), sharey=True,
                               gridspec_kw={"width_ratios": [1.3, 1], "wspace": 0.12})
for ax, table, colors in [(ax1, left, left_colors), (ax2, right, TME_COLORS)]:
    bottom = np.zeros(len(nhoods))
    for col in table.columns:
        ax.barh(y, table[col].values, left=bottom, color=colors[col], edgecolor="white",
                linewidth=0.3, height=0.82)
        bottom += table[col].values
    ax.set_xlim(0, 100)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(axis="y", length=0)

ax1.axvline(50, color="gray", linestyle="--", linewidth=0.8, alpha=0.6)
ax1.set_yticks(y)
ax1.set_yticklabels(nhoods)
ax1.invert_yaxis()
ax1.set_ylabel("Neighbourhoods", fontsize=12, fontweight="bold")
ax1.set_xlabel("% of all cells in neighborhood")
ax2.set_xlabel("% within the TME in neighborhood")
ax1.set_title("Tumour states", fontweight="bold")
ax2.set_title("Tumour microenvironment", fontweight="bold")

ax1.legend(handles=[mpatches.Patch(color=left_colors[c], label=c.replace("Tumour ", ""))
                    for c in left.columns],
           title="Tumour states / TME", loc="upper left", bbox_to_anchor=(0, -0.25),
           frameon=False, ncol=2, fontsize=8, title_fontsize=9)
ax2.legend(handles=[mpatches.Patch(color=TME_COLORS[c], label=c) for c in tme_types],
           title="TME cell types", loc="upper left", bbox_to_anchor=(0, -0.25),
           frameon=False, ncol=2, fontsize=8, title_fontsize=9)
fig.savefig(OUT_DIR / "EDFig10C_neighbourhood_composition.pdf", bbox_inches="tight")
plt.close(fig)
pd.concat({"tumour_states_and_TME_pct_of_all": left, "TME_cell_types_pct_of_TME": right},
          axis=1).to_csv(OUT_DIR / "EDFig10C_source_data.csv")
print("Done:", OUT_DIR)
