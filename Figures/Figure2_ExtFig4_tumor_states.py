"""
Figure 2 and Extended Data Figure 4: tumour states in adult gliomas.

  Fig. 2A    UMAPs of tumour cells, coloured by diagnostic group and tumour state
  Fig. 2B    tumour state characterisation (mean z-score of marker genes per state)
  Fig. 2C    marker gene expression on the tumour UMAP
  Fig. 2D    tumour state abundance per diagnostic group
  Fig. 2E    spatial plots of two example cores (tumour states and marker genes)
  ED Fig. 4  mean marker expression per tumour state, per diagnostic group

Inputs: see "Paths" below.
    tumour object  -> Fig. 2A-D and ED Fig. 4
    full object    -> Fig. 2E (all cells of the example cores)
Outputs: results/Figure2/ (figures and source data tables)

Run from the repository root:  python Figure2/figure2_and_edfig4.py
"""

# %% Imports and setup
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
from scipy import sparse

# %% Paths (relative to the repository root; edit if your data live elsewhere)
ADATA_TUMOR = Path("data/rdata_5tx_TUMOR_pca30_euclidean_nn10_umap_DEFAULT.h5ad")
ADATA_FULL = Path("data/rdata_5tx_pca_euclidean_umap.h5ad")
OUT_DIR = Path("results/Figure2")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# keep text editable in Illustrator/Inkscape
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42

# %% Settings
STATE_COL = "Annotation_merged_scanVI"
DX_COL = "Diagnosis"
LAYER = "lognorm"

# Annotation labels in the objects -> names used in the paper (also the plotting order)
STATE_NAMES = {
    "Tumor Factor (Cortical Glia)": "OPC-like I",
    "Tumor OPC-like": "OPC-like II",
    "Tumor NPC-like": "NPC-like",
    "Tumor Proliferating": "Proliferating",
    "Tumor AC-like": "AC-like",
    "Tumor MES1": "MES1",
    "Tumor MES2": "MES2",
    "Tumor Stressed": "Stressed",
    "Tumor Factor (Spinal Glia)": "EPN-like",
}
STATE_ORDER = list(STATE_NAMES.values())

STATE_COLORS = {
    "OPC-like I": "#b6a179",
    "OPC-like II": "#feb144",
    "NPC-like": "#5e8c61",
    "Proliferating": "#e55934",
    "AC-like": "#c97d63",
    "MES1": "#c97dba",
    "MES2": "#8687c1",
    "Stressed": "#3e1929",
    "EPN-like": "#8a7d72",
}

DIAGNOSIS_COLORS = {
    "Glioblastoma": "#b46f70",
    "Astrocytoma high-grade": "#ecb25f",
    "Astrocytoma low-grade": "#a8cf9e",
    "Oligodendroglioma": "#e8e47f",
    "Spinal ependymoma": "#c9afd5",
    "Myxopapillary ependymoma": "#74449b",
    "Subependymoma": "#ad97c6",
}

# marker genes for Fig. 2B and ED Fig. 4
MARKER_GENES = ["PLP1", "NES", "OLIG1", "BCAN", "EGFR", "SOX4", "SOX11", "TOP2A", "MKI67",
                "AQP4", "SPON1", "CD44", "A2M", "VEGFA", "IGFBP5", "HSPA1A", "HSPA1B",
                "FANK1", "KCNMA1"]

# genes shown on the UMAP in Fig. 2C
UMAP_GENES = ["OLIG1", "EGFR", "SOX11", "TOP2A", "AQP4", "CD44", "VEGFA", "FANK1"]

# x-axis order in Fig. 2D
DX_ORDER_2D = ["Oligodendroglioma", "Astrocytoma low-grade", "Astrocytoma high-grade",
               "Glioblastoma", "Myxopapillary ependymoma", "Spinal ependymoma", "Subependymoma"]

# panel order in ED Fig. 4
DX_ORDER_ED4 = ["Astrocytoma low-grade", "Astrocytoma high-grade", "Glioblastoma",
                "Oligodendroglioma", "Subependymoma", "Myxopapillary ependymoma",
                "Spinal ependymoma"]

# example cores for Fig. 2E: fill in the core names (as in obs["core"])
SPATIAL_CASES = {
    "Oligodendroglioma case": {
        "core": "FILL_IN",
        "states": ["MES1", "OPC-like I", "OPC-like II"],
        "genes": ["CD44", "PLP1", "OLIG1"],
    },
    "Glioblastoma case": {
        "core": "FILL_IN",
        "states": ["MES1", "MES2", "OPC-like I"],
        "genes": ["CD44", "VEGFA", "OLIG1"],
    },
}

# colour map for expression (Fig. 2C, 2E) and ED Fig. 4
WARM_RED = LinearSegmentedColormap.from_list(
    "warm_red", ["#E6E6E6", "#FDE0D2", "#F4A582", "#D6604D", "#B2182B"])
WARM_RED_HEATMAP = LinearSegmentedColormap.from_list(
    "warm_red_heatmap", ["#F0F0F0", "#FDE0D2", "#F4A582", "#D6604D", "#B2182B"])


def save_rasterized_umap(adata, color, filename, size=0.15):
    """UMAP with rasterised points and vector text/legend."""
    fig = sc.pl.umap(adata, color=color, size=size, show=False, return_fig=True)
    for ax in fig.axes:
        for coll in ax.collections:
            coll.set_rasterized(True)
    fig.savefig(OUT_DIR / filename, format="pdf", bbox_inches="tight", dpi=300)
    plt.close(fig)
    print("Saved:", OUT_DIR / filename)


def gene_values(adata, gene, layer=LAYER):
    """Expression of one gene as a dense 1D array."""
    x = adata[:, gene].layers[layer] if layer else adata[:, gene].X
    return x.toarray().ravel() if sparse.issparse(x) else np.asarray(x).ravel()


# %% Load tumour object
adata = sc.read_h5ad(ADATA_TUMOR)
adata = adata[adata.obs[STATE_COL].astype(str).isin(STATE_NAMES)].copy()

adata.obs["tumour_state"] = pd.Categorical(
    adata.obs[STATE_COL].astype(str).map(STATE_NAMES), categories=STATE_ORDER)
adata.uns["tumour_state_colors"] = [STATE_COLORS[s] for s in STATE_ORDER]

adata.obs[DX_COL] = adata.obs[DX_COL].astype("category")
adata.uns[f"{DX_COL}_colors"] = [DIAGNOSIS_COLORS[d] for d in adata.obs[DX_COL].cat.categories]

print(adata)
print(adata.obs["tumour_state"].value_counts())

# %% Fig. 2A: tumour UMAPs
save_rasterized_umap(adata, DX_COL, "Fig2A_umap_tumour_diagnosis.pdf")
save_rasterized_umap(adata, "tumour_state", "Fig2A_umap_tumour_states.pdf")

# %% Fig. 2B: mean z-score of marker genes per tumour state
# z-scores are computed per gene across all tumour cells
zdata = adata[:, MARKER_GENES].copy()
sc.pp.scale(zdata)

mp = sc.pl.matrixplot(zdata, MARKER_GENES, "tumour_state", categories_order=STATE_ORDER,
                      colorbar_title="Mean z-score", cmap="RdBu_r",
                      vcenter=0, vmin=-1, vmax=1.5, show=False, return_fig=True)
mp.savefig(OUT_DIR / "Fig2B_tumour_state_markers.pdf", bbox_inches="tight")
mp.values_df.to_csv(OUT_DIR / "Fig2B_source_data_mean_zscore.csv")
del zdata

# %% Fig. 2C: marker genes on the tumour UMAP
sc.set_figure_params(vector_friendly=True)   # rasterise points only
fig = sc.pl.umap(adata, color=UMAP_GENES, layer=LAYER, size=0.15, vmax=2, cmap=WARM_RED,
                 ncols=4, show=False, return_fig=True)
fig.savefig(OUT_DIR / "Fig2C_umap_marker_genes.pdf", bbox_inches="tight", dpi=300)
plt.close(fig)
matplotlib.rcParams["pdf.fonttype"] = 42      # set_figure_params resets rcParams
matplotlib.rcParams["ps.fonttype"] = 42

# %% Fig. 2D: tumour state abundance per diagnostic group
# fraction of each tumour state among the tumour cells of each core
counts = pd.crosstab([adata.obs[DX_COL], adata.obs["core"]], adata.obs["tumour_state"])
counts = counts[counts.sum(axis=1) > 0]
fractions = counts.div(counts.sum(axis=1), axis=0).reset_index().set_index("core")

core_ad = sc.AnnData(fractions[STATE_ORDER].astype(float))
core_ad.obs[DX_COL] = pd.Categorical(fractions[DX_COL].astype(str).values, categories=DX_ORDER_2D)

# dot size: fraction of cores with >2% of the state; colour: mean fraction per core
dp = sc.pl.dotplot(core_ad, var_names=STATE_ORDER, groupby=DX_COL,
                   categories_order=DX_ORDER_2D, expression_cutoff=0.02,
                   dendrogram=False, swap_axes=True, color_map="Reds", vmax=0.5,
                   colorbar_title="Mean fraction\nper core",
                   size_title="Fraction of cores\nwith >2% per state",
                   show=False, return_fig=True)
dp.savefig(OUT_DIR / "Fig2D_tumour_state_abundance.pdf", bbox_inches="tight")
pd.concat({"mean_fraction_per_core": dp.dot_color_df,
           "fraction_of_cores_above_2pct": dp.dot_size_df}, axis=1).to_csv(
    OUT_DIR / "Fig2D_source_data.csv")
fractions.reset_index(drop=True).to_csv(OUT_DIR / "Fig2D_source_data_per_core_fractions.csv",
                                        index=False)

# %% ED Fig. 4: mean marker expression per tumour state, per diagnostic group
# (plain mean lognorm expression; no gene weights)
expr = pd.DataFrame(
    np.column_stack([gene_values(adata, g) for g in MARKER_GENES]),
    columns=MARKER_GENES, index=adata.obs_names)
expr["state"] = adata.obs["tumour_state"].astype(str).values
expr["dx"] = adata.obs[DX_COL].astype(str).values

ed4_means = {
    dx: (expr[expr["dx"] == dx].groupby("state")[MARKER_GENES].mean()
         .reindex(STATE_ORDER).fillna(0))
    for dx in DX_ORDER_ED4
}
del expr

fig, axes = plt.subplots(4, 2, figsize=(8.27, 11.69))
axes = axes.ravel()
for ax, dx in zip(axes, DX_ORDER_ED4):
    df = ed4_means[dx]
    # colours scaled 0-1 per gene within each diagnostic group
    df_scaled = (df - df.min(axis=0)) / (df.max(axis=0) - df.min(axis=0) + 1e-8)
    sns.heatmap(df_scaled, cmap=WARM_RED_HEATMAP, vmin=0, vmax=1, cbar=False,
                xticklabels=True, yticklabels=True,
                linewidths=0.3, linecolor="white", ax=ax)
    ax.set_title(dx, fontsize=9, fontweight="bold")
    ax.set_xlabel("Genes", fontsize=7)
    ax.set_ylabel("Tumour states", fontsize=7)
    ax.tick_params(axis="x", labelsize=6, rotation=90)
    ax.tick_params(axis="y", labelsize=6, rotation=0)
    for label in ax.get_xticklabels():
        label.set_fontstyle("italic")

# colour bar in the empty last panel
axes[-1].axis("off")
cax = axes[-1].inset_axes([0.1, 0.1, 0.06, 0.8])
cb = fig.colorbar(matplotlib.cm.ScalarMappable(norm=Normalize(0, 1), cmap=WARM_RED_HEATMAP),
                  cax=cax, ticks=[0, 0.5, 1])
cb.set_label("Mean expression\n(scaled per gene)", fontsize=7)
cb.ax.tick_params(labelsize=6)

plt.tight_layout()
plt.savefig(OUT_DIR / "EDFig4_tumour_states_per_diagnosis.pdf", bbox_inches="tight")
plt.show()
plt.close(fig)

with pd.ExcelWriter(OUT_DIR / "EDFig4_source_data_mean_expression.xlsx") as xw:
    for dx, df in ed4_means.items():
        df.to_excel(xw, sheet_name=dx[:31])

del adata

# %% Fig. 2E: spatial plots of the example cores (full object, all cells)
missing = [name for name, case in SPATIAL_CASES.items() if case["core"] == "FILL_IN"]
if missing:
    raise SystemExit(f"Fig. 2E: set the core names in SPATIAL_CASES first ({missing})")

full = sc.read_h5ad(ADATA_FULL, backed="r")   # only the example cores are loaded into memory

for case_name, case in SPATIAL_CASES.items():
    core = full[full.obs["core"].astype(str) == case["core"]].to_memory()
    if core.n_obs == 0:
        raise ValueError(f"core {case['core']} not found")
    states = core.obs[STATE_COL].astype(str).map(STATE_NAMES)   # NaN for non-tumour cells
    x, y = core.obs["x_centroid"].to_numpy(), core.obs["y_centroid"].to_numpy()

    ncols = max(len(case["states"]), len(case["genes"]))
    fig, axes = plt.subplots(2, ncols, figsize=(4 * ncols, 8.5))

    # row 1: tumour states, highlighted on all cells in grey
    for ax, state in zip(axes[0], case["states"]):
        mask = (states == state).to_numpy()
        ax.scatter(x, y, s=2, c="#D3D3D3", edgecolors="none", rasterized=True)
        ax.scatter(x[mask], y[mask], s=2, c=STATE_COLORS[state], edgecolors="none", rasterized=True)
        ax.set_title(state, fontsize=11)

    # row 2: gene expression, highest values drawn on top
    for ax, gene in zip(axes[1], case["genes"]):
        values = gene_values(core, gene)
        order = np.argsort(values)
        im = ax.scatter(x[order], y[order], c=values[order], cmap=WARM_RED, vmin=0, vmax=2,
                        s=2, edgecolors="none", rasterized=True)
        ax.set_title(gene, fontsize=11, fontstyle="italic")

    for ax in axes.ravel():
        ax.set_xlim(x.min(), x.max())
        ax.set_ylim(y.max(), y.min())
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
    for ax in list(axes[0][len(case["states"]):]) + list(axes[1][len(case["genes"]):]):
        ax.remove()

    axes[0][0].set_ylabel("Tumour states", fontsize=11)
    axes[1][0].set_ylabel("Genes", fontsize=11)
    fig.suptitle(case_name, fontsize=12, fontstyle="italic")
    cb = fig.colorbar(im, ax=axes[1].tolist(), orientation="horizontal",
                      fraction=0.04, pad=0.04, shrink=0.4)
    cb.set_label("Norm. expression", fontsize=8)

    out = OUT_DIR / f"Fig2E_spatial_{case_name.split()[0].lower()}.pdf"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("Saved:", out)
