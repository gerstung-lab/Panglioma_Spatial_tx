"""
Figure 1D and 1E: cell-level UMAPs and lineage marker heatmap.

  1D  UMAPs coloured by cell type, diagnostic group and primary/recurrent status
  1E  z-scored mean expression of lineage markers per cell type
  plus a table of the top 10 marker genes per cell type (Wilcoxon)

Input: see "Paths" below (edit to match your data location).

Outputs (results/Figure1/):
    Fig1D_umap_celltypes.pdf, Fig1D_umap_diagnosis.pdf, Fig1D_umap_primary_recurrent.pdf
    Fig1E_marker_zscore_heatmap.pdf
    markers_per_cell_type_top10.xlsx
"""

# %% Imports and setup
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
from scipy import sparse

# %% Paths (relative to the repository root; edit if your data live elsewhere)
ADATA_ANALYSIS = Path("data/rdata_5tx_pca_euclidean_umap.h5ad")
OUT_DIR = Path("results/Figure1")
OUT_DIR.mkdir(parents=True, exist_ok=True)

CT_COL = "Annotation_merged_scanVI"
LAYER = "lognorm"

# keep text editable in Illustrator/Inkscape
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42

# %% Colour palettes
CELL_TYPE_COLORS = {
    # Tumour states
    "Tumor Factor (Cortical Glia)": "#b6a179",
    "Tumor OPC-like": "#feb144",
    "Tumor NPC-like": "#5e8c61",
    "Tumor Proliferating": "#e55934",
    "Tumor AC-like": "#c97d63",
    "Tumor MES1": "#c97dba",
    "Tumor MES2": "#8687c1",
    "Tumor Stressed": "#3e1929",
    "Tumor Factor (Spinal Glia)": "#8a7d72",
    # Glia / neurons
    "Astrocytes": "#e04f39",
    "Neurons-exh.": "#1976D2",
    "Neurons-inh.": "#64B5F6",
    "OPC/OC": "#fea022",
    # Myeloid
    "Myeloid-P2RY12+": "#eaccd1",
    "Myeloid-CD163+": "#ff9cae",
    "Myeloid-CLEC10A+": "#ff7196",
    "Myeloid-RNASE1+": "#febccc",
    "Neutrophils": "#cc87fa",
    # Vasculature
    "Vascular EC": "#fad644",
    "Perivascular fibroblasts": "#d7b259",
    # Lymphocytes
    "T-CD8+": "#A8E6A3",
    "T-CD4+": "#4CAF50",
    "B cells": "#388E3C",
    "NK cells": "#81C784",
    "Plasma cells": "#45a1a0",
    "Mast cells": "#839976",
    "T-cycling": "#839976",
}
TUMOR_COLOR = "#7fcdbb"

DIAGNOSIS_COLORS = {
    "Glioblastoma": "#b46f70",
    "Astrocytoma high-grade": "#ecb25f",
    "Astrocytoma low-grade": "#a8cf9e",
    "Oligodendroglioma": "#e8e47f",
    "Spinal ependymoma": "#c9afd5",
    "Myxopapillary ependymoma": "#74449b",
    "Subependymoma": "#ad97c6",
}

PRIMARY_RECURRENT_COLORS = {
    "primary": "#bfe6ea",
    "recurrence": "#4c84a8",
    "recurrent": "#4c84a8",
}

# %% Load data and set colours
rdata = sc.read_h5ad(ADATA_ANALYSIS)

# merge astrocyte subclusters
rdata.obs[CT_COL] = (rdata.obs[CT_COL].astype(str)
                     .replace({"Astrocytes1": "Astrocytes", "Astrocytes2": "Astrocytes"})
                     .astype("category"))

# all tumour states share one colour
rdata.uns[f"{CT_COL}_colors"] = [
    TUMOR_COLOR if label.startswith("Tumor") else CELL_TYPE_COLORS[label]
    for label in rdata.obs[CT_COL].cat.categories
]
rdata.uns["Diagnosis_colors"] = [DIAGNOSIS_COLORS[label] for label in rdata.obs["Diagnosis"].cat.categories]
rdata.uns["Primary/recurrent_colors"] = [
    PRIMARY_RECURRENT_COLORS[label] for label in rdata.obs["Primary/recurrent"].cat.categories
]

# all tumour states merged into a single "Tumor" group
TUMOR_MERGED_COL = "celltype_tumor_merged"
rdata.obs[TUMOR_MERGED_COL] = (rdata.obs[CT_COL].astype(str)
                               .where(lambda s: ~s.str.startswith("Tumor"), "Tumor")
                               .astype("category"))
print(rdata.obs[TUMOR_MERGED_COL].value_counts())

# %% Marker gene table (top 10 per cell type)
N_TOP = 10

# max 20k cells per cell type, to keep it fast
idx = (rdata.obs.groupby(TUMOR_MERGED_COL, observed=True, group_keys=False)
             .apply(lambda d: d.sample(min(len(d), 20000), random_state=0)).index)
sub_ct = rdata[idx].copy()

sc.tl.rank_genes_groups(sub_ct, TUMOR_MERGED_COL, layer=LAYER, method="wilcoxon", pts=True)
markers_ct = sc.get.rank_genes_groups_df(sub_ct, group=None)

top_ct = (markers_ct[markers_ct.logfoldchanges > 0]
          .groupby("group", observed=True).head(N_TOP)
          .copy())
top_ct.insert(1, "rank", top_ct.groupby("group", observed=True).cumcount() + 1)
top_ct = top_ct.rename(columns={"group": "cell_type", "names": "gene"})
top_ct.to_excel(OUT_DIR / f"markers_per_cell_type_top{N_TOP}.xlsx", index=False)
del sub_ct

# %% Figure 1D: UMAPs (points rasterised, text and legend kept as vector)
def save_rasterized_umap(adata, color, filename, size=0.15):
    fig = sc.pl.umap(adata, color=color, size=size, show=False, return_fig=True)
    for ax in fig.axes:
        for coll in ax.collections:
            coll.set_rasterized(True)
    out = OUT_DIR / filename
    fig.savefig(out, format="pdf", bbox_inches="tight", dpi=300)
    plt.close(fig)
    print("Saved:", out)


save_rasterized_umap(rdata, CT_COL, "Fig1D_umap_celltypes.pdf")
save_rasterized_umap(rdata, "Diagnosis", "Fig1D_umap_diagnosis.pdf")
save_rasterized_umap(rdata, "Primary/recurrent", "Fig1D_umap_primary_recurrent.pdf")

# %% Figure 1E: lineage marker heatmap
GROUPBY = "Annotation_tumor_collapsed"
if GROUPBY not in rdata.obs.columns:
    # same grouping as TUMOR_MERGED_COL: all tumour states collapsed into "Tumor"
    rdata.obs[GROUPBY] = rdata.obs[TUMOR_MERGED_COL]

group_order = [
    # brain resident / glia
    "Astrocytes", "OPC/OC", "Neurons-exh.", "Neurons-inh.",
    # myeloid
    "Myeloid-CD163+", "Myeloid-P2RY12+", "Myeloid-CLEC10A+", "Myeloid-RNASE1+", "Neutrophils",
    # lymphoid / immune
    "T-CD4+", "NK cells", "T-CD8+", "T-cycling", "B cells", "Plasma cells", "Mast cells",
    # vasculature / stromal
    "Vascular EC", "Perivascular fibroblasts",
    # tumour last
    "Tumor",
]
group_order = [g for g in group_order if g in rdata.obs[GROUPBY].astype(str).unique()]

genes_ordered = [
    "AQP4", "FGFR3",                    # astrocytes
    "PLP1", "OLIG1",                    # oligodendrocytes / OPC
    "STMN2", "GAD1",                    # neurons
    "C1QC", "CD163", "LYZ", "P2RY12", "CLEC10A", "RNASE1", "S100A9",   # myeloid
    "PTPRC", "KLRB1", "GZMA", "CD3G", "CD8A", "TOP2A", "CD19", "MZB1",  # lymphoid / immune
    "KIT",                              # mast cells
    "VWF", "MYO1B", "COL3A1",           # vasculature / stromal
    "EGFR", "NES",                      # tumour / progenitor
]
genes_ordered = list(dict.fromkeys(genes_ordered))
genes_present = [g for g in genes_ordered if g in rdata.var_names]
genes_missing = [g for g in genes_ordered if g not in rdata.var_names]
if genes_missing:
    print("Genes not in panel:", genes_missing)

X = rdata[:, genes_present].layers[LAYER]
if sparse.issparse(X):
    X = X.toarray()

expr = pd.DataFrame(X, index=rdata.obs_names, columns=genes_present)
expr[GROUPBY] = rdata.obs[GROUPBY].astype(str).values
mean_expr = expr.groupby(GROUPBY)[genes_present].mean().loc[group_order]

# z-score per gene across groups
z = (mean_expr - mean_expr.mean(axis=0)) / mean_expr.std(axis=0)
z = z.replace([np.inf, -np.inf], np.nan).fillna(0)

plt.figure(figsize=(14, max(5, 0.35 * len(group_order))))
ax = sns.heatmap(z, cmap="RdBu_r", center=0, vmax=3, linewidths=0.3, linecolor="white",
                 cbar_kws={"label": "z-score of mean expression"})
ax.set_xlabel("Marker genes")
ax.set_ylabel("")
ax.set_title("Lineage marker expression across cell annotations")
plt.xticks(rotation=90, ha="center")
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig(OUT_DIR / "Fig1E_marker_zscore_heatmap.pdf", format="pdf", bbox_inches="tight")
plt.show()

z.to_csv(OUT_DIR / "Fig1E_marker_zscore_source_data.csv")

