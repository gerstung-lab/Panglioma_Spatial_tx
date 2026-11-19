"""
Neighbourhood labels for the Xenium 5K astrocytoma dataset (input for Extended Data Fig. 10C).

The 5K data are too few cells to define neighbourhoods de novo, so the neighbourhood
labels of the discovery cohort are transferred:

  1. Cell types of both datasets are harmonised to a shared vocabulary.
  2. Features per cell: counts of each shared cell type among its direct
     neighbours in the spatial (Delaunay) graph, the cell itself excluded.
  3. A multinomial logistic regression (standardised features, balanced class
     weights) is trained on low-grade astrocytoma cells of the discovery cohort,
     using their neighbourhood labels, and applied to all 5K cells.

Inputs:
    data/rdata_5tx_pca_euclidean_umap.h5ad       discovery cohort (reference)
    data/xenium5k_astrocytoma_annotated.h5ad     5K dataset (needs obsp["spatial_connectivities"])
Outputs (results/analysis/xenium5k_nhood_transfer/):
    xenium5k_neighbourhood_labels.csv            label + confidence per 5K cell
    reference_self_prediction_confusion.csv      % per true reference label
"""

# %% Imports and setup
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc
from scipy import sparse
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# %% Paths (relative to the repository root)
REFERENCE = Path("data/xenium_custom_panglioma.h5ad")
QUERY = Path("data/xenium5k_astrocytoma_annotated.h5ad")
OUT_DIR = Path("results/analysis/xenium5k_nhood_transfer")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# %% Settings
SEED = 42
CONNECTIVITY_KEY = "spatial_connectivities"

REF_CELLTYPE_COL = "Annotation_merged_scanVI"
REF_NHOOD_COL = "leiden_nhood_annot"
REF_DIAGNOSES = ["Astrocytoma low-grade"]
QUERY_CELLTYPE_COL = "annotation_5k"

# neighbourhoods present in low-grade astrocytoma that are transferred
# (reference labels -> names in the paper)
NHOOD_NAMES = {
    "Vascular": "Vascular",
    "Non-malignant CNS": "TBI",
    "TBI": "TBI",
    "OPC-like": "OPC-like",
    "AC-like": "AC-like",
    "MES1": "Gliosis",
    "Proliferating/Immune": "Proliferating",
    "Stressed": "Stressed",
}

# cell types -> shared vocabulary; "Other" is not used as a feature
REF_TO_COMMON = {
    "T-CD8+": "T-cells", "T-CD4+": "T-cells", "T-cycling": "T-cells",
    "B cells": "Other", "NK cells": "Other", "Plasma cells": "Other", "Mast cells": "Other",
    "Neurons-inh.": "Inhibitory Neurons", "Neurons-exh.": "Excitatory Neurons",
    "Astrocytes1": "Astrocytes", "Astrocytes2": "Astrocytes", "Astrocytes": "Astrocytes",
    "OPC/OC": "OPC/Oligodendrocytes",
    "Myeloid-P2RY12+": "Myeloid-P2RY12+", "Myeloid-CD163+": "Myeloid-CD163+",
    "Myeloid-CLEC10A+": "Other", "Myeloid-RNASE1+": "Other", "Neutrophils": "Neutrophils",
    "Vascular EC": "Endothelia", "Perivascular fibroblasts": "Perivascular",
    "Tumor AC-like": "Tumor AC-like", "Tumor MES1": "Tumor MES1", "Tumor OPC-like": "Tumor OPC-like",
    "Tumor Proliferating": "Tumor Proliferating", "Tumor Stressed": "Tumor Stressed",
    # tumour states not annotated in the 5K data
    "Tumor MES2": "Other", "Tumor NPC-like": "Other",
    "Tumor Factor (Spinal Glia)": "Other", "Tumor Factor (Cortical Glia)": "Other",
}
QUERY_TO_COMMON = {
    "T cells": "T-cells", "Endothelia": "Endothelia", "Pericytes": "Perivascular",
    "Myeloid CD163+": "Myeloid-CD163+", "Myeloid P2RY12+": "Myeloid-P2RY12+",
    "Neutrophils": "Neutrophils", "OPC/Oligodendrocytes": "OPC/Oligodendrocytes",
    "Excitatory Neurons": "Excitatory Neurons", "Inhibitory Neurons": "Inhibitory Neurons",
    "Astrocytes": "Astrocytes",
    "Tumor AC-like": "Tumor AC-like", "Tumor OPC-like": "Tumor OPC-like",
    "Tumor Proliferating": "Tumor Proliferating", "Tumor MES1": "Tumor MES1",
    "Tumor Stressed": "Tumor Stressed",
}


def neighbour_counts(adata, celltypes, categories):
    """Counts of each cell type among the direct spatial-graph neighbours (self excluded)."""
    A = adata.obsp[CONNECTIVITY_KEY].tocsr().copy()
    A.data[:] = 1                       # binarise weighted graphs
    A.setdiag(0)
    A.eliminate_zeros()
    onehot = pd.get_dummies(celltypes).reindex(columns=categories, fill_value=0)
    X = A @ sparse.csr_matrix(onehot.to_numpy(dtype=np.float32))
    return np.asarray(X.todense(), dtype=np.float32)


# %% Reference: low-grade astrocytoma cells with one of the transferred neighbourhoods
ref = sc.read_h5ad(REFERENCE)
ref_nhood = ref.obs[REF_NHOOD_COL].astype(str)
keep = ref.obs["Diagnosis"].astype(str).isin(REF_DIAGNOSES) & ref_nhood.isin(NHOOD_NAMES)
ref = ref[keep.values]
ref_ct = ref.obs[REF_CELLTYPE_COL].astype(str).map(REF_TO_COMMON)
unmapped = sorted(set(ref.obs[REF_CELLTYPE_COL].astype(str)[ref_ct.isna()]))
if unmapped:
    raise ValueError(f"Reference cell types without a mapping: {unmapped}")
y_ref = ref.obs[REF_NHOOD_COL].astype(str).map(NHOOD_NAMES).values
print(f"Reference: {ref.n_obs:,} cells")

# %% Query: Xenium 5K
query = sc.read_h5ad(QUERY)
query_ct = query.obs[QUERY_CELLTYPE_COL].astype(str).map(QUERY_TO_COMMON)
unmapped = sorted(set(query.obs[QUERY_CELLTYPE_COL].astype(str)[query_ct.isna()]))
if unmapped:
    raise ValueError(f"5K cell types without a mapping: {unmapped}")
print(f"Query: {query.n_obs:,} cells")

# features = cell types present in both datasets
categories = sorted((set(ref_ct) & set(query_ct)) - {"Other"})
print(f"{len(categories)} features:", categories)

X_ref = neighbour_counts(ref, ref_ct.values, categories)
X_query = neighbour_counts(query, query_ct.values, categories)

# %% Train on the reference, predict the 5K cells
clf = make_pipeline(
    StandardScaler(),
    LogisticRegression(max_iter=5000, class_weight="balanced", random_state=SEED),
)
clf.fit(X_ref, y_ref)

# reference self-prediction (on the training cells)
confusion = pd.crosstab(pd.Series(y_ref, name="true"), pd.Series(clf.predict(X_ref), name="predicted"),
                        normalize="index") * 100
print("\nReference self-prediction (% of true label):\n", confusion.round(1))
confusion.round(2).to_csv(OUT_DIR / "reference_self_prediction_confusion.csv")

proba = clf.predict_proba(X_query)
labels = pd.DataFrame({
    "neighbourhood": clf.classes_[proba.argmax(axis=1)],
    "confidence": proba.max(axis=1),
}, index=query.obs_names)
print("\n5K neighbourhoods:\n", labels["neighbourhood"].value_counts())

# agreement with the labels stored in the 5K object (if present)
if "neighbourhood" in query.obs.columns:
    same = (labels["neighbourhood"] == query.obs["neighbourhood"].astype(str)).mean()
    print(f"\nAgreement with obs['neighbourhood'] in the 5K object: {same:.1%}")

labels.to_csv(OUT_DIR / "xenium5k_neighbourhood_labels.csv")
print("Saved to", OUT_DIR)
