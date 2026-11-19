"""
Extended Data Figure 1: within- and across-patient sampling similarity.

Pseudobulk per core (raw counts summed, logCPM), Pearson correlation between
cores on gene-wise standardised logCPM, grouped by pair category:
same patient/same status, same patient/different status, different patient
with the same diagnosis and status, and all other pairs.
Differences in median r are tested by permutation (labels permuted at the
level of cores/patients, because pairs sharing a core are not independent).

Also writes (supplementary, CSV only):
    cell-type-level pseudobulk similarity, gene-level ICC,
    and core-vs-core scatter plots for every replicate pair.

Input: see "Paths" below (edit to match your data location).
Outputs (results/ExtDataFig1/):
    EDFig1_core_similarity_by_category.pdf
    within_patient_pairs_core_level.csv, within_patient_pairs_by_celltype.csv
    gene_icc_*.csv, core_vs_core_scatter.pdf
Note: the CSV files and the scatter plot titles contain patient IDs.

Run from the repository root:  python ExtDataFig1/edfig1_core_similarity.py
"""
# %% Imports and setup
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt

# %% Paths (relative to the repository root; edit if your data live elsewhere)
ADATA_ANALYSIS = Path("data/xenium_custom_panglioma.h5ad")
OUTDIR = Path("results/ExtDataFig1")
OUTDIR.mkdir(parents=True, exist_ok=True)

# keep text editable in Illustrator/Inkscape
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42

# %% Configuration
COUNTS_LAYER = "counts"         # raw counts layer; None -> use .X (must be raw counts)
CORE_COL = "core"               # core names already contain the TMA id
PATIENT_COL = "patient_id"
STATUS_COL = "Primary/recurrent"
DX_COL = "Diagnosis"
CT_COL = "Annotation_merged_scanVI"
MERGE_IDS = {"58661": "7557"}   # old IDs that are the same patient; {} to disable
MIN_CELLS_CORE = 200            # minimum cells for a core to be used
MIN_CELLS_CT = 30               # minimum cells of a cell type in a core for cell-type pseudobulk
MIN_CORES_CT = 6                # minimum cores with that cell type to analyse it
N_PERM = 2000                   # permutations for the boxplot tests

BG = "diff patient, same dx+status"
WITHIN = "same patient, same status"
OTHER = "diff patient, other"

# %% Load data
rdata = sc.read_h5ad(ADATA_ANALYSIS)

# merge astrocyte subclusters
rdata.obs[CT_COL] = (rdata.obs[CT_COL].astype(str)
                     .replace({"Astrocytes1": "Astrocytes", "Astrocytes2": "Astrocytes"})
                     .astype("category"))

# overview: patients with more than one core
cores_per_patient = rdata.obs.groupby(PATIENT_COL, observed=True)[CORE_COL].nunique()
multi = cores_per_patient[cores_per_patient > 1]
print(f"{len(multi)} of {len(cores_per_patient)} patients have >1 core")

# %% Analysis
ad = rdata

obs = ad.obs
pid = obs[PATIENT_COL].astype(str).str.replace(r"\.0$", "", regex=True)
obs["pid"] = pid.replace(MERGE_IDS)
obs["core_s"] = obs[CORE_COL].astype(str)
obs["status_s"] = obs[STATUS_COL].astype(str).str.strip().str.lower()

# each core must belong to one patient; drop stray cells with a minority patient label
major = (obs.groupby(["core_s", "pid"], observed=True).size().reset_index(name="n")
            .sort_values("n", ascending=False).drop_duplicates("core_s")
            .set_index("core_s")["pid"])
stray = obs["pid"].values != obs["core_s"].map(major).values
if stray.any():
    print(f"WARNING: dropped {stray.sum()} cells ({stray.mean():.2%}) with a minority "
          f"patient label within their core")
    ad = ad[~stray].copy()
    obs = ad.obs

core_meta = obs.groupby("core_s", observed=True).agg(
    pid=("pid", "first"), status=("status_s", "first"),
    dx=(DX_COL, "first"), n_cells=("core_s", "size"))
core_meta["dx"] = core_meta["dx"].astype(str)
core_meta = core_meta[core_meta.n_cells >= MIN_CELLS_CORE]
core_meta["group"] = core_meta.pid + " | " + core_meta.status
gsize = core_meta.group.value_counts()
core_meta["is_rep"] = core_meta.group.map(gsize) >= 2

X = ad.layers[COUNTS_LAYER] if COUNTS_LAYER and COUNTS_LAYER in ad.layers else ad.X
X = sp.csr_matrix(X)
if not np.allclose(X[:1000].data, np.round(X[:1000].data)):
    print("WARNING: expression matrix does not look like raw counts")

# %% helpers
def pseudobulk(keys, min_cells):
    """Sum raw counts per key; keys is an array with NaN for cells to skip."""
    keys = pd.Series(keys)
    ok = keys.notna().values
    cats, codes = np.unique(keys[ok].values, return_inverse=True)
    rows = np.where(ok)[0]
    G = sp.csr_matrix((np.ones(len(rows)), (codes, rows)), shape=(len(cats), X.shape[0]))
    pb = pd.DataFrame(np.asarray((G @ X).todense()), index=cats, columns=ad.var_names)
    ncell = np.bincount(codes, minlength=len(cats))
    return pb.loc[ncell >= min_cells]


def logcpm(pb):
    return np.log1p(pb.div(pb.sum(axis=1), axis=0) * 1e6)


def corr_matrix(L):
    """Pearson r between cores on gene-wise standardised logCPM."""
    Z = (L - L.mean()) / L.std(ddof=0).replace(0, np.nan)
    return np.corrcoef(Z.dropna(axis=1).values)


def similarity_pairs(L, info):
    C = corr_matrix(L)
    idx = L.index
    i, j = np.triu_indices(len(idx), 1)
    m = info.loc[idx]
    sp_ = m.pid.values[i] == m.pid.values[j]
    ss = m.status.values[i] == m.status.values[j]
    sd = m.dx.values[i] == m.dx.values[j]
    p = pd.DataFrame({"core1": idx[i], "core2": idx[j], "r": C[i, j],
                      "dx": m.dx.values[i], "status": m.status.values[i]})
    p["category"] = np.select([sp_ & ss, sp_, sd & ss],
                              [WITHIN, "same patient, diff status", BG], OTHER)
    p["group"] = np.where(sp_ & ss, m.group.values[i], None)
    bg = p[p.category == BG]
    w = p.category == WITHIN
    p.loc[w, "pct_vs_bg"] = [
        (bg.loc[(bg.dx == d) & (bg.status == s), "r"] < r).mean() * 100
        if ((bg.dx == d) & (bg.status == s)).any() else np.nan
        for r, d, s in zip(p.loc[w, "r"], p.loc[w, "dx"], p.loc[w, "status"])]
    return p


def permutation_tests(L, info, n_perm=N_PERM, seed=0):
    """Pairs share cores, so they are not independent and a Mann-Whitney test on
    pairs would be far too optimistic. Instead, labels are permuted at the level
    of cores / patients and the difference in median r is recomputed.

    Test 1 (same patient vs diff patient, same dx+status):
        shuffle patient labels among cores within each diagnosis x status stratum.
    Test 2 (diff patient same dx+status vs diff patient other):
        shuffle diagnosis x status labels among patient groups (a patient's cores move together).
    """
    rng = np.random.default_rng(seed)
    C = corr_matrix(L)
    m = info.loc[L.index]
    i, j = np.triu_indices(len(m), 1)
    r = C[i, j]
    pid = m.pid.values
    strat = (m.dx + "|" + m.status).values
    grp = m.group.values

    def stats(pid_, strat_):
        same_p = pid_[i] == pid_[j]
        same_s = strat_[i] == strat_[j]
        w, b, o = same_p & same_s, ~same_p & same_s, ~same_p & ~same_s
        return np.median(r[w]) - np.median(r[b]), np.median(r[b]) - np.median(r[o])

    obs1, obs2 = stats(pid, strat)

    strata_idx = [np.where(strat == s)[0] for s in np.unique(strat)]
    null1 = np.empty(n_perm)
    for k in range(n_perm):
        p_ = pid.copy()
        for w in strata_idx:
            p_[w] = rng.permutation(p_[w])
        null1[k] = stats(p_, strat)[0]

    groups, inv = np.unique(grp, return_inverse=True)
    g_strat = pd.Series(strat).groupby(inv).first().values
    null2 = np.empty(n_perm)
    for k in range(n_perm):
        null2[k] = stats(pid, rng.permutation(g_strat)[inv])[1]

    p1 = (np.sum(null1 >= obs1) + 1) / (n_perm + 1)
    p2 = (np.sum(null2 >= obs2) + 1) / (n_perm + 1)
    return {(WITHIN, BG): p1, (BG, OTHER): p2}


def fmt_p(p, n_perm=N_PERM):
    return f"p < {1 / (n_perm + 1):.1g}" if p <= 1 / (n_perm + 1) + 1e-12 else f"p = {p:.2g}"


def add_bracket(ax, x1, x2, y, text, h=0.03):
    ax.plot([x1, x1, x2, x2], [y, y + h, y + h, y], lw=1, c="k")
    ax.text((x1 + x2) / 2, y + h, text, ha="center", va="bottom", fontsize=9)


def gene_icc(L, groups):
    """One-way ICC(1) per gene across replicate groups."""
    g = pd.Series(groups, index=L.index)
    n_i = g.value_counts()
    a, N = len(n_i), len(g)
    gm = L.groupby(g).mean()
    grand = L.mean()
    msb = (gm.sub(grand) ** 2).mul(n_i.loc[gm.index], axis=0).sum() / (a - 1)
    msw = ((L - gm.loc[g.values].values) ** 2).sum() / (N - a)
    k0 = (N - (n_i ** 2).sum() / N) / (a - 1)
    icc = (msb - msw) / (msb + (k0 - 1) * msw)
    return pd.DataFrame({"ICC": icc, "within_patient_sd": np.sqrt(msw),
                         "between_patient_sd": np.sqrt(msb.clip(lower=0) / k0),
                         "mean_logcpm": grand}).sort_values("ICC")

# %% 1. core-level boxplot + tests
core_keys = obs["core_s"].where(obs["core_s"].isin(core_meta.index))
L_core = logcpm(pseudobulk(core_keys, MIN_CELLS_CORE))
pairs = similarity_pairs(L_core, core_meta)
within = pairs[pairs.category == WITHIN].sort_values("pct_vs_bg")
within["flag_inconsistent"] = within.pct_vs_bg < 50
within.to_csv(f"{OUTDIR}/within_patient_pairs_core_level.csv", index=False)

pvals = permutation_tests(L_core, core_meta)

order = [WITHIN, "same patient, diff status", BG, OTHER]
order = [o for o in order if o in pairs.category.values]
counts = pairs.category.value_counts()
fig, ax = plt.subplots(figsize=(6, 5.5))
sns.boxplot(data=pairs, x="category", y="r", order=order, showfliers=False, ax=ax)
sns.stripplot(data=pairs[pairs.category.str.startswith("same patient")], x="category", y="r",
              order=order, color="k", size=4, ax=ax)
top = pairs.r.max()
for k, ((a, b), p) in enumerate(pvals.items()):
    if a in order and b in order:
        add_bracket(ax, order.index(a), order.index(b), top + 0.05 + 0.12 * k, fmt_p(p))
ax.set_ylim(top=top + 0.35)
ax.set_xticks(range(len(order)))
ax.set_xticklabels([f"{o}\n(n = {counts[o]} pairs)" for o in order],
                   rotation=30, ha="right")
ax.set_ylabel("Pearson r (standardised logCPM)")
ax.set_xlabel("")
ax.set_title(f"Core similarity; permutation test on median difference ({N_PERM} perm.)",
             fontsize=9)
plt.tight_layout()
plt.savefig(OUTDIR / "EDFig1_core_similarity_by_category.pdf")
plt.show()
