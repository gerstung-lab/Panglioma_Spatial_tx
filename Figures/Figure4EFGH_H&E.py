"""
Figure 4F-H: H&E histology (TITAN embeddings), histology clusters and neighbourhoods.

  Fig. 4F  UMAP of the TITAN H&E embeddings: Leiden histology clusters and diagnostic group
  Fig. 4G  the same UMAP coloured by neighbourhood fraction (TBI, AC-like, Gliosis, Hypoxia)
  Fig. 4H  do neighbourhoods explain the histology clusters beyond diagnosis?
           (astrocytoma LG/HG and glioblastoma; patient-grouped cross-validation,
           within-diagnosis permutation test, joint multinomial model)

Inputs:
    data/titan_features.parquet              TITAN slide embedding per core image
    source_data/Figure4_HE_core_table.csv    core, patient (pseudonym), diagnosis,
                                             neighbourhood fractions (he_prepare_core_table.py)

Note: To adhere to local regulations, H&E data can be shared after completion of a data transfer agreement for noncommercial use. 
Requests should be made to the corresponding author(s), with responses expected within 14 days of receipt of request.

"""

# %% Imports and setup
import warnings
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
from matplotlib.colors import LinearSegmentedColormap, LogNorm
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import FixedLocator, NullLocator
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, log_loss, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from umap import UMAP

warnings.filterwarnings("ignore", category=ConvergenceWarning)

# %% Paths (relative to the repository root)
TITAN_FEATURES = Path("data/titan_features.parquet")
CORE_TABLE = Path("source_data/Figure4_HE_core_table.csv")
OUT_DIR = Path("results/Figure4_HE")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# keep text editable in Illustrator/Inkscape
mpl.rcParams["pdf.fonttype"] = 42
mpl.rcParams["ps.fonttype"] = 42

# %% Settings
NHOOD_COLS = ["Vascular", "Non-malignant CNS", "AC-like", "MES1", "MES2", "OPC-like",
              "NPC-like", "Cortical glia", "Proliferating/Immune", "Stressed", "Spinal-glia"]
NHOOD_NAMES = {"Non-malignant CNS": "TBI", "MES1": "Gliosis", "MES2": "Hypoxia",
               "OPC-like": "OPC-like II", "Cortical glia": "OPC-like I",
               "Proliferating/Immune": "Proliferating", "Spinal-glia": "EPN-like"}
G_NHOODS = ["Non-malignant CNS", "AC-like", "MES1", "MES2"]          # shown in Fig. 4G

DIAGNOSIS_COLORS = {
    "Astrocytoma high-grade": "#EFB87C", "Astrocytoma low-grade": "#B1DBA1",
    "Glioblastoma": "#A56563", "Myxopapillary ependymoma": "#6415AA",
    "Oligodendroglioma": "#F5F19F", "Spinal ependymoma": "#E2C7FB", "Subependymoma": "#A789C2",
}
DX_SHORT = {"Astrocytoma high-grade": "Astrocytoma HG", "Astrocytoma low-grade": "Astrocytoma LG",
            "Myxopapillary ependymoma": "Myxopapillary EPN", "Spinal ependymoma": "Spinal EPN",
            "Subependymoma": "Sub-EPN"}

# Leiden cluster id -> cluster number in the figure, and colours
CLUSTER_ORDER = ["0", "4", "3", "1", "5", "2"]
CLUSTER_RENAME = {"0": "1", "4": "2", "3": "3", "1": "4", "5": "5", "2": "6"}
CLUSTER_COLORS = {"0": "#9d4edd", "4": "#2ec4b6", "3": "#eb5e28",
                  "1": "#04724d", "5": "#ffbf69", "2": "#f08080"}

FRACTION_CMAP = LinearSegmentedColormap.from_list(
    "he_fraction", ["#f2dae3", "#dfa8c0", "#bd78ad", "#8950a5", "#54227c"], N=256)


def umap_axes(n_legend_ratio=2.5):
    """UMAP panel (4.2 in) plus a separate legend panel."""
    fig = plt.figure(figsize=(6.7, 3.8))
    gs = GridSpec(1, 2, figure=fig, width_ratios=[4.2, n_legend_ratio], wspace=0.02)
    ax, legend_ax = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])
    legend_ax.axis("off")
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    return fig, ax, legend_ax


def move_legend(ax, legend_ax, title, labels=None):
    handles, auto_labels = ax.get_legend_handles_labels()
    if ax.legend_ is not None:
        ax.legend_.remove()
    legend_ax.legend(handles, labels or auto_labels, title=title, loc="center left",
                     frameon=False, borderaxespad=0, fontsize=8, title_fontsize=9,
                     markerscale=0.8, handletextpad=0.5, labelspacing=0.7)


def savefig(fig, name, **kw):
    fig.savefig(OUT_DIR / name, format="pdf", transparent=True, **kw)
    plt.close(fig)
    print("Saved:", OUT_DIR / name)


# %% Load data
titan = pd.read_parquet(TITAN_FEATURES)
cores = pd.read_csv(CORE_TABLE)

frame = pd.DataFrame({"slide_id": titan.index.astype(str)})
frame["core"] = (frame["slide_id"].str.split("_").str[0] + "_core_"
                 + frame["slide_id"].str.split("_").str[-1])
frame = frame.merge(cores, on="core", how="left", validate="one_to_one")

# %% UMAP and Leiden clusters on the TITAN embeddings
umapped = UMAP(n_components=2, random_state=5, metric="cosine").fit_transform(titan)
frame["UMAP1"], frame["UMAP2"] = umapped[:, 0], umapped[:, 1]

titan_ad = ad.AnnData(titan)
sc.pp.neighbors(titan_ad, use_rep="X", metric="cosine")
sc.tl.leiden(titan_ad, resolution=0.5)
frame["leiden"] = titan_ad.obs.loc[frame["slide_id"], "leiden"].astype(str).values
print(frame["leiden"].value_counts().sort_index())

frame.assign(cluster_in_figure=frame["leiden"].map(CLUSTER_RENAME))[
    ["slide_id", "core", "patient", "Diagnosis", "UMAP1", "UMAP2", "leiden",
     "cluster_in_figure"]].to_csv(OUT_DIR / "Fig4FG_source_data_umap_clusters.csv", index=False)

# %% Fig. 4F: histology clusters and diagnostic group (cores with a diagnosis)
with_dx = frame.dropna(subset=["Diagnosis"])
dx_order = [d for d in DIAGNOSIS_COLORS if d in set(with_dx["Diagnosis"])]
cl_order = [c for c in CLUSTER_ORDER if c in set(with_dx["leiden"])]

fig, ax, legend_ax = umap_axes()
sns.scatterplot(data=with_dx, x="UMAP1", y="UMAP2", hue="leiden", hue_order=cl_order,
                palette=CLUSTER_COLORS, s=20, linewidth=0, ax=ax)
ax.set_title("Histology clusters", fontsize=11)
move_legend(ax, legend_ax, "Cluster", [CLUSTER_RENAME[c] for c in cl_order])
savefig(fig, "Fig4F_umap_histology_clusters.pdf")

fig, ax, legend_ax = umap_axes()
sns.scatterplot(data=with_dx, x="UMAP1", y="UMAP2", hue="Diagnosis", hue_order=dx_order,
                palette=DIAGNOSIS_COLORS, s=20, linewidth=0, ax=ax)
ax.set_title("Diagnostic group", fontsize=11)
move_legend(ax, legend_ax, "Diagnostic group", [DX_SHORT.get(d, d) for d in dx_order])
savefig(fig, "Fig4F_umap_diagnostic_group.pdf")

# %% Fig. 4G: neighbourhood fraction on the H&E UMAP (all cores; grey = no spatial data)
for nhood in G_NHOODS:
    values = pd.to_numeric(frame[nhood], errors="coerce")
    missing = values.isna()
    valid = frame.loc[~missing, ["UMAP1", "UMAP2"]].assign(value=values[~missing]).sort_values("value")

    fig, ax, legend_ax = umap_axes()
    ax.scatter(frame.loc[missing, "UMAP1"], frame.loc[missing, "UMAP2"], color="#eeeeee", s=5,
               alpha=0.7, linewidths=0, rasterized=True)
    points = ax.scatter(valid["UMAP1"], valid["UMAP2"], c=valid["value"], cmap=FRACTION_CMAP,
                        vmin=0, vmax=1, s=20, linewidths=0)       # high fractions on top
    ax.set_title(NHOOD_NAMES.get(nhood, nhood), fontsize=11)
    cax = inset_axes(legend_ax, width="12%", height="60%", loc="center left", borderpad=0)
    cbar = fig.colorbar(points, cax=cax)
    cbar.set_label("Neighbourhood fraction", fontsize=9)
    cbar.ax.tick_params(labelsize=8)
    savefig(fig, f"Fig4G_umap_{NHOOD_NAMES.get(nhood, nhood)}.pdf")

# %% Fig. 4H: data for the cluster model
DIAGNOSES_KEEP = ["Glioblastoma", "Astrocytoma low-grade", "Astrocytoma high-grade"]
MIN_CLUSTER_SIZE = 10        # clusters with fewer cores in this subset are dropped
MIN_NHOOD_PRESENCE = 0.05    # nhoods present (>1% of a core) in <5% of cores are dropped
N_JOBS = -1
N_REPEATS = 20               # repeated patient-grouped 5-fold CV
N_PERM_GLOBAL = 200          # within-diagnosis permutations, global test
N_BOOT = 500                 # patient bootstrap, joint model

data = frame.copy()
data[NHOOD_COLS] = data[NHOOD_COLS].apply(pd.to_numeric, errors="coerce")
no_nhood = data[NHOOD_COLS].isna().all(axis=1)        # cores without spatial data
data = data.loc[~no_nhood].copy()
data[NHOOD_COLS] = data[NHOOD_COLS].fillna(0)        # nhood absent from a core = 0
data = data.dropna(subset=["Diagnosis", "leiden"])
data = data[data["Diagnosis"].isin(DIAGNOSES_KEEP)].copy()

# group = patient (cores of one patient always in the same CV fold); core if unknown
data["group"] = np.where(data["patient"].isna(), "core_" + data["core"], data["patient"].astype(str))

cl_n = data["leiden"].value_counts()
data = data[~data["leiden"].isin(cl_n[cl_n < MIN_CLUSTER_SIZE].index)].copy()
cluster_order = [c for c in CLUSTER_ORDER if c in set(data["leiden"])]

present = (data[NHOOD_COLS] > 0.01).mean()
nhood_cols = [n for n in NHOOD_COLS if present[n] >= MIN_NHOOD_PRESENCE]

classes = sorted(cluster_order)        # sklearn label order
y = data["leiden"].values
groups = data["group"].values
print(f"\nCluster model: {len(data)} cores, {data['group'].nunique()} patients, "
      f"clusters {cluster_order}, nhoods {nhood_cols}")


# %% Fig. 4H: helper functions
def make_model(C=1.0):
    return make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=5000))


def oof_proba(X, y, groups, classes, seed, n_splits=5, C=1.0):
    """Out-of-fold predicted probabilities, folds split by patient."""
    X, y = np.asarray(X, float), np.asarray(y)
    proba = np.zeros((len(y), len(classes)))
    cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for tr, te in cv.split(X, y, groups):
        m = make_model(C).fit(X[tr], y[tr])
        proba[np.ix_(te, [classes.index(c) for c in m.classes_])] = m.predict_proba(X[te])
    proba = np.clip(proba, 1e-6, None)
    return proba / proba.sum(1, keepdims=True)


def score_proba(y, proba, classes):
    out = {"macro AUC (OvR)": roc_auc_score(y, proba, multi_class="ovr", average="macro",
                                            labels=classes),
           "balanced accuracy": balanced_accuracy_score(y, np.asarray(classes)[proba.argmax(1)]),
           "log loss": log_loss(y, proba, labels=classes)}
    for k, c in enumerate(classes):
        out[f"AUC cluster {c}"] = roc_auc_score((np.asarray(y) == c).astype(int), proba[:, k])
    return out


def permute_within(values, strata, rng):
    """Shuffle rows among cores of the same diagnosis."""
    values = np.asarray(values)
    out = values.copy()
    for s in pd.unique(strata):
        idx = np.where(strata == s)[0]
        out[idx] = values[rng.permutation(idx)]
    return out


def global_permutation_test(df, nhood_cols, y, groups, classes, n_perm=200, n_splits=5,
                            C=1.0, seeds=(0, 1, 2), n_jobs=-1):
    """CV log-loss gain of (Dx + nhoods) over (Dx only); null = nhoods shuffled within Dx."""
    diag = pd.get_dummies(df["Diagnosis"], dtype=float).values
    nh = df[nhood_cols].values.astype(float)
    strata = df["Diagnosis"].values

    def delta_logloss(nh_mat):
        d = []
        for sd in seeds:
            p0 = oof_proba(diag, y, groups, classes, sd, n_splits, C)
            p1 = oof_proba(np.hstack([diag, nh_mat]), y, groups, classes, sd, n_splits, C)
            d.append(log_loss(y, p0, labels=classes) - log_loss(y, p1, labels=classes))
        return float(np.mean(d))

    observed = delta_logloss(nh)
    null = np.array(Parallel(n_jobs=n_jobs)(
        delayed(lambda i: delta_logloss(permute_within(nh, strata, np.random.default_rng(10_000 + i))))(i)
        for i in range(n_perm)))
    return observed, null, (1 + np.sum(null >= observed)) / (1 + n_perm)


def bh_fdr(p):
    p = np.asarray(p, float)
    n = len(p)
    order = np.argsort(p)
    ranked = np.minimum.accumulate((p[order] * n / np.arange(1, n + 1))[::-1])[::-1]
    q = np.empty(n)
    q[order] = np.minimum(ranked, 1)
    return q


def _joint_fit(X, y, classes, C):
    m = LogisticRegression(C=C, max_iter=5000).fit(StandardScaler().fit_transform(X), y)
    coef = np.full((len(classes), X.shape[1]), np.nan)
    for r, c in enumerate(m.classes_):
        coef[classes.index(c)] = m.coef_[r]
    return coef


def joint_model_coefficients(df, nhood_cols, classes, C=1.0, n_boot=500, n_jobs=-1, seed=0):
    """Multinomial model cluster ~ diagnosis + nhoods (standardised); patient-bootstrap CIs."""
    D = pd.get_dummies(df["Diagnosis"], dtype=float)
    names = list(D.columns) + list(nhood_cols)
    blocks = ["Diagnosis"] * D.shape[1] + ["Neighborhood"] * len(nhood_cols)
    X = np.hstack([D.values, df[nhood_cols].values.astype(float)])
    yy = df["leiden"].astype(str).values
    g = df["group"].values
    uniq = pd.unique(g)
    gidx = {u: np.where(g == u)[0] for u in uniq}
    coef = _joint_fit(X, yy, classes, C)

    def one(b):
        rng = np.random.default_rng([seed, b])
        idx = np.concatenate([gidx[u] for u in rng.choice(uniq, len(uniq), replace=True)])
        if X[idx].std(0).min() == 0:       # a predictor became constant
            return None
        return _joint_fit(X[idx], yy[idx], classes, C)

    boots = np.stack([b for b in Parallel(n_jobs=n_jobs)(delayed(one)(b) for b in range(n_boot))
                      if b is not None])
    rows = []
    for i, c in enumerate(classes):
        for j, (fname, blk) in enumerate(zip(names, blocks)):
            bj = boots[:, i, j]
            bj = bj[~np.isnan(bj)]
            lo, hi = np.percentile(bj, [2.5, 97.5])
            p = max(2 * min(np.mean(bj <= 0), np.mean(bj >= 0)), 1 / len(bj))
            rows.append(dict(cluster=c, feature=fname, block=blk, coef=coef[i, j],
                             rel_odds=np.exp(coef[i, j]), CI_low=np.exp(lo), CI_high=np.exp(hi),
                             p_boot=min(p, 1.0), n_boot_used=len(bj)))
    res = pd.DataFrame(rows)
    res["q_BH"] = bh_fdr(res["p_boot"])
    return res


# %% Fig. 4H: prediction of the cluster from diagnosis vs. neighbourhoods
diag_X = pd.get_dummies(data["Diagnosis"], prefix="dx", dtype=float)
feature_sets = {"Diagnosis": diag_X, "Neighborhoods": data[nhood_cols].astype(float),
                "Diagnosis + Neighborhoods": pd.concat([diag_X, data[nhood_cols].astype(float)], axis=1)}
cv_res = pd.DataFrame([
    {**score_proba(y, oof_proba(X, y, groups, classes, seed), classes), "model": name, "repeat": seed}
    for seed in range(N_REPEATS) for name, X in feature_sets.items()])
cv_res.to_csv(OUT_DIR / "Fig4H_cv_repeats.csv", index=False)

obs_gain, null_gain, p_global = global_permutation_test(data, nhood_cols, y, groups, classes,
                                                        n_perm=N_PERM_GLOBAL, n_jobs=N_JOBS)
print(f"Log-loss gain from adding nhoods to diagnosis: {obs_gain:.3f}, permutation p = {p_global:.3g}")
pd.DataFrame({"observed_gain": [obs_gain], "p_value": [p_global], "n_perm": [N_PERM_GLOBAL]}).to_csv(
    OUT_DIR / "Fig4H_global_permutation.csv", index=False)

joint = joint_model_coefficients(data, nhood_cols, classes, n_boot=N_BOOT, n_jobs=N_JOBS)
joint.to_csv(OUT_DIR / "Fig4H_joint_model_coefficients.csv", index=False)

# %% Fig. 4H: plot (AUC comparison on top, joint-model relative odds below)
FONT = 6.5
mpl.rcParams.update({"font.size": FONT, "axes.linewidth": 0.6,
                     "xtick.major.width": 0.6, "ytick.major.width": 0.6})
DX_ROW_NAMES = {"Astrocytoma low-grade": "Astrocytoma LG", "Astrocytoma high-grade": "Astrocytoma HG"}
lab = lambda c: CLUSTER_RENAME.get(c, c)

metric_cols = ["macro AUC (OvR)"] + [f"AUC cluster {c}" for c in cluster_order]
summ = cv_res.groupby("model")[metric_cols].agg(["mean", "std"])
wide = cv_res.set_index(["repeat", "model"])[metric_cols].unstack("model")

# best case for diagnosis alone: AUC of the in-sample cluster frequencies per diagnosis
p_dx = pd.crosstab(data["Diagnosis"], data["leiden"], normalize="index")
ceiling = {f"AUC cluster {c}": roc_auc_score(data["leiden"] == c, data["Diagnosis"].map(p_dx[c]))
           for c in cluster_order}
ceiling["macro AUC (OvR)"] = np.mean(list(ceiling.values()))

show_models = ["Diagnosis", "Neighborhoods"]
colors = {"Diagnosis": "#9e9e9e", "Neighborhoods": "#8950a5"}

tab = joint.rename(columns={"feature": "row", "rel_odds": "OR"})
blocks = [("Diagnosis", list(tab.loc[tab.block == "Diagnosis", "row"].unique())),
          ("Neighborhood", list(nhood_cols))]
heat_h = [0.14 * len(r) for _, r in blocks]
fig = plt.figure(figsize=(2.1, 1.5 + 0.75 + sum(heat_h) + 0.15 * len(blocks) + 0.5), dpi=300)
gs = GridSpec(3 + len(blocks), 1, height_ratios=[1.5, 0.75] + heat_h + [0.05], hspace=0.3, figure=fig)
ax = fig.add_subplot(gs[0])
leg_ax = fig.add_subplot(gs[1])
leg_ax.axis("off")
heat_axes = [fig.add_subplot(gs[2 + i]) for i in range(len(blocks))]
cax = fig.add_subplot(gs[-1])

x = np.arange(len(metric_cols))
off = {"Diagnosis": -0.14, "Neighborhoods": 0.14}
for xi, col in zip(x, metric_cols):
    ax.plot([xi + off[m] for m in show_models], [summ.loc[m, (col, "mean")] for m in show_models],
            color="black", lw=0.5, zorder=1)
for m in show_models:
    ax.errorbar(x + off[m], [summ.loc[m, (c, "mean")] for c in metric_cols],
                yerr=[summ.loc[m, (c, "std")] for c in metric_cols], fmt="o", ms=3.5,
                color=colors[m], mec="black", mew=0.3, elinewidth=0.5, capsize=1.2, label=m, zorder=2)
ax.scatter(x + off["Diagnosis"], [ceiling[c] for c in metric_cols], marker="_", s=40,
           linewidths=1.0, color="black", zorder=3, label="Diagnosis, best case")
for xi, col in zip(x, metric_cols):    # repeats in which nhoods beat diagnosis
    d = wide[(col, "Neighborhoods")] - wide[(col, "Diagnosis")]
    ax.text(xi, 1.0, f"{(d > 0).sum()}/{len(d)}", ha="center", va="bottom", fontsize=FONT - 1.5,
            color="dimgrey")
ax.axhline(0.5, ls="--", lw=0.5, color="grey")
ax.axvline(0.5, lw=0.4, color="lightgrey")
ax.set_ylim(0.4, 1.0)
ax.set_xlim(-0.5, x[-1] + 0.5)
ax.set_xticks(x)
ax.set_xticklabels(["All"] + [lab(c) for c in cluster_order])
for t, c in zip(ax.get_xticklabels()[1:], cluster_order):
    t.set_color(CLUSTER_COLORS.get(c, "black"))
    t.set_fontweight("bold")
ax.set_ylabel("Cluster prediction (AUC)")
ax.spines[["top", "right"]].set_visible(False)
h, l = ax.get_legend_handles_labels()
order = [l.index(n) for n in ["Diagnosis", "Diagnosis, best case", "Neighborhoods"]]
leg_ax.legend([h[i] for i in order], [l[i] for i in order], frameon=False, fontsize=FONT - 1,
              loc="upper center", bbox_to_anchor=(0.5, 0.85), handletextpad=0.2,
              borderaxespad=0, labelspacing=0.2)
ax.set_title("Histology cluster prediction\n(Astrocytoma LG/HG, GBM)", fontsize=FONT + 0.5, pad=8)
ax.text(0.02, 0.97, f"Nhood–cluster association\nbeyond diagnosis: p = {p_global:.2g}",
        transform=ax.transAxes, ha="left", va="top", fontsize=FONT - 1.5)

star = lambda v: "***" if v < 0.001 else "**" if v < 0.01 else "*" if v < 0.05 else ""
OR_all = tab.pivot(index="row", columns="cluster", values="OR").reindex(columns=cluster_order)
q_all = tab.pivot(index="row", columns="cluster", values="q_BH").reindex(columns=cluster_order)
or_max = 2 ** max(1, np.ceil(np.nanpercentile(np.abs(np.log2(OR_all.values)), 98)))
norm = LogNorm(vmin=1 / or_max, vmax=or_max)       # symmetric log scale around 1

for k, (axh, (blk, rows)) in enumerate(zip(heat_axes, blocks)):
    sns.heatmap(OR_all.reindex(rows).clip(1 / or_max, or_max), cmap="RdBu_r", norm=norm,
                annot=q_all.reindex(rows).apply(lambda s: s.map(star)), fmt="",
                annot_kws={"fontsize": FONT, "color": "black", "va": "center"},
                linewidths=0.3, linecolor="white", xticklabels=True, yticklabels=True,
                cbar=(k == 0), cbar_ax=cax if k == 0 else None,
                cbar_kws={"orientation": "horizontal"}, ax=axh)
    names = DX_ROW_NAMES if blk == "Diagnosis" else NHOOD_NAMES
    axh.set_yticklabels([names.get(r, r) for r in rows], rotation=0)
    axh.set_ylabel("Diagnostic group" if blk == "Diagnosis" else "Neighbourhood", fontsize=FONT,
                   fontweight="bold")
    axh.set_xlabel("")
    axh.tick_params(length=0)
    if k == 0:
        axh.xaxis.tick_top()
        axh.set_xticklabels([lab(c) for c in cluster_order], rotation=0)
        for t, c in zip(axh.get_xticklabels(), cluster_order):
            t.set_color(CLUSTER_COLORS.get(c, "black"))
            t.set_fontweight("bold")
    else:
        axh.set_xticks([])

kmax = int(np.log2(or_max))
ticks = [2.0 ** k for k in range(-kmax, kmax + 1) if k % (1 if kmax <= 2 else 2) == 0]
cax.xaxis.set_major_locator(FixedLocator(ticks))
cax.xaxis.set_minor_locator(NullLocator())
cax.set_xticklabels([f"{t:g}" if t >= 1 else f"1/{1 / t:g}" for t in ticks])
cax.set_xlabel("Relative odds (per SD)", fontsize=FONT - 0.5, labelpad=1)
cax.tick_params(labelsize=FONT - 1, length=1.5)
savefig(fig, "Fig4H_cluster_prediction.pdf", bbox_inches="tight")
