"""
Figure 4 and Extended Data Figures 7-8: neighbourhoods and clinical/molecular parameters.

  Fig. 4B    molecular features vs. neighbourhood abundance (Welch t on logit fractions)
  Fig. 4C    (boxplots) neighbourhood fractions in glioblastoma by methylation class
  Fig. 4D    neighbourhood composition by diagnostic group and WHO grade
  ED Fig. 7A molecular features and neighbourhood fractions per sample
  ED Fig. 7B as Fig. 4B, stratified by diagnostic group
  ED Fig. 8A neighbourhood fractions by WHO grade in IDH-mutant astrocytoma

All panels are drawn from Supplementary Table 5 (one row per sample = patient x
timepoint, already in the column order of ED Fig. 7A).

Not in this script:
  Fig. 4A          schematic
  Fig. 4E-H        H&E analysis (separate script)
  ED Fig. 8B       Kaplan-Meier curves (separate script)

Input:  source_data/SuppTable5_NhoodFractionsPerSample_withSurvival.xlsx
"""

# %% Imports and setup
from pathlib import Path

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.cm import ScalarMappable
from matplotlib.colors import ListedColormap, Normalize, TwoSlopeNorm, to_rgba
import warnings
from scipy.stats import mannwhitneyu, ttest_ind
from statsmodels.stats.multitest import multipletests

# %% Paths (relative to the repository root)
SOURCE_DATA = Path("source_data/SuppTable5_NhoodFractionsPerSample_withSurvival.xlsx")
OUT_DIR = Path("results/Figure4")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# keep text editable in Illustrator/Inkscape
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42

# %% Settings
# neighbourhood columns in Supp. Table 5 -> labels in the figures (also the order)
NHOODS = {
    "nhood_fraction_Vascular": "Vascular",
    "nhood_fraction_TBI": "TBI",
    "nhood_fraction_OPC-like-I": "OPC-like I",
    "nhood_fraction_OPC-like-II": "OPC-like II",
    "nhood_fraction_NPC-like": "NPC-like",
    "nhood_fraction_AC-like": "AC-like",
    "nhood_fraction_Gliosis": "Gliosis",
    "nhood_fraction_Hypoxia": "Hypoxia",
    "nhood_fraction_Proliferating": "Proliferating",
    "nhood_fraction_Stressed": "Stressed",
    "nhood_fraction_EPN-like": "EPN-like",
}

GRADE_COLORS = {"1.0": "#1f78b4", "2.0": "#15c79a", "3.0": "#f0c85f", "4.0": "#ef476f"}
METH_CLASS_COLORS = {"MES": "#B4D4B6", "RTK1": "#9B649B", "RTK2": "#F5A100"}

DX_COLLAPSE = {"Astrocytoma low-grade": "Astrocytoma", "Astrocytoma high-grade": "Astrocytoma"}


def clean_grade(x):
    """1, 1.0, '1' -> '1.0'; missing -> None."""
    if pd.isna(x) or str(x).strip().lower() in ["", "nan", "none", "<na>", "na"]:
        return None
    s = str(x).strip().removesuffix(".0")
    return f"{s}.0" if s in ["1", "2", "3", "4"] else s


def savefig(fig, name, **kw):
    fig.savefig(OUT_DIR / name, bbox_inches="tight", **kw)
    plt.close(fig)
    print("Saved:", OUT_DIR / name)


# %% Load source data
df = pd.read_excel(SOURCE_DATA)
nhood_cols = [c for c in NHOODS if c in df.columns]
df[nhood_cols] = df[nhood_cols].apply(pd.to_numeric, errors="coerce")
df["grade"] = df["WHO grade (2021)"].apply(clean_grade)
df["dx_collapsed"] = df["Diagnosis"].astype(str).str.strip().replace(DX_COLLAPSE)
print(df.shape[0], "samples")
print(pd.crosstab(df["Diagnosis"], df["grade"].fillna("NA")))

# %% Molecular features (Fig. 4B, ED Fig. 7B)
# Supp. Table 5 values -> feature levels; "unknown" = not tested (excluded)
def _na(s):
    s = s.astype(str).str.strip()
    return s.where(~s.str.lower().isin(["unknown", "nan", "none", ""]), "NA")


feat = pd.DataFrame(index=df.index)
feat["IDH"] = _na(df["IDH"])
feat["codel_1p19q"] = np.where(df["1p_loss"].eq("yes") & df["19q_loss"].eq("yes"),
                               "codeleted", "not_codeleted")
feat["loss_19q"] = np.where(df["19q_loss"].eq("yes"), "19q_loss", "no_19q_loss")
feat["MGMT"] = _na(df["MGMT"])
feat["EGFR"] = _na(df["EGFR_status"])
feat["PDGFRA"] = _na(df["PDGFRA_gain"]).replace({"mutation": "NA"})   # only gain/amp vs no
feat["TP53"] = _na(df["TP53_status"])
feat["ATRX"] = _na(df["ATRX_status"])
feat["TERT"] = _na(df["TERT_status"])
feat["CDKN2AB"] = _na(df["CDKN2A/B"])
feat["chr7_gain"] = np.where(df["7_gain"].eq("yes"), "chr7_gain", "no_chr7_gain")
feat["chr10_loss"] = np.where(df["10_loss"].eq("yes"), "chr10_loss", "no_chr10_loss")
feat["chr7_chr10"] = np.where(df["7_gain"].eq("yes") & df["10_loss"].eq("yes"),
                              "chr7_gain_chr10_loss", "no_chr7_gain_chr10_loss")
feat["MDM4"] = _na(df["MDM4_amp"])
feat["Diagnosis"] = df["Diagnosis"].astype(str).str.strip()
feat[nhood_cols] = df[nhood_cols]

MIN_GROUP_N = 3
T_COLOR_LIMIT = 4.0
Q_STAR_LEVELS = [(0.01, "***"), (0.05, "**"), (0.1, "*")]


def welch_t_logit(x_test, x_ref):
    """Welch t-test on logit(fraction). Zeros get a pseudocount of half the smallest
    non-zero fraction in the two compared groups."""
    x_test, x_ref = np.asarray(x_test, float), np.asarray(x_ref, float)
    if len(x_test) < 2 or len(x_ref) < 2:
        return np.nan, np.nan
    v = np.concatenate([x_test, x_ref])
    v = v[v > 0]
    eps = v.min() / 2 if len(v) else 1e-4
    lt = np.log(np.clip(x_test, eps, 1 - eps) / (1 - np.clip(x_test, eps, 1 - eps)))
    lr = np.log(np.clip(x_ref, eps, 1 - eps) / (1 - np.clip(x_ref, eps, 1 - eps)))
    if np.std(lt) < 1e-8 and np.std(lr) < 1e-8:      # both groups constant: t undefined
        return np.nan, np.nan
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        res = ttest_ind(lt, lr, equal_var=False)
    return float(res.statistic), float(res.pvalue)


def add_fdr(res, by):
    """BH-FDR across neighbourhoods, within each comparison."""
    res["q_value_BH"] = np.nan
    for _, idx in res.groupby(by).groups.items():
        valid = res.loc[idx].index[res.loc[idx, "p_value"].notna()]
        if len(valid):
            res.loc[valid, "q_value_BH"] = multipletests(res.loc[valid, "p_value"], method="fdr_bh")[1]
    return res


def q_to_star(q):
    for thr, star in Q_STAR_LEVELS:
        if pd.notna(q) and q < thr:
            return star
    return ""


def heatmap_cmap(middle_lightness=1.1, intensity=0.9):
    """RdBu_r, slightly lighter around 0 and softer at the extremes."""
    d = np.linspace(-1, 1, 512)
    src = np.sign(d) * np.abs(d) ** middle_lightness * intensity
    cmap = ListedColormap(plt.get_cmap("RdBu_r")(0.5 + 0.5 * src))
    cmap.set_bad("#eeeeee")
    return cmap


T_CMAP = heatmap_cmap()
T_NORM = TwoSlopeNorm(vmin=-T_COLOR_LIMIT, vcenter=0, vmax=T_COLOR_LIMIT)


def draw_t_heatmap(ax, t_mat, q_mat, xlabels, star_size=10.5):
    """t-statistic heatmap with FDR stars (black or white depending on the cell)."""
    im = ax.pcolormesh(np.arange(t_mat.shape[1] + 1), np.arange(t_mat.shape[0] + 1),
                       np.ma.masked_invalid(t_mat.values.astype(float)), cmap=T_CMAP, norm=T_NORM,
                       edgecolors="white", linewidth=1.0)
    ax.set_xlim(0, t_mat.shape[1])
    ax.set_ylim(t_mat.shape[0], 0)
    ax.set_xticks(np.arange(t_mat.shape[1]) + 0.5)
    ax.set_xticklabels(xlabels, rotation=90, fontsize=8)
    ax.set_yticks(np.arange(t_mat.shape[0]) + 0.5)
    ax.set_yticklabels(t_mat.index, fontsize=10)
    ax.tick_params(length=0)
    for i in range(t_mat.shape[0]):
        for j in range(t_mat.shape[1]):
            t, star = t_mat.iloc[i, j], q_to_star(q_mat.iloc[i, j])
            if star and pd.notna(t):
                r, g, b, _ = T_CMAP(T_NORM(np.clip(t, -T_COLOR_LIMIT, T_COLOR_LIMIT)))
                ax.text(j + 0.5, i + 0.5, star, ha="center", va="center", fontsize=star_size,
                        fontweight="bold",
                        color="white" if 0.299 * r + 0.587 * g + 0.114 * b < 0.5 else "black")
    for s in ax.spines.values():
        s.set_visible(False)
    return im


def t_colorbar(fig, im, ax, **kw):
    cbar = fig.colorbar(im, ax=ax, extend="both", **kw)
    cbar.set_label("Welch t statistic", fontsize=10, fontweight="bold")
    cbar.ax.tick_params(labelsize=8)
    cbar.ax.text(0, -0.12, "BH-FDR\n" + "\n".join(f"{s} q < {t:g}" for t, s in Q_STAR_LEVELS[::-1]),
                 transform=cbar.ax.transAxes, ha="left", va="top", fontsize=7)


# %% Fig. 4B: IDH-mutant glioma and IDH-wildtype glioblastoma
# each feature level vs. its reference; all non-reference levels with >= 3 samples are tested
B_COHORTS = {
    "IDH mutant glioma": {
        "diagnoses": ["Oligodendroglioma", "Astrocytoma low-grade", "Astrocytoma high-grade"],
        "features": [("IDH", "IDH_other_mut"), ("codel_1p19q", "not_codeleted"),
                     ("loss_19q", "no_19q_loss"), ("MGMT", "unmeth"), ("EGFR", "no"),
                     ("PDGFRA", "no"), ("TP53", "wildtype"), ("ATRX", "retained"),
                     ("CDKN2AB", "bal"), ("chr7_gain", "no_chr7_gain"),
                     ("chr10_loss", "no_chr10_loss"), ("chr7_chr10", "no_chr7_gain_chr10_loss"),
                     ("MDM4", "no")],
    },
    "IDH wt glioblastoma": {
        "diagnoses": ["Glioblastoma"],
        "features": [("codel_1p19q", "not_codeleted"), ("loss_19q", "no_19q_loss"),
                     ("MGMT", "unmeth"), ("EGFR", "no"), ("PDGFRA", "no"), ("TP53", "wildtype"),
                     ("ATRX", "retained"), ("CDKN2AB", "bal"), ("chr7_gain", "no_chr7_gain"),
                     ("chr10_loss", "no_chr10_loss"), ("chr7_chr10", "no_chr7_gain_chr10_loss"),
                     ("TERT", "wildtype"), ("MDM4", "no")],
    },
}
B_LABELS = {   # (feature, tested level) -> column label
    ("IDH", "IDH1_R132H"): "IDH1 R132H vs other", ("codel_1p19q", "codeleted"): "chr. 1p/19q codel.",
    ("loss_19q", "19q_loss"): "19q loss vs no", ("MGMT", "meth"): "MGMT meth.",
    ("EGFR", "altered"): "EGFR altered", ("PDGFRA", "gain/amp"): "PDGFRA gain/amp",
    ("TP53", "mutated"): "TP53 mutation", ("ATRX", "loss"): "ATRX loss",
    ("CDKN2AB", "het"): "CDKN2A/B het. del.", ("CDKN2AB", "homo"): "CDKN2A/B homo. del.",
    ("chr7_gain", "chr7_gain"): "chr7 gain", ("chr10_loss", "chr10_loss"): "chr10 loss",
    ("chr7_chr10", "chr7_gain_chr10_loss"): "chr7 gain + 10 loss", ("TERT", "mutated"): "TERT mutation",
    ("MDM4", "amp"): "MDM4 amp.",
}

b_rows = []
for cohort, spec in B_COHORTS.items():
    sub = feat[feat["Diagnosis"].isin(spec["diagnoses"])]
    for f, ref in spec["features"]:
        counts = sub.loc[sub[f].ne("NA"), f].value_counts()
        if counts.get(ref, 0) < MIN_GROUP_N:
            continue
        for level in sorted(l for l in counts.index if l != ref and counts[l] >= MIN_GROUP_N):
            for col in nhood_cols:
                x_test = sub.loc[sub[f].eq(level), col].dropna()
                x_ref = sub.loc[sub[f].eq(ref), col].dropna()
                t, p = welch_t_logit(x_test, x_ref)
                b_rows.append({"cohort": cohort, "feature": f, "level": level, "reference": ref,
                               "comparison": B_LABELS.get((f, level), f"{f}: {level} vs {ref}"),
                               "neighbourhood": NHOODS[col], "n_level": len(x_test),
                               "n_reference": len(x_ref), "mean_level": x_test.mean(),
                               "mean_reference": x_ref.mean(), "t_statistic": t, "p_value": p})
b_res = add_fdr(pd.DataFrame(b_rows), ["cohort", "comparison"])
b_res.to_csv(OUT_DIR / "Fig4B_statistics.csv", index=False)

nh_order = [NHOODS[c] for c in nhood_cols]
widths = [b_res.loc[b_res["cohort"] == c, "comparison"].nunique() for c in B_COHORTS]
fig, axes = plt.subplots(1, 2, figsize=(0.5 * sum(widths) + 3, 6.5),
                         gridspec_kw={"width_ratios": widths, "wspace": 0.03})
for k, (ax, cohort) in enumerate(zip(axes, B_COHORTS)):
    r = b_res[b_res["cohort"] == cohort]
    cols = list(dict.fromkeys(r["comparison"]))          # order of the feature plan
    t_mat = r.pivot(index="neighbourhood", columns="comparison", values="t_statistic").reindex(
        index=nh_order, columns=cols)
    q_mat = r.pivot(index="neighbourhood", columns="comparison", values="q_value_BH").reindex(
        index=nh_order, columns=cols)
    n = r.groupby("comparison")[["n_level", "n_reference"]].max()
    im = draw_t_heatmap(ax, t_mat, q_mat,
                        [f"{c}\n(n={n.loc[c, 'n_level']} vs {n.loc[c, 'n_reference']})" for c in cols])
    ax.set_title(cohort, fontsize=13, fontweight="bold", loc="left")
    if k > 0:
        ax.set_yticklabels([])
axes[0].set_ylabel("Neighbourhood", fontsize=12, fontweight="bold")
fig.supxlabel("Feature level versus reference", fontsize=11, fontweight="bold", y=-0.12)
t_colorbar(fig, im, list(axes), fraction=0.02, pad=0.02)
savefig(fig, "Fig4B_molecular_features_heatmap.pdf")

# %% ED Fig. 7B: as Fig. 4B, stratified by diagnostic group
ASTRO_FEATURES = [   # (label, feature, reference, test level)
    ("IDH1 R132H vs other IDH", "IDH", "IDH_other_mut", "IDH1_R132H"),
    ("chr7 gain vs no", "chr7_gain", "no_chr7_gain", "chr7_gain"),
    ("TP53 mut vs wt", "TP53", "wildtype", "mutated"),
    ("CDKN2A/B het vs bal", "CDKN2AB", "bal", "het"),
    ("CDKN2A/B homo vs bal", "CDKN2AB", "bal", "homo"),
    ("19q loss vs no", "loss_19q", "no_19q_loss", "19q_loss"),
    ("ATRX loss vs retained", "ATRX", "retained", "loss"),
    ("MGMT meth vs unmeth", "MGMT", "unmeth", "meth"),
    ("PDGFRA gain/amp vs no", "PDGFRA", "no", "gain/amp"),
]
IDH_MUT = ["Oligodendroglioma", "Astrocytoma low-grade", "Astrocytoma high-grade"]
CONTRASTS = (   # (label, diagnoses, feature, reference, test level)
    [(f"Oligo {l}", ["Oligodendroglioma"], f, r, t) for l, f, r, t in [
        ("IDH1 R132H vs other IDH", "IDH", "IDH_other_mut", "IDH1_R132H"),
        ("CDKN2A/B het vs bal", "CDKN2AB", "bal", "het"),
        ("CDKN2A/B homo vs bal", "CDKN2AB", "bal", "homo"),
        ("MGMT meth vs unmeth", "MGMT", "unmeth", "meth"),
        ("PDGFRA gain/amp vs no", "PDGFRA", "no", "gain/amp"),
        ("EGFR altered vs no", "EGFR", "no", "altered"),
        ("chr7 gain vs no", "chr7_gain", "no_chr7_gain", "chr7_gain"),
        ("chr10 loss vs no", "chr10_loss", "no_chr10_loss", "chr10_loss"),
        ("TP53 mut vs wt", "TP53", "wildtype", "mutated"),
        ("TERT mut vs wt", "TERT", "wildtype", "mutated"),
        ("MDM4 amp vs no", "MDM4", "no", "amp")]]
    + [(f"LG Astro {l}", ["Astrocytoma low-grade"], f, r, t) for l, f, r, t in ASTRO_FEATURES]
    + [(f"HG Astro {l}", ["Astrocytoma high-grade"], f, r, t) for l, f, r, t in ASTRO_FEATURES]
    + [("IDH-mut glioma MDM4 amp vs no", IDH_MUT, "MDM4", "no", "amp")]
    + [(f"GB {l}", ["Glioblastoma"], f, r, t) for l, f, r, t in [
        ("chr7 gain vs no", "chr7_gain", "no_chr7_gain", "chr7_gain"),
        ("chr10 loss vs no", "chr10_loss", "no_chr10_loss", "chr10_loss"),
        ("chr7 gain + chr10 loss vs no", "chr7_chr10", "no_chr7_gain_chr10_loss", "chr7_gain_chr10_loss"),
        ("TP53 mut vs wt", "TP53", "wildtype", "mutated"),
        ("CDKN2A/B het vs bal", "CDKN2AB", "bal", "het"),
        ("CDKN2A/B homo vs bal", "CDKN2AB", "bal", "homo"),
        ("EGFR altered vs no", "EGFR", "no", "altered"),
        ("PDGFRA gain/amp vs no", "PDGFRA", "no", "gain/amp"),
        ("TERT mut vs wt", "TERT", "wildtype", "mutated"),
        ("MGMT meth vs unmeth", "MGMT", "unmeth", "meth"),
        ("MDM4 amp vs no", "MDM4", "no", "amp")]]
    + [("Spinal EPN chr7 gain vs no", ["Spinal ependymoma"], "chr7_gain", "no_chr7_gain", "chr7_gain"),
       ("Myxopapillary EPN chr7 gain vs no", ["Myxopapillary ependymoma"], "chr7_gain", "no_chr7_gain", "chr7_gain"),
       ("Myxopapillary EPN chr10 loss vs no", ["Myxopapillary ependymoma"], "chr10_loss", "no_chr10_loss", "chr10_loss"),
       ("Myxopapillary EPN MGMT meth vs unmeth", ["Myxopapillary ependymoma"], "MGMT", "unmeth", "meth")]
)

e_rows, dropped = [], []
for label, dxs, f, ref, test in CONTRASTS:
    sub = feat[feat["Diagnosis"].isin(dxs)]
    n_ref, n_test = int(sub[f].eq(ref).sum()), int(sub[f].eq(test).sum())
    if n_ref < MIN_GROUP_N or n_test < MIN_GROUP_N:     # skipped: too few samples
        dropped.append(label)
        continue
    for col in nhood_cols:
        x_test, x_ref = sub.loc[sub[f].eq(test), col].dropna(), sub.loc[sub[f].eq(ref), col].dropna()
        t, p = welch_t_logit(x_test, x_ref)
        e_rows.append({"contrast": label, "diagnoses": "; ".join(dxs), "feature": f, "level": test,
                       "reference": ref, "neighbourhood": NHOODS[col], "n_level": len(x_test),
                       "n_reference": len(x_ref), "mean_level": x_test.mean(),
                       "mean_reference": x_ref.mean(), "t_statistic": t, "p_value": p})
print("\nED Fig. 7B contrasts skipped (<3 samples per group):", dropped)
e_res = add_fdr(pd.DataFrame(e_rows), ["contrast"])
e_res.to_csv(OUT_DIR / "EDFig7B_statistics.csv", index=False)

cols = list(dict.fromkeys(e_res["contrast"]))
t_mat = e_res.pivot(index="neighbourhood", columns="contrast", values="t_statistic").reindex(
    index=nh_order, columns=cols)
q_mat = e_res.pivot(index="neighbourhood", columns="contrast", values="q_value_BH").reindex(
    index=nh_order, columns=cols)
n = e_res.groupby("contrast")[["n_level", "n_reference"]].max()
fig, ax = plt.subplots(figsize=(max(14, len(cols) * 0.45), 6))
im = draw_t_heatmap(ax, t_mat, q_mat,
                    [f"{c}\n(n={n.loc[c, 'n_level']}/{n.loc[c, 'n_reference']})" for c in cols],
                    star_size=8)
ax.set_title("Diagnosis-stratified differences in spatial neighborhood abundance", fontsize=12.5,
             fontweight="bold", pad=14)
ax.set_xlabel("Feature level versus reference", fontsize=10, fontweight="bold")
ax.set_ylabel("Neighbourhood", fontsize=10, fontweight="bold")
t_colorbar(fig, im, ax, fraction=0.022, pad=0.02)
plt.tight_layout()
savefig(fig, "EDFig7B_molecular_features_per_diagnosis.pdf")

# %% Fig. 4C (boxplots): glioblastoma, neighbourhood fraction by methylation class
# two-sided Mann-Whitney U between each pair of methylation classes
C_NHOODS = ["nhood_fraction_Gliosis", "nhood_fraction_OPC-like-II", "nhood_fraction_AC-like"]
MC_ORDER = ["MES", "RTK1", "RTK2"]
PAIRS = [("MES", "RTK1"), ("MES", "RTK2"), ("RTK1", "RTK2")]

gbm = df[df["Diagnosis"].eq("Glioblastoma") & df["Meth group"].isin(MC_ORDER)].copy()
print("\nGlioblastoma samples per methylation class:\n", gbm["Meth group"].value_counts())

pvals = []
fig, axes = plt.subplots(1, len(C_NHOODS), figsize=(3.6 * len(C_NHOODS), 4.4))
for ax, col in zip(axes, C_NHOODS):
    sns.boxplot(data=gbm, x="Meth group", y=col, order=MC_ORDER, hue="Meth group",
                palette=METH_CLASS_COLORS, legend=False, width=0.58, showfliers=False,
                linewidth=1.2, boxprops={"edgecolor": "black"},
                whiskerprops={"color": "black"}, capprops={"color": "black"},
                medianprops={"color": "black"}, ax=ax)
    sns.stripplot(data=gbm, x="Meth group", y=col, order=MC_ORDER, color="black", size=3.5,
                  jitter=0.13, ax=ax)

    ymax = gbm[col].max() if np.isfinite(gbm[col].max()) else 0
    yrange = max(ymax, 0.05)
    base_y, step, h = ymax + 0.08 * yrange, 0.13 * yrange, 0.025 * yrange
    for i, (g1, g2) in enumerate(PAIRS):
        x, y = gbm.loc[gbm["Meth group"] == g1, col].dropna(), gbm.loc[gbm["Meth group"] == g2, col].dropna()
        p = mannwhitneyu(x, y, alternative="two-sided").pvalue if len(x) and len(y) else np.nan
        pvals.append({"neighbourhood": NHOODS[col], "group_1": g1, "group_2": g2,
                      "n_1": len(x), "n_2": len(y), "p_value": p})
        x1, x2, yb = MC_ORDER.index(g1), MC_ORDER.index(g2), base_y + i * step
        ax.plot([x1, x1, x2, x2], [yb, yb + h, yb + h, yb], color="black", linewidth=1)
        ax.text((x1 + x2) / 2, yb + h, f"p = {p:.3f}", ha="center", va="bottom", fontsize=8)

    ax.set_ylim(-0.02, base_y + len(PAIRS) * step + 0.08 * yrange)
    ax.set_title(NHOODS[col], fontsize=12, fontweight="bold")
    ax.set_xlabel("Methylation class", fontsize=10, fontweight="bold")
    ax.set_ylabel("Neighbourhood fraction", fontsize=10, fontweight="bold")
    sns.despine(ax=ax)
plt.tight_layout()
savefig(fig, "Fig4C_glioblastoma_methylation_class_boxplots.pdf", dpi=300)
pd.DataFrame(pvals).to_csv(OUT_DIR / "Fig4C_pvalues.csv", index=False)

# %% Fig. 4D: neighbourhood composition by diagnostic group and WHO grade
# dot colour = mean fraction per core; dot size = fraction of cores with fraction > 0.02
EXPRESSION_CUTOFF = 0.02
VMAX = 0.8
D_DIAGNOSES = ["Oligodendroglioma", "Astrocytoma", "Glioblastoma"]

d = df[df["dx_collapsed"].isin(D_DIAGNOSES) & df["grade"].isin(["1.0", "2.0", "3.0", "4.0"])].copy()
groups = (d.groupby(["dx_collapsed", "grade"]).size().rename("n").reset_index())
groups["dx_collapsed"] = pd.Categorical(groups["dx_collapsed"], categories=D_DIAGNOSES, ordered=True)
groups = groups.sort_values(["dx_collapsed", "grade"]).reset_index(drop=True)

rows = []
for gi, g in groups.iterrows():
    sub = d[(d["dx_collapsed"] == g["dx_collapsed"]) & (d["grade"] == g["grade"])]
    for ni, col in enumerate(nhood_cols):
        vals = sub[col].dropna()
        rows.append({"x": gi, "y": ni, "diagnosis": g["dx_collapsed"], "WHO_grade": g["grade"],
                     "neighbourhood": NHOODS[col], "n_cores": g["n"],
                     "mean_fraction": vals.mean(), "median_fraction": vals.median(),
                     "fraction_cores": (vals > EXPRESSION_CUTOFF).mean(),
                     "n_cores_above_cutoff": int((vals > EXPRESSION_CUTOFF).sum())})
dot_df = pd.DataFrame(rows)

MIN_DOT, MAX_DOT = 8, 420
fig, ax = plt.subplots(figsize=(max(4.5, len(groups) * 0.85), max(5, len(nhood_cols) * 0.55)))
sc_ = ax.scatter(dot_df["x"], dot_df["y"],
                 s=MIN_DOT + dot_df["fraction_cores"].fillna(0) * (MAX_DOT - MIN_DOT),
                 c=dot_df["mean_fraction"].clip(0, VMAX), cmap=plt.cm.Reds,
                 norm=Normalize(0, VMAX), edgecolors="black", linewidths=0.25, alpha=0.95)
ax.set_xticks(range(len(groups)))
ax.set_xticklabels([f"Grade {g[0]}\n(n={n})" for g, n in zip(groups["grade"], groups["n"])],
                   rotation=90, fontsize=8)
for label, g in zip(ax.get_xticklabels(), groups["grade"]):
    label.set_color(GRADE_COLORS[g])
ax.set_yticks(range(len(nhood_cols)))
ax.set_yticklabels([NHOODS[c] for c in nhood_cols], fontsize=9)
ax.set_xlim(-0.6, len(groups) - 0.4)
ax.set_ylim(len(nhood_cols) - 0.5, -0.5)
ax.set_ylabel("Neighbourhood", fontsize=10)
ax.set_title("Neighbourhood composition", fontsize=13, fontweight="bold", pad=14)
ax.set_axisbelow(True)
ax.grid(color="lightgrey", linewidth=0.35, alpha=0.45)

# diagnostic-group labels under the grade labels
for dx in D_DIAGNOSES:
    xs = groups.index[groups["dx_collapsed"] == dx]
    if len(xs):
        ax.annotate(dx, xy=((xs.min() + xs.max()) / 2, 0), xycoords=("data", "axes fraction"),
                    xytext=(0, -62), textcoords="offset points", ha="center", va="top",
                    fontsize=8, color="grey")

cbar = plt.colorbar(sc_, ax=ax, fraction=0.025, pad=0.02)
cbar.set_label("Mean fraction\nper core", fontsize=9)
handles = [ax.scatter([], [], s=MIN_DOT + f * (MAX_DOT - MIN_DOT), facecolors="white",
                      edgecolors="black", linewidths=0.4, label=f"{int(f * 100)}%")
           for f in [0.25, 0.50, 0.75, 1.00]]
ax.add_artist(ax.legend(handles=handles, title=f"Fraction of cores\n> {EXPRESSION_CUTOFF:g}",
                        loc="upper left", bbox_to_anchor=(1.03, 0.70), frameon=False,
                        fontsize=8, title_fontsize=9))
plt.tight_layout()
savefig(fig, "Fig4D_neighbourhood_composition.pdf")
dot_df.drop(columns=["x", "y"]).to_csv(OUT_DIR / "Fig4D_source_data.csv", index=False)

# %% ED Fig. 7A: molecular features and neighbourhood fractions per sample
# samples (columns) are plotted in the row order of Supp. Table 5
META_ROWS = {   # column in Supp. Table 5 -> row label
    "Diagnosis": "Diagnostic group",
    "WHO grade (2021)": "WHO grade (2021)",
    "Meth group": "Methylation group",
    "IDH": "IDH mutation",
    "7_gain": "Chr. 7 gain",
    "10_loss": "Chr. 10 loss",
    "1p_loss": "Chr. 1p loss",
    "19q_loss": "Chr. 19q loss",
    "MGMT": "MGMT status",
    "EGFR_status": "EGFR",
    "PDGFRA_gain": "PDGFRA gain",
    "TP53_status": "TP53",
    "ATRX_status": "ATRX",
    "TERT_status": "TERT",
    "CDKN2A/B": "CDKN2A/B",
}
ED7_NHOOD_LABELS = {"Vascular": "Vasculature", "Stressed": "Stress-response"}

MISSING_COLOR = "#d0d0d0"
UNMATCHED_COLOR = "#999999"
CAT_COLORS = {
    "Diagnosis": {
        "Glioblastoma": "#b26b6b", "Astrocytoma high-grade": "#e6aa5f",
        "Astrocytoma low-grade": "#a8d39c", "Oligodendroglioma": "#e8e283",
        "Spinal ependymoma": "#d8b8f1", "Myxopapillary ependymoma": "#7d00c4",
        "Subependymoma": "#b499c8",
    },
    "WHO grade (2021)": GRADE_COLORS,
    "Meth group": {
        "EPN_SPINE": "#9F84C1", "EPN_PF_SE": "#8F8FD5", "EPN_ST_SE": "#A6A7DC",
        "EPN_MPE": "#D497A4", "EPN_STE_SE": "#B9838C", "EPN_SPINE_SE_A": "#9B7C88",
        "EPN_SPINE_MYCN": "#AF77D7", "RTK2": "#F5A100", "MES": "#B4D4B6", "RTK1": "#9B649B",
        "Astro LG": "#B1DBA1", "Astro HG": "#EFB87C", "Oligo": "#F5F19F",
        "Subependymoma": "#A789C2",
    },
    "MGMT": {"meth": "#76c893", "unmeth": "#d9ed92", "unknown": "#bdbdbd"},
    "IDH": {"IDH1_R132H": "#f4a261", "IDH_wt": "#2a9d8f", "IDH_other_mut": "#e9c46a",
            "unknown": "#bdbdbd"},
    "TP53_status": {"mutated": "#d62828", "wildtype": "#d9d9d9", "unknown": "#bdbdbd"},
    "ATRX_status": {"loss": "#6a4c93", "retained": "#bde0fe", "unknown": "#bdbdbd"},
    "TERT_status": {"mutated": "#fb8500", "wildtype": "#d9d9d9", "unknown": "#bdbdbd"},
    "CDKN2A/B": {"bal": "#ff99c8", "gain": "#c77dff", "het": "#ffb3c6", "homo": "#7b2cbf"},
    "1p_loss": {"yes": "#f18f01", "no": "#d9d9d9", "unknown": "#bdbdbd"},
    "19q_loss": {"yes": "#e95d0f", "no": "#d9d9d9", "unknown": "#bdbdbd"},
    "7_gain": {"yes": "#c8553d", "no": "#d9d9d9", "unknown": "#bdbdbd"},
    "10_loss": {"yes": "#f28f3b", "no": "#d9d9d9", "unknown": "#bdbdbd"},
    "EGFR_status": {"altered": "#c9184a", "no": "#d9d9d9", "unknown": "#bdbdbd"},
    "PDGFRA_gain": {"gain/amp": "#06d6a0", "no": "#d9d9d9"},
}


def category_color(col, value):
    if pd.isna(value) or str(value).strip().lower() in ["", "nan", "none", "<na>", "na"]:
        return to_rgba(MISSING_COLOR)
    value = clean_grade(value) if col == "WHO grade (2021)" else str(value).strip()
    return to_rgba(CAT_COLORS.get(col, {}).get(value, UNMATCHED_COLOR))


meta_cols = [c for c in META_ROWS if c in df.columns]
n_samples, n_meta, n_nh = len(df), len(meta_cols), len(nhood_cols)
heat_norm = Normalize(0, max(1e-6, np.nanmax(df[nhood_cols].values)))

fig, ax = plt.subplots(figsize=(max(16, n_samples * 0.12), max(8, (n_meta + n_nh) * 0.38)))
for i, col in enumerate(meta_cols):
    for j, v in enumerate(df[col]):
        ax.add_patch(plt.Rectangle((j, i), 1, 1, facecolor=category_color(col, v),
                                   edgecolor="white", linewidth=0.15))
for r, col in enumerate(nhood_cols):
    for j, v in enumerate(df[col]):
        color = to_rgba(MISSING_COLOR) if pd.isna(v) else plt.cm.Blues(heat_norm(v))
        ax.add_patch(plt.Rectangle((j, n_meta + r), 1, 1, facecolor=color, edgecolor="white",
                                   linewidth=0.15))

row_labels = [META_ROWS[c] for c in meta_cols] + [
    ED7_NHOOD_LABELS.get(NHOODS[c], NHOODS[c]) for c in nhood_cols]
ax.set_xlim(0, n_samples)
ax.set_ylim(0, n_meta + n_nh)
ax.invert_yaxis()
ax.set_xticks([])
ax.set_yticks(np.arange(n_meta + n_nh) + 0.5)
ax.set_yticklabels(row_labels, fontsize=9)
ax.tick_params(length=0)
ax.axhline(n_meta, color="black", linewidth=0.8)
for s in ax.spines.values():
    s.set_visible(False)

legend_items = {}
for col in meta_cols:
    present = {clean_grade(v) if col == "WHO grade (2021)" else str(v).strip()
               for v in df[col].dropna()}
    for label, color in CAT_COLORS.get(col, {}).items():
        if label in present:
            legend_items.setdefault(f"{META_ROWS[col]}: {label}", color)
fig.legend([mpatches.Patch(facecolor=c, edgecolor="none") for c in legend_items.values()],
           list(legend_items), loc="lower center", ncol=5, fontsize=7, frameon=False,
           bbox_to_anchor=(0.5, -0.04))
cax = fig.add_axes([0.72, 0.065, 0.18, 0.015])
cb = plt.colorbar(ScalarMappable(norm=heat_norm, cmap=plt.cm.Blues), cax=cax,
                  orientation="horizontal")
cb.set_label("Nhood fraction per core", fontsize=8)
cb.ax.tick_params(labelsize=7)
plt.subplots_adjust(left=0.18, right=0.98, top=0.88, bottom=0.22)
savefig(fig, "EDFig7A_molecular_features_heatmap.pdf")

# %% ED Fig. 8A: neighbourhood fraction by WHO grade, IDH-mutant astrocytoma
# two-sided Mann-Whitney U (grade 2 vs 3, 3 vs 4); BH-FDR across neighbourhoods per comparison
GRADE_ORDER = ["2.0", "3.0", "4.0"]
COMPARISONS = [("2.0", "3.0"), ("3.0", "4.0")]
MIN_GROUP_N = 3

astro = df[df["dx_collapsed"].eq("Astrocytoma") & df["grade"].isin(GRADE_ORDER)]
print("\nAstrocytoma samples per WHO grade:\n", astro["grade"].value_counts().sort_index())

stats = []
for col in nhood_cols:
    for ga, gb in COMPARISONS:
        a = 100 * astro.loc[astro["grade"] == ga, col].dropna()
        b = 100 * astro.loc[astro["grade"] == gb, col].dropna()
        p = (mannwhitneyu(a, b, alternative="two-sided").pvalue
             if len(a) >= MIN_GROUP_N and len(b) >= MIN_GROUP_N else np.nan)
        stats.append({"neighbourhood": NHOODS[col], "comparison": f"{ga[0]}v{gb[0]}",
                      "n_a": len(a), "n_b": len(b), "median_a_pct": a.median(),
                      "median_b_pct": b.median(), "p_value": p})
stats = pd.DataFrame(stats)
stats["q_value_BH"] = np.nan
for comp, idx in stats.groupby("comparison").groups.items():
    valid = stats.loc[idx].index[stats.loc[idx, "p_value"].notna()]
    stats.loc[valid, "q_value_BH"] = multipletests(stats.loc[valid, "p_value"], method="fdr_bh")[1]
stats.to_csv(OUT_DIR / "EDFig8A_statistics.csv", index=False)


def fmt(v, letter):
    return f"{letter}=NA" if pd.isna(v) else (f"{letter}<0.001" if v < 0.001 else f"{letter}={v:.3f}")


N_COLS = 4
n_rows = int(np.ceil(len(nhood_cols) / N_COLS))
rng = np.random.default_rng(123)
fig, axes = plt.subplots(n_rows, N_COLS, figsize=(N_COLS * 2.9, n_rows * 3.4))
axes = axes.ravel()
for i, col in enumerate(nhood_cols):
    ax = axes[i]
    data = [100 * astro.loc[astro["grade"] == g, col].dropna().values for g in GRADE_ORDER]
    bp = ax.boxplot(data, positions=range(3), widths=0.55, patch_artist=True, showfliers=False,
                    medianprops={"color": "black", "linewidth": 1.3},
                    boxprops={"linewidth": 0.8, "color": "black"},
                    whiskerprops={"linewidth": 0.8, "color": "black"},
                    capprops={"linewidth": 0.8, "color": "black"})
    for patch, g in zip(bp["boxes"], GRADE_ORDER):
        patch.set_facecolor(GRADE_COLORS[g])
        patch.set_alpha(0.75)
    for k, vals in enumerate(data):
        ax.scatter(k + rng.normal(0, 0.04, len(vals)), vals, s=5, color="black", alpha=0.30,
                   linewidths=0, zorder=3)

    st = stats[stats["neighbourhood"] == NHOODS[col]].set_index("comparison")
    ax.text(0.5, 1.02, "\n".join(f"{c}: {fmt(st.loc[c, 'p_value'], 'p')}, "
                                 f"{fmt(st.loc[c, 'q_value_BH'], 'q')}" for c in ["2v3", "3v4"]),
            transform=ax.transAxes, ha="center", va="bottom", fontsize=6.4, linespacing=1.15)
    ax.set_title(NHOODS[col], fontsize=9.5, fontweight="bold", pad=24)
    ax.set_xticks(range(3))
    ax.set_xticklabels([f"WHO {g[0]}\n(n={len(v)})" for g, v in zip(GRADE_ORDER, data)], fontsize=7)
    ax.set_xlabel("WHO grade", fontsize=9, fontweight="bold")
    ax.set_ylabel("Neighbourhood fraction (%)" if i % N_COLS == 0 else "", fontsize=9,
                  fontweight="bold")
    ymax = np.nanmax(np.concatenate(data)) if sum(len(v) for v in data) else 0
    ax.set_ylim(0, ymax * 1.30 if ymax > 0 else 1)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
for ax in axes[len(nhood_cols):]:
    fig.delaxes(ax)
fig.suptitle("IDH-mutant astrocytoma", fontsize=13, fontweight="bold", y=1.01)
plt.tight_layout()
savefig(fig, "EDFig8A_who_grade_boxplots.pdf")
