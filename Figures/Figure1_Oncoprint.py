"""
Figure 1C: cohort overview (oncoprint).

Draws the cohort overview from the pseudonymised source data, so the figure
can be reproduced without access to the clinical metadata.

Input:  source_data/Figure1_source_data.xlsx, sheet "Fig1C_cohort_overview"
        (one column per patient and timepoint, already in plotting order)
Output: results/Figure1/Fig1C_cohort_overview.pdf
"""

# %% Imports and setup
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

# %% Paths (relative to the repository root)
SOURCE_DATA = Path("source_data/Figure1_source_data.xlsx")
OUT_DIR = Path("results/Figure1")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# keep text editable in Illustrator/Inkscape
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42

# %% Load source data (row order = plotting order; do not re-sort)
df = pd.read_excel(SOURCE_DATA, sheet_name="Fig1C_cohort_overview")
meta_rows = [c for c in df.columns if c not in ("column", "patient")]
n_samples = len(df)

# Excel may return WHO grades as 2 or 2.0; use "2.0" to match the palette
if "WHO grade (2021)" in df.columns:
    grade = pd.to_numeric(df["WHO grade (2021)"], errors="coerce")
    df["WHO grade (2021)"] = grade.map(lambda v: f"{v:.1f}" if pd.notna(v) else pd.NA)

# %% Value harmonisation and colours
VALUE_MAP = {
    "yes": "yes", "no": "no",
    "loss": "loss", "deleted": "loss", "deletion": "loss",
    "gain": "gain", "gained": "gain",
    "amp": "amp", "amplified": "amp", "amplification": "amp",
    "wt": "wt", "wildtype": "wt", "wild-type": "wt", "wild type": "wt",
    "mut": "mut", "mutant": "mut",
    "m": "m", "male": "m", "f": "f", "female": "f",
    "primary": "primary", "recurrent": "recurrent",
    "idh1 r132h": "IDH1 R132H", "idh1/2 wt": "IDH1/2 WT",
    "undeterminable": "undeterminable", "indeterminate": "undeterminable",
    "unknown": "unknown",
}


def normalize_value(x):
    """Harmonise spelling; unknown values are returned unchanged."""
    if pd.isna(x):
        return None
    x = str(x).strip()
    if x in ["", "nan", "None", "<NA>"]:
        return None
    return VALUE_MAP.get(x.lower(), x)


continuous_rows = [c for c in ["age"] if c in meta_rows]
categorical_rows = [c for c in meta_rows if c not in continuous_rows]

CAT_COLORS = {
    "Diagnosis": {
        "Glioblastoma": "#b26b6b",
        "Astrocytoma high-grade": "#e6aa5f",
        "Astrocytoma low-grade": "#a8d39c",
        "Oligodendroglioma": "#e8e283",
        "Ependymoma": "#9b7bbd",
        "Spinal ependymoma": "#d8b8f1",
        "Myxopapillary ependymoma": "#7d00c4",
        "Subependymoma": "#b499c8",
    },
    "WHO grade (2021)": {"1.0": "#1f78b4", "2.0": "#15c79a", "3.0": "#f0c85f", "4.0": "#ef476f"},
    "Primary/recurrent": {"primary": "#bfe5e8", "recurrent": "#447aa3", "recurrence": "#447aa3"},
    "Meth group": {
        "EPN_SPINE": "#9F84C1", "EPN_PF_SE": "#8F8FD5", "EPN_ST_SE": "#A6A7DC",
        "EPN_MPE": "#D497A4", "EPN_STE_SE": "#B9838C", "EPN_SPINE_SE_A": "#9B7C88",
        "EPN_SPINE_MYCN": "#AF77D7", "RTK2": "#F5A100", "MES": "#B4D4B6", "RTK1": "#9B649B",
        "Astro LG": "#B1DBA1", "Astro HG": "#EFB87C", "Oligo": "#F5F19F",
        "Subependymoma": "#A789C2",
    },
    "MGMT": {"meth": "#76c893", "unmeth": "#d9ed92", "unknown": "#bdbdbd"},
    "sex": {"m": "#c993e6", "f": "#a7c957"},
    "IDH_group": {"IDH1_R132H": "#f4a261", "IDH_wt": "#2a9d8f", "IDH_other_mut": "#e9c46a"},
    "CDKN2A/B": {"bal": "#ff99c8", "gain": "#c77dff", "het": "#ffb3c6", "homo": "#7b2cbf"},
    "1p_loss": {"yes": "#f18f01", "no": "#d9d9d9"},
    "19q_loss": {"yes": "#e95d0f", "no": "#d9d9d9"},
    "7_gain": {"yes": "#c8553d", "no": "#d9d9d9"},
    "10_loss": {"yes": "#f28f3b", "no": "#d9d9d9"},
    "EGFR_status": {"gain": "#c9184a", "Mutation/Gain?": "#ff595e", "no": "#d9d9d9"},
    "PDGFRA_gain": {"gain": "#06d6a0", "amp": "#2a9d8f", "no": "#d9d9d9", "mutation": "#2ec4b6"},
}
CONT_MAPS = {"age": plt.cm.Blues}
MISSING_COLOR = "#d0d0d0"
UNMATCHED_COLOR = "#999999"

# %% Plot
# Values without a colour in CAT_COLORS are drawn in UNMATCHED_COLOR; print them for checking
for col in categorical_rows:
    vals = sorted({normalize_value(v) for v in df[col].dropna()} - {None})
    missing = [v for v in vals if v not in CAT_COLORS.get(col, {})]
    if missing:
        print(f"{col}: no colour for {missing}")

n_rows = len(meta_rows)
fig, ax = plt.subplots(figsize=(max(18, n_samples * 0.08), max(4, n_rows * 0.5)))

for i in range(n_rows):
    for j in range(n_samples):
        ax.add_patch(plt.Rectangle((j, i), 1, 1, facecolor=MISSING_COLOR, edgecolor="white", linewidth=0.2))

for i, col in enumerate(meta_rows):
    if col in continuous_rows:
        vals = pd.to_numeric(df[col], errors="coerce")
        valid = vals.notna()
        cmap = CONT_MAPS.get(col, plt.cm.Greys)
        if valid.sum() > 0:
            norm = Normalize(vmin=vals[valid].min(), vmax=vals[valid].max())
            for j, v in enumerate(vals):
                color = MISSING_COLOR if pd.isna(v) else cmap(norm(v))
                ax.add_patch(plt.Rectangle((j, i), 1, 1, facecolor=color, edgecolor="white", linewidth=0.2))
    else:
        palette = CAT_COLORS.get(col, {})
        for j, v in enumerate(df[col]):
            val = normalize_value(v)
            color = MISSING_COLOR if val is None else palette.get(val, UNMATCHED_COLOR)
            ax.add_patch(plt.Rectangle((j, i), 1, 1, facecolor=color, edgecolor="white", linewidth=0.2))

ax.set_xlim(0, n_samples)
ax.set_ylim(0, n_rows)
ax.invert_yaxis()
ax.set_xticks([])
ax.set_yticks(np.arange(n_rows) + 0.5)
ax.set_yticklabels(meta_rows, fontsize=10)
ax.tick_params(length=0)
for spine in ax.spines.values():
    spine.set_visible(False)

# group headers by diagnosis
diag_norm = [normalize_value(x) or "NA" for x in df["Diagnosis"]]
groups, start = [], 0
for idx in range(1, len(diag_norm) + 1):
    if idx == len(diag_norm) or diag_norm[idx] != diag_norm[start]:
        groups.append((diag_norm[start], start, idx - start))
        start = idx
for name, start, count in groups:
    ax.text(start + count / 2, -0.6, f"{name}\n(n={count})", ha="center", va="bottom",
            fontsize=9, fontweight="bold", clip_on=False)

# legends
legend_items = {}
for col in categorical_rows:
    present = {normalize_value(v) for v in df[col].dropna()}
    for label, color in CAT_COLORS.get(col, {}).items():
        if label in present:
            legend_items.setdefault(f"{col}: {label}", color)
if legend_items:
    handles = [plt.Rectangle((0, 0), 1, 1, facecolor=c, edgecolor="none") for c in legend_items.values()]
    fig.legend(handles, list(legend_items), loc="lower center", ncol=6, fontsize=8,
               frameon=False, bbox_to_anchor=(0.5, -0.08))

for idx, col in enumerate(continuous_rows):
    vals = pd.to_numeric(df[col], errors="coerce").dropna()
    if vals.empty:
        continue
    sm = ScalarMappable(norm=Normalize(vmin=vals.min(), vmax=vals.max()),
                        cmap=CONT_MAPS.get(col, plt.cm.Greys))
    sm.set_array([])
    cax = fig.add_axes([0.72, 0.12 - idx * 0.05, 0.18, 0.015])
    cb = plt.colorbar(sm, cax=cax, orientation="horizontal")
    cb.set_label(col, fontsize=8)
    cb.ax.tick_params(labelsize=7)

plt.subplots_adjust(left=0.20, right=0.98, top=0.83, bottom=0.25)
plt.savefig(OUT_DIR / "Fig1C_cohort_overview.pdf", format="pdf", bbox_inches="tight")
plt.show()
