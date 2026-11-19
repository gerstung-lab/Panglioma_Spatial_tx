"""
Figure 4F-H: preparation of the per-core table (run where the clinical metadata are available).

Combines, per tissue core: pseudonymised patient, diagnostic group and neighbourhood
fractions. The output is all that figure4FGH_he_titan.py needs besides the TITAN
embeddings.

Inputs (not public):
    data/metadata/pan-glioma_metadata.csv               block -> core
    data/metadata/Glioma_TMA_metadata_20260327.xlsx     sheet "Plotting"
    data/niche_fraction_per_core.csv                    neighbourhood fractions per core
    results/Figure1/internal/INTERNAL_patient_key_DO_NOT_SHARE.xlsx   (from Figure 1)
Output (pseudonymised):
    source_data/Figure4_HE_core_table.csv

Run from the repository root:  python Figure4/he_prepare_core_table.py
"""

from pathlib import Path

import pandas as pd

# %% Paths (relative to the repository root)
META_BLOCKS = Path("data/metadata/pan-glioma_metadata.csv")
META_TMA = Path("data/metadata/Glioma_TMA_metadata_20260327.xlsx")
NHOOD_FRACTIONS = Path("data/niche_fraction_per_core.csv")
PATIENT_KEY = Path("results/Figure1/internal/INTERNAL_patient_key_DO_NOT_SHARE.xlsx")
OUT = Path("source_data/Figure4_HE_core_table.csv")
OUT.parent.mkdir(parents=True, exist_ok=True)

# %% Core -> patient and diagnostic group
blocks = pd.read_csv(META_BLOCKS, index_col=0)
meta = pd.read_excel(META_TMA, sheet_name="Plotting")
meta = meta.merge(blocks[["block_id", "core"]], how="left", left_on="Block No",
                  right_on="block_id", validate="one_to_one")
meta["patient_id"] = meta["patient_id_new"].astype(str).str.split(".").str[0]
meta = meta.dropna(subset=["core"])[["core", "patient_id", "Diagnosis"]]

# %% Pseudonymise with the Figure 1 key (patients not in the key get Q001, Q002, ...)
key = pd.read_excel(PATIENT_KEY)[["patient_id_new", "patient_pseudo"]].drop_duplicates()
key_map = dict(zip(key["patient_id_new"].astype(str).str.replace(r"\.0$", "", regex=True),
                   key["patient_pseudo"]))
meta["patient"] = meta["patient_id"].map(key_map)
unkeyed = sorted(meta.loc[meta["patient"].isna() & ~meta["patient_id"].isin(["nan", "None", ""]),
                          "patient_id"].unique())
meta.loc[meta["patient_id"].isin(unkeyed), "patient"] = meta["patient_id"].map(
    {p: f"Q{i + 1:03d}" for i, p in enumerate(unkeyed)})
print(f"{len(unkeyed)} patients not in the Figure 1 key (given Q-pseudonyms)")

# %% Add neighbourhood fractions and save
fractions = pd.read_csv(NHOOD_FRACTIONS)
table = fractions.merge(meta.drop(columns="patient_id"), on="core", how="outer",
                        validate="one_to_one")
table = table[["core", "patient", "Diagnosis"] + [c for c in fractions.columns if c != "core"]]
table.to_csv(OUT, index=False)
print("Saved:", OUT, table.shape)
