"""
Neighbourhood-to-neighbourhood spatial co-localisation (input for Fig. 3G and ED Fig. 6).

Workflow:
  cells -> same-neighbourhood connected domains (per core)
        -> observed domain contacts per core
        -> merge-based permutation null (expected contacts per core)
        -> Poisson rate ratio per neighbourhood pair across cores
           (all cores, and separately per diagnostic group)

Input:  full spatial object (obs + obsp["spatial_connectivities"])
Outputs:
    poisson_niche_pair_results.csv all cores           -> Fig. 3G
    poisson_niche_pair_results_by_diagnosis.csv  per diagnostic group -> ED Fig. 6
    per_core_observed_expected.csv
    domain_core_summary.csv

Runtime: ~20 min with 10,000 permutations per core.
"""

# %% Imports and setup
import time
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests
from tqdm.auto import tqdm

# %% Paths (relative to the repository root; edit if your data live elsewhere)
ADATA_FULL = Path("data/rdata_5tx_pca_euclidean_umap.h5ad")
OUT_DIR = Path("results/analysis/nhood_colocalisation")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# %% Parameters
NICHE_COL = "leiden_nhood_annot"
CORE_COL = "core"
DX_COL = "Diagnosis"
CONNECTIVITY_KEY = "spatial_connectivities"

NICHE_ORDER = ["Vascular", "Non-malignant CNS", "Cortical glia", "OPC-like", "NPC-like", "AC-like",
               "MES1", "MES2", "Proliferating/Immune", "Stressed", "Spinal-glia"]

MIN_domain_SIZE = 15            # minimum cells per same-neighbourhood domain
N_PERMUTATIONS = 10_000       # per core (use 100 for a quick test)
PERM_SEED = 42
MIN_CORES_PER_DIAGNOSIS = 2   # diagnostic groups with fewer usable cores are skipped

# %% Load data (only obs and the spatial graph are needed; X stays on disk)
adata = sc.read_h5ad(ADATA_FULL, backed="r")
states = [n for n in NICHE_ORDER if n in set(adata.obs[NICHE_COL].astype(str))]

keep = adata.obs[NICHE_COL].astype(str).isin(states).to_numpy()
obs = adata.obs.loc[keep, [CORE_COL, NICHE_COL, DX_COL]].copy()
obs[NICHE_COL] = obs[NICHE_COL].astype(str)
adj_all = adata.obsp[CONNECTIVITY_KEY].tocsr()[keep][:, keep]
del adata

print("Cells retained:", len(obs), "| cores:", obs[CORE_COL].nunique())


# %% 1. Same-neighbourhood domains per core
def process_single_core_domains(adj, labels_all, states, min_domain_size=15):
    """
    Find connected same-neighbourhood domains in one core and build the domain adjacency.

    Returns
    -------
    domain_adjacency : {domain_id: [neighbour_domain_id, ...]} (both directions present)
    domain_meta_valid : {domain_id: {"type": niche, "size": n_cells}}
    """
    n_cells = len(labels_all)
    cell_to_domain = np.full(n_cells, -1, dtype=np.int64)
    domain_meta = {}
    next_domain_id = 0

    for niche in states:
        niche_idx = np.flatnonzero(labels_all == str(niche))
        if len(niche_idx) == 0:
            continue
        subgraph = adj[niche_idx][:, niche_idx]
        n_comp, comp_labels = connected_components(subgraph, directed=False, return_labels=True)
        comp_sizes = np.bincount(comp_labels)
        for local_pos, comp_id in enumerate(comp_labels):
            domain_id = next_domain_id + int(comp_id)
            cell_to_domain[niche_idx[local_pos]] = domain_id
            if domain_id not in domain_meta:
                domain_meta[domain_id] = {"type": str(niche), "size": int(comp_sizes[comp_id])}
        next_domain_id += int(n_comp)

    # only domains >= threshold participate
    valid_domain_ids = {bid for bid, meta in domain_meta.items() if meta["size"] >= min_domain_size}
    if len(valid_domain_ids) < 2:
        return None, None

    # cell-cell spatial edges -> unique domain-domain edges
    rows, cols = adj.nonzero()
    src_domain, dst_domain = cell_to_domain[rows], cell_to_domain[cols]
    keep = (np.isin(src_domain, list(valid_domain_ids)) & np.isin(dst_domain, list(valid_domain_ids))
            & (src_domain != dst_domain))
    if not np.any(keep):
        return None, None
    directed_edges = np.unique(np.column_stack([src_domain[keep], dst_domain[keep]]), axis=0)

    domain_adjacency = {int(bid): [] for bid in sorted(valid_domain_ids)}
    for src, dst in directed_edges:
        src, dst = int(src), int(dst)
        if src in domain_adjacency and dst in valid_domain_ids:
            domain_adjacency[src].append(dst)

    # remove isolated domains so the permutation only uses the actual contact graph
    active = {bid for bid, nbrs in domain_adjacency.items() if len(nbrs) > 0}
    if len(active) < 2:
        return None, None
    domain_adjacency = {bid: [nbr for nbr in nbrs if nbr in active]
                      for bid, nbrs in domain_adjacency.items() if bid in active}
    domain_adjacency = {bid: nbrs for bid, nbrs in domain_adjacency.items() if len(nbrs) > 0}

    active = set(domain_adjacency)
    return domain_adjacency, {bid: domain_meta[bid] for bid in active}


domain_adjacency_store, core_summary = {}, []
cores = pd.unique(obs[CORE_COL])
for core in tqdm(cores, desc="domain detection"):
    mask = (obs[CORE_COL] == core).to_numpy()
    if mask.sum() < 2:
        continue
    idx = np.flatnonzero(mask)
    domain_adjacency, domain_meta = process_single_core_domains(
        adj_all[idx][:, idx], obs[NICHE_COL].to_numpy()[idx], states, MIN_domain_SIZE)
    if domain_adjacency is None:
        continue
    domain_adjacency_store[core] = (domain_adjacency, domain_meta)
    core_summary.append({CORE_COL: core, "n_valid_domains": len(domain_meta),
                         "n_directed_domain_edges": sum(len(v) for v in domain_adjacency.values())})

pd.DataFrame(core_summary).to_csv(OUT_DIR / "domain_core_summary.csv", index=False)
print("Cores retained:", len(domain_adjacency_store))


# %% 2. Observed neighbourhood-to-neighbourhood domain contacts per core
def compute_observed_edge_counts(domain_adjacency_store, states):
    niche_to_idx = {str(n): i for i, n in enumerate(states)}
    obs_store, total_obs = {}, {}
    for core, (domain_adjacency, domain_meta) in domain_adjacency_store.items():
        mat = np.zeros((len(states), len(states)), dtype=float)
        for src_domain, neighbors in domain_adjacency.items():
            i = niche_to_idx.get(str(domain_meta[src_domain]["type"]))
            if i is None:
                continue
            for dst_domain in neighbors:
                if dst_domain not in domain_meta:
                    continue
                j = niche_to_idx.get(str(domain_meta[dst_domain]["type"]))
                if j is None:
                    continue
                mat[i, j] += 1
        np.fill_diagonal(mat, 0)
        obs_store[core] = pd.DataFrame(mat, index=states, columns=states)
        total_obs[core] = float(mat.sum())
    return obs_store, total_obs


obs_store, total_obs = compute_observed_edge_counts(domain_adjacency_store, states)


# %% 3. Merge-based permutation null
def compute_permutation_expected(domain_adjacency_store, states, total_obs,
                                 n_permutations=10_000, seed=42):
    """Shuffle domain labels within each core, merge adjacent domains that now share
    a label, count contacts, and average (rescaled to the observed total)."""
    n_niches = len(states)
    niche_to_idx = {str(n): i for i, n in enumerate(states)}
    perm_E_store = {}
    master_rng = np.random.default_rng(seed)

    for core, (domain_adjacency, domain_meta) in tqdm(domain_adjacency_store.items(),
                                                  desc="Permutation null"):
        t_obs = total_obs.get(core, 0.0)
        if t_obs <= 0 or not domain_adjacency:
            continue
        domain_ids = list(domain_adjacency.keys())
        n_domains = len(domain_ids)
        if n_domains < 2:
            continue
        domain_to_i = {bid: i for i, bid in enumerate(domain_ids)}
        type_idx = np.array([niche_to_idx.get(str(domain_meta[bid]["type"]), -1)
                             for bid in domain_ids], dtype=np.int32)
        valid_pos = np.flatnonzero(type_idx >= 0)
        if len(valid_pos) < 2:
            continue

        edge_src, edge_dst = [], []
        for src_i, src_bid in enumerate(domain_ids):
            for nbr_bid in domain_adjacency[src_bid]:
                dst_i = domain_to_i.get(nbr_bid)
                if dst_i is not None:
                    edge_src.append(src_i)
                    edge_dst.append(dst_i)
        edge_src = np.asarray(edge_src, dtype=np.int32)
        edge_dst = np.asarray(edge_dst, dtype=np.int32)
        if len(edge_src) == 0:
            continue

        # one orientation of each undirected edge, for same-label merging
        undirected = edge_src < edge_dst
        edge_u_src, edge_u_dst = edge_src[undirected], edge_dst[undirected]

        rng = np.random.default_rng(int(master_rng.integers(0, np.iinfo(np.int32).max)))
        accum = np.zeros((n_niches, n_niches), dtype=float)
        n_valid_perm = 0

        for _ in range(n_permutations):
            shuffled = type_idx.copy()
            shuffled[valid_pos] = rng.permutation(type_idx[valid_pos])

            # merge adjacent domains that now have the same label
            if len(edge_u_src) > 0:
                same = ((shuffled[edge_u_src] >= 0) & (shuffled[edge_u_dst] >= 0)
                        & (shuffled[edge_u_src] == shuffled[edge_u_dst]))
            else:
                same = np.zeros(0, dtype=bool)

            if same.any():
                s, d = edge_u_src[same], edge_u_dst[same]
                same_label_adj = csr_matrix(
                    (np.ones(2 * len(s), dtype=np.float32),
                     (np.concatenate([s, d]), np.concatenate([d, s]))),
                    shape=(n_domains, n_domains))
                n_components, component_id = connected_components(
                    same_label_adj, directed=False, return_labels=True)
            else:
                component_id = np.arange(n_domains, dtype=np.int32)
                n_components = n_domains

            component_type = np.full(n_components, -1, dtype=np.int32)
            for i in range(n_domains):
                if shuffled[i] >= 0:
                    component_type[component_id[i]] = shuffled[i]

            # directed contacts between distinct merged components
            c_src, c_dst = component_id[edge_src], component_id[edge_dst]
            inter = c_src != c_dst
            t_src, t_dst = component_type[c_src[inter]], component_type[c_dst[inter]]
            valid = (t_src >= 0) & (t_dst >= 0)

            perm_mat = np.zeros((n_niches, n_niches), dtype=float)
            np.add.at(perm_mat, (t_src[valid], t_dst[valid]), 1)
            np.fill_diagonal(perm_mat, 0)

            # rescale to the observed total number of directed edges
            t_perm = float(perm_mat.sum())
            if t_perm > 0:
                accum += perm_mat * (t_obs / t_perm)
                n_valid_perm += 1

        if n_valid_perm > 0:
            perm_E_store[core] = pd.DataFrame(accum / n_valid_perm, index=states, columns=states)

    print(f"Expected matrices computed for {len(perm_E_store)} cores")
    return perm_E_store


t0 = time.time()
perm_E_store = compute_permutation_expected(domain_adjacency_store, states, total_obs,
                                            n_permutations=N_PERMUTATIONS, seed=PERM_SEED)
print(f"Permutation time: {(time.time() - t0) / 60:.1f} min")


# %% 4. Poisson rate ratio per neighbourhood pair across cores
def compute_poisson_rate_ratios(obs_store, perm_E_store, states):
    """log(E[O]) = beta + log(E), fitted across cores; rate ratio = exp(beta)."""
    result_rows, raw_rows = [], []

    for a, b in combinations(states, 2):
        obs_values, exp_values = [], []
        for core, obs_mat in obs_store.items():
            if core not in perm_E_store:
                continue
            e = float(perm_E_store[core].loc[a, b])
            if not np.isfinite(e) or e <= 0:      # only cores with nonzero expected contribute
                continue
            o = float(obs_mat.loc[a, b])
            obs_values.append(o)
            exp_values.append(e)
            raw_rows.append({"central_niche": a, "paired_niche": b, CORE_COL: core,
                             "observed": o, "expected": e, "oe": o / e})

        if len(obs_values) < 2:                   # at least 2 cores
            continue

        o_arr, e_arr = np.asarray(obs_values, float), np.asarray(exp_values, float)
        try:
            fit = sm.GLM(o_arr, np.ones((len(o_arr), 1)), family=sm.families.Poisson(),
                         offset=np.log(e_arr)).fit(disp=0)
            beta = float(fit.params[0])
            ci = np.asarray(fit.conf_int())[0]
            p_value = float(fit.pvalues[0])
        except Exception as exc:
            print(f"GLM failed for {a} <-> {b}: {exc}")
            beta, ci, p_value = np.nan, [np.nan, np.nan], np.nan

        result_rows.append({
            "niche_A": a, "niche_B": b,
            "log_rate_ratio": beta,
            "rate_ratio": np.exp(beta) if np.isfinite(beta) else np.nan,
            "ci_lo_log": float(ci[0]), "ci_hi_log": float(ci[1]),
            "rate_ratio_ci_lo": np.exp(ci[0]) if np.isfinite(ci[0]) else np.nan,
            "rate_ratio_ci_hi": np.exp(ci[1]) if np.isfinite(ci[1]) else np.nan,
            "p_value": p_value,
            "n_cores": len(o_arr),
            "n_zero_observed": int(np.sum(o_arr == 0)),
            "sum_observed": float(o_arr.sum()),
            "sum_expected": float(e_arr.sum()),
        })

    poisson_df, raw_oe_df = pd.DataFrame(result_rows), pd.DataFrame(raw_rows)
    if len(poisson_df) > 0:
        poisson_df["p_adj"] = np.nan
        valid = poisson_df["p_value"].notna()
        if valid.any():
            poisson_df.loc[valid, "p_adj"] = multipletests(
                poisson_df.loc[valid, "p_value"].to_numpy(), method="fdr_bh")[1]
    return poisson_df, raw_oe_df


poisson_df, raw_oe_df = compute_poisson_rate_ratios(obs_store, perm_E_store, states)
poisson_df.to_csv(OUT_DIR / "poisson_niche_pair_results.csv", index=False)
raw_oe_df.to_csv(OUT_DIR / "per_core_observed_expected.csv", index=False)

# %% 5. Per diagnostic group (same per-core O and E, refitted on the cores of each group)
core_diag = obs[[CORE_COL, DX_COL]].astype(str).drop_duplicates()
multi = core_diag[CORE_COL][core_diag[CORE_COL].duplicated(keep=False)].unique()
if len(multi):
    raise ValueError(f"Cores with >1 diagnosis label: {list(multi)[:10]}")
core_to_diag = core_diag.set_index(CORE_COL)[DX_COL]

usable_cores = [c for c in obs_store if c in perm_E_store]
by_dx = []
for diag, sub in core_to_diag.groupby(core_to_diag):
    cores_d = [c for c in usable_cores if str(c) in set(sub.index)]
    if len(cores_d) < MIN_CORES_PER_DIAGNOSIS:
        print(f"Skipping {diag}: {len(cores_d)} usable core(s)")
        continue
    p_df, _ = compute_poisson_rate_ratios({c: obs_store[c] for c in cores_d},
                                          {c: perm_E_store[c] for c in cores_d}, states)
    if len(p_df) == 0:
        continue
    p_df.insert(0, "diagnosis", diag)
    p_df["all_zero_observed"] = p_df["sum_observed"] == 0
    by_dx.append(p_df)
    print(f"{diag:<28} cores={len(cores_d):>4}  pairs fitted={len(p_df)}")

pd.concat(by_dx, ignore_index=True).to_csv(
    OUT_DIR / "poisson_niche_pair_results_by_diagnosis.csv", index=False)
print("Saved to", OUT_DIR)
