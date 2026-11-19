"""
Figure 3 and Extended Data Figures 5-6: neighbourhood definition and co-localisation.

  Fig. 3A    tissue space (cell types, neighbour graph, neighbourhoods) of one core,
             and UMAP of the spatial-neighbour composition
  Fig. 3B    zooms into three neighbourhoods of the same core, with cell-type composition
  Fig. 3C    cell-type composition per neighbourhood
  Fig. 3D    neighbourhood enrichment (odds ratio) per cell type and per TME compartment
  Fig. 3E    neighbourhood composition per core
  Fig. 3G    neighbourhood co-localisation (Poisson rate ratio)
  ED Fig. 5A tumour-state neighbourhood specificity (entropy-based, per core)
  ED Fig. 5B neighbourhood enrichment (odds ratio) per diagnostic group
  ED Fig. 6A co-localisation similarity between diagnostic groups (Jaccard)
  ED Fig. 6B co-localisation per diagnostic group

Fig. 3G and ED Fig. 6 need the output of Analysis/nhood_colocalisation_poisson.py (run that first). 
All other panels use the full spatial object.
"""

# %% Imports and setup
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp
import seaborn as sns
import shapely
from shapely.geometry import box
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from matplotlib import collections as mc
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.collections import PolyCollection
from matplotlib.colors import LinearSegmentedColormap, LogNorm, PowerNorm, TwoSlopeNorm
from matplotlib.lines import Line2D
from matplotlib.patches import PathPatch, Rectangle
from matplotlib.path import Path as MplPath
from scipy.cluster.hierarchy import leaves_list, linkage
from scipy.spatial import cKDTree
from scipy.spatial.distance import pdist
from scipy.stats import entropy

# %% Paths (relative to the repository root; edit if your data live elsewhere)
ADATA_FULL = Path("data/rdata_5tx_pca_euclidean_umap.h5ad")
COLOC_DIR = Path("results/analysis/nhood_colocalisation")   # from analysis/nhood_colocalisation_poisson.py
OUT_DIR = Path("results/Figure3")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# keep text editable in Illustrator/Inkscape
mpl.rcParams["pdf.fonttype"] = 42
mpl.rcParams["ps.fonttype"] = 42

# %% Settings
NICHE_COL = "leiden_nhood_annot"
CT_COL = "Annotation_merged_scanVI"
CORE_COL = "core"
DX_COL = "Diagnosis"

EXAMPLE_CORE = "0010586_core_18"     # Fig. 3A and 3B
EDGE_LIMIT = 50                      # max edge length drawn in Fig. 3A

# neighbourhood labels in the object -> names in the paper (also the plotting order)
NHOOD_NAMES = {
    "Vascular": "Vascular",
    "TBI": "TBI",
    "Cortical glia": "OPC-like I",
    "OPC-like": "OPC-like II",
    "NPC-like": "NPC-like",
    "AC-like": "AC-like",
    "MES1": "Gliosis",
    "MES2": "Hypoxia",
    "Proliferating/Immune": "Proliferating",
    "Stressed": "Stress-response",
    "Spinal-glia": "EPN-like",
}
NHOOD_ORDER = list(NHOOD_NAMES.values())
NHOOD_COLORS = {
    "Vascular": "#C1121F", "TBI": "#6a4c93", "OPC-like I": "#ccd5ae", "OPC-like II": "#80ed99",
    "NPC-like": "#1A659E", "AC-like": "#00CED1", "Gliosis": "#E76F51", "Hypoxia": "#588157",
    "Proliferating": "#edae49", "Stress-response": "#2A9D8F", "EPN-like": "#F5CAC3",
}

# cell-type labels in the object -> names in the paper
CT_NAMES = {
    "Tumor Factor (Cortical Glia)": "OPC-like I",
    "Tumor OPC-like": "OPC-like II",
    "Tumor NPC-like": "NPC-like",
    "Tumor AC-like": "AC-like",
    "Tumor MES1": "MES1",
    "Tumor MES2": "MES2",
    "Tumor Proliferating": "Proliferating",
    "Tumor Stressed": "Stressed",
    "Tumor Factor (Spinal Glia)": "EPN-like",
    "Astrocytes1": "Astrocytes",
    "Astrocytes2": "Astrocytes",
    "Perivascular fibroblasts": "Perivascular fibro.",
}
TUMOR_STATES = ["OPC-like I", "OPC-like II", "NPC-like", "AC-like", "MES1", "MES2",
                "Proliferating", "Stressed", "EPN-like"]
CT_COLORS = {
    "OPC-like I": "#b6a179", "OPC-like II": "#feb144", "NPC-like": "#5e8c61",
    "Proliferating": "#e55934", "AC-like": "#c97d63", "MES1": "#c97dba", "MES2": "#8687c1",
    "Stressed": "#3e1929", "EPN-like": "#8a7d72",
    "Astrocytes": "#e04f39", "Neurons-exh.": "#1976D2", "Neurons-inh.": "#64B5F6",
    "OPC/OC": "#fea022",
    "Myeloid-P2RY12+": "#eaccd1", "Myeloid-CD163+": "#ff9cae", "Myeloid-CLEC10A+": "#ff7196",
    "Myeloid-RNASE1+": "#febccc", "Neutrophils": "#cc87fa",
    "Vascular EC": "#fad644", "Perivascular fibro.": "#d7b259",
    "T-CD8+": "#A8E6A3", "T-CD4+": "#4CAF50", "B cells": "#388E3C", "NK cells": "#81C784",
    "Plasma cells": "#45a1a0", "Mast cells": "#839976", "T-cycling": "#839976",
}
# tumour states drawn in the colour of their neighbourhood (ED Fig. 5A)
TUMOR_STATE_NHOOD_COLORS = {
    "AC-like": "#00CED1", "OPC-like I": "#ccd5ae", "MES1": "#E76F51", "MES2": "#588157",
    "NPC-like": "#1A659E", "OPC-like II": "#80ed99", "EPN-like": "#F5CAC3",
    "Stressed": "#2A9D8F", "Proliferating": "#edae49",
}

# TME columns of the enrichment heatmaps (Fig. 3D, ED Fig. 5B)
TME_COMPONENTS = ["Vascular EC", "Perivascular fibro.", "OPC/OC", "Neurons-inh.", "Neurons-exh.",
                  "Astrocytes", "Myeloid-P2RY12+", "Myeloid-CD163+", "Myeloid-RNASE1+",
                  "Myeloid-CLEC10A+", "Neutrophils", "T-CD8+", "T-CD4+", "NK cells", "B cells",
                  "Mast cells", "Plasma cells", "T-cycling"]
BRAIN_RESIDENT = ["OPC/OC", "Neurons-inh.", "Neurons-exh.", "Astrocytes", "Myeloid-P2RY12+"]
PERIPHERAL_IMMUNE = ["Myeloid-CD163+", "Myeloid-RNASE1+", "Myeloid-CLEC10A+", "Neutrophils",
                     "T-CD8+", "T-CD4+", "NK cells", "B cells", "Mast cells", "Plasma cells",
                     "T-cycling"]

DX_COLORS = {
    "Oligodendroglioma": "#e8e283", "Astrocytoma low-grade": "#a8d39c",
    "Astrocytoma high-grade": "#e6aa5f", "Glioblastoma": "#b26b6b",
    "Subependymoma": "#b499c8", "Myxopapillary ependymoma": "#7d00c4",
    "Spinal ependymoma": "#d8b8f1",
}
DX_ORDER = list(DX_COLORS)
DX_SHORT = {"Astrocytoma low-grade": "Astrocytoma LG", "Astrocytoma high-grade": "Astrocytoma HG",
            "Subependymoma": "Sub-EPN", "Spinal ependymoma": "Spinal EPN",
            "Myxopapillary ependymoma": "Myxopapillary EPN"}

# odds-ratio colour scale (Fig. 3D, ED Fig. 5B): RdBu_r without the darkest blues
OR_VMIN, OR_VMAX = 0.1, 15
OR_CMAP = LinearSegmentedColormap.from_list(
    "RdBu_r_lighterblue", mpl.colormaps["RdBu_r"](np.linspace(0.10, 1.0, 256)))
OR_CMAP.set_bad("#d9d9d9")

# rate-ratio colour scale (Fig. 3G, ED Fig. 6B)
RR_CMAP = LinearSegmentedColormap.from_list("rr_soft_center", [
    (0.00, "#084594"), (0.22, "#2b79b9"), (0.36, "#9ecae1"), (0.44, "#e3f0f8"),
    (0.48, "#f6fbfe"), (0.50, "#ffffff"), (0.52, "#fffaf8"), (0.56, "#fce9e1"),
    (0.64, "#f7b89c"), (0.78, "#d6604d"), (1.00, "#99000d")])
RR_NORM = TwoSlopeNorm(vmin=0, vcenter=1, vmax=2)


def savefig(fig, name, **kw):
    fig.savefig(OUT_DIR / name, bbox_inches="tight", **kw)
    plt.close(fig)
    print("Saved:", OUT_DIR / name)


# %% Load data: obs, spatial coordinates and spatial graph (expression is not needed)
backed = sc.read_h5ad(ADATA_FULL, backed="r")
rdata = ad.AnnData(
    obs=backed.obs.copy(),
    obsm={"spatial": np.asarray(backed.obsm["spatial"])},
    obsp={k: backed.obsp[k] for k in ("spatial_connectivities", "spatial_distances")},
)
del backed

rdata.obs[CT_COL] = rdata.obs[CT_COL].astype(str).replace(CT_NAMES)
ct_order = [c for c in CT_COLORS if c in set(rdata.obs[CT_COL])]
rdata.obs[CT_COL] = pd.Categorical(rdata.obs[CT_COL], categories=ct_order)
rdata.uns[f"{CT_COL}_colors"] = [CT_COLORS[c] for c in ct_order]

rdata.obs[NICHE_COL] = rdata.obs[NICHE_COL].astype(str).map(NHOOD_NAMES)
nhood_order = [n for n in NHOOD_ORDER if n in set(rdata.obs[NICHE_COL].dropna())]
rdata.obs[NICHE_COL] = pd.Categorical(rdata.obs[NICHE_COL], categories=nhood_order)
rdata.uns[f"{NICHE_COL}_colors"] = [NHOOD_COLORS[n] for n in nhood_order]

cat_palette = rdata.uns[f"{NICHE_COL}_colors"]
anno_palette = rdata.uns[f"{CT_COL}_colors"]

obs_all = rdata.obs[[CORE_COL, DX_COL, NICHE_COL, CT_COL]].astype(str)   # incl. cells without a neighbourhood
obs_df = obs_all[obs_all[NICHE_COL] != "nan"]
dx_order = [d for d in DX_ORDER if d in set(obs_df[DX_COL])]


# %% Fig. 3A (left): tissue space of the example core
w = (rdata.obs[CORE_COL] == EXAMPLE_CORE).to_numpy()
xy = rdata.obsm["spatial"][w, :2]
obs_core = rdata.obs[w]
connect_core = rdata.obsp["spatial_connectivities"][w][:, w].tocsr()
dist_core = rdata.obsp["spatial_distances"][w][:, w].tocsr()
ann_codes_core = obs_core[CT_COL].cat.codes.values

with PdfPages(OUT_DIR / "Fig3A_tissue_space.pdf") as pdf:
    # page 1: cell types + all spatial connections
    fig, ax = plt.subplots(figsize=(20, 20))
    lines = [[tuple(xy[i]), tuple(xy[j])]
             for i in range(len(obs_core)) for j in connect_core[i].indices
             if j != i and dist_core[i, j] < EDGE_LIMIT]
    if lines:
        ax.add_collection(mc.LineCollection(lines, colors="lightgrey", linewidths=0.5,
                                            alpha=0.5, zorder=1))
    ax.scatter(xy[:, 0], xy[:, 1], marker=".", c=np.asarray(anno_palette)[ann_codes_core],
               s=10, zorder=2)
    ax.legend(handles=[mpatches.Patch(color=anno_palette[j], label=a)
                       for j, a in enumerate(obs_core[CT_COL].cat.categories)
                       if (obs_core[CT_COL] == a).any()],
              bbox_to_anchor=(1, 0.5), loc="center left", title="Cell types")
    ax.set_title(f"{EXAMPLE_CORE}\nCell types + neighbour graph")
    ax.axis("off")
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)

    # page 2: neighbourhoods (within-neighbourhood edges + territories) + cell types
    fig, ax = plt.subplots(figsize=(20, 20))
    for j, nhood in enumerate(obs_core[NICHE_COL].cat.categories):
        idx = np.where(obs_core[NICHE_COL] == nhood)[0]
        if len(idx) == 0:
            continue
        c_n, d_n = connect_core[idx][:, idx], dist_core[idx][:, idx]
        lines = [[tuple(xy[k]), tuple(xy[idx[l]])]
                 for li, k in enumerate(idx) for l in c_n[li].indices
                 if l != li and d_n[li, l] < EDGE_LIMIT]
        if lines:
            ax.add_collection(mc.LineCollection(lines, colors=cat_palette[j], linewidths=0.5,
                                                zorder=1))

    # triangles whose 3 cells share a neighbourhood and whose edges are short
    tri = mpl.tri.Triangulation(*xy.T)
    nt = obs_core[NICHE_COL].cat.codes.values[tri.triangles]
    xy_tri = xy[tri.triangles]
    max_edge = np.sqrt(((xy_tri - np.roll(xy_tri, -1, axis=1)) ** 2).sum(axis=2)).max(axis=1)
    tri.set_mask(~((nt == nt[:, [0]]).all(axis=1) & (max_edge < EDGE_LIMIT)))
    ax.tripcolor(tri, nt[:, 0], cmap=mcolors.ListedColormap(cat_palette),
                 vmax=len(cat_palette) - 1, alpha=0.25, zorder=0)

    ax.scatter(xy[:, 0], xy[:, 1], marker=".", c=np.asarray(anno_palette)[ann_codes_core],
               s=10, zorder=2)
    ax.legend(handles=[mpatches.Patch(color=cat_palette[j], label=n)
                       for j, n in enumerate(obs_core[NICHE_COL].cat.categories)
                       if (obs_core[NICHE_COL] == n).any()],
              bbox_to_anchor=(1, 0.5), loc="center left", title="Neighbourhoods")
    ax.set_title(f"{EXAMPLE_CORE}\nNeighbourhoods + cell types")
    ax.axis("off")
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)
print("Saved:", OUT_DIR / "Fig3A_tissue_space.pdf")

# %% Fig. 3A (right): UMAP of the spatial-neighbour composition of each cell
# each cell: fraction of each cell type among its neighbours (2 hops, excluding itself)
HOPS = 2
N_PER_NHOOD = 30000      # UMAP on a subsample, balanced across neighbourhoods
MIN_NEIGHBORS = 3

codes = rdata.obs[CT_COL].cat.codes.values
ok = codes >= 0
onehot = sp.csr_matrix((np.ones(ok.sum(), np.float32), (np.where(ok)[0], codes[ok])),
                       shape=(rdata.n_obs, len(ct_order)))
A = (rdata.obsp["spatial_connectivities"] > 0).astype(np.float32).tocsr()
A.setdiag(0)
A.eliminate_zeros()
A_h = A.copy()
for _ in range(HOPS - 1):
    A_h = A_h + A_h @ A
A = (A_h > 0).astype(np.float32).tocsr()
A.setdiag(0)
A.eliminate_zeros()

counts = np.asarray((A @ onehot).todense(), dtype=np.float32)
n_nb = counts.sum(1)
comp = counts / np.maximum(n_nb, 1)[:, None]
del A, A_h, counts

rng = np.random.default_rng(0)
keep = []
for nh in rdata.obs[NICHE_COL].cat.categories:
    idx = np.where((rdata.obs[NICHE_COL] == nh).values & (n_nb >= MIN_NEIGHBORS))[0]
    keep.append(rng.choice(idx, min(N_PER_NHOOD, len(idx)), replace=False))
keep = np.sort(np.concatenate(keep))

sub = ad.AnnData(X=comp[keep], obs=rdata.obs.iloc[keep][[CT_COL, NICHE_COL, CORE_COL]].copy())
sub.var_names = ct_order
del comp
sc.pp.neighbors(sub, use_rep="X", n_neighbors=30, metric="euclidean")
sc.tl.umap(sub, min_dist=0.3, random_state=0)


def umap_panel(ax, sub, col, palette, title):
    cats = sub.obs[col].cat.categories
    c = sub.obs[col].cat.codes.values
    order = np.random.default_rng(1).permutation(sub.n_obs)   # no category drawn on top
    u = sub.obsm["X_umap"][order]
    ax.scatter(u[:, 0], u[:, 1], s=0.5, c=np.asarray(palette)[c[order]],
               linewidths=0, rasterized=True)
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel("UMAP1")
    ax.set_ylabel("UMAP2")
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    ax.legend(handles=[Line2D([], [], marker="o", ls="", mfc=palette[i], mec="none", ms=6,
                              label=cats[i]) for i in np.unique(c)],
              loc="center left", bbox_to_anchor=(1, 0.5), frameon=False, fontsize=8)


fig, axes = plt.subplots(1, 2, figsize=(20, 8), gridspec_kw={"wspace": 0.6})
umap_panel(axes[0], sub, CT_COL, anno_palette, "Tumour states")
umap_panel(axes[1], sub, NICHE_COL, cat_palette, "Neighbourhoods")
savefig(fig, "Fig3A_umap_spatial_neighbours.pdf", dpi=600)

# %% Fig. 3B: neighbourhood zooms (cell type = dot, neighbourhood = capped Voronoi territory)
TERRITORY_ALPHA = 0.55   # 1.0 = full neighbourhood colour; lower = lighter


def _blend(c, alpha):
    """Colour pre-blended onto white (opaque, no PDF transparency)."""
    r, g, b, _ = mcolors.to_rgba(c)
    return (r * alpha + 1 - alpha, g * alpha + 1 - alpha, b * alpha + 1 - alpha)


def _polys(geom):
    return [p for p in shapely.get_parts(geom) if p.geom_type == "Polygon" and not p.is_empty]


def _to_path(geom):
    verts, path_codes = [], []
    for p in _polys(geom):
        for ring in [p.exterior, *p.interiors]:
            c = np.asarray(ring.coords)
            verts.append(c)
            path_codes += [MplPath.MOVETO] + [MplPath.LINETO] * (len(c) - 2) + [MplPath.CLOSEPOLY]
    return MplPath(np.concatenate(verts), path_codes) if verts else None


def capped_voronoi(xy, max_dist, bounds):
    """Voronoi cell of each point, intersected with a circle (max_dist) and the window."""
    x0, x1, y0, y1 = bounds
    pts = shapely.points(xy)
    env = box(x0 - 5 * max_dist, y0 - 5 * max_dist, x1 + 5 * max_dist, y1 + 5 * max_dist)
    vor = np.asarray(shapely.get_parts(
        shapely.voronoi_polygons(shapely.multipoints(xy), extend_to=env)))
    p_idx, c_idx = shapely.STRtree(vor).query(pts, predicate="intersects")
    cells = np.empty(len(xy), dtype=object)
    cells[p_idx] = vor[c_idx]
    circles = shapely.buffer(pts, max_dist, quad_segs=8)
    missing = np.array([c is None for c in cells])        # duplicated coordinates
    cells[missing] = circles[missing]
    cells = shapely.intersection(cells, circles)
    return shapely.intersection(cells, box(x0, y0, x1, y1))


def draw_region(ax, xy, nh_codes, ann_codes, bounds, *, max_dist=12, dot_size=25,
                cell_border=True, nhood_border=True):
    x0, x1, y0, y1 = bounds
    pad = 2 * max_dist
    sel = ((xy[:, 0] >= x0 - pad) & (xy[:, 0] < x1 + pad) &
           (xy[:, 1] >= y0 - pad) & (xy[:, 1] < y1 + pad))
    cells = capped_voronoi(xy[sel], max_dist, bounds)
    codes_sel = nh_codes[sel]

    for code in np.unique(codes_sel):
        m = codes_sel == code
        col = _blend(cat_palette[code], TERRITORY_ALPHA) if code >= 0 else (0.85, 0.85, 0.85)
        merged = None
        if cell_border:
            verts = [np.asarray(p.exterior.coords) for g in cells[m] for p in _polys(g)]
            ax.add_collection(PolyCollection(verts, facecolors=[col], edgecolors="white",
                                             linewidths=0.4, zorder=0))
        else:
            merged = shapely.union_all(cells[m], grid_size=1e-3)
            path = _to_path(merged)
            if path is not None:
                ax.add_patch(PathPatch(path, fc=col, ec="none", zorder=0))
        if nhood_border:
            if merged is None:
                merged = shapely.union_all(cells[m], grid_size=1e-3)
            path = _to_path(merged)
            if path is not None:
                ax.add_patch(PathPatch(path, fc="none", ec=(0.15, 0.15, 0.15), lw=0.7, zorder=1))

    inb = (xy[:, 0] >= x0) & (xy[:, 0] < x1) & (xy[:, 1] >= y0) & (xy[:, 1] < y1)
    ax.scatter(xy[inb, 0], xy[inb, 1], s=dot_size, c=np.asarray(anno_palette)[ann_codes[inb]],
               edgecolors="k", linewidths=0.3 if dot_size > 5 else 0, zorder=2)
    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])


def window_coverage(tree, x0, y0, win, max_dist, n=15):
    """Fraction of the window that is tissue (within max_dist of a cell)."""
    g = np.linspace(0, win, n)
    gx, gy = np.meshgrid(x0 + g, y0 + g)
    d, _ = tree.query(np.c_[gx.ravel(), gy.ravel()])
    return (d < max_dist).mean()


def pick_window_for(xy, nh_codes, target_code, win=250, min_cells=80, avoid=(),
                    tree=None, max_dist=None, min_coverage=0.95):
    """Window with the highest fraction of target-neighbourhood cells (ties -> more cells)."""
    step = win // 5
    best, best_score = None, (-1, -1)
    for x0 in np.arange(xy[:, 0].min(), xy[:, 0].max() - win + step, step):
        for y0 in np.arange(xy[:, 1].min(), xy[:, 1].max() - win + step, step):
            if any(abs(x0 - a) < win and abs(y0 - c) < win for a, _, c, _ in avoid):
                continue
            m = ((xy[:, 0] >= x0) & (xy[:, 0] < x0 + win) &
                 (xy[:, 1] >= y0) & (xy[:, 1] < y0 + win))
            n = m.sum()
            if n < min_cells:
                continue
            if tree is not None and window_coverage(tree, x0, y0, win, max_dist) < min_coverage:
                continue
            frac = (nh_codes[m] == target_code).mean()
            if (frac, n) > best_score:
                best_score, best = (frac, n), (x0, x0 + win, y0, y0 + win)
    return best, best_score[0]


def composition_bar(ax, ann_codes_sel, an_cats, title, top_n_labels=3):
    counts = np.bincount(ann_codes_sel, minlength=len(an_cats))
    frac = counts / counts.sum()
    order = np.argsort(frac)[::-1]
    order = order[frac[order] > 0]
    left = 0
    for i, j in enumerate(order):
        ax.barh(0, frac[j], left=left, color=anno_palette[j], edgecolor="white",
                linewidth=0.5, height=0.8)
        if i < top_n_labels and frac[j] > 0.07:
            ax.text(left + frac[j] / 2, 0, f"{frac[j] * 100:.0f}%", ha="center", va="center",
                    fontsize=8, bbox=dict(fc="white", ec="none", pad=0.5))
        left += frac[j]
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.5, 0.5)
    ax.set_yticks([])
    ax.set_xticks([0, 0.5, 1])
    ax.set_xticklabels(["0", "50", "100"], fontsize=7)
    ax.set_xlabel("Cell type composition (%)", fontsize=7)
    for s in ["top", "right", "left"]:
        ax.spines[s].set_visible(False)
    ax.set_title(title, fontsize=8.5, loc="left")
    return order


targets = ("Hypoxia", "OPC-like I", "Vascular")
nh_cats = list(obs_core[NICHE_COL].cat.categories)
an_cats = list(obs_core[CT_COL].cat.categories)
nh_codes = obs_core[NICHE_COL].cat.codes.values
tree = cKDTree(xy)
max_dist = 1.5 * np.median(tree.query(xy, k=2)[0][:, 1])

windows = []
for t in targets:
    b, f = pick_window_for(xy, nh_codes, nh_cats.index(t), avoid=windows, tree=tree,
                           max_dist=max_dist)
    if b is None:
        raise ValueError(f"No fully-tissue window found for {t} in {EXAMPLE_CORE}")
    windows.append(b)

fig = plt.figure(figsize=(8 + 4.2 * len(targets), 8))
gs = fig.add_gridspec(1, 1 + len(targets), width_ratios=[2.4] + [1] * len(targets), wspace=0.08)
ax_ov = fig.add_subplot(gs[0, 0])
ov_bounds = (xy[:, 0].min() - 20, xy[:, 0].max() + 20, xy[:, 1].min() - 20, xy[:, 1].max() + 20)
draw_region(ax_ov, xy, nh_codes, ann_codes_core, ov_bounds, max_dist=max_dist, dot_size=1.5,
            cell_border=False, nhood_border=False)

for k, (t, b) in enumerate(zip(targets, windows)):
    tcode = nh_cats.index(t)
    tcol = cat_palette[tcode]
    lab = ["I", "II", "III"][k]
    ax = fig.add_subplot(gs[0, 1 + k])
    draw_region(ax, xy, nh_codes, ann_codes_core, b, max_dist=max_dist, dot_size=28)
    for s in ax.spines.values():
        s.set_edgecolor(tcol)
        s.set_linewidth(4)
    ax.set_title(f"{lab}. {t} neighbourhood", fontsize=11, loc="left")
    ax.plot([b[1] - 60, b[1] - 10], [b[2] + 10] * 2, "k-", lw=3, zorder=6)   # 50 µm scale bar
    ax.text(b[1] - 35, b[2] + 14, "50", ha="center", fontsize=8)

    ax_ov.add_patch(Rectangle((b[0], b[2]), b[1] - b[0], b[3] - b[2], fill=False, ec="k",
                              lw=1.5, zorder=6))
    ax_ov.text(b[0], b[3], lab, fontsize=13, weight="bold", va="bottom", ha="left", zorder=7)

    # cell types of the target-neighbourhood cells inside the zoom
    inb = (xy[:, 0] >= b[0]) & (xy[:, 0] < b[1]) & (xy[:, 1] >= b[2]) & (xy[:, 1] < b[3])
    sel = inb & (nh_codes == tcode)
    composition_bar(ax.inset_axes([0, -0.30, 1, 0.13]), ann_codes_core[sel], an_cats,
                    f"n={sel.sum()} cells")

nh_handles = [mpatches.Patch(color=_blend(cat_palette[i], TERRITORY_ALPHA), label=c)
              for i, c in enumerate(nh_cats) if (nh_codes == i).any()]
an_handles = [Line2D([], [], marker="o", ls="", mfc=anno_palette[i], mec="k", mew=0.3, ms=7,
                     label=an_cats[i]) for i in np.unique(ann_codes_core)]
leg1 = fig.legend(handles=nh_handles, title="Neighbourhoods", loc="upper left",
                  bbox_to_anchor=(0.905, 0.9), frameon=False)
fig.legend(handles=an_handles, title="Cell types", loc="upper left",
           bbox_to_anchor=(0.905, 0.6), frameon=False)
fig.add_artist(leg1)
savefig(fig, "Fig3B_neighbourhood_zooms.pdf")

del sub, rdata.obsp["spatial_connectivities"], rdata.obsp["spatial_distances"]

# %% Fig. 3C: cell-type composition per neighbourhood
celltype_groups = {
    "tumor": TUMOR_STATES,
    "yellow": ["Vascular EC", "Perivascular fibro.", "OPC/OC"],
    "neurons_astro": ["Neurons-exh.", "Neurons-inh.", "Astrocytes"],
    "myeloid": ["Myeloid-P2RY12+", "Myeloid-RNASE1+", "Myeloid-CD163+", "Myeloid-CLEC10A+",
                "Neutrophils"],
    "lymphoid": ["B cells", "T-CD4+", "T-CD8+", "T-cycling", "NK cells", "Plasma cells",
                 "Mast cells"],
}
present = set(obs_df[CT_COL])
celltype_groups = {g: [c for c in cts if c in present] for g, cts in celltype_groups.items()}
celltype_groups = {g: cts for g, cts in celltype_groups.items() if cts}

prop = pd.crosstab(obs_df[NICHE_COL], obs_df[CT_COL], normalize="index")
prop = prop.reindex(index=nhood_order).fillna(0)

# row labels used in this panel
C_ROW_LABELS = {"Vascular": "Vasculature", "TBI": "Tumor-brain interface"}


def get_ylim(group_name, niche_name):
    if group_name == "tumor":
        return 0.30 if niche_name in ["Vascular", "TBI"] else 0.70
    return {"yellow": 0.25, "neurons_astro": 0.04, "myeloid": 0.15, "lymphoid": 0.01}.get(
        group_name, 0.10)


fig = plt.figure(figsize=(10, 8))
gs = gridspec.GridSpec(len(nhood_order), len(celltype_groups),
                       width_ratios=[len(v) for v in celltype_groups.values()],
                       wspace=0, hspace=0)
for row_idx, niche in enumerate(nhood_order):
    for col_idx, (group_name, cts) in enumerate(celltype_groups.items()):
        ax = fig.add_subplot(gs[row_idx, col_idx])
        x = np.arange(len(cts))
        ax.bar(x, prop.reindex(columns=cts).fillna(0).loc[niche].values,
               color=[CT_COLORS[c] for c in cts], width=0.8, edgecolor="none")
        ymax = get_ylim(group_name, niche)
        ax.set_ylim(0, ymax)
        ax.set_yticks([0, ymax])
        ax.set_yticklabels(["0", f"{ymax:g}"], fontsize=7)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(axis="y", labelsize=7, length=2, pad=1)
        ax.tick_params(axis="x", length=0)
        if row_idx == len(nhood_order) - 1:
            ax.set_xticks(x)
            ax.set_xticklabels(cts, rotation=90, ha="center", va="top", fontsize=9)
        else:
            ax.set_xticks([])
        if col_idx == 0:
            ax.set_ylabel(C_ROW_LABELS.get(niche, niche), rotation=0, ha="right", va="center",
                          fontsize=10, labelpad=55)
fig.text(0.035, 0.53, "Neighbourhoods", rotation=90, va="center", ha="center", fontsize=14,
         fontweight="bold")
fig.text(0.55, 0.0, "Cell types", ha="center", fontsize=14, fontweight="bold")
fig.suptitle("Neighbourhood definition", fontsize=14, fontweight="bold")
plt.tight_layout(rect=[0.07, 0.05, 1, 1])
savefig(fig, "Fig3C_neighbourhood_composition.pdf")
prop.to_csv(OUT_DIR / "Fig3C_source_data.csv")


# %% Fig. 3D and ED Fig. 5B: neighbourhood enrichment (odds ratio)
def calculate_or(df, celltypes, niches, background_all_niches=False):
    """
    Odds ratio per neighbourhood x cell type, within one compartment
    (tumour states vs. other tumour cells; TME vs. other TME cells):

                       cell type   other cell types
        in nhood           a              b
        outside nhood      c              d

    Haldane-Anscombe correction (+0.5) only where a zero occurs.
    background_all_niches=True: cells of any other neighbourhood label in df also count
    as "outside" (Fig. 3D); False: only the neighbourhoods in `niches` (ED Fig. 5B).
    """
    tmp = df[df[CT_COL].isin(celltypes)]
    N = pd.crosstab(tmp[NICHE_COL], tmp[CT_COL]).reindex(columns=celltypes, fill_value=0)
    if not background_all_niches:
        N = N.reindex(index=niches, fill_value=0)
    a_all = N.values.astype(float)
    col_tot, grand = a_all.sum(0, keepdims=True), a_all.sum()
    N = N.reindex(index=niches, fill_value=0)
    a = N.values.astype(float)
    b = a.sum(1, keepdims=True) - a
    c = col_tot - a
    d = grand - a - b - c
    zero = (a == 0) | (b == 0) | (c == 0) | (d == 0)
    a, b, c, d = [np.where(zero, x + 0.5, x) for x in (a, b, c, d)]
    return pd.DataFrame(a * d / (b * c), index=niches, columns=celltypes)


def calculate_group_or(df, group, niches):
    """Odds ratio per neighbourhood for one group of TME cell types vs. the other TME cells."""
    is_group = df[CT_COL].isin(group)
    out = []
    for niche in niches:
        in_n = df[NICHE_COL] == niche
        a, b = np.sum(in_n & is_group), np.sum(in_n & ~is_group)
        c, d = np.sum(~in_n & is_group), np.sum(~in_n & ~is_group)
        if min(a, b, c, d) == 0:
            a, b, c, d = a + 0.5, b + 0.5, c + 0.5, d + 0.5
        out.append((a * d) / (b * c))
    return out


tumor_cols = [c for c in TUMOR_STATES if c in present]
tme_cols = [c for c in TME_COMPONENTS if c in present]
n_t = len(tumor_cols)
OR_NORM = LogNorm(vmin=OR_VMIN, vmax=OR_VMAX)


def draw_or_heatmap(ax, OR, title, show_xlabels=True, show_ylabels=True, xlabels=None,
                    lw=0.7, fontsize=10):
    mesh = ax.pcolormesh(np.arange(OR.shape[1] + 1), np.arange(OR.shape[0] + 1),
                         np.ma.masked_invalid(OR.values), cmap=OR_CMAP, norm=OR_NORM,
                         edgecolors="black", linewidth=lw)
    ax.invert_yaxis()
    ax.set_xticks(np.arange(OR.shape[1]) + 0.5)
    ax.set_yticks(np.arange(OR.shape[0]) + 0.5)
    ax.set_xticklabels((xlabels or list(OR.columns)) if show_xlabels else [], rotation=90,
                       fontsize=fontsize)
    ax.set_yticklabels(OR.index if show_ylabels else [], fontsize=fontsize)
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    for xpos, w_ in [(n_t, 3), (n_t + 6, 1.5), (n_t + 11, 1.5)]:   # block separators
        if xpos < OR.shape[1]:
            ax.axvline(xpos, color="black", linewidth=w_)
    ax.set_title(title, fontsize=11 if fontsize < 10 else 14, fontweight="bold")
    return mesh


def or_colorbar(fig, mesh, **kw):
    cbar = fig.colorbar(mesh, **kw)
    cbar.set_label("Odds ratio", fontsize=11)
    cbar.set_ticks([0.1, 1, 10])
    cbar.set_ticklabels([r"$10^{-1}$", r"$10^{0}$", r"$10^{1}$"])
    cbar.ax.tick_params(labelsize=9)
    return cbar


# Fig. 3D, left: per cell type
or_plot = pd.concat([calculate_or(obs_all, tumor_cols, nhood_order, background_all_niches=True),
                     calculate_or(obs_all, tme_cols, nhood_order, background_all_niches=True)],
                    axis=1)
fig, ax = plt.subplots(figsize=(11, 4.8))
mesh = draw_or_heatmap(ax, or_plot, "Neighborhood enrichment")
ax.set_ylabel("Neighbourhoods", fontsize=14, fontweight="bold")
ax.set_xlabel("Cell types", fontsize=14, fontweight="bold")
or_colorbar(fig, mesh, ax=ax, fraction=0.025, pad=0.04)
plt.tight_layout()
savefig(fig, "Fig3D_neighbourhood_enrichment.pdf")
or_plot.to_csv(OUT_DIR / "Fig3D_source_data_odds_ratio.csv")

# Fig. 3D, right: brain-resident vs. peripheral-immune TME cells
tme_df = obs_all[obs_all[CT_COL].isin(BRAIN_RESIDENT + PERIPHERAL_IMMUNE)]
or_comp = pd.DataFrame({
    "Brain resident": calculate_group_or(tme_df, [c for c in BRAIN_RESIDENT if c in present],
                                         nhood_order),
    "Peripheral immune": calculate_group_or(tme_df, [c for c in PERIPHERAL_IMMUNE if c in present],
                                            nhood_order),
}, index=nhood_order)
fig, ax = plt.subplots(figsize=(2.5, 5.5))
mesh = ax.pcolormesh(np.arange(3), np.arange(len(nhood_order) + 1), or_comp.values,
                     cmap=OR_CMAP, norm=OR_NORM, edgecolors="black", linewidth=0.8)
ax.invert_yaxis()
ax.set_xticks(np.arange(2) + 0.5)
ax.set_xticklabels(or_comp.columns, rotation=90, fontsize=11, fontweight="bold")
ax.set_yticks(np.arange(len(nhood_order)) + 0.5)
ax.set_yticklabels(or_comp.index, fontsize=11)
ax.tick_params(length=0)
for s in ax.spines.values():
    s.set_visible(False)
or_colorbar(fig, mesh, ax=ax, fraction=0.07, pad=0.08)
plt.tight_layout()
savefig(fig, "Fig3D_compartment_enrichment.pdf")
or_comp.to_csv(OUT_DIR / "Fig3D_source_data_compartments.csv")

# ED Fig. 5B: odds ratio per diagnostic group
# grey = neighbourhood with <30 cells or cell type with <20 cells in that diagnostic group
MIN_CELLS_NICHE, MIN_CELLS_CELLTYPE = 30, 20
or_by_dx = {}
for dx in dx_order:
    d = obs_df[obs_df[DX_COL] == dx]
    OR = pd.concat([calculate_or(d, tumor_cols, nhood_order),
                    calculate_or(d, tme_cols, nhood_order)], axis=1)
    OR.loc[d[NICHE_COL].value_counts().reindex(nhood_order, fill_value=0) < MIN_CELLS_NICHE, :] = np.nan
    OR.loc[:, d[CT_COL].value_counts().reindex(OR.columns, fill_value=0) < MIN_CELLS_CELLTYPE] = np.nan
    or_by_dx[dx] = OR

ed5b_order = ["Oligodendroglioma", "Astrocytoma low-grade", "Astrocytoma high-grade",
              "Glioblastoma", "Subependymoma", "Myxopapillary ependymoma", "Spinal ependymoma"]
ed5b_order = [d for d in ed5b_order if d in or_by_dx]
xlabels = [f"Tumour {c}" if c in TUMOR_STATES else c for c in or_plot.columns]
ncols = 2
nrows = int(np.ceil(len(ed5b_order) / ncols))
fig, axes = plt.subplots(nrows, ncols, figsize=(8 * ncols, 3.9 * nrows), squeeze=False,
                         layout="constrained")
for i, dx in enumerate(ed5b_order):
    mesh = draw_or_heatmap(axes.flat[i], or_by_dx[dx], dx, lw=0.5, fontsize=8,
                           show_xlabels=(i >= len(ed5b_order) - ncols),
                           show_ylabels=(i % ncols == 0), xlabels=xlabels)
    if i % ncols == 0:
        axes.flat[i].set_ylabel("Neighbourhoods", fontsize=10, fontweight="bold")
    if i >= len(ed5b_order) - ncols:
        axes.flat[i].set_xlabel("Cell types", fontsize=10, fontweight="bold")
for ax in axes.flat[len(ed5b_order):]:
    ax.axis("off")
cbar = or_colorbar(fig, mesh, ax=axes, fraction=0.015, pad=0.01, shrink=0.5)
cbar.set_label("Niche-level enrichment\n(Odds ratio)", fontsize=11)
savefig(fig, "EDFig5B_neighbourhood_enrichment_per_diagnosis.pdf")
pd.concat(or_by_dx, names=["diagnosis", "neighbourhood"]).to_csv(
    OUT_DIR / "EDFig5B_source_data_odds_ratio.csv")

# %% ED Fig. 5A: tumour-state neighbourhood specificity (entropy-based, per core)
MIN_CELLS_PER_STATE_PER_CORE = 20   # tumour state needs >= 20 cells in a core
MIN_PCT_IN_NHOOD = 5.0              # ignore a state in a neighbourhood where it is <5% of tumour cells

obs_tumor = obs_df[obs_df[CT_COL].isin(TUMOR_STATES)]
results = []
for core, obs_s in obs_tumor.groupby(CORE_COL):
    counts = pd.crosstab(obs_s[NICHE_COL], obs_s[CT_COL])
    if counts.shape[0] < 2:
        continue
    frac_in_nhood = counts.div(counts.sum(axis=1), axis=0) * 100
    counts_filt = counts.mask(frac_in_nhood < MIN_PCT_IN_NHOOD, 0)
    for state in counts_filt.columns:
        total, total_raw = counts_filt[state].sum(), counts[state].sum()
        if total_raw < MIN_CELLS_PER_STATE_PER_CORE:
            continue
        if total == 0:
            spec = np.nan
        else:
            p = counts_filt[state] / total
            p = p[p > 0]
            spec = 1 - entropy(p) / np.log(counts_filt.shape[0])   # 1 = in one neighbourhood only
        results.append({CORE_COL: core, "tumour_state": state, "entropy_specificity": spec,
                        "n_cells_raw": total_raw, "n_cells_after_filter": total,
                        "n_nhoods_total": counts.shape[0],
                        "n_nhoods_used": (counts_filt[state] > 0).sum()})
entropy_by_core = pd.DataFrame(results)
entropy_by_core.to_csv(OUT_DIR / "EDFig5A_source_data_per_core.csv", index=False)

order = (entropy_by_core.groupby("tumour_state")["entropy_specificity"].median()
         .sort_values().index)
fig, ax = plt.subplots(figsize=(5, 3.5))
sns.pointplot(data=entropy_by_core, y="tumour_state", x="entropy_specificity", order=order,
              estimator=np.median, errorbar=("pi", 50), linestyle="none",
              hue="tumour_state", palette=TUMOR_STATE_NHOOD_COLORS, legend=False, ax=ax)
ax.axvline(0.5, color="gray", linestyle="--", linewidth=0.8, alpha=0.6)
ax.set_xlim(0, 1.05)
ax.set_xlabel("Entropy-based nhood specificity\n(per core)")
ax.set_ylabel("Tumour states")
ax.set_title("Neighborhood specificity")
plt.tight_layout()
savefig(fig, "EDFig5A_neighbourhood_specificity.pdf", dpi=300)

# %% Fig. 3E: neighbourhood composition per core
comp_core = pd.crosstab(obs_df[CORE_COL], obs_df[NICHE_COL], normalize="index") * 100
comp_core = comp_core[[n for n in nhood_order if n in comp_core.columns]]

# cores ordered by hierarchical clustering of their composition
core_order = comp_core.index[leaves_list(linkage(pdist(comp_core.values, "euclidean"), "ward"))]
comp_core = comp_core.loc[core_order]
dx_by_core = (obs_df.groupby(CORE_COL)[DX_COL].agg(lambda x: x.value_counts().idxmax())
              .loc[core_order])

fig = plt.figure(figsize=(14, 4))
gs = fig.add_gridspec(2, 1, height_ratios=[12, 0.5], hspace=0.03)
ax = fig.add_subplot(gs[0])
ax_dx = fig.add_subplot(gs[1], sharex=ax)
x = np.arange(comp_core.shape[0])
bottom = np.zeros(comp_core.shape[0])
for nhood in comp_core.columns:
    ax.bar(x, comp_core[nhood].values, bottom=bottom, color=NHOOD_COLORS[nhood], width=1.0,
           edgecolor="none")
    bottom += comp_core[nhood].values
ax.set_ylim(0, 100)
ax.set_ylabel("Fraction (%)")
ax.set_xticks([])
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.set_title("Spatial neighborhood composition", fontsize=13, fontweight="bold")

for i, dx in enumerate(dx_by_core):
    ax_dx.bar(i, 1, color=DX_COLORS.get(dx, "#BDBDBD"), width=1.0, edgecolor="none")
ax_dx.set_yticks([])
ax_dx.set_xticks([])
ax_dx.set_xlim(-0.5, comp_core.shape[0] - 0.5)
ax_dx.set_ylabel("Diagnostic\ngroups", rotation=0, ha="right", va="center", fontsize=9)
for s in ax_dx.spines.values():
    s.set_visible(False)

leg1 = ax.legend(handles=[mpatches.Patch(color=NHOOD_COLORS[n], label=n) for n in comp_core.columns],
                 title="Neighbourhoods", bbox_to_anchor=(1.01, 1), loc="upper left",
                 frameon=False, fontsize=8, title_fontsize=9)
ax.add_artist(leg1)
ax.legend(handles=[mpatches.Patch(color=DX_COLORS[d], label=DX_SHORT.get(d, d))
                   for d in dx_order if d in set(dx_by_core)],
          title="Diagnostic groups", bbox_to_anchor=(1.01, 0.45), loc="upper left",
          frameon=False, fontsize=8, title_fontsize=9)
savefig(fig, "Fig3E_neighbourhood_composition_per_core.pdf", dpi=300)
comp_core.to_csv(OUT_DIR / "Fig3E_source_data.csv")

# %% Fig. 3G: neighbourhood co-localisation (Poisson rate ratio, all cores)
poisson_df = pd.read_csv(COLOC_DIR / "poisson_niche_pair_results.csv")
states = [n for n in NHOOD_NAMES if n in set(poisson_df["niche_A"]) | set(poisson_df["niche_B"])]
state_to_idx = {s: i for i, s in enumerate(states)}
labels = [NHOOD_NAMES[s] for s in states]

plot_df = poisson_df[np.isfinite(poisson_df["rate_ratio"]) & np.isfinite(poisson_df["n_cores"])].copy()
plot_df["x_idx"] = plot_df["niche_B"].map(state_to_idx)
plot_df["y_idx"] = plot_df["niche_A"].map(state_to_idx)
swap = plot_df["y_idx"] < plot_df["x_idx"]            # every pair in the lower triangle
plot_df.loc[swap, ["x_idx", "y_idx"]] = plot_df.loc[swap, ["y_idx", "x_idx"]].values

# dot size = number of contributing cores
min_dot, max_dot = 40, 600
nmin, nmax = plot_df["n_cores"].min(), plot_df["n_cores"].max()


def dot_size(n):
    return min_dot + (n - nmin) / (nmax - nmin) * (max_dot - min_dot) if nmax > nmin else 200


fig, ax = plt.subplots(figsize=(9, 8))
sc_ = ax.scatter(plot_df["x_idx"], plot_df["y_idx"], s=dot_size(plot_df["n_cores"]),
                 c=plot_df["rate_ratio"], cmap=RR_CMAP, norm=RR_NORM, edgecolors="black",
                 linewidths=0.6, zorder=3)
ax.set_xticks(range(len(states)))
ax.set_xticklabels(labels, rotation=58, ha="left", fontsize=10)
ax.set_yticks(range(len(states)))
ax.set_yticklabels(labels, fontsize=10)
ax.xaxis.tick_top()
ax.tick_params(top=True, labeltop=True, bottom=False, labelbottom=False, length=0)
ax.set_xlim(-0.5, len(states) - 0.5)
ax.set_ylim(len(states) - 0.5, -0.5)
ax.set_aspect("equal")
ax.set_xlabel("Neighbourhood", fontsize=12, fontweight="bold", labelpad=15)
ax.xaxis.set_label_position("top")
ax.set_ylabel("Neighbourhood", fontsize=12, fontweight="bold")
ax.set_title("Neighbourhood co-localisation", fontsize=15, fontweight="bold", pad=45)
ax.set_xticks(np.arange(-0.5, len(states), 1), minor=True)
ax.set_yticks(np.arange(-0.5, len(states), 1), minor=True)
ax.grid(which="minor", linewidth=0.7, alpha=0.25)
ax.tick_params(which="minor", length=0)
cbar = plt.colorbar(sc_, ax=ax, fraction=0.046, pad=0.04)
cbar.set_label("Rate ratio", fontsize=11)
cbar.set_ticks([0, 0.5, 1, 1.5, 2.0])
handles = [ax.scatter([], [], s=dot_size(v), facecolors="lightgray", edgecolors="black",
                      linewidths=0.6, label=str(v))
           for v in np.unique(np.round(np.linspace(nmin, nmax, 4)).astype(int))]
ax.legend(handles=handles, title="Cores contributing", bbox_to_anchor=(1.23, 0.35),
          loc="center left", frameon=False)
plt.tight_layout()
savefig(fig, "Fig3G_neighbourhood_colocalisation.pdf", dpi=300)

# %% ED Fig. 6A: co-localisation similarity between diagnostic groups (Jaccard)
# a pair is "supported" in a group if it is assessable in >= 2 cores;
# J = shared supported pairs / all supported pairs in either group
MIN_CORES_PER_PAIR = 2
poisson_dx = pd.read_csv(COLOC_DIR / "poisson_niche_pair_results_by_diagnosis.csv")
ed6_order = ["Oligodendroglioma", "Astrocytoma low-grade", "Astrocytoma high-grade",
             "Glioblastoma", "Subependymoma", "Spinal ependymoma", "Myxopapillary ependymoma"]
diagnoses = [d for d in ed6_order if d in set(poisson_dx["diagnosis"])]

supported = {}
for d in diagnoses:
    df = poisson_dx[(poisson_dx["diagnosis"] == d) & (poisson_dx["n_cores"] >= MIN_CORES_PER_PAIR)
                    & np.isfinite(poisson_dx["log_rate_ratio"])]
    supported[d] = {" ↔ ".join(sorted([str(a), str(b)])) for a, b in zip(df["niche_A"], df["niche_B"])}

jaccard = pd.DataFrame(np.nan, index=diagnoses, columns=diagnoses)
for d1 in diagnoses:
    for d2 in diagnoses:
        union = supported[d1] | supported[d2]
        if union:
            jaccard.loc[d1, d2] = len(supported[d1] & supported[d2]) / len(union)
jaccard.to_csv(OUT_DIR / "EDFig6A_source_data_jaccard.csv")

jaccard_cmap = LinearSegmentedColormap.from_list("jaccard_light", [
    (0.00, "#ffffff"), (0.30, "#fef3ee"), (0.50, "#fcd9c8"), (0.70, "#f5a582"),
    (0.85, "#de5a45"), (1.00, "#99000d")])
short = [DX_SHORT.get(d, d) for d in diagnoses]
fig, ax = plt.subplots(figsize=(8, 7))
sns.heatmap(jaccard.set_axis(short, axis=0).set_axis(short, axis=1),
            mask=np.triu(np.ones(jaccard.shape, dtype=bool), k=1),
            cmap=jaccard_cmap, norm=PowerNorm(gamma=0.8, vmin=0, vmax=1), square=True,
            annot=True, fmt=".2f", linewidths=1.2, linecolor="white",
            cbar_kws={"label": "Jaccard similarity", "ticks": [0, 0.2, 0.4, 0.6, 0.8, 1.0]},
            ax=ax)
ax.set_title("Co-localisation similarity", fontsize=15, fontweight="bold")
ax.set_xlabel("Diagnostic groups", fontsize=12, fontweight="bold")
ax.set_ylabel("Diagnostic groups", fontsize=12, fontweight="bold")
plt.xticks(rotation=90)
plt.yticks(rotation=0)
plt.tight_layout()
savefig(fig, "EDFig6A_colocalisation_similarity.pdf", dpi=300)

# %% ED Fig. 6B: co-localisation per diagnostic group
ED6_NAMES = {"MES1": "Gliosis", "MES2": "Hypoxia", "OPC-like": "OPC-like II",
             "Cortical glia": "OPC-like I", "Stressed": "Stress response",
             "TBI": "TBI", "Spinal-glia": "EPN"}

plot_df = poisson_dx[poisson_dx["diagnosis"].isin(diagnoses)].copy()
first_second = [(a, b) if state_to_idx[a] <= state_to_idx[b] else (b, a)
                for a, b in zip(plot_df["niche_A"].astype(str), plot_df["niche_B"].astype(str))]
plot_df["pair"] = [f"{a} ↔ {b}" for a, b in first_second]

pair_order, pair_labels = [], []
for i in range(len(states)):
    for j in range(i + 1, len(states)):
        pair = f"{states[i]} ↔ {states[j]}"
        if pair in set(plot_df["pair"]):
            pair_order.append(pair)
            pair_labels.append(f"{ED6_NAMES.get(states[i], states[i])} ↔ "
                               f"{ED6_NAMES.get(states[j], states[j])}")

plot_df["x_idx"] = pd.Categorical(plot_df["diagnosis"], categories=diagnoses).codes
plot_df["y_idx"] = pd.Categorical(plot_df["pair"], categories=pair_order).codes
plot_df = plot_df[(plot_df["x_idx"] >= 0) & (plot_df["y_idx"] >= 0)]

# dot size: 1-100 contributing cores (capped at 100)
SIZE_MIN, SIZE_MAX, min_dot, max_dot = 1, 100, 20, 360


def dot_size_ed6(n):
    return min_dot + (np.clip(n, SIZE_MIN, SIZE_MAX) - SIZE_MIN) / (SIZE_MAX - SIZE_MIN) * (max_dot - min_dot)


fig, ax = plt.subplots(figsize=(11, max(10, len(pair_order) * 0.28)))
sc_ = ax.scatter(plot_df["x_idx"], plot_df["y_idx"], s=dot_size_ed6(plot_df["n_cores"]),
                 c=plot_df["rate_ratio"], cmap=RR_CMAP, norm=RR_NORM, edgecolors="black",
                 linewidths=0.5, zorder=3)
ax.set_xticks(range(len(diagnoses)))
ax.set_xticklabels(diagnoses, rotation=55, ha="right", fontsize=10)
ax.set_yticks(range(len(pair_labels)))
ax.set_yticklabels(pair_labels, fontsize=9)
ax.set_xlim(-0.5, len(diagnoses) - 0.5)
ax.set_ylim(len(pair_order) - 0.5, -0.5)
ax.set_xlabel("Diagnostic groups", fontsize=13, fontweight="bold")
ax.set_ylabel("Neighbourhood pair", fontsize=13, fontweight="bold")
ax.set_title("Spatial co-localisation", fontsize=15, fontweight="bold", pad=15)
ax.set_xticks(np.arange(-0.5, len(diagnoses), 1), minor=True)
ax.set_yticks(np.arange(-0.5, len(pair_order), 1), minor=True)
ax.grid(which="minor", linewidth=0.6, alpha=0.20)
ax.tick_params(which="minor", length=0)
cbar = plt.colorbar(sc_, ax=ax, fraction=0.025, pad=0.03)
cbar.set_label("Poisson rate ratio\n1 = expected, >1 = enriched, <1 = depleted", fontsize=11)
cbar.set_ticks([0, 0.5, 1, 1.5, 2])
handles = [ax.scatter([], [], s=dot_size_ed6(v), facecolors="lightgray", edgecolors="black",
                      linewidths=0.5, label=str(v))
           for v in [1, 5, 10, 20, 40, 60, 80, 100]]
ax.legend(handles=handles, title="Cores contributing", bbox_to_anchor=(1.17, 0.27),
          loc="center left", frameon=False, labelspacing=1.0)
plt.tight_layout()
savefig(fig, "EDFig6B_colocalisation_per_diagnosis.pdf", dpi=300)
