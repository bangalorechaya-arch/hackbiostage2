# hackbiostage2

Cell 1 — imports and plotting settings
Python
# Cell 1
import scanpy as sc
import anndata as ad
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import sparse

sc.settings.verbosity = 3
sc.set_figure_params(dpi=100, figsize=(5,5))
Cell 2 — read the 10x H5 (or skip if you already have fresh_blood_adata in memory)
Python
# Cell 2
# If you already loaded fresh_blood_adata, skip this cell.
# Replace the filename below if different.
fn = '10k_5p_Human_diseased_PBMC_ALL_Fresh_count_filtered_feature_bc_matrix.h5'
fresh_blood_adata = sc.read_10x_h5(fn)
fresh_blood_adata.var_names_make_unique()
print(fresh_blood_adata)
Cell 3 — quick shape/info and first checks
Python
# Cell 3
print("Shape:", fresh_blood_adata.shape)
print("Obs keys:", list(fresh_blood_adata.obs_keys()))
print("Var keys:", list(fresh_blood_adata.var_keys()))
fresh_blood_adata.var.head()
Cell 4 — compute QC metrics (n_genes, total_counts, pct mitochondrial)
Python
# Cell 4
# robust handling for sparse/dense X
X = fresh_blood_adata.X
if sparse.issparse(X):
    total_counts = np.array(X.sum(axis=1)).ravel()
    n_genes_by_counts = np.array((X > 0).sum(axis=1)).ravel()
else:
    total_counts = X.sum(axis=1)
    n_genes_by_counts = (X > 0).sum(axis=1)

fresh_blood_adata.obs['total_counts'] = total_counts
fresh_blood_adata.obs['n_genes_by_counts'] = n_genes_by_counts

# mitochondrial genes: assume gene symbols begin with MT- (human)
mt_mask = fresh_blood_adata.var_names.str.upper().str.startswith('MT-')
if mt_mask.sum() == 0:
    print("No MT- genes found in var_names. Inspect var to find mitochondrial genes if needed.")
else:
    if sparse.issparse(X):
        mt_counts = np.array(fresh_blood_adata[:, mt_mask].X.sum(axis=1)).ravel()
    else:
        mt_counts = fresh_blood_adata[:, mt_mask].X.sum(axis=1)
    fresh_blood_adata.obs['pct_counts_mt'] = 100 * mt_counts / fresh_blood_adata.obs['total_counts']

# show QC violin
sc.pl.violin(fresh_blood_adata, ['n_genes_by_counts','total_counts','pct_counts_mt'],
             jitter=0.4, multi_panel=True)
Cell 5 — filtering cells and genes (adjust thresholds after inspecting plots)
Python
# Cell 5
# Typical starting thresholds; change to fit your dataset
min_genes = 200
max_pct_mt = 10.0
min_counts = 500

sc.pp.filter_cells(fresh_blood_adata, min_genes=min_genes)        # drops cells with few genes
fresh_blood_adata = fresh_blood_adata[fresh_blood_adata.obs['total_counts'] >= min_counts].copy()
if 'pct_counts_mt' in fresh_blood_adata.obs:
    fresh_blood_adata = fresh_blood_adata[fresh_blood_adata.obs['pct_counts_mt'] < max_pct_mt].copy()

# filter genes seen in few cells
sc.pp.filter_genes(fresh_blood_adata, min_cells=3)

print("After filtering:", fresh_blood_adata.shape)
Cell 6 — normalization and log transform
Python
# Cell 6
sc.pp.normalize_total(fresh_blood_adata, target_sum=1e4)
sc.pp.log1p(fresh_blood_adata)
# store raw counts (useful for later DE / visualization)
fresh_blood_adata.raw = fresh_blood_adata
Cell 7 — find highly variable genes and subset
Python
# Cell 7
sc.pp.highly_variable_genes(fresh_blood_adata,
                            n_top_genes=3000,
                            flavor='seurat_v3',
                            subset=True,
                            inplace=True)
print("HVGs kept:", fresh_blood_adata.n_vars)
Cell 8 — regress out technical effects and scale
Python
# Cell 8
# Regress out total_counts and percent mitochondrial (if present)
vars_to_regress = []
if 'total_counts' in fresh_blood_adata.obs:
    vars_to_regress.append('total_counts')
if 'pct_counts_mt' in fresh_blood_adata.obs:
    vars_to_regress.append('pct_counts_mt')
if vars_to_regress:
    sc.pp.regress_out(fresh_blood_adata, vars_to_regress)
sc.pp.scale(fresh_blood_adata, max_value=10)
Cell 9 — PCA and variance explained
Python
# Cell 9
sc.tl.pca(fresh_blood_adata, svd_solver='arpack')
sc.pl.pca_variance_ratio(fresh_blood_adata, log=True)
Cell 10 — neighborhood graph, UMAP, and clustering
Python
# Cell 10
n_pcs = 30     # choose based on PCA elbow
sc.pp.neighbors(fresh_blood_adata, n_pcs=n_pcs)
sc.tl.umap(fresh_blood_adata)
sc.tl.leiden(fresh_blood_adata, resolution=0.5, key_added='leiden')
sc.pl.umap(fresh_blood_adata, color=['leiden'], legend_loc='on data')
Cell 11 — find cluster marker genes (DE)
Python
# Cell 11
sc.tl.rank_genes_groups(fresh_blood_adata, groupby='leiden', method='t-test')
sc.pl.rank_genes_groups(fresh_blood_adata, n_genes=20, sharey=False)
# get results as DataFrame
ranked = sc.get.rank_genes_groups_df(fresh_blood_adata, group=None)
ranked.head()
Cell 12 — optional: run CellTypist for automated annotation (best-effort; may need a model name change)
Python
# Cell 12
try:
    import celltypist
    from celltypist import models, annotate
    # pick a model appropriate for your data; this is an example model name
    model_name = 'Immune_All_Low-Data'
    model = models.get_model(model_name)
    # CellTypist expects cells x genes DataFrame with gene symbols
    expr_df = fresh_blood_adata.to_df()
    pred = annotate(expr_df, model=model, majority_voting=True)
    # predicted_labels is a pandas Series (or similar)
    fresh_blood_adata.obs['celltypist_label'] = pred.predicted_labels.values
    sc.pl.umap(fresh_blood_adata, color=['celltypist_label'], legend_loc='on data')
except Exception as e:
    print("CellTypist annotation failed (you may need to install models or pick a different model):", e)
Cell 13 — optional: pathway / activity analysis using decoupler (minimal example)
Python
# Cell 13
try:
    import decoupler as dc
    # Basic example: compute a gene-set activity with wmean (requires a gene set resource)
    # Replace this with loading a gene set collection you want (GMT or built-in)
    # Example shows framework only.
    expr = fresh_blood_adata.to_df().T  # genes x cells for decoupler
    print("Decoupler ready; refer to decoupler docs for running specific methods (mlm, wmean, etc).")
except Exception as e:
    print("Decoupler not available or failed:", e)
Cell 14 — save processed AnnData
Python
# Cell 14
fresh_blood_adata.write('fresh_blood_adata_processed.h5ad')
print("Saved to fresh_blood_adata_processed.h5ad")
