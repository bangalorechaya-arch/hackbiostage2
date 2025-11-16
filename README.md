# hackbiostage2


- filter_cells → filter cells by number of genes - filter_genes → filter genes by number of cells

- sc.pp.filter_genes(adata, min_cells=3);sc.pp.filter_cells(adata, min_genes=200)

- UAMP in single cell analysis- visualize high dimensional data

- PCA vs UAMP
- UAMP captures non linear relations while PCA is for linear

- Leidein clustering in scanpy workflow
- to detect groups of similar cells based on shared gene expression

- 1. What cell types did you identify?
List each annotated cluster. No need for a novel classification system. Just show you understand what you’re looking at.

•	Inspect ranked genes per cluster
print(adata.uns['rank_genes_groups']['names'])
2.	Automated annotation options
•	CellTypist (Python, fast, good human blood references): pip install celltypist import celltypist model = celltypist.models.download_model('Immune_All_Low.pkl') # example predictions = celltypist.annotate(adata, model=model)
•	SingleR (R, reference-based): Use SingleR with HumanPrimaryCellAtlasData or BlueprintEncodeData.
•	scmap (R) or Azimuth (Seurat-based / requires reference)
3.	Common bone-marrow / blood marker genes to map clusters to cell types
•	Hematopoietic stem / progenitor: CD34, KIT, GATA2
•	Early erythroid / erythrocytes: HBB, HBA1, GYPA
•	Megakaryocytes / platelets: PPBP, PF4, ITGA2B
•	Monocytes / macrophages: LYZ, CD14, FCGR3A (CD16)
•	Neutrophils/granulocytes: S100A8, S100A9, MPO, LTF
•	T cells: CD3D, CD3E; helper T: CD4; cytotoxic T: CD8A
•	B cells / plasma cells: CD19, MS4A1(CD20), IGHM, JCHAIN, MZB1
•	NK cells: NKG7, GNLY, NCAM1 (CD56)
•	Dendritic cells: CLEC9A (cDC1), LILRA4 (pDC)
•	Endothelial / stromal: PECAM1 (CD31), VWF, PDGFRB (pericytes/MSC markers vary)

4.	to run an annotation
•	notebook for Scanpy + CellTypist 
•	S100A8 and S100A9: abundant Ca2+-binding proteins in neutrophils (also found in activated monocytes).
•	MPO (myeloperoxidase): a neutrophil azurophilic granule enzyme — highly specific for neutrophil lineage in myeloid cells.
•	LTF (lactotransferrin): a secondary granule protein of neutrophils.
Taken together (S100A8/9 + MPO + LTF) is a neutrophil signature; if S100A8/9 are high but MPO/LTF are absent, consider activated monocytes or immature myeloid cells.

- 
- 2. Explain the biological role of each cell type
For every annotated label, give a short explanation of what that cell type actually does in bone marrow or peripheral immunity.

Examples:

Neutrophils: short-lived phagocytes, first responders to infection

Plasma cells: antibody factories derived from B cells

Platelets: fragments of megakaryocytes, support clotting

Keep each explanation tight. You are not writing a review paper.


3  3. Is the tissue source really bone marrow? Justify your answer
Your job is to reason your way toward (or away from) that conclusion using:
•	expected vs. missing lineage populations
•	typical frequency distributions
•	presence or absence of progenitors
If you claim bone marrow, explain the flaws in your logic. Otherwise, justify it biologically. Hand-waving is a fail.

Runnable code Scanpy to run on bone_marrow.h5ad, 
•	QC & filtering: ensure you didn’t over-filter (min_genes/min_counts).
•	Clustering + UMAP to find transcriptional groups.
•	Inspect canonical markers per cluster (dotplot/heatmap / ranked genes).
•	Compute signature scores for lineages & for progenitors (score_genes).
•	Get per-sample / per-cluster frequency tables and stacked-bar plots.
•	Flag clusters with mixed markers (doublets/ambient RNA).
•	If looking for progenitors: check CD34/KIT/PROM1 and progenitor signature; optionally run SingleR/CellTypist/Azimuth for reference-based labels.
2.	Marker panels (use these to call lineages)
•	Hematopoietic stem / progenitors (HSPC): CD34, KIT, PROM1 (CD133), GATA2, MEIS1, HLF, MLLT3
•	Early myeloid / granulocyte progenitors: ELANE, PRTN3, MPO, AZU1
•	Neutrophils / granulocytes: S100A8, S100A9, MPO, LTF, FCGR3B (CD16b)
•	Notes: interpret combos — e.g., S100A8/9 + MPO + LTF = strong neutrophil signature. S100A8/9 without MPO may indicate monocytes or immature myeloid.
3.	Typical frequency ranges in adult bone marrow (approximate; depends on sampling, patient, enrichment)
•	HSPCs (CD34+): rare — often <1–5% of total BM cells in un-enriched BM (can be enriched in CD34+ preps).
•	Granulocytic lineage (neutrophils + precursors): often the largest fraction — tens of % to >50% depending on stage and sample prep.
•	Erythroid precursors: variable — from low to ~10–30% (in erythroid-active marrow higher).
•	Monocytes: ~5–15% (rough estimate).
•	Lymphocytes (T + B + NK): often 5–30% combined, depends on marrow site and age.
•	Megakaryocytes: very rare (<1%); platelets captured as fragments can appear as low-UMI cells. Caveat: single cell prep method (density gradient, Ficoll, RBC lysis, FACS enrichment) dramatically shifts these numbers.


4.	How to quantify frequencies (Scanpy example)
•	compute scores, label cells, then proportion per cluster/sample: import scanpy as sc adata = sc.read_h5ad("bone_marrow.h5ad")


•	basic preprocessing assumed done (norm/log/HVG/PCA)
neutro_genes = ["S100A8","S100A9","MPO","LTF"] hspc_genes = ["CD34","KIT","PROM1","GATA2","MEIS1"] sc.tl.score_genes(adata, neutro_genes, score_name="neutro_score") sc.tl.score_genes(adata, hspc_genes, score_name="hspc_score")
classify by simple rules
adata.obs['predicted'] = 'unassigned' adata.obs.loc[(adata.obs['neutro_score']>0.5),'predicted'] = 'neutrophil' adata.obs.loc[(adata.obs['hspc_score']>0.5),'predicted'] = 'HSPC'
table of proportions
freq = adata.obs.groupby(['sample','predicted']).size().unstack(fill_value=0) freq_prop = freq.div(freq.sum(axis=1), axis=0) print(freq_prop)
visual checks
sc.pl.umap(adata, color=['predicted','neutro_score','hspc_score'], wspace=0.4)
Adjust score thresholds based on distributions (use medians / percentiles).
5.	Seurat (R) quick steps
•	After Seurat object (seu): neutro_genes <- c("S100A8","S100A9","MPO","LTF") hspc_genes <- c("CD34","KIT","PROM1","GATA2","MEIS1") seu <- NormalizeData(seu) seu <- FindVariableFeatures(seu) seu <- ScaleData(seu) seu <- RunPCA(seu) seu <- RunUMAP(seu, dims=1:20) seu <- AddModuleScore(seu, features = list(neutro_genes), name = "neutro_score") seu <- AddModuleScore(seu, features = list(hspc_genes), name = "hspc_score") VlnPlot(seu, c("neutro_score1","hspc_score1"), group.by="seurat_clusters")
frequency table
table(seu$sample, Idents(seu)) -> convert to proportions per sample.
6.	Identify presence/absence of progenitors
•	Direct markers: count CD34+ cells (percent expressing at given threshold e.g. >0 counts or >1 normalized).
•	Use a multi-gene HSPC score (recommended) rather than single-gene calls.
•	Confirm progenitor clusters express low lineage-specific terminal markers (lack MPO/HBB/MS4A1 etc) and express transcription factors (GATA2, MEIS1).
•	Use reference annotation (SingleR / CellTypist / Azimuth) tuned for hematopoiesis to flag HSC/MPP/CLP/GMP/ErythroidProgenitor.
•	Pseudotime / trajectory (Monocle3, scVelo, Palantir) can show progression from HSPC -> committed progenitors -> mature lineages.
7.	Dealing with missing expected populations
•	Check sample prep: was there CD34 enrichment or depletion? RBC lysis may remove erythroid precursors if done incorrectly.
•	Check filtering: aggressive mitochondrial or low-UMI filtering can remove low-RNA cell types (e.g., mature granulocytes often have low RNA).
•	Check clustering resolution: small populations may be merged—try lower PCA cutoffs or higher clustering resolution.
•	Check doublet removal: some pipelines misclassify small clusters as doublets and remove them.
•	Check gene detection sensitivity: if key markers are not detected, consider whether reads/UMIs are low; consider imputation cautiously.
•	If an expected cell type is entirely absent: verify experimental metadata (sorted/enriched sample), or use bulk marker expression across all cells to ensure it’s not present at very low frequency.
8.	Reporting guidance (what to include)
•	Per-sample and overall frequency table (counts + proportions).
•	UMAP with lineage marker overlays.
•	Dotplot/heatmap of canonical markers per cluster.
•	A short rule list used to call cell types (markers & thresholds).
•	Number/percent of cells classified as progenitors (and gene expression summary for that group).
•	Notes on caveats: enrichment, sample handling, QC filters, doublet removal, ambient RNA.
9.	Quick interpretation rules (practical)
•	Strong MPO + LTF + S100A8/9 -> confident neutrophils/granulocytes.
•	CD34-high cluster + low lineage markers -> HSPC / progenitor.
•	If CD34+ cells are <0.1% in an un-enriched BM, that may be expected; if you expected many, confirm enrichment protocol.
•	Very low megakaryocyte calls: expected—rare cells need larger input or targeted enrichment


4. Based on the relative abundance of cell types, is the patient healthy or infected?
Use the cluster proportions to make a call.
Your job: defend your conclusion using deviations in:
•	neutrophils
•	monocytes
•	NK cell activation states
•	lymphocyte depletion or expansion
Do not just guess. Interpret the landscape like a scientist.

The data set I analysed is diseased

