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
- 

