import os
import gzip
import pandas as pd
import numpy as np
import scanpy as sc
from scipy.stats import pearsonr
from scipy import sparse
from anndata import AnnData
from scipy.optimize import curve_fit

def load_counts(data_dir: str) -> pd.DataFrame:
    """加载所有单细胞计数，返回 genes×cells 的 DataFrame"""
    cell_series = []
    for fn in sorted(os.listdir(data_dir)):
        if not fn.endswith('.csv.gz'):
            continue
        cell_id = fn.replace('.csv.gz', '')
        path = os.path.join(data_dir, fn)
        with gzip.open(path, 'rt') as f:
            df = pd.read_csv(f, sep=None, engine='python')
        genes = df.iloc[:, 0]
        counts = df.iloc[:, 1]
        counts.index = genes
        counts.name = cell_id
        cell_series.append(counts)

    # 合并，缺失值填 0
    counts_matrix = pd.concat(cell_series, axis=1).fillna(0).astype(int)
    print(f"Counts matrix shape: {counts_matrix.shape} (genes × cells)")
    return counts_matrix

def load_cell_types(ct_file: str, cells: pd.Index) -> pd.Series:
    """读取 cell_types.csv 并对齐到给定的 cells 顺序，返回 cell_id→cell_type 的 Series"""
    # 假设文件无 header，或者有 header 列名为 cell_id,cell_type
    try:
        ct_df = pd.read_csv(ct_file, index_col=0)
        if ct_df.shape[1] != 1:
            ct_df.columns = ['cell_type']
    except:
        ct_df = pd.read_csv(ct_file, header=None, names=['cell_id','cell_type'], index_col=0)
    # 对齐顺序
    ct = ct_df['cell_type'].reindex(cells)
    if ct.isna().any():
        missing = ct[ct.isna()].index.tolist()
        raise ValueError(f"以下细胞在标签文件中不存在：{missing}")
    return ct

def select_features_LinearDrop(X: np.ndarray, var_names: pd.Index, n_features: int=500):
    """基于 log(dropout) ~ log(mean) 残差，选取 top n_features 基因"""
    # X: (n_cells, n_genes)
    mean_expr = X.mean(axis=0)
    dropout_rate = np.sum(X == 0, axis=0) / X.shape[0]

    log_mean = np.log2(mean_expr + 1e-6)
    log_dropout = np.log2(dropout_rate + 1e-6)

    # 线性拟合
    coeffs = np.polyfit(log_mean, log_dropout, deg=1)
    fitted = np.polyval(coeffs, log_mean)
    residuals = log_dropout - fitted

    # top genes
    idx = np.argsort(-residuals)[:n_features]
    top_genes = var_names[idx]
    return top_genes

def select_features_linear(X, var_names, n_features=500):
    """
    输入：
      X          : np.ndarray 或 sparse matrix, shape (n_cells, n_genes)
      var_names  : pd.Index，长度 = n_genes
    返回：
      top_genes  : pd.Index，长度 = n_features
    """
    # 兼容稀疏
    if sparse.issparse(X):
        X = X.toarray()
    mean_expr    = X.mean(axis=0)
    dropout_rate = (X == 0).sum(axis=0) / X.shape[0]
    log_mean     = np.log2(mean_expr    + 1e-6)
    log_dropout  = np.log2(dropout_rate + 1e-6)
    # 线性拟合
    a, b         = np.polyfit(log_mean, log_dropout, deg=1)
    fitted       = a * log_mean + b
    residuals    = log_dropout - fitted
    idx          = np.argsort(-residuals)[:n_features]
    top_genes    = var_names[idx]
    print(f"[Linear] selected {n_features} genes")
    return top_genes

def select_features_hvg(
    X,
    var_names,
    n_features: int = 500,
    flavor: str = 'seurat'
):
    # 构造临时 AnnData
    if sparse.issparse(X):
        adata_tmp = AnnData(X=X.tocsr())
    else:
        adata_tmp = AnnData(X=X)
    adata_tmp.var_names = var_names

    # 调用 Scanpy 高变基因筛选
    sc.pp.highly_variable_genes(
        adata_tmp,
        flavor=flavor,
        n_top_genes=n_features,
        subset=False,
        inplace=True
    )

    mask = adata_tmp.var['highly_variable'].values
    top_genes = adata_tmp.var_names[mask]
    print(f"[HVG] flavor={flavor}, selected {len(top_genes)} genes")
    return top_genes

def select_features_random(var_names, n_features=500, seed=None):
    """
    直接随机挑 var_names 中的 n_features 个。
    """
    rng       = np.random.default_rng(seed)
    idx       = rng.choice(len(var_names), size=n_features, replace=False)
    top_genes = var_names[idx]
    print(f"[Random] seed={seed}, selected {n_features} genes")
    return top_genes


def build_centroids(adata: sc.AnnData, use_genes: pd.Index) -> pd.DataFrame:
    """对每个 cell_type 计算所选基因的平均表达，返回 types×genes 的 DataFrame"""
    df = pd.DataFrame(
        adata.X[:, [adata.var_names.get_loc(g) for g in use_genes]],
        index=adata.obs_names,
        columns=use_genes
    )
    centroids = df.join(adata.obs['cell_type']).groupby('cell_type').mean()
    return centroids

def project_cells(adata: sc.AnnData, centroids: pd.DataFrame, use_genes: pd.Index):
    """对每个细胞计算其与各 centroid 的 Pearson 相关，打标签"""
    X = adata[:, use_genes].X
    if sparse.issparse(X):
        X = X.toarray()
    data_feat = pd.DataFrame(X, index=adata.obs_names, columns=use_genes)

    preds = {}
    for cell in data_feat.index:
        best_type, best_r = 'unassigned', -np.inf
        v = data_feat.loc[cell].values
        for ctype, centroid in centroids.iterrows():
            r, _ = pearsonr(v, centroid.values)
            if np.isnan(r):
                continue
            if r > best_r:
                best_type, best_r = ctype, r
        preds[cell] = best_type
    return pd.Series(preds, name='scmap_selfproj')
