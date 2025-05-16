import scanpy as sc
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import accuracy_score
from utils import *
import time

def self_projection(raw_dir, ct_file, n_features, feature_selection='linear', seed=42):
    # 加载数据
    # counts = load_counts(raw_dir)            # genes × cells
    # ct = load_cell_types(ct_file, counts.columns)

    # 构建 AnnData
    # adata = sc.AnnData(X=counts.T)            # 转置后 cells × genes
    # adata.obs['cell_type'] = ct
    # adata.var_names = counts.index
    # adata.write('self_projection_results.h5ad')

    adata = sc.read_h5ad('self_projection_results.h5ad')

    # 预处理：归一化 + 对数化
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)

    # 特征选择
    if feature_selection == 'linear':
        top_genes = select_features_linear(adata.X, adata.var_names, n_features=n_features)

    elif feature_selection == 'hvg':
        top_genes = select_features_hvg(adata.X, adata.var_names, n_features=n_features)

    elif feature_selection == 'random':
        top_genes = select_features_random(adata.var_names, n_features=n_features, seed=seed)

    else:
        print("Invalid feature selection method")
        return 0

    adata.var['use_for_scmap'] = adata.var_names.isin(top_genes)

    # 构建 centroids
    centroids = build_centroids(adata, top_genes)

    # 自投影预测
    t0 = time.perf_counter()
    preds = project_cells(adata, centroids, top_genes)
    adata.obs['scmap_selfproj'] = preds

    t1 = time.perf_counter()
    elapsed = t1 - t0
    print(f"Feature selection → classification elapsed time: {elapsed:.2f} seconds")

    y_true = adata.obs['cell_type']
    y_pred = adata.obs['scmap_selfproj']
    y_pred_filled = y_pred.fillna('unassigned')

    # 准确率
    acc = accuracy_score(y_true, y_pred)
    print(f"Accuracy (including unassigned): {acc:.2%}")

    # Cohen's κ
    kappa = cohen_kappa_score(y_true, y_pred)
    print(f"Cohen’s kappa: {kappa:.3f}")

    # 保存结果
    # adata.obs[['cell_type','scmap_selfproj']].to_csv('self_projection_results.csv')
    # print("结果已保存到 self_projection_results.csv")
