import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, cohen_kappa_score, classification_report
from anndata import AnnData
from scipy import sparse
import scanpy as sc
from sklearn.model_selection import train_test_split
import time

from utils import select_features_hvg, select_features_random, select_features_linear


def projection_rf_svm(raw_dir, ct_file, n_features, feature_selection='linear', seed=42, projection='rf'):

    adata = sc.read_h5ad('self_projection_results.h5ad')

    if feature_selection == 'linear':
        top_genes = select_features_linear(adata.X, adata.var_names, n_features=n_features)

    elif feature_selection == 'hvg':
        top_genes = select_features_hvg(adata.X, adata.var_names, n_features=n_features)

    elif feature_selection == 'random':
        top_genes = select_features_random(adata.var_names, n_features=n_features, seed=seed)

    else:
        print("Invalid feature selection method")
        return 0

    # 提取训练矩阵
    X = adata[:, top_genes].X
    if sparse.issparse(X):
        X = X.toarray()
    y = adata.obs['cell_type'].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, train_size=0.7, stratify=y, random_state=seed
    )

    # 训练并预测
    if projection == 'rf':
        t0 = time.perf_counter()

        model = RandomForestClassifier(n_estimators=50, random_state=seed, n_jobs=-1)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        t1 = time.perf_counter()
        elapsed = t1 - t0
        print(f"Feature selection → classification elapsed time: {elapsed:.2f} seconds")

    elif projection == 'svm':
        t0 = time.perf_counter()

        model = SVC(kernel='linear', C=1.0, random_state=seed)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        t1 = time.perf_counter()
        elapsed = t1 - t0
        print(f"Feature selection → classification elapsed time: {elapsed:.2f} seconds")

    else:
        raise ValueError(f"Invalid projection: {projection}")

    # 计算指标
    acc = accuracy_score(y_test, y_pred)
    kappa = cohen_kappa_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, zero_division=0)

    print(f"[{projection.upper():3s}] feature={feature_selection:6s} | "
          f"Accuracy: {acc:.2%}, κ: {kappa:.3f}")
    print(report)
    return {
        'accuracy': acc,
        'kappa': kappa,
        'report': report
    }
