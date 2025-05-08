# 参数设置
raw_dir = 'GSE67835_RAW'
ct_file = 'cell_types.csv'
n_features = 500
seed = 42

from self_projection import self_projection
from rf_svm import projection_rf_svm

self_projection(raw_dir, ct_file, n_features, 'linear')

print("————————————————****究极分割符****————————————————")

self_projection(raw_dir, ct_file, n_features, 'hvg')

print("————————————————****究极分割符****————————————————")

self_projection(raw_dir, ct_file, n_features, 'random', seed)

print("————————————————****究极分割符****————————————————")

projection_rf_svm(raw_dir, ct_file, n_features, feature_selection='linear', seed=42, projection='rf')

print("————————————————****究极分割符****————————————————")

projection_rf_svm(raw_dir, ct_file, n_features, feature_selection='linear', seed=42, projection='svm')
