import joblib
import numpy as np
import os

model_path = 'models/pipeline.pkl'
if not os.path.exists(model_path):
    print(f"Model not found: {model_path}")
    exit(1)

data = joblib.load(model_path)

print("=" * 45)
print("       NOI DUNG FILE pipeline.pkl")
print("=" * 45)

pca = data['pca']
print(f"\n--- PCA ---")
print(f"  Input dims     : 784")
print(f"  Output dims    : {pca.n_components_}")
print(f"  Whitening      : {pca.whiten}")
print(f"  Variance kept  : {pca.explained_variance_ratio_.sum()*100:.1f}%")
print(f"  Top 5 var ratio: {pca.explained_variance_ratio_[:5].round(4)}")

clf = data['clf']
print(f"\n--- RPCA KNN Classifier ---")
print(f"  k (neighbors)  : {clf.k}")
print(f"  distance metric: {clf.metric}")
print(f"  n_trees        : {clf.forest.n_trees}")
print(f"  leaf_size      : {clf.forest.leaf_size}")
print(f"  Train shape    : {clf.X_train.shape}")
print(f"  n_classes      : {len(np.unique(clf.y_train))}")

if 'labels' in data:
    print(f"  Labels         : {data['labels']}")

size = os.path.getsize(model_path)
print(f"\n--- File info ---")
print(f"  File size      : {size / 1024 / 1024:.1f} MB")
print("=" * 45)
