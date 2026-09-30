import os
import numpy as np
import joblib
from sklearn.metrics import accuracy_score, classification_report
from tqdm import tqdm
import time

from src.preprocessing import straighten_image, build_pca_pipeline
from src.classifier    import RPCAKNNClassifier
from src.load_emnist   import load_emnist


def main():
    print("=" * 50)
    print("  Handwritten Character Recognition")
    print("  Dataset: EMNIST digits + letters (36 classes)")
    print("=" * 50)

    # 1. Load data
    print("\n[1/4] Loading EMNIST...")
    X_tr, y_tr, X_te, y_te, labels = load_emnist(data_dir='./data')

    # 2. Preprocessing
    cache = 'data/emnist_preprocessed.npz'
    if os.path.exists(cache):
        print("\n[2/4] Loading preprocessed cache...")
        d = np.load(cache)
        X_tr_s, X_te_s = d['X_tr'], d['X_te']
    else:
        print("\n[2/4] Improved PCA straightening (first run ~10 min)...")
        X_tr_s = np.array([straighten_image(x) for x in tqdm(X_tr)])
        X_te_s = np.array([straighten_image(x) for x in tqdm(X_te)])
        os.makedirs('data', exist_ok=True)
        np.savez(cache, X_tr=X_tr_s, X_te=X_te_s)

    # 3. PCA
    print("\n[3/4] PCA dimensionality reduction...")
    pca, X_tr_pca = build_pca_pipeline(X_tr_s, variance=0.95)
    X_te_pca = pca.transform(X_te_s)

    # 4. Train
    print("\n[4/4] Training RPCA Forest + KNN...")
    clf = RPCAKNNClassifier(
        n_trees=20,
        k=7,
        metric='euclidean',
        leaf_size=200
    )
    clf.fit(X_tr_pca, y_tr)

    # 5. Evaluate
    print("\nEvaluating...")
    t0 = time.perf_counter()
    preds = clf.predict(X_te_pca)
    elapsed = time.perf_counter() - t0

    acc = accuracy_score(y_te, preds)
    print(f"\nAccuracy : {acc:.4f} ({acc*100:.1f}%)")
    print(f"Time     : {elapsed:.1f}s ({elapsed/len(y_te)*1000:.1f}ms/sample)")
    print("\nClassification Report:")
    print(classification_report(y_te, preds, target_names=labels))

    # 6. Save
    os.makedirs('models', exist_ok=True)
    joblib.dump({
        'pca':    pca,
        'clf':    clf,
        'labels': labels,
    }, 'models/pipeline.pkl')
    print("Model saved: models/pipeline.pkl")


if __name__ == '__main__':
    main()
