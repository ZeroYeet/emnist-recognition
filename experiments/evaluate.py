"""
Đánh giá hiệu suất pipeline và xuất biểu đồ.
Chạy: python experiments/evaluate.py
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import numpy as np
import joblib
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import time
from tqdm import tqdm

from src.preprocessing import straighten_image, build_pca_pipeline
from src.load_emnist   import load_emnist

matplotlib.rcParams['font.family'] = 'DejaVu Sans'
os.makedirs('results', exist_ok=True)


def plot_confusion_matrix(y_true, y_pred, labels, save_path):
    cm = confusion_matrix(y_true, y_pred)
    cm_pct = cm.astype('float') / cm.sum(axis=1, keepdims=True) * 100

    fig, ax = plt.subplots(figsize=(18, 16))
    sns.heatmap(
        cm_pct, annot=True, fmt='.0f', cmap='Blues',
        xticklabels=labels, yticklabels=labels,
        linewidths=0.3, ax=ax, annot_kws={'size': 7}
    )
    ax.set_title('Confusion Matrix (%)', fontsize=16, fontweight='bold', pad=15)
    ax.set_xlabel('Predicted Label', fontsize=12)
    ax.set_ylabel('True Label', fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")


def plot_accuracy_per_class(y_true, y_pred, labels, save_path):
    cm  = confusion_matrix(y_true, y_pred)
    acc = cm.diagonal() / cm.sum(axis=1) * 100

    colors = ['#e74c3c' if a < 70 else '#f39c12' if a < 85 else '#2ecc71'
              for a in acc]

    fig, ax = plt.subplots(figsize=(16, 6))
    bars = ax.bar(labels, acc, color=colors, edgecolor='white', linewidth=0.5)
    ax.axhline(y=acc.mean(), color='#3498db', linestyle='--',
               linewidth=1.5, label=f'Mean: {acc.mean():.1f}%')
    ax.axhline(y=85, color='#95a5a6', linestyle=':', linewidth=1,
               label='Target: 85%')
    ax.set_title('Accuracy per Class', fontsize=15, fontweight='bold')
    ax.set_xlabel('Class', fontsize=12)
    ax.set_ylabel('Accuracy (%)', fontsize=12)
    ax.set_ylim(0, 110)

    for bar, val in zip(bars, acc):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'{val:.0f}', ha='center', va='bottom', fontsize=7)

    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#2ecc71', label='>= 85% (Tot)'),
        Patch(facecolor='#f39c12', label='70-85% (Trung binh)'),
        Patch(facecolor='#e74c3c', label='< 70% (Yeu)'),
    ]
    ax.legend(handles=legend_elements, loc='lower right', fontsize=9)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")


def plot_pca_variance(pca, save_path):
    cumvar = pca.explained_variance_ratio_.cumsum() * 100
    indvar = pca.explained_variance_ratio_ * 100

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(range(1, len(cumvar)+1), cumvar, color='#3498db', linewidth=2)
    ax1.axhline(95, color='#e74c3c', linestyle='--', linewidth=1.5, label='95% variance')
    ax1.axhline(99, color='#f39c12', linestyle='--', linewidth=1.5, label='99% variance')
    ax1.fill_between(range(1, len(cumvar)+1), cumvar, alpha=0.1, color='#3498db')
    ax1.set_title('Cumulative Explained Variance', fontsize=13, fontweight='bold')
    ax1.set_xlabel('Number of Components')
    ax1.set_ylabel('Variance Explained (%)')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)

    top = 30
    ax2.bar(range(1, top+1), indvar[:top], color='#9b59b6', edgecolor='white', linewidth=0.5)
    ax2.set_title(f'Top {top} Components - Individual Variance', fontsize=13, fontweight='bold')
    ax2.set_xlabel('Component Index')
    ax2.set_ylabel('Variance Explained (%)')
    ax2.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")


def plot_sample_predictions(X_te_raw, y_true, y_pred, labels, n=40, save_path='results/sample_predictions.png'):
    fig, axes = plt.subplots(5, 8, figsize=(16, 10))
    fig.suptitle('Sample Predictions (Green=Correct, Red=Wrong)',
                 fontsize=14, fontweight='bold')

    indices = np.random.choice(len(y_true), n, replace=False)

    for ax, idx in zip(axes.flatten(), indices):
        img = X_te_raw[idx].reshape(28, 28)
        true_lbl = labels[y_true[idx]]
        pred_lbl = labels[y_pred[idx]]
        correct  = true_lbl == pred_lbl

        ax.imshow(img, cmap='gray')
        ax.set_title(
            f'T:{true_lbl} P:{pred_lbl}',
            color='green' if correct else 'red',
            fontsize=9, fontweight='bold'
        )
        ax.axis('off')

    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")


def plot_training_size_vs_accuracy(X_tr, y_tr, X_te, y_te, labels, save_path):
    from src.classifier import RPCAKNNClassifier

    sizes  = [0.1, 0.2, 0.4, 0.6, 0.8, 1.0]
    accs   = []

    print("  Generating learning curve...")
    for frac in sizes:
        n = max(int(len(y_tr) * frac), 10)
        idx = np.random.choice(len(y_tr), n, replace=False)

        clf_tmp = RPCAKNNClassifier(n_trees=10, k=5, leaf_size=100)
        clf_tmp.fit(X_tr[idx], y_tr[idx])

        preds_tmp = clf_tmp.predict(X_te)
        accs.append(accuracy_score(y_te, preds_tmp) * 100)
        print(f"    {int(frac*100)}% data ({n} anh): {accs[-1]:.1f}%")

    fig, ax = plt.subplots(figsize=(9, 5))
    x_vals = [int(s * len(y_tr)) for s in sizes]
    ax.plot(x_vals, accs, 'o-', color='#3498db',
            linewidth=2, markersize=8, markerfacecolor='white', markeredgewidth=2)

    for x, a in zip(x_vals, accs):
        ax.annotate(f'{a:.1f}%', (x, a),
                    textcoords='offset points', xytext=(0, 10),
                    ha='center', fontsize=9)

    ax.set_title('Learning Curve: Training Size vs Accuracy', fontsize=13, fontweight='bold')
    ax.set_xlabel('Number of Training Samples')
    ax.set_ylabel('Test Accuracy (%)')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 105)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")


def plot_inference_speed(X_te_pca, clf, save_path):
    sample_sizes = [1, 10, 50, 100, 200, 500]
    times_ms = []

    for n in sample_sizes:
        if n > len(X_te_pca):
            break
        t0 = time.perf_counter()
        clf.predict(X_te_pca[:n])
        elapsed = (time.perf_counter() - t0) * 1000
        times_ms.append(elapsed / n)
        print(f"    n={n:4d}: {elapsed/n:.2f}ms/sample")

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(sample_sizes[:len(times_ms)], times_ms,
            's-', color='#e74c3c', linewidth=2,
            markersize=8, markerfacecolor='white', markeredgewidth=2)
    ax.set_title('Inference Speed per Sample', fontsize=13, fontweight='bold')
    ax.set_xlabel('Batch Size')
    ax.set_ylabel('Time per Sample (ms)')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")


def main():
    print("=" * 55)
    print("  EVALUATION & VISUALIZATION")
    print("=" * 55)

    # 1. Load data
    print("\n[1/6] Loading EMNIST...")
    X_tr, y_tr, X_te, y_te, labels = load_emnist(data_dir='./data')

    # 2. Preprocessing
    print("\n[2/6] Preprocessing...")
    cache = 'data/emnist_preprocessed.npz'
    if os.path.exists(cache):
        d = np.load(cache)
        X_tr_s, X_te_s = d['X_tr'], d['X_te']
    else:
        X_tr_s = np.array([straighten_image(x) for x in tqdm(X_tr)])
        X_te_s = np.array([straighten_image(x) for x in tqdm(X_te)])
        np.savez(cache, X_tr=X_tr_s, X_te=X_te_s)

    # 3. Load or train model
    model_path = 'models/pipeline.pkl'
    if os.path.exists(model_path):
        print("\n[3/6] Loading saved model...")
        saved    = joblib.load(model_path)
        clf      = saved['clf']
        pca      = saved['pca']
        labels   = saved.get('labels', labels)
        X_tr_pca = pca.transform(X_tr_s)
        X_te_pca = pca.transform(X_te_s)
    else:
        print("\n[3/6] Training model...")
        from src.classifier import RPCAKNNClassifier
        pca, X_tr_pca = build_pca_pipeline(X_tr_s, variance=0.95)
        X_te_pca = pca.transform(X_te_s)
        clf = RPCAKNNClassifier(n_trees=20, k=7, leaf_size=200)
        clf.fit(X_tr_pca, y_tr)

    # 4. Predict
    print("\n[4/6] Predicting on test set...")
    t0    = time.perf_counter()
    preds = clf.predict(X_te_pca)
    elapsed = time.perf_counter() - t0

    acc = accuracy_score(y_te, preds)
    print(f"\n  Overall Accuracy : {acc*100:.2f}%")
    print(f"  Total Time       : {elapsed:.1f}s")
    print(f"  Per Sample       : {elapsed/len(y_te)*1000:.1f}ms")

    # 5. Plots
    print("\n[5/6] Generating plots...")

    print("\n  1. Confusion Matrix...")
    plot_confusion_matrix(y_te, preds, labels, 'results/confusion_matrix.png')

    print("  2. Accuracy per Class...")
    plot_accuracy_per_class(y_te, preds, labels, 'results/accuracy_per_class.png')

    print("  3. PCA Variance...")
    plot_pca_variance(pca, 'results/pca_variance.png')

    print("  4. Sample Predictions...")
    plot_sample_predictions(X_te_s, y_te, preds, labels,
                            save_path='results/sample_predictions.png')

    print("  5. Learning Curve...")
    plot_training_size_vs_accuracy(
        X_tr_pca, y_tr, X_te_pca, y_te, labels,
        'results/learning_curve.png'
    )

    print("  6. Inference Speed...")
    plot_inference_speed(X_te_pca, clf, 'results/inference_speed.png')

    # 6. Summary
    print("\n[6/6] Summary Report")
    print("=" * 55)
    print(classification_report(y_te, preds, target_names=labels))

    print("\nTat ca bieu do da luu vao thu muc results/:")
    for f in os.listdir('results'):
        if f.endswith('.png'):
            size = os.path.getsize(f'results/{f}') // 1024
            print(f"  {f} ({size} KB)")


if __name__ == '__main__':
    main()
