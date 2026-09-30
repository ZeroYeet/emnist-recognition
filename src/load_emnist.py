import numpy as np
from torchvision import datasets

# 36 classes: 0-9 (digits) + a-z (letters, case-insensitive)
CLASSES = [str(i) for i in range(10)] + list('abcdefghijklmnopqrstuvwxyz')


def _to_array(ds):
    X = ds.data.numpy()               # (N, 28, 28) uint8
    X = np.rot90(X, k=1, axes=(1, 2)) # fix EMNIST 90° rotation
    return X.reshape(-1, 784).astype('float32') / 255.


def _subsample_per_class(X, y, n_per_class, rng):
    """Lấy tối đa n_per_class mẫu cho mỗi class, giữ phân bố đều."""
    idx = []
    for c in np.unique(y):
        where = np.where(y == c)[0]
        chosen = rng.choice(where, size=min(n_per_class, len(where)), replace=False)
        idx.append(chosen)
    idx = np.concatenate(idx)
    rng.shuffle(idx)
    return X[idx], y[idx]


def load_emnist(data_dir='./data', random_state=42):
    """
    Kết hợp EMNIST digits (0-9) + letters (a-z), cân bằng class.
    Trả về X_tr, y_tr, X_te, y_te, labels (36 classes).
    """
    rng = np.random.default_rng(random_state)

    # Load digits (labels 0-9)
    d_tr = datasets.EMNIST(data_dir, split='digits',  train=True,  download=True)
    d_te = datasets.EMNIST(data_dir, split='digits',  train=False, download=True)
    # Load letters (torchvision labels 1-26 → remap thành 10-35)
    l_tr = datasets.EMNIST(data_dir, split='letters', train=True,  download=True)
    l_te = datasets.EMNIST(data_dir, split='letters', train=False, download=True)

    X_tr_d = _to_array(d_tr);  y_tr_d = d_tr.targets.numpy()
    X_te_d = _to_array(d_te);  y_te_d = d_te.targets.numpy()
    X_tr_l = _to_array(l_tr);  y_tr_l = l_tr.targets.numpy() - 1 + 10  # 1-26 → 10-35
    X_te_l = _to_array(l_te);  y_te_l = l_te.targets.numpy() - 1 + 10

    # Cân bằng: lấy số mẫu bằng nhau mỗi class
    # digits: 24000/class (train), letters: 4800/class (train)
    n_tr = int(np.unique(y_tr_l, return_counts=True)[1].min())  # ~4800
    n_te = int(np.unique(y_te_l, return_counts=True)[1].min())  # ~800

    X_tr_d, y_tr_d = _subsample_per_class(X_tr_d, y_tr_d, n_tr, rng)
    X_te_d, y_te_d = _subsample_per_class(X_te_d, y_te_d, n_te, rng)

    # Ghép digits + letters
    X_tr = np.vstack([X_tr_d, X_tr_l])
    y_tr = np.hstack([y_tr_d, y_tr_l])
    X_te = np.vstack([X_te_d, X_te_l])
    y_te = np.hstack([y_te_d, y_te_l])

    # Shuffle
    tr_idx = rng.permutation(len(y_tr))
    te_idx = rng.permutation(len(y_te))
    X_tr, y_tr = X_tr[tr_idx], y_tr[tr_idx]
    X_te, y_te = X_te[te_idx], y_te[te_idx]

    print(f"Train: {X_tr.shape[0]} anh | Test: {X_te.shape[0]} anh | Classes: {len(CLASSES)}")
    print(f"  ~{n_tr} anh/class (train), ~{n_te} anh/class (test)")
    return X_tr, y_tr, X_te, y_te, CLASSES
