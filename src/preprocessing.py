import numpy as np
from sklearn.decomposition import PCA
from scipy.ndimage import rotate


def straighten_image(img_flat: np.ndarray) -> np.ndarray:
    img = img_flat.reshape(28, 28)
    coords = np.column_stack(np.where(img > 0.1))
    if len(coords) < 10:
        return img_flat

    centered = coords - coords.mean(axis=0)
    cov = np.cov(centered.T)
    eigvals, eigvecs = np.linalg.eigh(cov)
    principal = eigvecs[:, -1]
    angle = np.degrees(np.arctan2(principal[1], principal[0]))

    if angle < -45 or angle > 135:
        angle += 180

    straightened = rotate(img, -angle, reshape=False, mode='constant')
    return straightened.flatten()


def build_pca_pipeline(X_train: np.ndarray, variance: float = 0.95):
    # Bước 1: fit full PCA để tìm số components cần thiết
    pca_full = PCA(svd_solver='full')
    pca_full.fit(X_train)
    cumvar = pca_full.explained_variance_ratio_.cumsum()
    n_components = int((cumvar < variance).sum()) + 1
    print(f"      Chọn {n_components} components ({variance*100:.0f}% variance)")

    # Bước 2: fit lại với số components xác định + randomized solver
    pca = PCA(n_components=n_components, svd_solver='randomized', whiten=True)
    X_out = pca.fit_transform(X_train)
    print(f"      PCA: {X_train.shape[1]} -> {X_out.shape[1]} dims")
    return pca, X_out