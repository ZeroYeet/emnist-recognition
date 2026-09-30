import numpy as np
from scipy.spatial.distance import cdist
from .rpca_forest import RPCAForest


class RPCAKNNClassifier:
    def __init__(self, n_trees: int = 10, k: int = 5,
                 metric: str = 'euclidean', leaf_size: int = 50):
        self.k = k
        self.metric = metric
        self.forest = RPCAForest(n_trees=n_trees, leaf_size=leaf_size)

    def fit(self, X: np.ndarray, y: np.ndarray):
        self.X_train = X
        self.y_train = y
        self.forest.fit(X)
        return self

    def predict_one(self, x: np.ndarray) -> int:
        candidates_idx = self.forest.get_candidates(x)
        if len(candidates_idx) < self.k:
            candidates_idx = np.arange(len(self.X_train))

        X_cand = self.X_train[candidates_idx]
        dists   = cdist(x.reshape(1, -1), X_cand, metric=self.metric)[0]

        top_k_local = np.argsort(dists)[:self.k]
        top_k_idx   = candidates_idx[top_k_local]
        top_k_dists = dists[top_k_local] + 1e-10

        weights = 1.0 / top_k_dists
        labels  = self.y_train[top_k_idx]
        vote    = np.bincount(labels, weights=weights,
                              minlength=int(self.y_train.max()) + 1)
        return int(np.argmax(vote))

    def predict(self, X: np.ndarray) -> np.ndarray:
        from tqdm import tqdm
        return np.array([self.predict_one(x)
                         for x in tqdm(X, desc='Predicting')])