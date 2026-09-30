import numpy as np
from sklearn.decomposition import TruncatedSVD


class RPCATree:
    def __init__(self, n_components: int = 3, leaf_size: int = 50):
        self.n_components = n_components
        self.leaf_size = leaf_size

    def fit(self, X: np.ndarray, indices: np.ndarray = None):
        if indices is None:
            indices = np.arange(len(X))
        self.X = X
        self.root = self._build(indices)
        del self.X
        return self

    def _build(self, indices: np.ndarray) -> dict:
        if len(indices) <= self.leaf_size:
            return {'leaf': True, 'indices': indices}

        n_comp = min(self.n_components, len(indices) - 1)
        svd = TruncatedSVD(n_components=n_comp)
        projected = svd.fit_transform(self.X[indices])
        pc1 = projected[:, 0]

        split = np.random.laplace(pc1.mean(), pc1.std() + 1e-8)
        left_mask  = pc1 < split
        right_mask = ~left_mask

        if left_mask.sum() == 0 or right_mask.sum() == 0:
            return {'leaf': True, 'indices': indices}

        return {
            'leaf':  False,
            'svd':   svd,
            'split': split,
            'left':  self._build(indices[left_mask]),
            'right': self._build(indices[right_mask]),
        }

    def query(self, x: np.ndarray) -> np.ndarray:
        node = self.root
        while not node['leaf']:
            proj = node['svd'].transform(x.reshape(1, -1))[0, 0]
            node = node['left'] if proj < node['split'] else node['right']
        return node['indices']


class RPCAForest:
    def __init__(self, n_trees: int = 10, n_components: int = 3,
                 leaf_size: int = 50):
        self.n_trees = n_trees
        self.n_components = n_components
        self.leaf_size = leaf_size
        self.trees = []

    def fit(self, X: np.ndarray):
        from tqdm import tqdm
        self.trees = []
        for _ in tqdm(range(self.n_trees), desc='Building RPCA Forest'):
            tree = RPCATree(self.n_components, self.leaf_size)
            tree.fit(X)
            self.trees.append(tree)
        return self

    def get_candidates(self, x: np.ndarray) -> np.ndarray:
        candidates = set()
        for tree in self.trees:
            candidates.update(tree.query(x).tolist())
        return np.array(list(candidates))