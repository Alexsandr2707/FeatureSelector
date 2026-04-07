import numpy as np
import pandas as pd
from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import r2_score
from sklearn.base import TransformerMixin
from scipy.cluster.hierarchy import ClusterNode


def _pls_explained_variance(pls, X, Y_true, do_plot=False):
    r2 = np.zeros(pls.n_components)
    x_transformed = pls.transform(X)
    for i in range(0, pls.n_components):
        Y_pred = (np.dot(x_transformed[:, i][:, np.newaxis], pls.y_loadings_[:, i][:, np.newaxis].T) * pls._y_std
                  + pls._y_mean)
        r2[i] = r2_score(Y_true, Y_pred)
    # Use all components together.
    overall_r2 = r2_score(Y_true, pls.predict(X))
    return r2, overall_r2


def _feature_explained_variance(pls, X, Y_true, do_plot=False):
    r2 = np.zeros(pls.n_features_in_)
    if len(r2) == 1:
        return pls.score(X, Y_true)
    scaled = (X - pls._x_mean) / pls._x_std
    transformed = np.stack([scaled * pls.x_rotations_[:, i]
                           for i in range(pls.n_components)])
    transformed = transformed.transpose((2, 1, 0))
    preds = [(np.dot(t, pls.y_loadings_.T) * pls._y_std + pls._y_mean)
             for t in transformed]
    preds = np.array(preds)
    scores = np.array([r2_score(Y_true, x) for x in preds])
    for i in range(len(r2)):
        r2[i] = (1 - scores[i])/(1 - max(np.delete(scores, i)))
    return r2


def _PLS_tree(X, y):
    map_index = {X.columns[i]: i for i in range(len(X.columns))}
    X_variance = np.var(X, axis=0).sum()

    def recursion(X, y):
        if X.shape[-1] == 1:
            c = ClusterNode(map_index[X.columns[0]], None, None, dist=0)
            c.tag = X.columns[0]
            return c
        pls = PLSRegression(n_components=2).fit(X, y)
        (var_left, var_right), (total) = _pls_explained_variance(pls, X, y)
        # print(var_left, var_right, total)
        mask = abs(pls.x_weights_)[:, 0] > abs(pls.x_weights_)[:, 1]
        lmask, rmask = X.columns[mask], X.columns[~mask]
        if X.shape[-1] == 2 and (len(lmask) == 0 or len(rmask) == 0):
            lmask, rmask = [X.columns[0]], [X.columns[1]]
        left = recursion(X[lmask], y)
        right = recursion(X[rmask], y)
        return ClusterNode(X.shape[-1], left, right, dist=float(max(total, 0)))
    return recursion(X, y)


def _get_level(tree, depth=10):
    leaf_index = -1
    clusters = {}

    def prune(tree, depth=10):
        nonlocal leaf_index
        if depth == 0 or tree.left is None and tree.right is None:
            leaf_index += 1
            clusters[leaf_index] = _collect_leaves(tree)
            return ClusterNode(leaf_index, None, None, dist=tree.dist)
        return ClusterNode(np.random.randint(1e8),
                           None if tree.left is None else prune(
                               tree.left, depth-1),
                           None if tree.right is None else prune(tree.right, depth-1), dist=tree.dist)
    return prune(tree, depth), clusters


def _cluster_info(X, y):
    if X.shape[-1] == 1:
        return pd.Series(data=[np.nan], index=X.columns)
    pls = PLSRegression(n_components=2).fit(X, y)
    rating = _feature_explained_variance(pls, X, y)
    return pd.Series(data=rating, index=X.columns).sort_values()


def _collect_leaves(tree):
    nodes = []

    def recursion(tree):
        if tree.left is None and tree.right is None:
            nodes.append(tree.id)
            return
        recursion(tree.left)
        recursion(tree.right)
    recursion(tree)
    return nodes


class PLSTransformer(TransformerMixin):
    def __init__(self, depth=4, dropna=False):
        self.depth = depth
        self.dropna = dropna
        self.is_fit = False

    def fit(self, X, y):
        if self.dropna:
            X = X.dropna()
            y = y.dropna()
            index = y.index.intersection(X.index)
            X = X.loc[index]
            y = y.loc[index]
        self.tree = _PLS_tree(X, y)
        _, self.clusters = _get_level(self.tree, depth=self.depth)
        self.feature_names_in_ = X.columns
        self.info = [_cluster_info(X.iloc[:, self.clusters[x]], y)
                     for x in self.clusters]
        self.feature_names_out_ = [x.index[0] for x in self.info]
        self.is_fit = True
        return self

    def transform(self, X, y=None):
        if y is None:
            return X[self.feature_names_out_]
        else:
            return X[self.feature_names_out_], y

    def get_feature_names_out(self, X, y):
        return self.feature_names_out_

    def to_json(self, crop_height=False):
        data = {}
        i = 0

        def recursion(data, tree, depth):
            nonlocal i
            if depth == 0:
                name = self.info[i].index[0]
                payload = self.info[i].fillna(0).to_dict()
                height = 0 if crop_height else tree.dist
                i += 1
                return [dict(name=name, height=height, payload=payload)]
            if tree.left is None and tree.right is None:
                # print("tree.tag", tree.tag)
                return [dict(name=tree.tag, height=0 if crop_height else tree.dist)]
            return [dict(name="C1", children=recursion(data, tree.left, depth-1), height=tree.left.dist),
                    dict(name="C2", children=recursion(data, tree.right, depth-1), height=tree.right.dist)]
        return dict(name="root", children=recursion(data, self.tree, self.depth), height=1)


if __name__ == "__main__":
    X = pd.DataFrame(np.random.random(size=(1000, 100)))
    y = pd.Series(np.random.random(size=(1000,)))
    pls = PLSTransformer(depth=1e100)
    pls.fit(X, y)
    print(pls.to_json())
