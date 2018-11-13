import pandas as pd
import numpy as np
import networkx as nx
import os
from scipy.sparse.linalg import eigsh
from sklearn.cluster import KMeans


class Partition:

    def __init__(self, df, k):
        self._graph = nx.from_pandas_edgelist(df)
        self.k_partition = k

    def execute(self, filename=None):
        adj_mat_laplacian = self.__laplacian()
        eigenvectors = self.__eigevectors(adj_mat_laplacian)
        norm_eigenvecs = self.normalize(eigenvectors)

        self._results = list(zip(self._graph.nodes(), self.__cluster(norm_eigenvecs)))

        if not filename:
            return self._results

        self.to_txt(filename)

    def __laplacian(self):
        return nx.linalg.normalized_laplacian_matrix(self._graph)

    def __eigevectors(self, X):
        _, vectors = eigsh(X, k=self.k_partition, which="LM")
        return vectors

    @staticmethod
    def normalize(X):
        l1norm = np.abs(X).sum(axis=1).reshape(X.shape[0], 1)
        return X / l1norm

    def __cluster(self, X):
        kmeans = KMeans(n_clusters=self.k_partition, init="k-means++", n_jobs=-1).fit(X)
        return kmeans.labels_

    def to_txt(self, filename):
        current_dir = os.path.dirname(os.path.abspath(__file__))
        path = os.path.abspath(os.path.join(current_dir, "results", filename))
        with open(path, "w") as f:
            f.write('\n'.join('%s %s' % x for x in self._results))



if __name__:
    graph_data = pd.read_csv(filepath_or_buffer="data/ca-GrQc.txt",
                             delimiter="\t",
                             skiprows=[0, 1, 2, 3],
                             header=None,
                             dtype=np.int32,
                             names=["source", "target"])

    p = Partition(graph_data, 6).execute(filename="results.txt")



