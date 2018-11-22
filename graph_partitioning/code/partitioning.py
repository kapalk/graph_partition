import pandas as pd
import numpy as np
import networkx as nx
import os
from scipy.sparse.linalg import eigsh
from sklearn.cluster import KMeans
from tqdm import tqdm


class Partition:

    def __init__(self, df, k):
        self._graph = nx.from_pandas_edgelist(df)
        self._number_of_nodes = nx.number_of_nodes(self._graph)
        self._number_of_edges = nx.number_of_edges(self._graph)
        self.k_partition = k

    def execute(self, filename=None):
        adj_mat_laplacian = self._laplacian()
        eigenvectors = self._eigevectors(adj_mat_laplacian)
        norm_eigenvecs = self.normalize(eigenvectors)

        self._results = list(zip(self._graph.nodes(), self._cluster(norm_eigenvecs)))

        if not filename:
            return self._results

        self._to_txt(filename)

    def _laplacian(self):
        return nx.linalg.normalized_laplacian_matrix(self._graph)

    def _eigevectors(self, X):
        _, vectors = eigsh(X, k=self.k_partition, which="LM")
        return vectors

    @staticmethod
    def normalize(X):
        l1norm = np.abs(X).sum(axis=1).reshape(X.shape[0], 1)
        return X / l1norm

    def _cluster(self, X):
        kmeans = KMeans(n_clusters=self.k_partition, init="k-means++", n_jobs=-1).fit(X)
        return kmeans.labels_

    def _to_txt(self, filename):
        current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        path = os.path.abspath(os.path.join(current_dir, "results", filename))
        with open(path, "w") as f:
            f.write("# {0} {1} {2} {3}\n".format(filename.split(".")[0], self._number_of_nodes, self._number_of_edges, self.k_partition))
            f.write('\n'.join('%s %s' % x for x in self._results))



if __name__:
    filenames = ["ca-GrQc.txt", "Oregon-1.txt", "soc-Epinions1.txt", "web-NotreDame.txt", "roadNet-CA.txt"]
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    for filename in tqdm(filenames):
        graph_filepath = os.path.join(root_dir, "data", "graphs", filename)
        output_filename = filename + ".output.txt"

        with open(graph_filepath) as f:
            first_line = f.readline()
            k = int(first_line.split(" ")[-1])

        graph_data = pd.read_csv(filepath_or_buffer=graph_filepath,
                                 delimiter=" ",
                                 skiprows=[0],
                                 header=None,
                                 names=["source", "target"])

        p = Partition(graph_data, k).execute(filename=output_filename)



