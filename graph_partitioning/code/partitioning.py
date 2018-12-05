from scipy.sparse.linalg import eigsh
from sklearn.cluster import KMeans
from networkx.algorithms.cuts import cut_size
from tqdm import tqdm

import pandas as pd
import numpy as np
import networkx as nx
import os
import argparse

class SpectralClustering:

    def __init__(self, df, n_clusters):
        self._graph = nx.from_pandas_edgelist(df)
        self._number_of_nodes = nx.number_of_nodes(self._graph)
        self._number_of_edges = nx.number_of_edges(self._graph)
        self.k_partition = n_clusters
        self._results = None
        self.conductance = None
        self._results = None

    def fit(self, output_path=None, normalized=True, return_cost=False):

        if normalized:
            adj_mat_laplacian = self._normalized_laplacian()
            k_eigenvectors = self.normalize(self._eigevectors(adj_mat_laplacian))
        else:
            adj_mat_laplacian = self._laplacian()
            k_eigenvectors = self._eigevectors(adj_mat_laplacian)

        self._results = list(zip(self._graph.nodes(), self._cluster(k_eigenvectors)))
        if not output_path:
            if return_cost:
                self.compute_cost()
                return self._results, self.conductance
            else:
                return self._results

        self._to_txt(output_path)

    def _normalized_laplacian(self):
        return nx.linalg.normalized_laplacian_matrix(self._graph)

    def _laplacian(self):
        return nx.linalg.laplacian_matrix(self._graph).asfptype()

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

    def _to_txt(self, output_path):

        with open(output_path, "w") as f:
            f.write("# {0} {1} {2} {3}\n".format(filename.split(".")[0],
                                                 self._number_of_nodes,
                                                 self._number_of_edges,
                                                 self.k_partition))
            f.write('\n'.join('%s %s' % x for x in self._results))

    def compute_cost(self):
        # conductance of the final partitions
        arr = np.array(self._results)
        cut_size_sum = 0

        smallest_partition_size = self._number_of_nodes
        for i in np.unique(arr[:, 1]):
            nodes_in_partition = [node_id for node_id, partition_id in self._results if partition_id == i]
            cut_size_sum += cut_size(self._graph, nodes_in_partition)

            partition_size = len(nodes_in_partition)
            if partition_size < smallest_partition_size:
                smallest_partition_size = partition_size

        self.conductance = cut_size_sum / smallest_partition_size



if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Graph partitioning')
    parser.add_argument("--k", help="the number of partitions")
    args = parser.parse_args()

    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    dataset_dir = os.path.join(root_dir, "data")
    result_dir = os.path.abspath(os.path.join(root_dir, "results"))

    datasets = [f for f in os.listdir(dataset_dir) if f.endswith(".txt")]
    for filename in tqdm(datasets):
        dataset_filepath = os.path.join(dataset_dir, filename)
        output_filepath = os.path.join(result_dir, filename.strip(".txt") + ".output.txt")



        graph_data = pd.read_csv(filepath_or_buffer=dataset_filepath,
                                 delimiter=" ",
                                 skiprows=[0],
                                 header=None,
                                 names=["source", "target"])

        SpectralClustering(graph_data, int(args.k)).fit(output_path=output_filepath, normalized=True)


