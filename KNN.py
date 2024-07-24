import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from sklearn.neighbors import kneighbors_graph

# Load the feature dataset
data = pd.read_csv('urbansound8k_yamnet_features.csv')

# Separate features and labels
X = data.drop(columns=['label'])
y = data['label']



# Number of neighbors
k_neighbors = 5

# Construct the KNN graph
knn_graph = kneighbors_graph(X, n_neighbors=k_neighbors, mode='connectivity', include_self=False)

# The result is a sparse matrix in CSR format
print(knn_graph)



# Reduce dimensionality for visualization
X_reduced = PCA(n_components=2).fit_transform(X)  # Or use TSNE for better visualization

# Create a NetworkX graph from the KNN graph
G = nx.from_scipy_sparse_array(knn_graph)

# Draw the graph
pos = {i: X_reduced[i] for i in range(len(X_reduced))}
nx.draw(G, pos, node_size=20, with_labels=False)

plt.title("KNN Graph Visualization")
plt.show()
