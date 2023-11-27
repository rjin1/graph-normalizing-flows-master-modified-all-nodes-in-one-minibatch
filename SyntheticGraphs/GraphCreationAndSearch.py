import networkx as nx
import numpy as np
from scipy.io import loadmat
import pickle
import graph_nets as gn

# # Load connections from .mat
ConnectionFilePath = 'GridGraph5neighbors'
variable_name = 'Node_pairs'
Node_pairs = loadmat(ConnectionFilePath)[variable_name].transpose()
# Add edges to graph, weights = 1
GridGraph = nx.Graph()
GridGraph.add_edges_from(list(Node_pairs))
# Relabel nodes to match orders of node features
GridGraph = nx.relabel.convert_node_labels_to_integers(GridGraph)

G = GridGraph
nx.write_gpickle(GridGraph, ConnectionFilePath)
# G = nx.read_gpickle('GridGraph5neighbors')
# # # Test k-hops neighbors graph
# g_ego = nx.generators.ego.ego_graph(G, 0, 1)
# key = gn.blocks.broadcast_sender_nodes_to_edges(g_ego)
