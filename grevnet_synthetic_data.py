from functools import partial
import random

from sklearn import datasets
from sklearn.preprocessing import normalize
import graph_nets as gn
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from scipy.io import loadmat

MAX_SEED = 2 ** 32 - 1
GAUSSIAN_MEAN = [0, 0]
GAUSSIAN_COV = [[1, 0], [0, 1]]


# Return a fully connected networkx graph that has edges from a given node to
# itself.
def fully_connected_nx_graph(num_nodes):
    g = nx.complete_graph(num_nodes, create_using=nx.DiGraph)
    r = range(num_nodes)
    g.add_edges_from(zip(r, r))
    # g = nx.Graph(create_using=nx.DiGraph)
    # g.add_edges_from(list([[0, 0]]))
    return g


class SyntheticDataset():
    def __init__(self, graph_generator_fn):
        self.graph_generator_fn = graph_generator_fn

    def get_next_batch_data_dicts(self, batch_size, iteration=0):
        data_dicts = []
        for _ in range(batch_size):
            data_dict = {}
            g, node_features = self.graph_generator_fn()
            num_nodes = g.number_of_nodes()
            data_dict['n_node'] = num_nodes
            data_dict['n_edge'] = g.number_of_edges()
            edges = list(zip(*g.edges()))
            data_dict['senders'] = np.array(edges[0])
            data_dict['receivers'] = np.array(edges[1])
            data_dict['nodes'] = node_features
            data_dict['globals'] = 0
            data_dict['edges'] = np.zeros(g.number_of_edges())
            data_dicts.append(data_dict)
        return data_dicts

    def get_next_batch(self, batch_size, iteration):
        return gn.utils_np.data_dicts_to_graphs_tuple(
            self.get_next_batch_data_dicts(batch_size, iteration))


def moons_sample(n_samples, noise=0.05):
    g = fully_connected_nx_graph(n_samples)
    return g, datasets.make_moons(
        n_samples=n_samples,
        shuffle=True,
        noise=noise,
        random_state=random.randrange(MAX_SEED))[0].astype(np.float32)


def mom_sample(n_samples_choices, noise=0.05):
    num_nodes = np.random.choice(n_samples_choices)
    g = fully_connected_nx_graph(num_nodes)
    return moons_sample(num_nodes, noise=noise)


def mog_sample(offsets_choices, rotate=False):
    offsets = random.choice(offsets_choices)
    num_nodes = len(offsets)
    g = fully_connected_nx_graph(num_nodes)
    np.random.shuffle(offsets)
    features = np.random.multivariate_normal(
        GAUSSIAN_MEAN, GAUSSIAN_COV, num_nodes).astype(np.float32) + offsets

    if rotate:
        angle = np.random.random() * np.pi
        rot_mat = [[np.cos(angle), -np.sin(angle)],
                   [np.sin(angle), np.cos(angle)]]
        features = np.transpose(np.matmul(
            rot_mat, np.transpose(features))).astype(np.float32)
    return g, features


OFFSETS_4 = np.array([5]).astype(np.float32)
# OFFSETS_4 = np.array([[-5, 5], [5, 5], [-5, -5], [5, -5]]).astype(np.float32)
OFFSETS_6 = np.array([[-5, 5], [5, 5], [-5, -5], [5, -5], [15, 5],
                      [15, -5]]).astype(np.float32)
OFFSETS_9 = np.array([[-5, 5], [5, 5], [-5, -5], [5, -5], [15, 5], [15, -5],
                      [-5, 15], [5, 15], [15, 15]]).astype(np.float32)

DATASETS_MAP = {
    'moons_100':
        SyntheticDataset(partial(moons_sample, n_samples=100)),
    'moons_10':
        SyntheticDataset(partial(moons_sample, n_samples=10)),
    'moons_6':
        SyntheticDataset(partial(moons_sample, n_samples=6)),
    'mom_6_10':
        SyntheticDataset(partial(mom_sample, n_samples_choices=[6, 10])),
    'mom_6_10_20':
        SyntheticDataset(partial(mom_sample, n_samples_choices=[6, 10, 20])),
    'mog_4':
        SyntheticDataset(partial(mog_sample, offsets_choices=[OFFSETS_4])),
    'mog_6':
        SyntheticDataset(partial(mog_sample, offsets_choices=[OFFSETS_6])),
    'mog_9':
        SyntheticDataset(partial(mog_sample, offsets_choices=[OFFSETS_9])),
    'mog_4_rotate':
        SyntheticDataset(
            partial(mog_sample, offsets_choices=[OFFSETS_4], rotate=True)),
    'mog_4_6':
        SyntheticDataset(
            partial(mog_sample, offsets_choices=[OFFSETS_4, OFFSETS_6])),
    'mog_4_9':
        SyntheticDataset(
            partial(mog_sample, offsets_choices=[OFFSETS_4, OFFSETS_9])),
}


# Load fMRI datasets and basic information into dicts.-Rui

class LOAD_FMRI:
    def __init__(self, fMRI_file_name, variable_topick_name, graph_file_name, ego_radius):
        # load fMRI datasets according to file name fMRI_file_name using scipy.io.-Rui
        # Note that directory is fMRI_data/
        self.fMRI_file_name = fMRI_file_name
        self.variable_topick_name = variable_topick_name
        self.features_all = loadmat('fMRI_data/' + self.fMRI_file_name)[self.variable_topick_name].transpose()
        self.features_all = self.features_all.astype(np.float32)
        self.features_all = normalize(self.features_all)
        # load the big one graph.-Rui
        # Note that directory is SyntheticGraph/
        self.graph_all = nx.read_gpickle('SyntheticGraphs/' + graph_file_name)
        # Only apply for the ego graph.-Rui
        self.ego_radius = ego_radius

    @staticmethod
    def get_validate_ground_truth(ground_truth_file, variable_name):
        val_ground_truth = loadmat('fMRI_data/' + ground_truth_file)[variable_name].transpose()
        return val_ground_truth.astype(np.float32)

    @staticmethod
    def get_predefined_distribution_params(Params1_file_name, Params1_variable_name, Params2_file_name, Params2_variable_name):
        dis_params_1 = loadmat('fMRI_data/' + Params1_file_name)[Params1_variable_name].transpose()
        dis_params_2 = loadmat('fMRI_data/' + Params2_file_name)[Params2_variable_name].transpose()
        return dis_params_1.astype(np.float32), dis_params_2.astype(np.float32)

    def get_next_batch_data_dicts(self):
        data_dicts = []
        data_dict = {}
        data_dict['nodes'] = self.features_all
        data_dict['n_node'] = self.graph_all.number_of_nodes()
        data_dict['n_edge'] = self.graph_all.number_of_edges()
        edges = list(zip(*self.graph_all.edges()))
        data_dict['senders'] = np.array(edges[0])
        data_dict['receivers'] = np.array(edges[1])
        data_dict['globals'] = 0
        data_dict['edges'] = np.zeros(self.graph_all.number_of_edges())
        data_dicts.append(data_dict)
        return data_dicts

    def get_next_batch(self):
        return gn.utils_np.data_dicts_to_graphs_tuple(
            self.get_next_batch_data_dicts())


# Visualize synthetic dataset.
def visualize(dataset_name):
    dataset = DATASETS_MAP[dataset_name]
    data = []

    # Aggregated data.
    for i in range(100):
        g, features = dataset.graph_generator_fn()
        data.append(features)
    all_data = np.concatenate(data, axis=0)
    plt.plot(all_data[..., 0], all_data[..., 1], 'ro')
    plt.show()

    # Individual samples.
    plot_ind = 0
    for i in range(0, 100, 5):
        plot_ind += 1
        plt.subplot(4, 5, plot_ind)
        plt.plot(data[i][..., 0], data[i][..., 1], 'ro')
    plt.show()
