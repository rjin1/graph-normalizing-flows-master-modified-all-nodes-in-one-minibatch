from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from functools import partial
from itertools import permutations
import collections
import math
import os
import random
import warnings
# Only for linux.-Rui
import sys

sys.path.append("~/anaconda3/bin/python")

from sklearn import datasets
from sklearn.manifold import spectral_embedding
from sonnet.python.modules import base
from absl import flags
import graph_nets as gn
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import sonnet as snt
import tensorflow as tf
import tensorflow_probability as tfp
from scipy.io import savemat

# import tfplot
tfd = tfp.distributions
tfb = tfp.bijectors

from gnn import *
from grevnet_synthetic_data import *
from utils import *

warnings.filterwarnings("ignore")

# FMRI datasets and graph loading params.-Rui
flags.DEFINE_bool('load_fMRI_datasets', True,
                  'True: Load fMRI_datasets, False: Use original GNF synthetic datasets.')
flags.DEFINE_string('fMRI_file_name', 'fMRI_SMpure_quadra_syn.mat',
                    'Indicate the file name of fMRI dataset.')
flags.DEFINE_string('variable_topick_name', 'fMRI_quadra_syn',
                    'Indicate the to-pick variable name in the fMRI dataset.')
flags.DEFINE_string('graph_file_name', 'GridGraph1neighbors',
                    'Indicate the to-pick variable name in the fMRI dataset.')
flags.DEFINE_integer('ego_radius', 1,
                     'The ego graph radius.')

# Latent sources distribution.-Rui
flags.DEFINE_string('latent_distribution', 'GumbelPerElements',
                    'The distribution of latent sources, options are listed here'
                    'GumbelPerElements, with loc=0, scale=1'
                    'MultivariateNormal, with mean =0, cov = I.')
flags.DEFINE_bool('Use_predefined_params', False,
                  'Whether to use predefined distribution parameters')
flags.DEFINE_string('Params1_file_name', 'GumbelLocatParams.mat',
                    'The name of distribution parameter 1 file.')
flags.DEFINE_string('Params1_variable_name', 'locat_params',
                    'The name of distribution parameter 1 file.')
flags.DEFINE_string('Params2_file_name', 'GumbelScaleParams.mat',
                    'The name of distribution parameter 2 file.')
flags.DEFINE_string('Params2_variable_name', 'scale_params',
                    'The name of distribution parameter 2 file.')

# Validation.-Rui
flags.DEFINE_bool('validation', True,
                  'Whether to do validation.')
flags.DEFINE_string('ground_truth_file_name', 'SMpuresyn.mat',
                    'Indicate the file name of ground truth.')
flags.DEFINE_string('ground_truth_variable_name', 'SM',
                    'Indicate the variable name of ground truth.'
                    'It should be in the same ground truth file.')
flags.DEFINE_integer('validate_every_n_steps', 5,
                     'How often to perform validation.')

# Graph params.
flags.DEFINE_integer('node_embedding_dim', 16,
                     'Number of dimensions in node embeddings.')  # 2-Rui
flags.DEFINE_integer('min_nodes', 100, 'Min nodes in graph.')
flags.DEFINE_integer('max_nodes', 101, 'Max nodes in graph.')
flags.DEFINE_string('generate_graphs_fn', 'fc', 'Can be fc or isolated.')

# GRevNet params.
flags.DEFINE_integer(
    'num_coupling_layers', 3,
    'Number of coupling layers in GRevNet. Each coupling layers '
    'consists of applying F and then G, where F and G are GNNs.')  # 12-Rui
flags.DEFINE_bool('weight_sharing', False, '')

# GNN params.
flags.DEFINE_string('make_gnn_fn', 'independent',
                    'Function that makes a GNN of a specific type.')
flags.DEFINE_integer('gnn_num_layers', 2,
                     'Number of layers to use in MLP in GRevNet.')  # 5.-Rui
flags.DEFINE_integer('gnn_latent_dim', 2,
                     'Latent dim for GNN used in GRevNet.')  # 256.-Rui
flags.DEFINE_float('gnn_bias_init_stddev', 0.1,
                   'Used to initialize biases in GRevNet MLPs.')
flags.DEFINE_float(
    'gnn_l2_regularizer_weight', 0.1,
    'How much to weight the L2 regularizer for the GNN MLP weights.')
flags.DEFINE_float(
    'gnn_avg_then_mlp_epsilon', 1,
    'How much to weight the current node embeddings compared to the aggregate'
    'of its neighbors.')

# Self-attention params.
flags.DEFINE_integer('attn_kq_dim', 10, '')
flags.DEFINE_integer('attn_output_dim', 10, '')
flags.DEFINE_integer('attn_num_heads', 8, '')
flags.DEFINE_integer('attn_multi_proj_dim', 40, '')
flags.DEFINE_bool('attn_concat', True, '')
flags.DEFINE_bool('attn_residual', False, '')
flags.DEFINE_bool('attn_layer_norm', False, '')

# Training params.
flags.DEFINE_bool('use_lr_schedule', False, '')
flags.DEFINE_integer('lr_schedule_ramp_up', 1000, '')
flags.DEFINE_integer('lr_schedule_hold_steady', 2000, '')

flags.DEFINE_bool('smaller_stddev_samples', False, '')
flags.DEFINE_float('smaller_stddev', 0.5, '')

flags.DEFINE_bool('use_batch_norm', False,
                  'Whether to use batch norm during training.')
flags.DEFINE_string('dataset', 'mog_4',
                    'Which dataset to use')
flags.DEFINE_string('logdir', 'test_runs/test_grevnet',
                    'Where to write training files.')
flags.DEFINE_integer('num_train_iters', 150000,
                     'Number of steps to run training.')
flags.DEFINE_integer('save_every_n_steps', 1, 'How often to save model.')
flags.DEFINE_integer('log_every_n_steps', 1, 'How often to log model stats.')
flags.DEFINE_integer('write_imgs_every_n_steps', 1000,
                     'How often to log model stats.')
flags.DEFINE_integer('write_data_every_n_steps', 10000,
                     'How often to log model stats.')
flags.DEFINE_integer('max_checkpoints_to_keep', 5,
                     'Max model checkpoints to save.')
flags.DEFINE_integer('max_individual_samples', 4,
                     'Max individual samples to display in Tensorboard.')
flags.DEFINE_integer('random_seed', 5, '')
flags.DEFINE_bool('include_histograms', False, '')
flags.DEFINE_bool('add_optimizer_summaries', False, '')
flags.DEFINE_bool('add_weight_summaries', True, '')

# Optimizer params.
flags.DEFINE_float('lr', 5e-3, 'Learning rate.')
flags.DEFINE_bool('use_lr_decay', False, 'Whether to decay learning rate.')
flags.DEFINE_integer('lr_decay_steps', 10,
                     'How often to decay learning rate.')
flags.DEFINE_float('lr_decay_rate', 0.92, 'How much to decay learning rate.')
flags.DEFINE_float('adam_beta1', 0.9, 'Adam optimizer beta1.')
flags.DEFINE_float('adam_beta2', 0.9, 'Adam optimizer beta2.')
flags.DEFINE_float('adam_epsilon', 1e-08, 'Adam optimizer epsilon.')
flags.DEFINE_bool('clip_gradient_by_value', False,
                  'Whether to use value-based gradient clipping.')
flags.DEFINE_float('clip_gradient_value_lower', -1.0,
                   'Lower threshold for valued-based gradient clipping.')
flags.DEFINE_float('clip_gradient_value_upper', 5.0,
                   'Upper threshold for value-based gradient clipping.')
flags.DEFINE_bool('clip_gradient_by_norm', False,
                  'Whether to use norm-based gradient clipping.')
flags.DEFINE_float('clip_gradient_norm', 10.0,
                   'Value for norm-based gradient clipping.')
FLAGS = tf.app.flags.FLAGS

tf.logging.set_verbosity(tf.logging.DEBUG)
tf.random.set_random_seed(FLAGS.random_seed)
np.random.seed(FLAGS.random_seed)
random.seed(FLAGS.random_seed)
np.set_printoptions(suppress=True, formatter={'float': '{: 0.3f}'.format})

MAX_SEED = 2 ** 32 - 1
MIN_MAX_NODES = (FLAGS.min_nodes, FLAGS.max_nodes)
logdir_prefix = os.environ.get('MLPATH')
if not logdir_prefix:
    logdir_prefix = '.'
LOGDIR = os.path.join(logdir_prefix, FLAGS.logdir)

# imgs_dir = os.path.join(LOGDIR, 'imgs')
# pickle_dir = os.path.join(LOGDIR, 'pickle_files')
# os.makedirs(imgs_dir)
# os.makedirs(pickle_dir)

# DATASET = DATASETS_MAP[FLAGS.dataset]

# Set the parameters of the latent distribution
dis_param_1_tensor = tf.zeros(FLAGS.node_embedding_dim)
dis_param_2_tensor = tf.ones(FLAGS.node_embedding_dim)
# Use fMRI dataset or not.-Rui
if FLAGS.load_fMRI_datasets:
    DATASET = LOAD_FMRI(FLAGS.fMRI_file_name, FLAGS.variable_topick_name, FLAGS.graph_file_name, FLAGS.ego_radius)
    # Use predefined distribution parameters
    if FLAGS.Use_predefined_params:
        dis_param_1, dis_param_2 = DATASET.get_predefined_distribution_params(FLAGS.Params1_file_name,
                                                                              FLAGS.Params1_variable_name,
                                                                              FLAGS.Params2_file_name,
                                                                              FLAGS.Params2_variable_name)
        dis_param_1_tensor = tf.squeeze(tf.constant(dis_param_1))
        dis_param_2_tensor = tf.squeeze(tf.constant(dis_param_2))

    # For validation purpose.-Rui
    if FLAGS.validation:
        # Initialize to-show validation iterations.-Rui
        iter_val_toprint = 0
        # Load validation datsets, basic information
        val_ground_truth = DATASET.get_validate_ground_truth(FLAGS.ground_truth_file_name,
                                                             FLAGS.ground_truth_variable_name)
        num__val_components = np.shape(val_ground_truth)[1]
        # Mean Pearson correlation.-Rui
        MPC = np.array([], dtype=np.float32)
        # Mean square error.-Rui
        MSE = np.array([], dtype=np.float32)


def make_avg_concat_then_mlp_gnn():
    return avg_concat_then_mlp_gnn(
        partial(make_mlp_model, FLAGS.gnn_latent_dim,
                FLAGS.node_embedding_dim / 2, FLAGS.gnn_num_layers,
                tf.nn.leaky_relu, FLAGS.gnn_l2_regularizer_weight,
                FLAGS.gnn_bias_init_stddev))


def make_sum_concat_then_mlp_gnn():
    return sum_concat_then_mlp_gnn(
        partial(make_mlp_model, FLAGS.gnn_latent_dim,
                FLAGS.node_embedding_dim / 2, FLAGS.gnn_num_layers,
                tf.nn.leaky_relu, FLAGS.gnn_l2_regularizer_weight,
                FLAGS.gnn_bias_init_stddev))


def make_gru_gnn():
    gru_block = GRUBlock(FLAGS.node_embedding_dim / 2)
    return NodeBlockGNN(gru_block)


def make_avg_then_mlp_gnn():
    return avg_then_mlp_gnn(
        partial(make_mlp_model, FLAGS.gnn_latent_dim,
                FLAGS.node_embedding_dim / 2, FLAGS.gnn_num_layers,
                tf.nn.leaky_relu, FLAGS.gnn_l2_regularizer_weight,
                FLAGS.gnn_bias_init_stddev), FLAGS.gnn_avg_then_mlp_epsilon)


def make_independent_gnn():
    return gn.modules.GraphIndependent(node_model_fn=partial(
        make_mlp_model, FLAGS.gnn_latent_dim, FLAGS.node_embedding_dim /
                                              2, FLAGS.gnn_num_layers, tf.nn.leaky_relu,
        FLAGS.gnn_l2_regularizer_weight, FLAGS.gnn_bias_init_stddev))


def make_dm_self_attn_gnn():
    return dm_self_attn_gnn(
        kq_dim=FLAGS.attn_kq_dim,
        v_dim=FLAGS.attn_v_dim,
        make_mlp_fn=partial(make_mlp_model, FLAGS.gnn_latent_dim,
                            FLAGS.node_embedding_dim / 2, FLAGS.gnn_num_layers,
                            tf.nn.relu, FLAGS.gnn_l2_regularizer_weight,
                            FLAGS.gnn_bias_init_stddev),
        num_heads=FLAGS.attn_num_heads,
        concat_heads_output_dim=FLAGS.attn_concat_heads_output_dim,
        concat=FLAGS.attn_concat,
        residual=FLAGS.attn_residual,
        layer_norm=FLAGS.attn_layer_norm)


def make_self_attn_gnn():
    return self_attn_gnn(kq_dim=FLAGS.attn_kq_dim,
                         v_dim=FLAGS.attn_v_dim,
                         make_mlp_fn=partial(make_mlp_model,
                                             FLAGS.gnn_latent_dim,
                                             FLAGS.node_embedding_dim / 2,
                                             FLAGS.gnn_num_layers, tf.nn.relu,
                                             FLAGS.gnn_l2_regularizer_weight,
                                             FLAGS.gnn_bias_init_stddev),
                         kq_dim_division=True)


def make_multihead_self_attn_gnn():
    return multihead_self_attn_gnn(
        kq_dim=FLAGS.attn_kq_dim,
        v_dim=FLAGS.attn_v_dim,
        concat_heads_output_dim=FLAGS.attn_concat_heads_output_dim,
        make_mlp_fn=partial(make_mlp_model, FLAGS.gnn_latent_dim,
                            FLAGS.node_embedding_dim / 2, FLAGS.gnn_num_layers,
                            tf.nn.relu, FLAGS.gnn_l2_regularizer_weight,
                            FLAGS.gnn_bias_init_stddev),
        num_heads=FLAGS.attn_num_heads,
        kq_dim_division=True)


def make_latest_self_attn_gnn():
    return latest_self_attention_gnn(
        kq_dim=FLAGS.attn_kq_dim,
        v_dim=FLAGS.attn_v_dim,
        concat_heads_output_dim=FLAGS.attn_concat_heads_output_dim,
        make_mlp_fn=partial(make_mlp_model, FLAGS.gnn_latent_dim,
                            FLAGS.node_embedding_dim / 2, FLAGS.gnn_num_layers,
                            tf.nn.relu, FLAGS.gnn_l2_regularizer_weight,
                            FLAGS.gnn_bias_init_stddev),
        train_batch_size=FLAGS.train_batch_size,
        max_n_node=6,
        num_heads=FLAGS.attn_num_heads,
        kq_dim_division=True)


MAKE_GNN_FN_MAP = {
    'gru': make_gru_gnn,
    'avg_then_mlp': make_avg_then_mlp_gnn,
    'avg_concat_then_mlp': make_avg_concat_then_mlp_gnn,
    'sum_concat_then_mlp': make_sum_concat_then_mlp_gnn,
    'independent': make_independent_gnn,
    'dm_self_attn': make_dm_self_attn_gnn,
    'self_attn': make_self_attn_gnn,
    'multihead_self_attn': make_multihead_self_attn_gnn,
    'latest_self_attn': make_latest_self_attn_gnn,
}
MAKE_GNN_FN = MAKE_GNN_FN_MAP[FLAGS.make_gnn_fn]

# Data placeholders.
data_dict = DATASET.get_next_batch_data_dicts()
graph_phs = gn.utils_tf.placeholders_from_data_dicts(data_dict)

grevnet = GRevNet(
    MAKE_GNN_FN,
    FLAGS.num_coupling_layers,
    FLAGS.node_embedding_dim,
    use_batch_norm=FLAGS.use_batch_norm,
    weight_sharing=FLAGS.weight_sharing)

# Log prob(z).
grevnet_reverse_output, log_det_jacobian = grevnet(graph_phs, inverse=True)

# Pick distributions.- Rui
DISTRIBUTION_MAP = {
    'GumbelPerElements': tfd.Gumbel(dis_param_1_tensor, dis_param_2_tensor),
    'MultivariateNormal': tfd.MultivariateNormalDiag(dis_param_1_tensor,
                                                     dis_param_2_tensor),
}
distribution_latent = DISTRIBUTION_MAP[FLAGS.latent_distribution]
log_prob_zs = tf.reduce_sum(distribution_latent.log_prob(grevnet_reverse_output.nodes))
log_prob_xs = log_prob_zs + log_det_jacobian
total_loss = -1 * log_prob_xs

num_nodes = tf.cast(tf.reduce_sum(graph_phs.n_node), tf.float32)
loss_per_node = total_loss / num_nodes
log_prob_xs_per_node = log_prob_xs / num_nodes
log_prob_zs_per_node = log_prob_zs / num_nodes
log_det_jacobian_per_node = log_det_jacobian / num_nodes

# Reconstruct observations.-Rui
# mvn = tfd.MultivariateNormalDiag(tf.zeros(FLAGS.node_embedding_dim),
#                                  tf.ones(FLAGS.node_embedding_dim))
#
# sample = mvn.sample(sample_shape=(tf.reduce_sum(graph_phs.n_node, )))
# sample_graph_phs = graph_phs.replace(nodes=sample)
# sample_log_prob = mvn.log_prob(sample)
# grevnet_top = grevnet(sample_graph_phs, inverse=False)
# grevnet_top_nodes = grevnet_top.nodes

# if FLAGS.smaller_stddev_samples:
#    smaller_mvn = tfd.MultivariateNormalDiag(
#        tf.zeros(FLAGS.node_embedding_dim),
#        tf.zeros(FLAGS.node_embedding_dim) + FLAGS.smaller_stddev)
#    smaller_sample = smaller_mvn.sample(
#        sample_shape=(tf.reduce_sum(graph_phs.n_node, )))
#    smaller_sample_graph_phs = graph_phs.replace(nodes=smaller_sample)
#    smaller_grevnet_top = grevnet(smaller_sample_graph_phs, inverse=False)
#    smaller_grevnet_top_nodes = smaller_grevnet_top.nodes

# Visualize tensors.
# tfplot.summary.plot("training_data", plot_data, [graph_phs.nodes])
# tfplot.summary.plot("single_training_example", plot_data,
#                    [single_training_graph.nodes])
# tfplot.summary.plot("zs", plot_data, [grevnet_reverse_output.nodes])
# tfplot.summary.plot("generated_sample", plot_data, [grevnet_top_nodes])

# if FLAGS.smaller_stddev_samples:
#    tfplot.summary.plot("smaller_variance_sample", plot_data,
#                        [smaller_grevnet_top_nodes])

# for i in range(min(FLAGS.train_batch_size, FLAGS.max_individual_samples)):
#    tfplot.summary.plot("single_generated_sample_{}".format(i), plot_data,
#                        [gn.utils_tf.get_graph(grevnet_top, i).nodes])
#
# if FLAGS.smaller_stddev_samples:
#    for i in range(min(FLAGS.train_batch_size, FLAGS.max_individual_samples)):
#        tfplot.summary.plot(
#            "smaller_single_generated_sample_{}".format(i), plot_data,
#            [gn.utils_tf.get_graph(smaller_grevnet_top, i).nodes])

# Optimizer.
global_step = tf.Variable(0, trainable=False, name='global_step')
decaying_learning_rate = tf.train.exponential_decay(
    learning_rate=FLAGS.lr,
    global_step=global_step,
    decay_steps=FLAGS.lr_decay_steps,
    decay_rate=FLAGS.lr_decay_rate,
    staircase=False)
learning_rate = decaying_learning_rate if FLAGS.use_lr_decay else FLAGS.lr

learning_rate_placeholder = tf.placeholder(
    tf.float32, [], name='learning_rate')
optimizer = tf.train.AdamOptimizer(
    learning_rate=learning_rate_placeholder
    if FLAGS.use_lr_schedule else learning_rate,
    beta1=FLAGS.adam_beta1,
    beta2=FLAGS.adam_beta2,
    epsilon=FLAGS.adam_epsilon)

# RMSprop
# optimizer = tf.train.RMSPropOptimizer(
#     learning_rate=learning_rate_placeholder
#     if FLAGS.use_lr_schedule else learning_rate)

with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
    grads_and_vars = optimizer.compute_gradients(total_loss)
    if FLAGS.clip_gradient_by_value:
        grads_and_vars = [
            (tf.clip_by_value(grad, FLAGS.clip_gradient_value_lower,
                              FLAGS.clip_gradient_value_upper), var)
            for grad, var in grads_and_vars
        ]

    if FLAGS.clip_gradient_by_norm:
        grads_and_vars = [(tf.clip_by_norm(grad,
                                           FLAGS.clip_gradient_norm), var)
                          for grad, var in grads_and_vars]

    step_op = optimizer.apply_gradients(grads_and_vars,
                                        global_step=global_step)

saver = tf.train.Saver(max_to_keep=FLAGS.max_checkpoints_to_keep)
# config = tf.ConfigProto()
# config.gpu_options.allow_growth = True
# sess = reset_sess(config)
sess = reset_sess()

# Training loss stats.
tf.summary.scalar('total_loss/loss', total_loss)
tf.summary.scalar('total_loss/log_prob_zs', log_prob_zs)
tf.summary.scalar('total_loss/log_det_jacobian', log_det_jacobian)
tf.summary.scalar('total_loss/log_prob_xs', log_prob_xs)
tf.summary.scalar('per_node_loss/loss', loss_per_node)
tf.summary.scalar('per_node_loss/log_prob_zs', log_prob_zs_per_node)
tf.summary.scalar('per_node_loss/log_det_jacobian', log_det_jacobian_per_node)
tf.summary.scalar('per_node_loss/log_prob_xs', log_prob_xs_per_node)

# Optimization stats.
if FLAGS.add_optimizer_summaries:
    for (g, v) in grads_and_vars:
        if g is not None:
            tf.summary.scalar("grads/{}/norm".format(v.name), tf.norm(g))
            tf.summary.scalar("adam_moment_1/{}/norm".format(v.name),
                              tf.norm(optimizer.get_slot(v, 'm')))
            tf.summary.scalar("adam_moment_2/{}/norm".format(v.name),
                              tf.norm(optimizer.get_slot(v, 'v')))
            if FLAGS.include_histograms:
                tf.summary.histogram("grads/{}".format(v.name), g)
                tf.summary.histogram("adam_moment_1/{}".format(v.name),
                                     optimizer.get_slot(v, 'm'))
                tf.summary.histogram("adam_moment_2/{}".format(v.name),
                                     optimizer.get_slot(v, 'v'))

if FLAGS.add_weight_summaries:
    for v in tf.trainable_variables():
        tf.summary.scalar("weights/{}/norm".format(v.name), tf.norm(v))
        if FLAGS.include_histograms:
            tf.summary.histogram("weights/{}".format(v.name), v)

merged = tf.summary.merge_all()
writer = tf.summary.FileWriter(LOGDIR, sess.graph)  # Too slow, comment it temporally-Rui

# flags_map = tf.app.flags.FLAGS.flag_values_dict()
# with open(os.path.join(LOGDIR, 'desc.txt'), 'w') as f:
#     for (k, v) in flags_map.items():
#         f.write("{}: {}\n".format(k, str(v)))

values_map = {
    "graph_phs": graph_phs,
    "grevnet_bottom": grevnet_reverse_output.nodes,
    "log_det_jacobian": log_det_jacobian,
    "log_prob_zs": log_prob_zs,
    "total_loss": total_loss,
    "loss_per_node": loss_per_node,
    "step_op": step_op,
    "merged": merged,
    # We don't need sample reverse generation.-Rui
    # "grevnet_top_nodes": grevnet_top_nodes,
    # "sample_log_prob": sample_log_prob,
}

# We don't need sample reverse generation.-Rui
# for i in range(min(FLAGS.train_batch_size, FLAGS.max_individual_samples)):
#     values_map["generated_sample_{}".format(i)] = gn.utils_tf.get_graph(
#         grevnet_top, i).nodes

for iteration in range(FLAGS.num_train_iters + 1):
    feed_dict = {}
    # Need iteration as input to load training fMRI data.-Rui
    # feed_dict[graph_phs] = DATASET.get_next_batch(FLAGS.train_batch_size)
    feed_dict[graph_phs] = DATASET.get_next_batch()
    if FLAGS.use_lr_schedule:
        feed_dict[learning_rate_placeholder] = get_learning_rate(
            iteration, FLAGS.lr, FLAGS.lr_schedule_ramp_up,
            FLAGS.lr_schedule_hold_steady)
    train_values = sess.run(values_map, feed_dict=feed_dict)

    # Validation.-Rui
    if FLAGS.validation:
        if iteration % FLAGS.validate_every_n_steps == 0:
            val_z = np.empty((0, FLAGS.node_embedding_dim), np.float32)
            feed_dict_val = {graph_phs: DATASET.get_next_batch()}
            val_z = np.append(val_z, sess.run(values_map["grevnet_bottom"], feed_dict=feed_dict_val), axis=0)
            # Mean Pearson correlation
            corr = np.abs(np.corrcoef(val_z.transpose(),
                                      val_ground_truth.transpose()))[:num__val_components, num__val_components:]
            MPC = np.append(MPC, np.mean(corr.diagonal()))
            # Mean square error
            mse = np.mean(np.square(val_z.transpose() - val_ground_truth.transpose()))
            MSE = np.append(MSE, mse)
            iter_val_toprint = iter_val_toprint + 1
            # Saving
            savemat('MPC.mat', {'MPC': MPC})
            savemat('MSE.mat', {'MSE': MSE})
            if MPC[iter_val_toprint - 1] > 0.5438:
                savemat('SM_RealNVP_estimates.mat', {'SM_estimates': val_z})

    if iteration % FLAGS.save_every_n_steps == 0:
        saver.save(sess,
                   os.path.join(LOGDIR, 'model'),
                   global_step,
                   write_meta_graph=False)

    if iteration % FLAGS.log_every_n_steps == 0:
        writer.add_summary(train_values["merged"], iteration)  # too slow, temporally comment it-Rui

        print('Training: iter = %.0f, TotalLoss = %.1f, LogDetJaco = %.1f, LogZs = %.1f'
              ' | Validation: iter = %.0f, MPC = %.4f, MSE = %.4f' %
              (iteration, train_values["total_loss"], train_values["log_det_jacobian"],
               train_values["log_prob_zs"], iter_val_toprint, MPC[iter_val_toprint - 1], MSE[iter_val_toprint - 1]))

        # print("*" * 50)
        # print("iteration num: {}".format(iteration))
        # print("total_loss: {}".format(train_values["total_loss"]))
        # print("loss per node: {}".format(train_values["loss_per_node"]))
        # # print("log det jacobian: {}".format(train_values["log_det_jacobian"]))
        # print("grevnet bottom: {}".format(train_values["grevnet_bottom"]))
        # print("original mean {} std dev {}".format(
        #     np.mean(train_values["graph_phs"].nodes, 0),
        #     np.std(train_values["graph_phs"].nodes, 0)))
        # print("transformed mean {} std dev {}".format(
        #     np.mean(train_values["grevnet_bottom"], 0),
        #     np.std(train_values["grevnet_bottom"], 0)))

    # if iteration % FLAGS.write_imgs_every_n_steps == 0:
    #     plot_data(train_values["grevnet_bottom"],
    #               os.path.join(imgs_dir, 'iter_{}_zs.png'.format(iteration)))
    #     plot_data(
    #         train_values["grevnet_top_nodes"],
    #         os.path.join(imgs_dir,
    #                      'iter_{}_aggregated_x.png'.format(iteration)))
    #     for i in range(
    #             min(FLAGS.train_batch_size, FLAGS.max_individual_samples)):
    #         name = "generated_sample_{}".format(i)
    #         outname = "iter_{}_generated_sample_{}.png".format(iteration, i)
    #         plot_data(train_values[name], os.path.join(imgs_dir, outname))

    # if iteration % FLAGS.write_data_every_n_steps == 0:
    #    pickle.dump(
    #        train_values["grevnet_top_nodes"],
    #        open(
    #            os.path.join(LOGDIR,
    #                         "pickle_files/samples_{}.dat".format(iteration)),
    #            'wb'))
    #    pickle.dump(
    #        train_values["graph_phs"].nodes,
    #        open(
    #            os.path.join(
    #                LOGDIR,
    #                "pickle_files/training_data_{}.dat".format(iteration)),
    #            'wb'))
    #    pickle.dump(
    #        train_values["grevnet_bottom"],
    #        open(
    #            os.path.join(LOGDIR,
    #                         "pickle_files/zs_{}.dat".format(iteration)),
    #            'wb'))
    #    pickle.dump(
    #        train_values["sample_log_prob"],
    #        open(
    #            os.path.join(
    #                LOGDIR,
    #                "pickle_files/sample_log_prob_{}.dat".format(iteration)),
    #            'wb'))
