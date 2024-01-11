import tensorflow as tf
import numpy as np
import math
import sys
import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, '../utils'))
sys.path.append(os.path.join(BASE_DIR, '../../utils'))

ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'utils'))
sys.path.append(os.path.join(ROOT_DIR, 'tf_ops/sampling'))
sys.path.append(os.path.join(ROOT_DIR, 'tf_ops/grouping'))
sys.path.append(os.path.join(ROOT_DIR, 'tf_ops/3d_interpolation'))
from tf_sampling import farthest_point_sample, gather_point
from tf_grouping import query_ball_point, group_point, knn_point
from tf_interpolate import three_nn, three_interpolate


import tf_util
from pointnet_util import *
import extra_loss
import pan_util


def placeholder_inputs(batch_size, num_point):
	pointclouds_pl = tf.placeholder(tf.float32, shape=(batch_size, num_point, 9))
	labels_pl = tf.placeholder(tf.int32, shape=(batch_size, num_point))
	return pointclouds_pl, labels_pl


def get_model(point_input, labels_pl, is_training, bn_decay=None, coarse_flag=0, 
	mmd_flag=0, pfs_flag=False, fully_concate=False):
	""" Seg, input is BxNx9, output BxNx13 """
	batch_size = point_input.get_shape()[0].value
	num_point1 = point_input.get_shape()[1].value
	num_point2 = int(np.floor(num_point1 / 4.0))
	num_point3 = int(np.floor(num_point2 / 4.0))
	end_points = {}
	k = 10
	pk = 10
	p1_1 = 0
	p1_2 = 0.4
	p2_1 = 0
	p2_2 = 1.6

	activation_fn = tf.nn.relu
	point_cloud1 = tf.slice(point_input, [0, 0, 0], [-1, -1, 3])
	hie_matrix1 = tf.math.maximum(tf.sqrt(tf_util.pairwise_distance(point_cloud1)), 1e-20)


	net1, idx = pan_util.feature_extractor(point_input,
										   block='inception',
										   n_blocks=3,
										   growth_rate=64,
										   k=16, d=1,
										   scope='feature_extraction1', is_training=is_training,
										   use_global_pooling=False,
										   bn_decay=None)


	net1 = tf.squeeze(net1)
	dist_threshold1 = False
	net, p1_idx, _, point_cloud2 = pan_util.edge_preserve_graph_sampling(net1, point_cloud1, num_point2, hie_matrix1,
		dist_threshold1, pk, 1, p1_1, p1_2, pfs_flag, atrous_flag=False)
	net = tf.squeeze(net)
	hie_matrix2 = tf.math.maximum(tf.sqrt(tf_util.pairwise_distance(point_cloud2)), 1e-20)
	net2, idx = pan_util.feature_extractor(net,
										   block='inception',
										   n_blocks=3,
										   growth_rate=128, k=16, d=1,
										   use_global_pooling=False,
										   scope='feature_extraction2', is_training=is_training,
										   bn_decay=None)

	net2 = tf.squeeze(net2)
	dist_threshold2 = False
	net, p2_idx, _, point_cloud3 = pan_util.edge_preserve_graph_sampling(net2, point_cloud2, num_point3, hie_matrix2,
		dist_threshold2, pk, 1, p2_1, p2_2, pfs_flag, atrous_flag=False)
	net = tf.squeeze(net)
	net3, idx = pan_util.feature_extractor(net,
										   block='inception',
										   n_blocks=3,
										   growth_rate=256, k=16, d=1,
										   use_global_pooling=False,
										   scope='feature_extraction3', is_training=is_training,
										   bn_decay=None)
	net = tf_util.conv2d(net3, 1024, [1,1],
					   padding='VALID', stride=[1,1], activation_fn=activation_fn,
					   use_bn=True, is_training=is_training,
					   scope='encoder', bn_decay=bn_decay)
	net = tf.reduce_max(net, axis=1, keepdims=True)
	net = tf.reshape(net, [batch_size, -1])
	net = tf_util.fully_connected(net, 512, bn=True, is_training=is_training, activation_fn=None,
								scope='rg1', bn_decay=bn_decay)
	if mmd_flag > 0:
		end_points['embedding'] = net
	else:
		end_points['embedding'] = None
	net = tf_util.dropout(net, keep_prob=0.5, is_training=is_training, scope='rgdp1')
	global_feature_size = 1024
	net = tf_util.fully_connected(net, global_feature_size, bn=True, is_training=is_training, activation_fn=activation_fn,
								scope='rg2', bn_decay=bn_decay)
	net = tf_util.dropout(net, keep_prob=0.5, is_training=is_training, scope='rgdp2')
	net = tf.reshape(net, [batch_size, 1, 1, global_feature_size])
	net = tf.tile(net, [1, num_point3, 1, 1])
	net = tf.concat([net, net3], axis=-1)
	net = tf_util.conv2d(net, 512, [1,1],
					   padding='VALID', stride=[1,1], activation_fn=activation_fn,
					   use_bn=True, is_training=is_training,
					   scope='decoder', bn_decay=bn_decay)
	if coarse_flag > 0:
		coarse_net = tf.squeeze(net)
		coarse_net = tf_util.conv1d(coarse_net, 128, 1, padding='VALID', bn=True, activation_fn=activation_fn,
			is_training=is_training, scope='coarse_fc1', bn_decay=bn_decay)
		coarse_net = tf_util.dropout(coarse_net, keep_prob=0.5, is_training=is_training, scope='cdp1')
		coarse_net = tf_util.conv1d(coarse_net, 50, 1, padding='VALID', activation_fn=None, scope='coarse_fc2')

		coarse_labels_pl = tf_util.gather_labels(labels_pl, p1_idx)
		coarse_labels_pl = tf_util.gather_labels(coarse_labels_pl, p2_idx)

		end_points['coarse_pred'] = coarse_net
		end_points['coarse_label'] = coarse_labels_pl

	else:
		end_points['coarse_pred'] = None
		end_points['coarse_label'] = None

	net4 = pan_util.edge_preserve_graph_unsampling(net, point_cloud2, num_point3, hie_matrix2,
									dist_threshold2, pk, 1, p2_1, p2_2, pfs_flag, atrous_flag=False)
	coord3 = pan_util.feature_extractor(net4,
										   block='inception',
										   n_blocks=3,
										   growth_rate=256, k=16, d=1,
										   use_global_pooling=False,
										   scope='feature_extraction3', is_training=is_training,
										   bn_decay=None)
	net4 = coord3
	net5 = pan_util.edge_preserve_graph_unsampling(net4, point_cloud2, num_point3, hie_matrix2,
									dist_threshold2, pk, 1, p2_1, p2_2, pfs_flag, atrous_flag=False)

	coord2 = pan_util.feature_extractor(net5,
										   block='inception',
										   n_blocks=3,
										   growth_rate=128, k=16, d=1,
										   use_global_pooling=False,
										   scope='feature_extraction3', is_training=is_training,
										   bn_decay=None)
	net5 = coord2
	coord1 = pan_util.feature_extractor(net5,
										   block='inception',
										   n_blocks=3,
										   growth_rate=128, k=16, d=1,
										   use_global_pooling=False,
										   scope='feature_extraction3', is_training=is_training,
										   bn_decay=None)
	net6 = tf.squeeze(coord1)
	net = tf_util.conv1d(net6, 128, 1, padding='VALID', bn=True, activation_fn=activation_fn,
		is_training=is_training, scope='fc1', bn_decay=bn_decay)
	
	net = tf_util.dropout(net, keep_prob=0.5, is_training=is_training, scope='dp1')
	net = tf_util.conv1d(net, 13, 1, padding='VALID', activation_fn=None, scope='fc2')

	return net, end_points

def get_loss(pred, label, end_points, coarse_flag, mmd_flag):
	""" pred: BxNxC,
		label: BxN, 12*4096"""

	print('get loss de  pred is %s' % (pred.shape))
	print('get loss de  label is %s' % (label.shape))

	seg_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=pred, labels=label)

	seg_loss = tf.reduce_mean(seg_loss)
	tf.summary.scalar('seg loss', seg_loss)

	loss = seg_loss

	if coarse_flag > 0:
		coarse_seg_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=end_points['coarse_pred'], labels=end_points['coarse_label'])


		coarse_seg_loss = tf.reduce_mean(coarse_seg_loss)
		coarse_seg_loss = coarse_seg_loss * coarse_flag
		tf.summary.scalar('coarse seg loss', coarse_seg_loss)

		loss = loss + coarse_seg_loss

	if mmd_flag > 0:
		batch_size = end_points['embedding'].get_shape()[0].value
		feature_size = end_points['embedding'].get_shape()[1].value

		true_samples = tf.random_normal(tf.stack([batch_size, feature_size]))
		mmd_loss = extra_loss.compute_mmd(end_points['embedding'], true_samples)
		mmd_loss = mmd_loss * mmd_flag
		tf.summary.scalar('mmd loss', mmd_loss)

		loss = loss + mmd_loss

	tf.add_to_collection('losses', loss)
	return loss



if __name__=='__main__':
	with tf.Graph().as_default():
		inputs = tf.zeros((32,2048,6))
		cls_labels = tf.zeros((32),dtype=tf.int32)
		output, ep = get_model(inputs, cls_labels, tf.constant(True))
		print(output)
