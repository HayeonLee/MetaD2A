###########################################################################################
# Copyright (c) Hayeon Lee, Eunyoung Hyung [GitHub MetaD2A], 2021
# Rapid Neural Architecture Search by Learning to Generate Graphs from Datasets, ICLR 2021
###########################################################################################
from __future__ import print_function
import os
import time
import igraph
import random
import numpy as np
import scipy.stats
import argparse
import torch


def load_graph_config(graph_data_name, nvt, data_path):
	if graph_data_name is not 'ofa_mbv3':
		raise NotImplementedError(graph_data_name)
	max_n=20
	graph_config = {}
	graph_config['num_vertex_type'] = nvt + 2  # original types + start/end types
	graph_config['max_n'] = max_n + 2  # maximum number of nodes
	graph_config['START_TYPE'] = 0  # predefined start vertex type
	graph_config['END_TYPE'] = 1  # predefined end vertex type
	
	return graph_config


type_dict = {'2-3-3': 0, '2-3-4': 1, '2-3-6': 2,
             '2-5-3': 3, '2-5-4': 4, '2-5-6': 5,
             '2-7-3': 6, '2-7-4': 7, '2-7-6': 8,
             '3-3-3': 9, '3-3-4': 10, '3-3-6': 11,
             '3-5-3': 12, '3-5-4': 13, '3-5-6': 14,
             '3-7-3': 15, '3-7-4': 16, '3-7-6': 17,
             '4-3-3': 18, '4-3-4': 19, '4-3-6': 20,
             '4-5-3': 21, '4-5-4': 22, '4-5-6': 23,
             '4-7-3': 24, '4-7-4': 25, '4-7-6': 26}

edge_dict = {2: (2, 3, 3), 3: (2, 3, 4), 4: (2, 3, 6),
             5: (2, 5, 3), 6: (2, 5, 4), 7: (2, 5, 6),
             8: (2, 7, 3), 9: (2, 7, 4), 10: (2, 7, 6),
             11: (3, 3, 3), 12: (3, 3, 4), 13: (3, 3, 6),
             14: (3, 5, 3), 15: (3, 5, 4), 16: (3, 5, 6),
             17: (3, 7, 3), 18: (3, 7, 4), 19: (3, 7, 6),
             20: (4, 3, 3), 21: (4, 3, 4), 22: (4, 3, 6),
             23: (4, 5, 3), 24: (4, 5, 4), 25: (4, 5, 6),
             26: (4, 7, 3), 27: (4, 7, 4), 28: (4, 7, 6)}


def decode_ofa_mbv3_to_igraph(matrix):
	# 5 stages, 4 layers for each stage
	# d: 2, 3, 4
	# e: 3, 4, 6
	# k: 3, 5, 7
	
	# stage_depth to one hot
	num_stage = 5
	num_layer = 4
	
	node_types = torch.zeros(num_stage * num_layer)
	
	d = []
	for i in range(num_stage):
		for j in range(num_layer):
			d.append(matrix['d'][i])
	for i, (ks, e, d) in enumerate(zip(
			matrix['ks'], matrix['e'], d)):
		node_types[i] = type_dict[f'{d}-{ks}-{e}']
	
	n = num_stage * num_layer
	g = igraph.Graph(directed=True)
	g.add_vertices(n + 2)  # + in/out nodes
	g.vs[0]['type'] = 0
	for i, v in enumerate(node_types):
		g.vs[i + 1]['type'] = v + 2  # in node: 0, out node: 1
		g.add_edge(i, i + 1)
	g.vs[n + 1]['type'] = 1
	g.add_edge(n, n + 1)
	return g, n + 2


def is_valid_ofa_mbv3(g, START_TYPE=0, END_TYPE=1):
	# first need to be a valid DAG computation graph
	msg = ''
	res = is_valid_DAG(g, START_TYPE, END_TYPE)
	# in addition, node i must connect to node i+1
	res = res and len(g.vs['type']) == 22
	if not res:
		return res
	msg += '{} ({}) '.format(g.vs['type'][1:-1], len(g.vs['type']))
	
	for i in range(5):
		if ((g.vs['type'][1:-1][i * 4]) - 2) // 9 == 0:
			for j in range(1, 4):
				res = res and ((g.vs['type'][1:-1][i * 4 + j]) - 2) // 9 == 0
		
		elif ((g.vs['type'][1:-1][i * 4]) - 2) // 9 == 1:
			for j in range(1, 4):
				res = res and ((g.vs['type'][1:-1][i * 4 + j]) - 2) // 9 == 1
		
		elif ((g.vs['type'][1:-1][i * 4]) - 2) // 9 == 2:
			for j in range(1, 4):
				res = res and ((g.vs['type'][1:-1][i * 4 + j]) - 2) // 9 == 2
		else:
			raise ValueError
	return res


def is_valid_DAG(g, START_TYPE=0, END_TYPE=1):
	res = g.is_dag()
	n_start, n_end = 0, 0
	for v in g.vs:
		if v['type'] == START_TYPE:
			n_start += 1
		elif v['type'] == END_TYPE:
			n_end += 1
		if v.indegree() == 0 and v['type'] != START_TYPE:
			return False
		if v.outdegree() == 0 and v['type'] != END_TYPE:
			return False
	return res and n_start == 1 and n_end == 1


def decode_igraph_to_ofa_mbv3(g):
	if not is_valid_ofa_mbv3(g, START_TYPE=0, END_TYPE=1):
		return None
	
	graph = {'ks': [], 'e': [], 'd': [4, 4, 4, 4, 4]}
	for i, edge_type in enumerate(g.vs['type'][1:-1]):
		edge_type = int(edge_type)
		d, ks, e = edge_dict[edge_type]
		graph['ks'].append(ks)
		graph['e'].append(e)
		graph['d'][i // 4] = d
	return graph


class Accumulator():
	def __init__(self, *args):
		self.args = args
		self.argdict = {}
		for i, arg in enumerate(args):
			self.argdict[arg] = i
		self.sums = [0] * len(args)
		self.cnt = 0
	
	def accum(self, val):
		val = [val] if type(val) is not list else val
		val = [v for v in val if v is not None]
		assert (len(val) == len(self.args))
		for i in range(len(val)):
			if torch.is_tensor(val[i]):
				val[i] = val[i].item()
			self.sums[i] += val[i]
		self.cnt += 1
	
	def clear(self):
		self.sums = [0] * len(self.args)
		self.cnt = 0
	
	def get(self, arg, avg=True):
		i = self.argdict.get(arg, -1)
		assert (i is not -1)
		if avg:
			return self.sums[i] / (self.cnt + 1e-8)
		else:
			return self.sums[i]
	
	def print_(self, header=None, time=None,
	           logfile=None, do_not_print=[], as_int=[],
	           avg=True):
		msg = '' if header is None else header + ': '
		if time is not None:
			msg += ('(%.3f secs), ' % time)
		
		args = [arg for arg in self.args if arg not in do_not_print]
		arg = []
		for arg in args:
			val = self.sums[self.argdict[arg]]
			if avg:
				val /= (self.cnt + 1e-8)
			if arg in as_int:
				msg += ('%s %d, ' % (arg, int(val)))
			else:
				msg += ('%s %.4f, ' % (arg, val))
		print(msg)
		
		if logfile is not None:
			logfile.write(msg + '\n')
			logfile.flush()
	
	def add_scalars(self, summary, header=None, tag_scalar=None,
	                step=None, avg=True, args=None):
		for arg in self.args:
			val = self.sums[self.argdict[arg]]
			if avg:
				val /= (self.cnt + 1e-8)
			else:
				val = val
			tag = f'{header}/{arg}' if header is not None else arg
			if tag_scalar is not None:
				summary.add_scalars(main_tag=tag,
				                    tag_scalar_dict={tag_scalar: val},
				                    global_step=step)
			else:
				summary.add_scalar(tag=tag,
				                   scalar_value=val,
				                   global_step=step)


class Log:
	def __init__(self, args, logf, summary=None):
		self.args = args
		self.logf = logf
		self.summary = summary
		self.stime = time.time()
		self.ep_sttime = None
	
	def print(self, logger, epoch, tag=None, avg=True):
		if tag == 'train':
			ct = time.time() - self.ep_sttime
			tt = time.time() - self.stime
			msg = f'[total {tt:6.2f}s (ep {ct:6.2f}s)] epoch {epoch:3d}'
			print(msg)
			self.logf.write(msg + '\n')
		logger.print_(header=tag, logfile=self.logf, avg=avg)
		
		if self.summary is not None:
			logger.add_scalars(
				self.summary, header=tag, step=epoch, avg=avg)
		logger.clear()
	
	def print_args(self):
		argdict = vars(self.args)
		print(argdict)
		for k, v in argdict.items():
			self.logf.write(k + ': ' + str(v) + '\n')
		self.logf.write('\n')
	
	def set_time(self):
		self.stime = time.time()
	
	def save_time_log(self):
		ct = time.time() - self.stime
		msg = f'({ct:6.2f}s) meta-training phase done'
		print(msg)
		self.logf.write(msg + '\n')
	
	def print_pred_log(self, loss, corr, tag, epoch=None, max_corr_dict=None):
		if tag == 'train':
			ct = time.time() - self.ep_sttime
			tt = time.time() - self.stime
			msg = f'[total {tt:6.2f}s (ep {ct:6.2f}s)] epoch {epoch:3d}'
			self.logf.write(msg + '\n');
			print(msg);
			self.logf.flush()
		# msg = f'ep {epoch:3d} ep time {time.time() - ep_sttime:8.2f} '
		# msg += f'time {time.time() - sttime:6.2f} '
		if max_corr_dict is not None:
			max_corr = max_corr_dict['corr']
			max_loss = max_corr_dict['loss']
			msg = f'{tag}: loss {loss:.6f} ({max_loss:.6f}) '
			msg += f'corr {corr:.4f} ({max_corr:.4f})'
		else:
			msg = f'{tag}: loss {loss:.6f} corr {corr:.4f}'
		self.logf.write(msg + '\n');
		print(msg);
		self.logf.flush()
	
	def max_corr_log(self, max_corr_dict):
		corr = max_corr_dict['corr']
		loss = max_corr_dict['loss']
		epoch = max_corr_dict['epoch']
		msg = f'[epoch {epoch}] max correlation: {corr:.4f}, loss: {loss:.6f}'
		self.logf.write(msg + '\n');
		print(msg);
		self.logf.flush()


def get_log(epoch, loss, y_pred, y, acc_std, acc_mean, tag='train'):
	msg = f'[{tag}] Ep {epoch} loss {loss.item() / len(y):0.4f} '
	msg += f'pacc {y_pred[0]:0.4f}'
	msg += f'({y_pred[0] * 100.0 * acc_std + acc_mean:0.4f}) '
	msg += f'acc {y[0]:0.4f}({y[0] * 100 * acc_std + acc_mean:0.4f})'
	return msg


def load_model(model, model_path, load_epoch=None, load_max_pt=None):
	if load_max_pt is not None:
		ckpt_path = os.path.join(model_path, load_max_pt)
	else:
		ckpt_path = os.path.join(model_path, f'ckpt_{load_epoch}.pt')
	print(f"==> load model from {ckpt_path} ...")
	model.cpu()
	model.load_state_dict(torch.load(ckpt_path))


def save_model(epoch, model, model_path, max_corr=None):
	print("==> save current model...")
	if max_corr is not None:
		torch.save(model.cpu().state_dict(),
		           os.path.join(model_path, 'ckpt_max_corr.pt'))
	else:
		torch.save(model.cpu().state_dict(),
		           os.path.join(model_path, f'ckpt_{epoch}.pt'))


def mean_confidence_interval(data, confidence=0.95):
	a = 1.0 * np.array(data)
	n = len(a)
	m, se = np.mean(a), scipy.stats.sem(a)
	h = se * scipy.stats.t.ppf((1 + confidence) / 2., n - 1)
	return m, h