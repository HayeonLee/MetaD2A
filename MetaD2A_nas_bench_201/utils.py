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
  if graph_data_name is not 'nasbench201':
    raise NotImplementedError(graph_data_name)
  g_list = []
  max_n = 0  # maximum number of nodes
  ms = torch.load(os.path.join(
          data_path, f'{graph_data_name}.pt'))['arch']['matrix']
  for i in range(len(ms)):
    g, n = decode_NAS_BENCH_201_8_to_igraph(ms[i])
    max_n = max(max_n, n)
    g_list.append((g, 0))
  # number of different node types including in/out node
  graph_config = {}
  graph_config['num_vertex_type'] = nvt # original types + start/end types
  graph_config['max_n'] = max_n # maximum number of nodes
  graph_config['START_TYPE'] = 0 # predefined start vertex type
  graph_config['END_TYPE'] = 1 # predefined end vertex type

  return graph_config


def decode_NAS_BENCH_201_8_to_igraph(row):
  if type(row) == str:
    row = eval(row)  # convert string to list of lists
  n = len(row)
  g = igraph.Graph(directed=True)
  g.add_vertices(n)
  for i, node in enumerate(row):
    g.vs[i]['type'] = node[0]
    if i < (n - 2) and i > 0:
      g.add_edge(i, i + 1)  # always connect from last node
    for j, edge in enumerate(node[1:]):
      if edge == 1:
        g.add_edge(j, i)
  return g, n


def is_valid_NAS201(g, START_TYPE=0, END_TYPE=1):
  # first need to be a valid DAG computation graph
  res = is_valid_DAG(g, START_TYPE, END_TYPE)
  # in addition, node i must connect to node i+1
  res = res and len(g.vs['type'])==8
  res = res and not (0 in g.vs['type'][1:-1])
  res = res and not (1 in g.vs['type'][1:-1])
  return res


def decode_igraph_to_NAS201_matrix(g):
  m = [[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]]
  xys = [(1, 0), (2, 0), (2, 1), (3, 0), (3, 1), (3, 2)]
  for i, xy in enumerate(xys):
    m[xy[0]][xy[1]] = float(g.vs[i + 1]['type']) - 2
  import numpy
  return numpy.array(m)


def decode_igraph_to_NAS_BENCH_201_string(g):
  if not is_valid_NAS201(g):
    return None
  m = decode_igraph_to_NAS201_matrix(g)
  types = ['none', 'skip_connect', 'nor_conv_1x1', 'nor_conv_3x3', 'avg_pool_3x3']
  return '|{}~0|+|{}~0|{}~1|+|{}~0|{}~1|{}~2|'.\
      format(types[int(m[1][0])],
             types[int(m[2][0])], types[int(m[2][1])],
             types[int(m[3][0])], types[int(m[3][1])], types[int(m[3][2])])


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
      self.logf.write(msg+'\n')
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
    self.logf.write(msg+'\n')

  def print_pred_log(self, loss, corr, tag, epoch=None, max_corr_dict=None):
    if tag == 'train':
      ct = time.time() - self.ep_sttime
      tt = time.time() - self.stime
      msg = f'[total {tt:6.2f}s (ep {ct:6.2f}s)] epoch {epoch:3d}'
      self.logf.write(msg+'\n'); print(msg); self.logf.flush() 
    #msg = f'ep {epoch:3d} ep time {time.time() - ep_sttime:8.2f} '
    #msg += f'time {time.time() - sttime:6.2f} '
    if max_corr_dict is not None:
      max_corr = max_corr_dict['corr']
      max_loss = max_corr_dict['loss']
      msg = f'{tag}: loss {loss:.6f} ({max_loss:.6f}) '
      msg += f'corr {corr:.4f} ({max_corr:.4f})'
    else:
      msg = f'{tag}: loss {loss:.6f} corr {corr:.4f}'
    self.logf.write(msg+'\n'); print(msg); self.logf.flush()   

  def max_corr_log(self, max_corr_dict):
    corr = max_corr_dict['corr']
    loss = max_corr_dict['loss']
    epoch = max_corr_dict['epoch']
    msg = f'[epoch {epoch}] max correlation: {corr:.4f}, loss: {loss:.6f}'
    self.logf.write(msg+'\n'); print(msg); self.logf.flush()


def get_log(epoch, loss, y_pred, y, acc_std, acc_mean, tag='train'):
  msg = f'[{tag}] Ep {epoch} loss {loss.item()/len(y):0.4f} '
  msg += f'pacc {y_pred[0]:0.4f}'
  msg += f'({y_pred[0]*100.0*acc_std+acc_mean:0.4f}) '
  msg += f'acc {y[0]:0.4f}({y[0]*100*acc_std+acc_mean:0.4f})'
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
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n-1)
    return m, h