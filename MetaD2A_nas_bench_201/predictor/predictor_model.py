######################################################################################
# Copyright (c) muhanzhang, D-VAE, NeurIPS 2019 [GitHub D-VAE]
# Modified by Hayeon Lee, Eunyoung Hyung, MetaD2A, ICLR2021, 2021. 03 [GitHub MetaD2A]
######################################################################################
import math
import random
import torch
from torch import nn
from torch.nn import functional as F
import numpy as np
import igraph

from set_encoder.setenc_models import SetPool


class PredictorModel(nn.Module):
  def __init__(self, args, graph_config):
    super(PredictorModel, self).__init__()
    self.max_n = graph_config['max_n']  # maximum number of vertices
    self.nvt = args.nvt  # number of vertex types
    self.START_TYPE = graph_config['START_TYPE']
    self.END_TYPE = graph_config['END_TYPE']
    self.hs = args.hs  # hidden state size of each vertex
    self.nz = args.nz  # size of latent representation z
    self.gs = args.hs  # size of graph state
    self.bidir = True  # whether to use bidirectional encoding
    self.vid = True
    self.device = None
    self.input_type = 'DG'
    self.num_sample = args.num_sample

    if self.vid:
      self.vs = self.hs + self.max_n  # vertex state size = hidden state + vid
    else:
      self.vs = self.hs
    
    # 0. encoding-related
    self.grue_forward = nn.GRUCell(self.nvt, self.hs)  # encoder GRU
    self.grue_backward = nn.GRUCell(self.nvt, self.hs)  # backward encoder GRU
    self.fc1 = nn.Linear(self.gs, self.nz)  # latent mean
    self.fc2 = nn.Linear(self.gs, self.nz)  # latent logvar

    # 2. gate-related
    self.gate_forward = nn.Sequential(
      nn.Linear(self.vs, self.hs),
      nn.Sigmoid()
    )
    self.gate_backward = nn.Sequential(
      nn.Linear(self.vs, self.hs),
      nn.Sigmoid()
    )
    self.mapper_forward = nn.Sequential(
      nn.Linear(self.vs, self.hs, bias=False),
    )  # disable bias to ensure padded zeros also mapped to zeros
    self.mapper_backward = nn.Sequential(
      nn.Linear(self.vs, self.hs, bias=False),
    )
    
    # 3. bidir-related, to unify sizes
    if self.bidir:
      self.hv_unify = nn.Sequential(
        nn.Linear(self.hs * 2, self.hs),
      )
      self.hg_unify = nn.Sequential(
        nn.Linear(self.gs * 2, self.gs),
      )

    # 4. other
    self.relu = nn.ReLU()
    self.sigmoid = nn.Sigmoid()
    self.tanh = nn.Tanh()
    self.logsoftmax1 = nn.LogSoftmax(1)
    
    # 6. predictor
    np = self.gs
    self.intra_setpool = SetPool(dim_input=512, 
                                 num_outputs=1, 
                                 dim_output=self.nz, 
                                 dim_hidden=self.nz, 
                                 mode='sabPF')
    self.inter_setpool = SetPool(dim_input=self.nz, 
                                 num_outputs=1, 
                                 dim_output=self.nz, 
                                 dim_hidden=self.nz, 
                                 mode='sabPF')
    self.set_fc = nn.Sequential(
        nn.Linear(512, self.nz),
        nn.ReLU())

    input_dim = 0
    if 'D' in self.input_type:
      input_dim += self.nz
    if 'G' in self.input_type:
      input_dim += self.nz

    self.pred_fc = nn.Sequential(
        nn.Linear(input_dim, self.hs),
        nn.Tanh(),
        nn.Linear(self.hs, 1)
      )
    self.mseloss = nn.MSELoss(reduction='sum')


  def predict(self, D_mu, G_mu):
    input_vec = []
    if 'D' in self.input_type:
      input_vec.append(D_mu)
    if 'G' in self.input_type:
      input_vec.append(G_mu)
    input_vec = torch.cat(input_vec, dim=1)
    return self.pred_fc(input_vec)

  def get_device(self):
    if self.device is None:
      self.device = next(self.parameters()).device
    return self.device
  
  def _get_zeros(self, n, length):
    return torch.zeros(n, length).to(self.get_device())  # get a zero hidden state
  
  def _get_zero_hidden(self, n=1):
    return self._get_zeros(n, self.hs)  # get a zero hidden state
  
  def _one_hot(self, idx, length):
    if type(idx) in [list, range]:
      if idx == []:
        return None
      idx = torch.LongTensor(idx).unsqueeze(0).t()
      x = torch.zeros((len(idx), length)).scatter_(1, idx, 1).to(self.get_device())
    else:
      idx = torch.LongTensor([idx]).unsqueeze(0)
      x = torch.zeros((1, length)).scatter_(1, idx, 1).to(self.get_device())
    return x
  
  def _gated(self, h, gate, mapper):
    return gate(h) * mapper(h)
  
  def _collate_fn(self, G):
    return [g.copy() for g in G]
  
  def _propagate_to(self, G, v, propagator, H=None, reverse=False, gate=None, mapper=None):
    # propagate messages to vertex index v for all graphs in G
    # return the new messages (states) at v
    G = [g for g in G if g.vcount() > v]
    if len(G) == 0:
      return
    if H is not None:
      idx = [i for i, g in enumerate(G) if g.vcount() > v]
      H = H[idx]
    v_types = [g.vs[v]['type'] for g in G]
    X = self._one_hot(v_types, self.nvt)
    if reverse:
      H_name = 'H_backward'  # name of the hidden states attribute
      H_pred = [[g.vs[x][H_name] for x in g.successors(v)] for g in G]
      if self.vid:
        vids = [self._one_hot(g.successors(v), self.max_n) for g in G]
      gate, mapper = self.gate_backward, self.mapper_backward
    else:
      H_name = 'H_forward'  # name of the hidden states attribute
      H_pred = [[g.vs[x][H_name] for x in g.predecessors(v)] for g in G]
      if self.vid:
        vids = [self._one_hot(g.predecessors(v), self.max_n) for g in G]
      if gate is None:
        gate, mapper = self.gate_forward, self.mapper_forward
    if self.vid:
      H_pred = [[torch.cat([x[i], y[i:i + 1]], 1) for i in range(len(x))] for x, y in zip(H_pred, vids)]
    # if h is not provided, use gated sum of v's predecessors' states as the input hidden state
    if H is None:
      max_n_pred = max([len(x) for x in H_pred])  # maximum number of predecessors
      if max_n_pred == 0:
        H = self._get_zero_hidden(len(G))
      else:
        H_pred = [torch.cat(h_pred +
                            [self._get_zeros(max_n_pred - len(h_pred), self.vs)], 0).unsqueeze(0)
                  for h_pred in H_pred]  # pad all to same length
        H_pred = torch.cat(H_pred, 0)  # batch * max_n_pred * vs
        H = self._gated(H_pred, gate, mapper).sum(1)  # batch * hs
    Hv = propagator(X, H)
    for i, g in enumerate(G):
      g.vs[v][H_name] = Hv[i:i + 1]
    return Hv
  
  def _propagate_from(self, G, v, propagator, H0=None, reverse=False):
    # perform a series of propagation_to steps starting from v following a topo order
    # assume the original vertex indices are in a topological order
    if reverse:
      prop_order = range(v, -1, -1)
    else:
      prop_order = range(v, self.max_n)
    Hv = self._propagate_to(G, v, propagator, H0, reverse=reverse)  # the initial vertex
    for v_ in prop_order[1:]:
      self._propagate_to(G, v_, propagator, reverse=reverse)
    return Hv
  
  def _get_graph_state(self, G, decode=False):
    # get the graph states
    # when decoding, use the last generated vertex's state as the graph state
    # when encoding, use the ending vertex state or unify the starting and ending vertex states
    Hg = []
    for g in G:
      hg = g.vs[g.vcount() - 1]['H_forward']
      if self.bidir and not decode:  # decoding never uses backward propagation
        hg_b = g.vs[0]['H_backward']
        hg = torch.cat([hg, hg_b], 1)
      Hg.append(hg)
    Hg = torch.cat(Hg, 0)
    if self.bidir and not decode:
      Hg = self.hg_unify(Hg)
    return Hg


  def set_encode(self, X):
    proto_batch = []
    for x in X: 
      cls_protos = self.intra_setpool(
        x.view(-1, self.num_sample, 512)).squeeze(1)
      proto_batch.append(
        self.inter_setpool(cls_protos.unsqueeze(0)))
    v = torch.stack(proto_batch).squeeze()
    return v


  def graph_encode(self, G):
    # encode graphs G into latent vectors
    if type(G) != list:
      G = [G]
    self._propagate_from(G, 0, self.grue_forward, H0=self._get_zero_hidden(len(G)),
                         reverse=False)
    if self.bidir:
      self._propagate_from(G, self.max_n - 1, self.grue_backward,
                           H0=self._get_zero_hidden(len(G)), reverse=True)
    Hg = self._get_graph_state(G)
    mu = self.fc1(Hg)
    #logvar = self.fc2(Hg)
    return mu #, logvar


  def reparameterize(self, mu, logvar, eps_scale=0.01):
    # return z ~ N(mu, std)
    if self.training:
      std = logvar.mul(0.5).exp_()
      eps = torch.randn_like(std) * eps_scale
      return eps.mul(std).add_(mu)
    else:
      return mu
  