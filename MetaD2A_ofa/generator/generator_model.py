######################################################################################
# Copyright (c) muhanzhang, D-VAE, NeurIPS 2019 [GitHub D-VAE]
# Modified by Hayeon Lee, Eunyoung Hyung, MetaD2A, ICLR2021, 2021. 03 [GitHub MetaD2A]
######################################################################################
import torch
from torch import nn
from torch.nn import functional as F
import numpy as np
import igraph
from set_encoder.setenc_models import SetPool


class GeneratorModel(nn.Module):
	def __init__(self, args, graph_config):
		super(GeneratorModel, self).__init__()
		self.max_n = graph_config['max_n']  # maximum number of vertices
		self.nvt = graph_config['num_vertex_type']  # number of vertex types
		self.START_TYPE = graph_config['START_TYPE']
		self.END_TYPE = graph_config['END_TYPE']
		self.hs = args.hs  # hidden state size of each vertex
		self.nz = args.nz  # size of latent representation z
		self.gs = args.hs  # size of graph state
		self.bidir = True  # whether to use bidirectional encoding
		self.vid = True
		self.device = None
		self.num_sample = args.num_sample
		
		if self.vid:
			self.vs = self.hs + self.max_n  # vertex state size = hidden state + vid
		else:
			self.vs = self.hs
		
		# 0. encoding-related
		self.grue_forward = nn.GRUCell(self.nvt, self.hs)  # encoder GRU
		self.grue_backward = nn.GRUCell(self.nvt, self.hs)  # backward encoder GRU
		self.enc_g_mu = nn.Linear(self.gs, self.nz)  # latent mean
		self.enc_g_var = nn.Linear(self.gs, self.nz)  # latent var
		self.fc1 = nn.Linear(self.gs, self.nz)  # latent mean
		self.fc2 = nn.Linear(self.gs, self.nz)  # latent logvar
		
		# 1. decoding-related
		self.grud = nn.GRUCell(self.nvt, self.hs)  # decoder GRU
		self.fc3 = nn.Linear(self.nz, self.hs)  # from latent z to initial hidden state h0
		self.add_vertex = nn.Sequential(
			nn.Linear(self.hs, self.hs * 2),
			nn.ReLU(),
			nn.Linear(self.hs * 2, self.nvt)
		)  # which type of new vertex to add f(h0, hg)
		self.add_edge = nn.Sequential(
			nn.Linear(self.hs * 2, self.hs * 4),
			nn.ReLU(),
			nn.Linear(self.hs * 4, 1)
		)  # whether to add edge between v_i and v_new, f(hvi, hnew)
		self.decoding_gate = nn.Sequential(
			nn.Linear(self.vs, self.hs),
			nn.Sigmoid()
		)
		self.decoding_mapper = nn.Sequential(
			nn.Linear(self.vs, self.hs, bias=False),
		)  # disable bias to ensure padded zeros also mapped to zeros
		
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
			x = torch.zeros((len(idx), length)
			                ).scatter_(1, idx, 1).to(self.get_device())
		else:
			idx = torch.LongTensor([idx]).unsqueeze(0)
			x = torch.zeros((1, length)
			                ).scatter_(1, idx, 1).to(self.get_device())
		return x
	
	def _gated(self, h, gate, mapper):
		return gate(h) * mapper(h)
	
	def _collate_fn(self, G):
		return [g.copy() for g in G]
	
	def _propagate_to(self, G, v, propagator,
	                  H=None, reverse=False, gate=None, mapper=None):
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
		H_name = 'H_forward'  # name of the hidden states attribute
		H_pred = [[g.vs[x][H_name] for x in g.predecessors(v)] for g in G]
		if self.vid:
			vids = [self._one_hot(g.predecessors(v), self.max_n) for g in G]
		if reverse:
			H_name = 'H_backward'  # name of the hidden states attribute
			H_pred = [[g.vs[x][H_name] for x in g.successors(v)] for g in G]
			if self.vid:
				vids = [self._one_hot(g.successors(v), self.max_n) for g in G]
			gate, mapper = self.gate_backward, self.mapper_backward
		else:
			H_name = 'H_forward'  # name of the hidden states attribute
			H_pred = [
				[g.vs[x][H_name] for x in g.predecessors(v)] for g in G]
			if self.vid:
				vids = [
					self._one_hot(g.predecessors(v), self.max_n) for g in G]
			if gate is None:
				gate, mapper = self.gate_forward, self.mapper_forward
		if self.vid:
			H_pred = [[torch.cat(
				[x[i], y[i:i + 1]], 1) for i in range(len(x))
			] for x, y in zip(H_pred, vids)]
		# if h is not provided, use gated sum of v's predecessors' states as the input hidden state
		if H is None:
			max_n_pred = max([len(x) for x in H_pred])  # maximum number of predecessors
			if max_n_pred == 0:
				H = self._get_zero_hidden(len(G))
			else:
				H_pred = [torch.cat(h_pred +
				                    [self._get_zeros(max_n_pred - len(h_pred),
				                                     self.vs)], 0).unsqueeze(0)
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
	
	def _update_v(self, G, v, H0=None):
		# perform a forward propagation step at v when decoding to update v's state
		# self._propagate_to(G, v, self.grud, H0, reverse=False)
		self._propagate_to(G, v, self.grud, H0,
		                   reverse=False, gate=self.decoding_gate,
		                   mapper=self.decoding_mapper)
		return
	
	def _get_vertex_state(self, G, v):
		# get the vertex states at v
		Hv = []
		for g in G:
			if v >= g.vcount():
				hv = self._get_zero_hidden()
			else:
				hv = g.vs[v]['H_forward']
			Hv.append(hv)
		Hv = torch.cat(Hv, 0)
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
	
	def graph_encode(self, G):
		# encode graphs G into latent vectors
		if type(G) != list:
			G = [G]
		self._propagate_from(G, 0, self.grue_forward,
		                     H0=self._get_zero_hidden(len(G)), reverse=False)
		if self.bidir:
			self._propagate_from(G, self.max_n - 1, self.grue_backward,
			                     H0=self._get_zero_hidden(len(G)), reverse=True)
		Hg = self._get_graph_state(G)
		mu, logvar = self.enc_g_mu(Hg), self.enc_g_var(Hg)
		return mu, logvar
	
	def set_encode(self, X):
		proto_batch = []
		for x in X:  # X.shape: [32, 400, 512]
			cls_protos = self.intra_setpool(
				x.view(-1, self.num_sample, 512)).squeeze(1)
			proto_batch.append(
				self.inter_setpool(cls_protos.unsqueeze(0)))
		v = torch.stack(proto_batch).squeeze()
		mu, logvar = self.fc1(v), self.fc2(v)
		return mu, logvar
	
	def reparameterize(self, mu, logvar, eps_scale=0.01):
		# return z ~ N(mu, std)
		if self.training:
			std = logvar.mul(0.5).exp_()
			eps = torch.randn_like(std) * eps_scale
			return eps.mul(std).add_(mu)
		else:
			return mu
	
	def _get_edge_score(self, Hvi, H, H0):
		# compute scores for edges from vi based on Hvi, H (current vertex) and H0
		# in most cases, H0 need not be explicitly included since Hvi and H contain its information
		return self.sigmoid(self.add_edge(torch.cat([Hvi, H], -1)))
	
	def graph_decode(self, z, stochastic=True):
		# decode latent vectors z back to graphs
		# if stochastic=True, stochastically sample each action from the predicted distribution;
		# otherwise, select argmax action deterministically.
		H0 = self.tanh(self.fc3(z))  # or relu activation, similar performance
		G = [igraph.Graph(directed=True) for _ in range(len(z))]
		for g in G:
			g.add_vertex(type=self.START_TYPE)
		self._update_v(G, 0, H0)
		finished = [False] * len(G)
		for idx in range(1, self.max_n):
			# decide the type of the next added vertex
			if idx == self.max_n - 1:  # force the last node to be end_type
				new_types = [self.END_TYPE] * len(G)
			else:
				Hg = self._get_graph_state(G, decode=True)
				type_scores = self.add_vertex(Hg)
				if stochastic:
					type_probs = F.softmax(type_scores, 1
					                       ).cpu().detach().numpy()
					new_types = [np.random.choice(range(self.nvt),
					                              p=type_probs[i]) for i in range(len(G))]
				else:
					new_types = torch.argmax(type_scores, 1)
					new_types = new_types.flatten().tolist()
			for i, g in enumerate(G):
				if not finished[i]:
					g.add_vertex(type=new_types[i])
			self._update_v(G, idx)
			
			# decide connections
			edge_scores = []
			for vi in range(idx - 1, -1, -1):
				Hvi = self._get_vertex_state(G, vi)
				H = self._get_vertex_state(G, idx)
				ei_score = self._get_edge_score(Hvi, H, H0)
				if stochastic:
					random_score = torch.rand_like(ei_score)
					decisions = random_score < ei_score
				else:
					decisions = ei_score > 0.5
				for i, g in enumerate(G):
					if finished[i]:
						continue
					if new_types[i] == self.END_TYPE:
						# if new node is end_type, connect it to all loose-end vertices (out_degree==0)
						end_vertices = set([
							v.index for v in g.vs.select(_outdegree_eq=0)
							if v.index != g.vcount() - 1])
						for v in end_vertices:
							g.add_edge(v, g.vcount() - 1)
						finished[i] = True
						continue
					if decisions[i, 0]:
						g.add_edge(vi, g.vcount() - 1)
				self._update_v(G, idx)
		
		for g in G:
			del g.vs['H_forward']  # delete hidden states to save GPU memory
		return G
	
	def loss(self, mu, logvar, G_true, beta=0.005):
		# compute the loss of decoding mu and logvar to true graphs using teacher forcing
		# ensure when computing the loss of step i, steps 0 to i-1 are correct
		z = self.reparameterize(mu, logvar)
		H0 = self.tanh(self.fc3(z))  # or relu activation, similar performance
		G = [igraph.Graph(directed=True) for _ in range(len(z))]
		for g in G:
			g.add_vertex(type=self.START_TYPE)
		self._update_v(G, 0, H0)
		res = 0  # log likelihood
		for v_true in range(1, self.max_n):
			# calculate the likelihood of adding true types of nodes
			# use start type to denote padding vertices since start type only appears for vertex 0
			# and will never be a true type for later vertices, thus it's free to use
			true_types = [g_true.vs[v_true]['type']
			              if v_true < g_true.vcount()
			              else self.START_TYPE for g_true in G_true]
			Hg = self._get_graph_state(G, decode=True)
			type_scores = self.add_vertex(Hg)
			# vertex log likelihood
			vll = self.logsoftmax1(type_scores)[
				np.arange(len(G)), true_types].sum()
			res = res + vll
			for i, g in enumerate(G):
				if true_types[i] != self.START_TYPE:
					g.add_vertex(type=true_types[i])
			self._update_v(G, v_true)
			
			# calculate the likelihood of adding true edges
			true_edges = []
			for i, g_true in enumerate(G_true):
				true_edges.append(g_true.get_adjlist(igraph.IN)[v_true]
				                  if v_true < g_true.vcount() else [])
			edge_scores = []
			for vi in range(v_true - 1, -1, -1):
				Hvi = self._get_vertex_state(G, vi)
				H = self._get_vertex_state(G, v_true)
				ei_score = self._get_edge_score(Hvi, H, H0)
				edge_scores.append(ei_score)
				for i, g in enumerate(G):
					if vi in true_edges[i]:
						g.add_edge(vi, v_true)
				self._update_v(G, v_true)
			edge_scores = torch.cat(edge_scores[::-1], 1)
			
			ground_truth = torch.zeros_like(edge_scores)
			idx1 = [i for i, x in enumerate(true_edges)
			        for _ in range(len(x))]
			idx2 = [xx for x in true_edges for xx in x]
			ground_truth[idx1, idx2] = 1.0
			
			# edges log-likelihood
			ell = - F.binary_cross_entropy(
				edge_scores, ground_truth, reduction='sum')
			res = res + ell
		
		res = -res  # convert likelihood to loss
		kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
		return res + beta * kld, res, kld