###########################################################################################
# Copyright (c) Hayeon Lee, Eunyoung Hyung [GitHub MetaD2A], 2021
# Rapid Neural Architecture Search by Learning to Generate Graphs from Datasets, ICLR 2021
###########################################################################################
from __future__ import print_function
import os
import random
from tqdm import tqdm
import numpy as np
import time

import torch
from torch import optim
from torch.optim.lr_scheduler import ReduceLROnPlateau

from utils import load_graph_config, decode_ofa_mbv3_to_igraph, decode_igraph_to_ofa_mbv3
from utils import Accumulator, Log
from utils import load_model, save_model
from loader import get_meta_train_loader, get_meta_test_loader

from .generator_model import GeneratorModel


class Generator:
	def __init__(self, args):
		self.args = args
		self.batch_size = args.batch_size
		self.data_path = args.data_path
		self.num_sample = args.num_sample
		self.max_epoch = args.max_epoch
		self.save_epoch = args.save_epoch
		self.model_path = args.model_path
		self.save_path = args.save_path
		self.model_name = args.model_name
		self.test = args.test
		self.device = args.device
		
		graph_config = load_graph_config(
			args.graph_data_name, args.nvt, args.data_path)
		self.model = GeneratorModel(args, graph_config)
		self.model.to(self.device)
		
		if self.test:
			self.data_name = args.data_name
			self.num_class = args.num_class
			self.load_epoch = args.load_epoch
			self.num_gen_arch = args.num_gen_arch
			load_model(self.model, self.model_path, self.load_epoch)
		
		else:
			self.optimizer = optim.Adam(self.model.parameters(), lr=1e-4)
			self.scheduler = ReduceLROnPlateau(self.optimizer, 'min',
			                                   factor=0.1, patience=10, verbose=True)
			self.mtrloader = get_meta_train_loader(
				self.batch_size, self.data_path, self.num_sample)
			self.mtrlog = Log(self.args, open(os.path.join(
				self.save_path, self.model_name, 'meta_train_generator.log'), 'w'))
			self.mtrlog.print_args()
			self.mtrlogger = Accumulator('loss', 'recon_loss', 'kld')
			self.mvallogger = Accumulator('loss', 'recon_loss', 'kld')
	
	def meta_train(self):
		sttime = time.time()
		for epoch in range(1, self.max_epoch + 1):
			self.mtrlog.ep_sttime = time.time()
			loss = self.meta_train_epoch(epoch)
			self.scheduler.step(loss)
			self.mtrlog.print(self.mtrlogger, epoch, tag='train')
			
			self.meta_validation()
			self.mtrlog.print(self.mvallogger, epoch, tag='valid')
			
			if epoch % self.save_epoch == 0:
				save_model(epoch, self.model, self.model_path)
		
		self.mtrlog.save_time_log()
	
	def meta_train_epoch(self, epoch):
		self.model.to(self.device)
		self.model.train()
		
		self.mtrloader.dataset.set_mode('train')
		pbar = tqdm(self.mtrloader)
		
		for batch in pbar:
			for x, g, acc in batch:
				self.optimizer.zero_grad()
				g = decode_ofa_mbv3_to_igraph(g)[0]
				x_ = x.unsqueeze(0).to(self.device)
				mu, logvar = self.model.set_encode(x_)
				loss, recon, kld = self.model.loss(mu.unsqueeze(0), logvar.unsqueeze(0), [g])
				loss.backward()
				self.optimizer.step()
				cnt = len(x)
				self.mtrlogger.accum([loss.item() / cnt,
				                      recon.item() / cnt,
				                      kld.item() / cnt])
			
		return self.mtrlogger.get('loss')
	
	
	def meta_validation(self):
		self.model.to(self.device)
		self.model.eval()
		
		self.mtrloader.dataset.set_mode('valid')
		pbar = tqdm(self.mtrloader)
		
		for batch in pbar:
			for x, g, acc in batch:
				with torch.no_grad():
					g = decode_ofa_mbv3_to_igraph(g)[0]
					x_ = x.unsqueeze(0).to(self.device)
					mu, logvar = self.model.set_encode(x_)
					loss, recon, kld = self.model.loss(mu.unsqueeze(0), logvar.unsqueeze(0), [g])
				
				cnt = len(x)
				self.mvallogger.accum([loss.item() / cnt,
				                       recon.item() / cnt,
				                       kld.item() / cnt])
			
		return self.mvallogger.get('loss')
	
	
	def meta_test(self, predictor):
		if self.data_name == 'all':
			for data_name in ['cifar100', 'cifar10', 'mnist', 'svhn', 'aircraft30', 'aircraft100', 'pets']:
				self.meta_test_per_dataset(data_name, predictor)
		else:
			self.meta_test_per_dataset(self.data_name, predictor)
	
	def meta_test_per_dataset(self, data_name, predictor):
		# meta_test_path = os.path.join(
		#   self.save_path, 'meta_test', data_name, 'generated_arch')
		meta_test_path = os.path.join(
			self.save_path, 'meta_test', data_name, f'{self.num_gen_arch}', 'generated_arch')
		if not os.path.exists(meta_test_path):
			os.makedirs(meta_test_path)
		
		meta_test_loader = get_meta_test_loader(
			self.data_path, data_name, self.num_sample, self.num_class)
		
		print(f'==> generate architectures for {data_name}')
		runs = 10 if data_name in ['cifar10', 'cifar100'] else 1
		# num_gen_arch = 500 if data_name in ['cifar100'] else self.num_gen_arch
		elasped_time = []
		for run in range(1, runs + 1):
			print(f'==> run {run}/{runs}')
			elasped_time.append(self.generate_architectures(
				meta_test_loader, data_name,
				meta_test_path, run, self.num_gen_arch, predictor))
			print(f'==> done\n')
		
		# time_path = os.path.join(self.save_path, 'meta_test', data_name, 'time.txt')
		time_path = os.path.join(self.save_path, 'meta_test', data_name, f'{self.num_gen_arch}', 'time.txt')
		with open(time_path, 'w') as f_time:
			msg = f'generator elasped time {np.mean(elasped_time):.2f}s'
			print(f'==> save time in {time_path}')
			f_time.write(msg + '\n');
			print(msg)
	
	def generate_architectures(self, meta_test_loader, data_name,
	                           meta_test_path, run, num_gen_arch, predictor):
		self.model.eval()
		self.model.to(self.device)
		
		architecture_string_lst, pred_acc_lst = [], []
		total_cnt, valid_cnt = 0, 0
		flag = False
		
		start = time.time()
		with torch.no_grad():
			for x in meta_test_loader:
				x_ = x.unsqueeze(0).to(self.device)
				mu, logvar = self.model.set_encode(x_)
				z = self.model.reparameterize(mu.unsqueeze(0), logvar.unsqueeze(0))
				g_recon = self.model.graph_decode(z)
				pred_acc = predictor.forward(x_, g_recon)
				architecture_string = decode_igraph_to_ofa_mbv3(g_recon[0])
				total_cnt += 1
				if architecture_string is not None:
					if not architecture_string in architecture_string_lst:
						valid_cnt += 1
						architecture_string_lst.append(architecture_string)
						pred_acc_lst.append(pred_acc.item())
						if valid_cnt == num_gen_arch:
							flag = True
							break
				if flag:
					break
		elapsed = time.time() - start
		pred_acc_lst, architecture_string_lst = zip(*sorted(zip(pred_acc_lst,
		                                                        architecture_string_lst),
		                                        key=lambda x: x[0], reverse=True))
		
		spath = os.path.join(meta_test_path, f"run_{run}.txt")
		with open(spath, 'w') as f:
			print(f'==> save generated architectures in {spath}')
			msg = f'elapsed time: {elapsed:6.2f}s '
			print(msg);
			f.write(msg + '\n')
			for i, architecture_string in enumerate(architecture_string_lst):
				f.write(f"{architecture_string}\n")
		return elapsed
