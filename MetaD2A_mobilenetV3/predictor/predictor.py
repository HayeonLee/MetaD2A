###########################################################################################
# Copyright (c) Hayeon Lee, Eunyoung Hyung [GitHub MetaD2A], 2021
# Rapid Neural Architecture Search by Learning to Generate Graphs from Datasets, ICLR 2021
###########################################################################################
from __future__ import print_function
import torch
import os
import random
from tqdm import tqdm
import numpy as np
import time
import os
import shutil

from torch import nn, optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from scipy.stats import pearsonr

from utils import load_graph_config, decode_ofa_mbv3_to_igraph
from utils import Log, get_log
from utils import load_model, save_model, mean_confidence_interval

from loader import get_meta_train_loader, get_meta_test_loader, MetaTestDataset
from .predictor_model import PredictorModel


class Predictor:
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
		self.max_corr_dict = {'corr': -1, 'epoch': -1}
		self.train_arch = args.train_arch
		
		graph_config = load_graph_config(
			args.graph_data_name, args.nvt, args.data_path)
		
		self.model = PredictorModel(args, graph_config)
		self.model.to(self.device)
		
		if self.test:
			self.data_name = args.data_name
			self.num_class = args.num_class
			self.load_epoch = args.load_epoch
			load_model(self.model, self.model_path, load_max_pt='ckpt_max_corr.pt')
			self.model.to(self.device)
		else:
			self.optimizer = optim.Adam(self.model.parameters(), lr=1e-4)
			self.scheduler = ReduceLROnPlateau(self.optimizer, 'min',
			                                   factor=0.1, patience=10, verbose=True)
			self.mtrloader = get_meta_train_loader(
				self.batch_size, self.data_path, self.num_sample, is_pred=True)
			
			self.acc_mean = self.mtrloader.dataset.mean
			self.acc_std = self.mtrloader.dataset.std
			
			self.mtrlog = Log(self.args, open(os.path.join(
				self.save_path, self.model_name, 'meta_train_predictor.log'), 'w'))
			self.mtrlog.print_args()
	
	def forward(self, x, arch):
		D_mu = self.model.set_encode(x.unsqueeze(0).to(self.device)).unsqueeze(0)
		G_mu = self.model.graph_encode(arch[0])
		y_pred = self.model.predict(D_mu, G_mu)
		return y_pred
	
	def meta_train(self):
		sttime = time.time()
		for epoch in range(1, self.max_epoch + 1):
			self.mtrlog.ep_sttime = time.time()
			loss, corr = self.meta_train_epoch(epoch)
			self.scheduler.step(loss)
			self.mtrlog.print_pred_log(loss, corr, 'train', epoch)
			valoss, vacorr = self.meta_validation(epoch)
			if self.max_corr_dict['corr'] < vacorr:
				self.max_corr_dict['corr'] = vacorr
				self.max_corr_dict['epoch'] = epoch
				self.max_corr_dict['loss'] = valoss
				save_model(epoch, self.model, self.model_path, max_corr=True)
			
			self.mtrlog.print_pred_log(
				valoss, vacorr, 'valid', max_corr_dict=self.max_corr_dict)
			
			if epoch % self.save_epoch == 0:
				save_model(epoch, self.model, self.model_path)
		
		self.mtrlog.save_time_log()
		self.mtrlog.max_corr_log(self.max_corr_dict)
	
	def meta_train_epoch(self, epoch):
		self.model.to(self.device)
		self.model.train()
		
		self.mtrloader.dataset.set_mode('train')
		
		dlen = len(self.mtrloader.dataset)
		trloss = 0
		y_all, y_pred_all = [], []
		pbar = tqdm(self.mtrloader)
		
		for batch in pbar:
			batch_loss = 0
			y_batch, y_pred_batch = [], []
			self.optimizer.zero_grad()
			for x, g, acc in batch:
				y_pred = self.forward(x, decode_ofa_mbv3_to_igraph(g))

				y = acc.to(self.device)
				batch_loss += self.model.mseloss(y_pred, y)
				
				y = y.squeeze().tolist()
				y_pred = y_pred.squeeze().tolist()
				
				y_batch.append(y)
				y_pred_batch.append(y_pred)
				y_all.append(y)
				y_pred_all.append(y_pred)
			
			batch_loss.backward()
			trloss += float(batch_loss)
			self.optimizer.step()
			pbar.set_description(get_log(
				epoch, batch_loss, y_pred_batch, y_batch, self.acc_std, self.acc_mean))
		
		return trloss / dlen, pearsonr(np.array(y_all),
			                               np.array(y_pred_all))[0]
	
	
	def meta_validation(self, epoch):
		self.model.to(self.device)
		self.model.eval()
		
		valoss = 0
		self.mtrloader.dataset.set_mode('valid')
		dlen = len(self.mtrloader.dataset)
		y_all, y_pred_all = [], []
		pbar = tqdm(self.mtrloader)
		
		with torch.no_grad():
			for batch in pbar:
				batch_loss = 0
				y_batch, y_pred_batch = [], []

				for x, g, acc in batch:
					y_pred = self.forward(x, decode_ofa_mbv3_to_igraph(g))
					
					y = acc.to(self.device)
					batch_loss += self.model.mseloss(y_pred, y)
					
					y = y.squeeze().tolist()
					y_pred = y_pred.squeeze().tolist()
					
					y_batch.append(y)
					y_pred_batch.append(y_pred)
					y_all.append(y)
					y_pred_all.append(y_pred)
				
				valoss += float(batch_loss)
				pbar.set_description(get_log(
					epoch, batch_loss, y_pred_batch, y_batch, self.acc_std, self.acc_mean, tag='val'))
		return valoss / dlen, pearsonr(np.array(y_all),
		                               np.array(y_pred_all))[0]
	
