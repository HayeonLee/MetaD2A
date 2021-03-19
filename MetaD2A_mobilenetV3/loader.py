###########################################################################################
# Copyright (c) Hayeon Lee, Eunyoung Hyung [GitHub MetaD2A], 2021
# Rapid Neural Architecture Search by Learning to Generate Graphs from Datasets, ICLR 2021
###########################################################################################
from __future__ import print_function
import os
import torch
from tqdm import tqdm
from torch.utils.data import Dataset
from torch.utils.data import DataLoader


def get_meta_train_loader(batch_size, data_path, num_sample, is_pred=False):
	dataset = MetaTrainDatabase(data_path, num_sample, is_pred)
	print(f'==> The number of tasks for meta-training: {len(dataset)}')
	
	loader = DataLoader(dataset=dataset,
	                    batch_size=batch_size,
	                    shuffle=True,
	                    num_workers=1,
	                    collate_fn=collate_fn)
	return loader


def get_meta_test_loader(data_path, data_name, num_class=None, is_pred=False):
	dataset = MetaTestDataset(data_path, data_name, num_class)
	print(f'==> Meta-Test dataset {data_name}')
	
	loader = DataLoader(dataset=dataset,
	                    batch_size=100,
	                    shuffle=False,
	                    num_workers=1)
	return loader


class MetaTrainDatabase(Dataset):
	def __init__(self, data_path, num_sample, is_pred=False):
		self.mode = 'train'
		self.acc_norm = True
		self.num_sample = num_sample
		self.x = torch.load(os.path.join(data_path, 'imgnet32bylabel.pt'))
		
		self.dpath = '{}/{}/processed/'.format(data_path, 'predictor' if is_pred else 'generator')
		self.dname = f'collected_database'
		
		if not os.path.exists(self.dpath + f'{self.dname}_train.pt'):
			database = torch.load(self.dpath + f'{self.dname}.pt')
			
			rand_idx = torch.randperm(len(database))
			test_len = int(len(database) * 0.15)
			idxlst = {'test': rand_idx[:test_len],
			          'valid': rand_idx[test_len:2 * test_len],
			          'train': rand_idx[2 * test_len:]}
			
			for m in ['train', 'valid', 'test']:
				acc, graph, cls, net, flops = [], [], [], [], []
				for idx in tqdm(idxlst[m].tolist(), desc=f'data-{m}'):
					acc.append(database[idx]['top1'])
					net.append(database[idx]['net'])
					cls.append(database[idx]['class'])
					flops.append(database[idx]['flops'])
				if m == 'train':
					mean = torch.mean(torch.tensor(acc)).item()
					std = torch.std(torch.tensor(acc)).item()
				torch.save({'acc': acc,
				            'class': cls,
				            'net': net,
				            'flops': flops,
				            'mean': mean,
				            'std': std},
				           self.dpath + f'{self.dname}_{m}.pt')
		
		self.set_mode(self.mode)
	
	def set_mode(self, mode):
		self.mode = mode
		data = torch.load(self.dpath + f'{self.dname}_{self.mode}.pt')
		self.acc = data['acc']
		self.cls = data['class']
		self.net = data['net']
		self.flops = data['flops']
		self.mean = data['mean']
		self.std = data['std']
	
	def __len__(self):
		return len(self.acc)
	
	def __getitem__(self, index):
		data = []
		classes = self.cls[index]
		acc = self.acc[index]
		graph = self.net[index]

		for i, cls in enumerate(classes):
			cx = self.x[cls.item()][0]
			ridx = torch.randperm(len(cx))
			data.append(cx[ridx[:self.num_sample]])
		x = torch.cat(data)
		if self.acc_norm:
			acc = ((acc - self.mean) / self.std) / 100.0
		else:
			acc = acc / 100.0
		return x, graph, torch.tensor(acc).view(1, 1)


class MetaTestDataset(Dataset):
	def __init__(self, data_path, data_name, num_sample, num_class=None):
		self.num_sample = num_sample
		self.data_name = data_name
		
		num_class_dict = {
			'cifar100': 100,
			'cifar10': 10,
			'mnist': 10,
			'svhn': 10,
			'aircraft30': 30,
			'aircraft100': 100,
			'pets': 37
		}
		
		if num_class is not None:
			self.num_class = num_class
		else:
			self.num_class = num_class_dict[data_name]
		
		self.x = torch.load(os.path.join(data_path, f'{data_name}bylabel.pt'))
	
	def __len__(self):
		return 1000000
	
	def __getitem__(self, index):
		data = []
		classes = list(range(self.num_class))
		for cls in classes:
			cx = self.x[cls][0]
			ridx = torch.randperm(len(cx))
			data.append(cx[ridx[:self.num_sample]])
		x = torch.cat(data)
		return x


def collate_fn(batch):
	# x = torch.stack([item[0] for item in batch])
	# graph = [item[1] for item in batch]
	# acc = torch.stack([item[2] for item in batch])
	return batch
