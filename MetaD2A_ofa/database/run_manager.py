######################################################################################
# Copyright (c) Han Cai, Once for All, ICLR 2020 [GitHub OFA]
# Modified by Hayeon Lee, Eunyoung Hyung, MetaD2A, ICLR2021, 2021. 03 [GitHub MetaD2A]
######################################################################################
import os
import json
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
from tqdm import tqdm
from utils import decode_ofa_mbv3_to_igraph
from ofa_local.utils import get_net_info, cross_entropy_loss_with_soft_target, cross_entropy_with_label_smoothing
from ofa_local.utils import AverageMeter, accuracy, write_log, mix_images, mix_labels, init_models

__all__ = ['RunManager']
import torchvision.models as models


class RunManager:
	
	def __init__(self, path, args, net, run_config, init=True, measure_latency=None,
	             no_gpu=False, data_loader=None, pp=None):
		self.path = path
		self.mode = args.model_name
		self.net = net
		self.run_config = run_config
		
		self.best_acc = 0
		self.start_epoch = 0
		
		os.makedirs(self.path, exist_ok=True)
		# dataloader
		if data_loader is not None:
			self.data_loader = data_loader
			cls_lst = self.data_loader.get_cls_idx()
			self.cls_lst = cls_lst
		else:
			self.data_loader = self.run_config.valid_loader
			self.data_loader.create_episode()
			cls_lst = self.data_loader.get_cls_idx()
			self.cls_lst = cls_lst
		
		state_dict = self.net.classifier.state_dict()
		new_state_dict = {'weight': state_dict['linear.weight'][cls_lst],
		                  'bias': state_dict['linear.bias'][cls_lst]}
		
		self.net.classifier = nn.Linear(1280, len(cls_lst), bias=True)
		self.net.classifier.load_state_dict(new_state_dict)
		
		# move network to GPU if available
		if torch.cuda.is_available() and (not no_gpu):
			self.device = torch.device('cuda:0')
			self.net = self.net.to(self.device)
			cudnn.benchmark = True
		else:
			self.device = torch.device('cpu')
		
		# net info
		net_info = get_net_info(
			self.net, self.run_config.data_provider.data_shape, measure_latency, False)
		self.net_info = net_info
		self.test_transform = self.run_config.data_provider.test.dataset.transform

		# criterion
		if isinstance(self.run_config.mixup_alpha, float):
			self.train_criterion = cross_entropy_loss_with_soft_target
		elif self.run_config.label_smoothing > 0:
			self.train_criterion = \
				lambda pred, target: cross_entropy_with_label_smoothing(pred, target, self.run_config.label_smoothing)
		else:
			self.train_criterion = nn.CrossEntropyLoss()
		self.test_criterion = nn.CrossEntropyLoss()
		
		# optimizer
		if self.run_config.no_decay_keys:
			keys = self.run_config.no_decay_keys.split('#')
			net_params = [
				self.network.get_parameters(keys, mode='exclude'),  # parameters with weight decay
				self.network.get_parameters(keys, mode='include'),  # parameters without weight decay
			]
		else:
			# noinspection PyBroadException
			try:
				net_params = self.network.weight_parameters()
			except Exception:
				net_params = []
				for param in self.network.parameters():
					if param.requires_grad:
						net_params.append(param)
		self.optimizer = self.run_config.build_optimizer(net_params)
		
		self.net = torch.nn.DataParallel(self.net)
		
		if self.mode == 'generator':
			# PP
			save_dir = f'{args.save_path}/predictor/model/ckpt_max_corr.pt'

			self.acc_predictor = pp.to('cuda')
			self.acc_predictor.load_state_dict(torch.load(save_dir))
			self.acc_predictor = torch.nn.DataParallel(self.acc_predictor)
			model = models.resnet18(pretrained=True).eval()
			feature_extractor = torch.nn.Sequential(*list(model.children())[:-1]).to(self.device)
			self.feature_extractor = torch.nn.DataParallel(feature_extractor)
	
	""" save path and log path """
	
	@property
	def save_path(self):
		if self.__dict__.get('_save_path', None) is None:
			save_path = os.path.join(self.path, 'checkpoint')
			os.makedirs(save_path, exist_ok=True)
			self.__dict__['_save_path'] = save_path
		return self.__dict__['_save_path']
	
	@property
	def logs_path(self):
		if self.__dict__.get('_logs_path', None) is None:
			logs_path = os.path.join(self.path, 'logs')
			os.makedirs(logs_path, exist_ok=True)
			self.__dict__['_logs_path'] = logs_path
		return self.__dict__['_logs_path']
	
	@property
	def network(self):
		return self.net.module if isinstance(self.net, nn.DataParallel) else self.net
	
	def write_log(self, log_str, prefix='valid', should_print=True, mode='a'):
		write_log(self.logs_path, log_str, prefix, should_print, mode)
	
	""" save and load models """
	
	def save_model(self, checkpoint=None, is_best=False, model_name=None):
		if checkpoint is None:
			checkpoint = {'state_dict': self.network.state_dict()}
		
		if model_name is None:
			model_name = 'checkpoint.pth.tar'
		
		checkpoint['dataset'] = self.run_config.dataset  # add `dataset` info to the checkpoint
		latest_fname = os.path.join(self.save_path, 'latest.txt')
		model_path = os.path.join(self.save_path, model_name)
		with open(latest_fname, 'w') as fout:
			fout.write(model_path + '\n')
		torch.save(checkpoint, model_path)
		
		if is_best:
			best_path = os.path.join(self.save_path, 'model_best.pth.tar')
			torch.save({'state_dict': checkpoint['state_dict']}, best_path)
	
	def load_model(self, model_fname=None):
		latest_fname = os.path.join(self.save_path, 'latest.txt')
		if model_fname is None and os.path.exists(latest_fname):
			with open(latest_fname, 'r') as fin:
				model_fname = fin.readline()
				if model_fname[-1] == '\n':
					model_fname = model_fname[:-1]
		# noinspection PyBroadException
		try:
			if model_fname is None or not os.path.exists(model_fname):
				model_fname = '%s/checkpoint.pth.tar' % self.save_path
				with open(latest_fname, 'w') as fout:
					fout.write(model_fname + '\n')
			print("=> loading checkpoint '{}'".format(model_fname))
			checkpoint = torch.load(model_fname, map_location='cpu')
		except Exception:
			print('fail to load checkpoint from %s' % self.save_path)
			return {}
		
		self.network.load_state_dict(checkpoint['state_dict'])
		if 'epoch' in checkpoint:
			self.start_epoch = checkpoint['epoch'] + 1
		if 'best_acc' in checkpoint:
			self.best_acc = checkpoint['best_acc']
		if 'optimizer' in checkpoint:
			self.optimizer.load_state_dict(checkpoint['optimizer'])
		
		print("=> loaded checkpoint '{}'".format(model_fname))
		return checkpoint
	
	def save_config(self, extra_run_config=None, extra_net_config=None):
		""" dump run_config and net_config to the model_folder """
		run_save_path = os.path.join(self.path, 'run.config')
		if not os.path.isfile(run_save_path):
			run_config = self.run_config.config
			if extra_run_config is not None:
				run_config.update(extra_run_config)
			json.dump(run_config, open(run_save_path, 'w'), indent=4)
			print('Run configs dump to %s' % run_save_path)
		
		try:
			net_save_path = os.path.join(self.path, 'net.config')
			net_config = self.network.config
			if extra_net_config is not None:
				net_config.update(extra_net_config)
			json.dump(net_config, open(net_save_path, 'w'), indent=4)
			print('Network configs dump to %s' % net_save_path)
		except Exception:
			print('%s do not support net config' % type(self.network))
	
	""" metric related """
	
	def get_metric_dict(self):
		return {
			'top1': AverageMeter(),
			'top5': AverageMeter(),
		}
	
	def update_metric(self, metric_dict, output, labels):
		acc1, acc5 = accuracy(output, labels, topk=(1, 5))
		metric_dict['top1'].update(acc1[0].item(), output.size(0))
		metric_dict['top5'].update(acc5[0].item(), output.size(0))
	
	def get_metric_vals(self, metric_dict, return_dict=False):
		if return_dict:
			return {
				key: metric_dict[key].avg for key in metric_dict
			}
		else:
			return [metric_dict[key].avg for key in metric_dict]
	
	def get_metric_names(self):
		return 'top1', 'top5'
	
	""" train and test """
	def validate(self, epoch=0, is_test=False, run_str='', net=None,
	             data_loader=None, no_logs=False, train_mode=False, net_setting=None):
		if net is None:
			net = self.net
		if not isinstance(net, nn.DataParallel):
			net = nn.DataParallel(net)
		
		if data_loader is not None:
			self.data_loader = data_loader
		
		if train_mode:
			net.train()
		else:
			net.eval()
		
		losses = AverageMeter()
		metric_dict = self.get_metric_dict()
		
		features_stack = []
		with torch.no_grad():
			with tqdm(total=len(self.data_loader),
			          desc='Validate Epoch #{} {}'.format(epoch + 1, run_str), disable=no_logs) as t:
				for i, (images, labels) in enumerate(self.data_loader):
					images, labels = images.to(self.device), labels.to(self.device)
					if self.mode == 'generator':
						features = self.feature_extractor(images).squeeze()
						features_stack.append(features)
					# compute output
					output = net(images)
					loss = self.test_criterion(output, labels)
					# measure accuracy and record loss
					self.update_metric(metric_dict, output, labels)
					
					losses.update(loss.item(), images.size(0))
					t.set_postfix({
						'loss': losses.avg,
						**self.get_metric_vals(metric_dict, return_dict=True),
						'img_size': images.size(2),
					})
					t.update(1)
					
		if self.mode == 'generator':
			features_stack = torch.cat(features_stack)
			igraph_g = decode_ofa_mbv3_to_igraph(net_setting)[0]
			D_mu = self.acc_predictor.module.set_encode(features_stack.unsqueeze(0).to('cuda'))
			G_mu = self.acc_predictor.module.graph_encode(igraph_g)
			pred_acc = self.acc_predictor.module.predict(D_mu.unsqueeze(0), G_mu).item()
			
		return losses.avg, self.get_metric_vals(metric_dict), \
		       pred_acc if self.mode == 'generator' else None
	

	def validate_all_resolution(self, epoch=0, is_test=False, net=None):
		if net is None:
			net = self.network
		if isinstance(self.run_config.data_provider.image_size, list):
			img_size_list, loss_list, top1_list, top5_list = [], [], [], []
			for img_size in self.run_config.data_provider.image_size:
				img_size_list.append(img_size)
				self.run_config.data_provider.assign_active_img_size(img_size)
				self.reset_running_statistics(net=net)
				loss, (top1, top5) = self.validate(epoch, is_test, net=net)
				loss_list.append(loss)
				top1_list.append(top1)
				top5_list.append(top5)
			return img_size_list, loss_list, top1_list, top5_list
		else:
			loss, (top1, top5) = self.validate(epoch, is_test, net=net)
			return [self.run_config.data_provider.active_img_size], [loss], [top1], [top5]
	
	def reset_running_statistics(self, net=None, subset_size=2000, subset_batch_size=200, data_loader=None):
		from ofa_local.imagenet_classification.elastic_nn.utils import set_running_statistics
		if net is None:
			net = self.network
		if data_loader is None:
			data_loader = self.run_config.random_sub_train_loader(subset_size, subset_batch_size)
		set_running_statistics(net, data_loader)
