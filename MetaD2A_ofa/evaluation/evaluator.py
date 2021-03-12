import os
import torch
import numpy as np
import random
import eval_utils
from codebase.networks import NSGANetV2
from codebase.run_manager import get_run_config
from ofa.elastic_nn.networks import OFAMobileNetV3
from ofa.imagenet_codebase.run_manager import RunManager
from ofa.elastic_nn.modules.dynamic_op import DynamicSeparableConv2d
from torchprofile import profile_macs
import copy
import json
import warnings

warnings.simplefilter("ignore")

DynamicSeparableConv2d.KERNEL_TRANSFORM_MODE = 1


class ArchManager:
	def __init__(self):
		self.num_blocks = 20
		self.num_stages = 5
		self.kernel_sizes = [3, 5, 7]
		self.expand_ratios = [3, 4, 6]
		self.depths = [2, 3, 4]
		self.resolutions = [160, 176, 192, 208, 224]
	
	def random_sample(self):
		sample = {}
		d = []
		e = []
		ks = []
		for i in range(self.num_stages):
			d.append(random.choice(self.depths))
		
		for i in range(self.num_blocks):
			e.append(random.choice(self.expand_ratios))
			ks.append(random.choice(self.kernel_sizes))
		
		sample = {
			'wid': None,
			'ks': ks,
			'e': e,
			'd': d,
			'r': [random.choice(self.resolutions)]
		}
		
		return sample
	
	def random_resample(self, sample, i):
		assert i >= 0 and i < self.num_blocks
		sample['ks'][i] = random.choice(self.kernel_sizes)
		sample['e'][i] = random.choice(self.expand_ratios)
	
	def random_resample_depth(self, sample, i):
		assert i >= 0 and i < self.num_stages
		sample['d'][i] = random.choice(self.depths)
	
	def random_resample_resolution(self, sample):
		sample['r'][0] = random.choice(self.resolutions)


def parse_string_list(string):
	if isinstance(string, str):
		# convert '[5 5 5 7 7 7 3 3 7 7 7 3 3]' to [5, 5, 5, 7, 7, 7, 3, 3, 7, 7, 7, 3, 3]
		return list(map(int, string[1:-1].split()))
	else:
		return string


def pad_none(x, depth, max_depth):
	new_x, counter = [], 0
	for d in depth:
		for _ in range(d):
			new_x.append(x[counter])
			counter += 1
		if d < max_depth:
			new_x += [None] * (max_depth - d)
	return new_x


def get_net_info(net, data_shape, measure_latency=None, print_info=True, clean=False, lut=None):
	net_info = eval_utils.get_net_info(
		net, data_shape, measure_latency, print_info=print_info, clean=clean, lut=lut)
	
	gpu_latency, cpu_latency = None, None
	for k in net_info.keys():
		if 'gpu' in k:
			gpu_latency = np.round(net_info[k]['val'], 2)
		if 'cpu' in k:
			cpu_latency = np.round(net_info[k]['val'], 2)
	
	return {
		'params': np.round(net_info['params'] / 1e6, 2),
		'flops': np.round(net_info['flops'] / 1e6, 2),
		'gpu': gpu_latency, 'cpu': cpu_latency
	}


def validate_config(config, max_depth=4):
	kernel_size, exp_ratio, depth = config['ks'], config['e'], config['d']
	
	if isinstance(kernel_size, str): kernel_size = parse_string_list(kernel_size)
	if isinstance(exp_ratio, str): exp_ratio = parse_string_list(exp_ratio)
	if isinstance(depth, str): depth = parse_string_list(depth)
	
	assert (isinstance(kernel_size, list) or isinstance(kernel_size, int))
	assert (isinstance(exp_ratio, list) or isinstance(exp_ratio, int))
	assert isinstance(depth, list)
	
	if len(kernel_size) < len(depth) * max_depth:
		kernel_size = pad_none(kernel_size, depth, max_depth)
	if len(exp_ratio) < len(depth) * max_depth:
		exp_ratio = pad_none(exp_ratio, depth, max_depth)
	
	# return {'ks': kernel_size, 'e': exp_ratio, 'd': depth, 'w': config['w']}
	return {'ks': kernel_size, 'e': exp_ratio, 'd': depth}


def set_nas_test_dataset(path, test_data_name, max_img):
	if not test_data_name in ['mnist', 'svhn', 'cifar10',
	                          'cifar100', 'aircraft', 'pets']: raise ValueError(test_data_name)
	
	dpath = path
	num_cls = 10  # mnist, svhn, cifar10
	if test_data_name in ['cifar100', 'aircraft']:
		num_cls = 100
	elif test_data_name == 'pets':
		num_cls = 37

	x = torch.load(dpath + f'/{test_data_name}bylabel')
	img_per_cls = min(int(max_img / num_cls), 20)
	return x, img_per_cls, num_cls


class OFAEvaluator:
	""" based on OnceForAll supernet taken from https://github.com/mit-han-lab/once-for-all """
	
	def __init__(self, args,
	             n_classes=1000,
	             model_path=None,
	             kernel_size=None, exp_ratio=None, depth=None):
		# default configurations
		self.kernel_size = [3, 5, 7] if kernel_size is None else kernel_size  # depth-wise conv kernel size
		self.exp_ratio = [3, 4, 6] if exp_ratio is None else exp_ratio  # expansion rate
		self.depth = [2, 3, 4] if depth is None else depth  # number of MB block repetition
		
		if 'w1.0' in model_path:
			self.width_mult = 1.0
		elif 'w1.2' in model_path:
			self.width_mult = 1.2
		else:
			raise ValueError
		
		self.engine = OFAMobileNetV3(
			n_classes=n_classes,
			dropout_rate=0, width_mult_list=self.width_mult, ks_list=self.kernel_size,
			expand_ratio_list=self.exp_ratio, depth_list=self.depth)
		
		init = torch.load(model_path, map_location='cpu')['state_dict']
		self.engine.load_weights_from_net(init)
		
		## metad2a
		self.arch_manager = ArchManager()
		self.num_gen_arch = args.num_gen_arch
		if not args.model_config.startswith('flops@'):
			meta_test_path = os.path.join(
				args.save_path, 'meta_test', args.data_name, f'{args.num_gen_arch}', 'generated_arch')
			path = f'{meta_test_path}/run_1.txt'
			with open(path, 'r') as f:
				self.lines = f.readlines()[1:]
	
	def sample_random_architecture(self):
		sampled_architecture = self.arch_manager.random_sample()
		return sampled_architecture
	
	def get_architecture(self, args):
		g_lst, pred_acc_lst, x_lst = [], [], []
		searched_g, max_pred_acc = None, 0
		
		with torch.no_grad():
			for n in range(self.num_gen_arch):
				file_acc = self.lines[n].split()[0]
				g_dict = ' '.join(self.lines[n].split())
				g = json.loads(g_dict.replace("'", "\""))
				
				if args.bound is not None:
					subnet, config = self.sample(g)
					net = NSGANetV2.build_from_config(subnet.config,
					                                  drop_connect_rate=args.drop_path)
					inputs = torch.randn(1, 3, args.img_size, args.img_size)
					flops = profile_macs(copy.deepcopy(net), inputs) / 1e6
					if flops <= args.bound:
						searched_g = g
						break
				else:
					searched_g = g
					pred_acc_lst.append(file_acc)
					break
		
		if searched_g is None:
			raise ValueError(searched_g)
		return searched_g, pred_acc_lst
	

	def sample(self, config=None):
		""" randomly sample a sub-network """
		if config is not None:
			config = validate_config(config)
			self.engine.set_active_subnet(ks=config['ks'], e=config['e'], d=config['d'])
		else:
			config = self.engine.sample_active_subnet()
		
		subnet = self.engine.get_active_subnet(preserve_weight=True)
		return subnet, config
	
	@staticmethod
	def save_net_config(path, net, config_name='net.config'):
		""" dump run_config and net_config to the model_folder """
		net_save_path = os.path.join(path, config_name)
		json.dump(net.config, open(net_save_path, 'w'), indent=4)
		print('Network configs dump to %s' % net_save_path)
	
	@staticmethod
	def save_net(path, net, model_name):
		""" dump net weight as checkpoint """
		if isinstance(net, torch.nn.DataParallel):
			checkpoint = {'state_dict': net.module.state_dict()}
		else:
			checkpoint = {'state_dict': net.state_dict()}
		model_path = os.path.join(path, model_name)
		torch.save(checkpoint, model_path)
		print('Network model dump to %s' % model_path)
