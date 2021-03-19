######################################################################################
# Copyright (c) Han Cai, Once for All, ICLR 2020 [GitHub OFA]
# Modified by Hayeon Lee, Eunyoung Hyung, MetaD2A, ICLR2021, 2021. 03 [GitHub MetaD2A]
######################################################################################
import torch
import torch.nn as nn
from torch.distributions import Bernoulli
from collections import OrderedDict
from ofa_local.utils import get_same_padding, min_divisible_value, SEModule, ShuffleLayer
from ofa_local.utils import MyNetwork, MyModule
from ofa_local.utils import build_activation, make_divisible

__all__ = [
	'set_layer_from_config',
	'ConvLayer', 'IdentityLayer', 'LinearLayer', 'MultiHeadLinearLayer', 'ZeroLayer', 'MBConvLayer',
	'ResidualBlock', 'ResNetBottleneckBlock',
]


class DropBlock(nn.Module):
	def __init__(self, block_size):
		super(DropBlock, self).__init__()
		
		self.block_size = block_size
		
	def forward(self, x, gamma):
		# shape: (bsize, channels, height, width)
		
		if self.training:
			batch_size, channels, height, width = x.shape
			
			bernoulli = Bernoulli(gamma)
			mask = bernoulli.sample(
				(batch_size, channels, height - (self.block_size - 1), width - (self.block_size - 1))).cuda()
			# print((x.sample[-2], x.sample[-1]))
			block_mask = self._compute_block_mask(mask)
			# print (block_mask.size())
			# print (x.size())
			countM = block_mask.size()[0] * block_mask.size()[1] * block_mask.size()[2] * block_mask.size()[3]
			count_ones = block_mask.sum()
			
			return block_mask * x * (countM / count_ones)
		else:
			return x
	
	def _compute_block_mask(self, mask):
		left_padding = int((self.block_size - 1) / 2)
		right_padding = int(self.block_size / 2)
		
		batch_size, channels, height, width = mask.shape
		# print ("mask", mask[0][0])
		non_zero_idxs = mask.nonzero()
		nr_blocks = non_zero_idxs.shape[0]
		
		offsets = torch.stack(
			[
				torch.arange(self.block_size).view(-1, 1).expand(self.block_size, self.block_size).reshape(-1),
				# - left_padding,
				torch.arange(self.block_size).repeat(self.block_size),  # - left_padding
			]
		).t().cuda()
		offsets = torch.cat((torch.zeros(self.block_size ** 2, 2).cuda().long(), offsets.long()), 1)
		
		if nr_blocks > 0:
			non_zero_idxs = non_zero_idxs.repeat(self.block_size ** 2, 1)
			offsets = offsets.repeat(nr_blocks, 1).view(-1, 4)
			offsets = offsets.long()
			
			block_idxs = non_zero_idxs + offsets
			# block_idxs += left_padding
			padded_mask = F.pad(mask, (left_padding, right_padding, left_padding, right_padding))
			padded_mask[block_idxs[:, 0], block_idxs[:, 1], block_idxs[:, 2], block_idxs[:, 3]] = 1.
		else:
			padded_mask = F.pad(mask, (left_padding, right_padding, left_padding, right_padding))
		
		block_mask = 1 - padded_mask  # [:height, :width]
		return block_mask
	
	
def set_layer_from_config(layer_config):
	if layer_config is None:
		return None

	name2layer = {
		ConvLayer.__name__: ConvLayer,
		IdentityLayer.__name__: IdentityLayer,
		LinearLayer.__name__: LinearLayer,
		MultiHeadLinearLayer.__name__: MultiHeadLinearLayer,
		ZeroLayer.__name__: ZeroLayer,
		MBConvLayer.__name__: MBConvLayer,
		'MBInvertedConvLayer': MBConvLayer,
		##########################################################
		ResidualBlock.__name__: ResidualBlock,
		ResNetBottleneckBlock.__name__: ResNetBottleneckBlock,
	}

	layer_name = layer_config.pop('name')
	layer = name2layer[layer_name]
	return layer.build_from_config(layer_config)


class My2DLayer(MyModule):

	def __init__(self, in_channels, out_channels,
				 use_bn=True, act_func='relu', dropout_rate=0, ops_order='weight_bn_act'):
		super(My2DLayer, self).__init__()
		self.in_channels = in_channels
		self.out_channels = out_channels

		self.use_bn = use_bn
		self.act_func = act_func
		self.dropout_rate = dropout_rate
		self.ops_order = ops_order

		""" modules """
		modules = {}
		# batch norm
		if self.use_bn:
			if self.bn_before_weight:
				modules['bn'] = nn.BatchNorm2d(in_channels)
			else:
				modules['bn'] = nn.BatchNorm2d(out_channels)
		else:
			modules['bn'] = None
		# activation
		modules['act'] = build_activation(self.act_func, self.ops_list[0] != 'act' and self.use_bn)
		# dropout
		if self.dropout_rate > 0:
			modules['dropout'] = nn.Dropout2d(self.dropout_rate, inplace=True)
		else:
			modules['dropout'] = None
		# weight
		modules['weight'] = self.weight_op()

		# add modules
		for op in self.ops_list:
			if modules[op] is None:
				continue
			elif op == 'weight':
				# dropout before weight operation
				if modules['dropout'] is not None:
					self.add_module('dropout', modules['dropout'])
				for key in modules['weight']:
					self.add_module(key, modules['weight'][key])
			else:
				self.add_module(op, modules[op])

	@property
	def ops_list(self):
		return self.ops_order.split('_')

	@property
	def bn_before_weight(self):
		for op in self.ops_list:
			if op == 'bn':
				return True
			elif op == 'weight':
				return False
		raise ValueError('Invalid ops_order: %s' % self.ops_order)

	def weight_op(self):
		raise NotImplementedError

	""" Methods defined in MyModule """

	def forward(self, x):
		# similar to nn.Sequential
		for module in self._modules.values():
			x = module(x)
		return x

	@property
	def module_str(self):
		raise NotImplementedError

	@property
	def config(self):
		return {
			'in_channels': self.in_channels,
			'out_channels': self.out_channels,
			'use_bn': self.use_bn,
			'act_func': self.act_func,
			'dropout_rate': self.dropout_rate,
			'ops_order': self.ops_order,
		}

	@staticmethod
	def build_from_config(config):
		raise NotImplementedError


class ConvLayer(My2DLayer):

	def __init__(self, in_channels, out_channels,
				 kernel_size=3, stride=1, dilation=1, groups=1, bias=False, has_shuffle=False, use_se=False,
				 use_bn=True, act_func='relu', dropout_rate=0, ops_order='weight_bn_act'):
		# default normal 3x3_Conv with bn and relu
		self.kernel_size = kernel_size
		self.stride = stride
		self.dilation = dilation
		self.groups = groups
		self.bias = bias
		self.has_shuffle = has_shuffle
		self.use_se = use_se

		super(ConvLayer, self).__init__(in_channels, out_channels, use_bn, act_func, dropout_rate, ops_order)
		if self.use_se:
			self.add_module('se', SEModule(self.out_channels))

	def weight_op(self):
		padding = get_same_padding(self.kernel_size)
		if isinstance(padding, int):
			padding *= self.dilation
		else:
			padding[0] *= self.dilation
			padding[1] *= self.dilation

		weight_dict = OrderedDict({
			'conv': nn.Conv2d(
				self.in_channels, self.out_channels, kernel_size=self.kernel_size, stride=self.stride, padding=padding,
				dilation=self.dilation, groups=min_divisible_value(self.in_channels, self.groups), bias=self.bias
			)
		})
		if self.has_shuffle and self.groups > 1:
			weight_dict['shuffle'] = ShuffleLayer(self.groups)

		return weight_dict

	@property
	def module_str(self):
		if isinstance(self.kernel_size, int):
			kernel_size = (self.kernel_size, self.kernel_size)
		else:
			kernel_size = self.kernel_size
		if self.groups == 1:
			if self.dilation > 1:
				conv_str = '%dx%d_DilatedConv' % (kernel_size[0], kernel_size[1])
			else:
				conv_str = '%dx%d_Conv' % (kernel_size[0], kernel_size[1])
		else:
			if self.dilation > 1:
				conv_str = '%dx%d_DilatedGroupConv' % (kernel_size[0], kernel_size[1])
			else:
				conv_str = '%dx%d_GroupConv' % (kernel_size[0], kernel_size[1])
		conv_str += '_O%d' % self.out_channels
		if self.use_se:
			conv_str = 'SE_' + conv_str
		conv_str += '_' + self.act_func.upper()
		if self.use_bn:
			if isinstance(self.bn, nn.GroupNorm):
				conv_str += '_GN%d' % self.bn.num_groups
			elif isinstance(self.bn, nn.BatchNorm2d):
				conv_str += '_BN'
		return conv_str

	@property
	def config(self):
		return {
			'name': ConvLayer.__name__,
			'kernel_size': self.kernel_size,
			'stride': self.stride,
			'dilation': self.dilation,
			'groups': self.groups,
			'bias': self.bias,
			'has_shuffle': self.has_shuffle,
			'use_se': self.use_se,
			**super(ConvLayer, self).config
		}

	@staticmethod
	def build_from_config(config):
		return ConvLayer(**config)


class IdentityLayer(My2DLayer):

	def __init__(self, in_channels, out_channels,
				 use_bn=False, act_func=None, dropout_rate=0, ops_order='weight_bn_act'):
		super(IdentityLayer, self).__init__(in_channels, out_channels, use_bn, act_func, dropout_rate, ops_order)

	def weight_op(self):
		return None

	@property
	def module_str(self):
		return 'Identity'

	@property
	def config(self):
		return {
			'name': IdentityLayer.__name__,
			**super(IdentityLayer, self).config,
		}

	@staticmethod
	def build_from_config(config):
		return IdentityLayer(**config)


class LinearLayer(MyModule):

	def __init__(self, in_features, out_features, bias=True,
				 use_bn=False, act_func=None, dropout_rate=0, ops_order='weight_bn_act'):
		super(LinearLayer, self).__init__()

		self.in_features = in_features
		self.out_features = out_features
		self.bias = bias

		self.use_bn = use_bn
		self.act_func = act_func
		self.dropout_rate = dropout_rate
		self.ops_order = ops_order

		""" modules """
		modules = {}
		# batch norm
		if self.use_bn:
			if self.bn_before_weight:
				modules['bn'] = nn.BatchNorm1d(in_features)
			else:
				modules['bn'] = nn.BatchNorm1d(out_features)
		else:
			modules['bn'] = None
		# activation
		modules['act'] = build_activation(self.act_func, self.ops_list[0] != 'act')
		# dropout
		if self.dropout_rate > 0:
			modules['dropout'] = nn.Dropout(self.dropout_rate, inplace=True)
		else:
			modules['dropout'] = None
		# linear
		modules['weight'] = {'linear': nn.Linear(self.in_features, self.out_features, self.bias)}

		# add modules
		for op in self.ops_list:
			if modules[op] is None:
				continue
			elif op == 'weight':
				if modules['dropout'] is not None:
					self.add_module('dropout', modules['dropout'])
				for key in modules['weight']:
					self.add_module(key, modules['weight'][key])
			else:
				self.add_module(op, modules[op])

	@property
	def ops_list(self):
		return self.ops_order.split('_')

	@property
	def bn_before_weight(self):
		for op in self.ops_list:
			if op == 'bn':
				return True
			elif op == 'weight':
				return False
		raise ValueError('Invalid ops_order: %s' % self.ops_order)

	def forward(self, x):
		for module in self._modules.values():
			x = module(x)
		return x

	@property
	def module_str(self):
		return '%dx%d_Linear' % (self.in_features, self.out_features)

	@property
	def config(self):
		return {
			'name': LinearLayer.__name__,
			'in_features': self.in_features,
			'out_features': self.out_features,
			'bias': self.bias,
			'use_bn': self.use_bn,
			'act_func': self.act_func,
			'dropout_rate': self.dropout_rate,
			'ops_order': self.ops_order,
		}

	@staticmethod
	def build_from_config(config):
		return LinearLayer(**config)


class MultiHeadLinearLayer(MyModule):

	def __init__(self, in_features, out_features, num_heads=1, bias=True, dropout_rate=0):
		super(MultiHeadLinearLayer, self).__init__()
		self.in_features = in_features
		self.out_features = out_features
		self.num_heads = num_heads

		self.bias = bias
		self.dropout_rate = dropout_rate

		if self.dropout_rate > 0:
			self.dropout = nn.Dropout(self.dropout_rate, inplace=True)
		else:
			self.dropout = None

		self.layers = nn.ModuleList()
		for k in range(num_heads):
			layer = nn.Linear(in_features, out_features, self.bias)
			self.layers.append(layer)

	def forward(self, inputs):
		if self.dropout is not None:
			inputs = self.dropout(inputs)

		outputs = []
		for layer in self.layers:
			output = layer.forward(inputs)
			outputs.append(output)

		outputs = torch.stack(outputs, dim=1)
		return outputs

	@property
	def module_str(self):
		return self.__repr__()

	@property
	def config(self):
		return {
			'name': MultiHeadLinearLayer.__name__,
			'in_features': self.in_features,
			'out_features': self.out_features,
			'num_heads': self.num_heads,
			'bias': self.bias,
			'dropout_rate': self.dropout_rate,
		}

	@staticmethod
	def build_from_config(config):
		return MultiHeadLinearLayer(**config)

	def __repr__(self):
		return 'MultiHeadLinear(in_features=%d, out_features=%d, num_heads=%d, bias=%s, dropout_rate=%s)' % (
			self.in_features, self.out_features, self.num_heads, self.bias, self.dropout_rate
		)


class ZeroLayer(MyModule):

	def __init__(self):
		super(ZeroLayer, self).__init__()

	def forward(self, x):
		raise ValueError

	@property
	def module_str(self):
		return 'Zero'

	@property
	def config(self):
		return {
			'name': ZeroLayer.__name__,
		}

	@staticmethod
	def build_from_config(config):
		return ZeroLayer()


class MBConvLayer(MyModule):

	def __init__(self, in_channels, out_channels,
				 kernel_size=3, stride=1, expand_ratio=6, mid_channels=None, act_func='relu6', use_se=False,
				 groups=None):
		super(MBConvLayer, self).__init__()

		self.in_channels = in_channels
		self.out_channels = out_channels

		self.kernel_size = kernel_size
		self.stride = stride
		self.expand_ratio = expand_ratio
		self.mid_channels = mid_channels
		self.act_func = act_func
		self.use_se = use_se
		self.groups = groups

		if self.mid_channels is None:
			feature_dim = round(self.in_channels * self.expand_ratio)
		else:
			feature_dim = self.mid_channels

		if self.expand_ratio == 1:
			self.inverted_bottleneck = None
		else:
			self.inverted_bottleneck = nn.Sequential(OrderedDict([
				('conv', nn.Conv2d(self.in_channels, feature_dim, 1, 1, 0, bias=False)),
				('bn', nn.BatchNorm2d(feature_dim)),
				('act', build_activation(self.act_func, inplace=True)),
			]))

		pad = get_same_padding(self.kernel_size)
		groups = feature_dim if self.groups is None else min_divisible_value(feature_dim, self.groups)
		depth_conv_modules = [
			('conv', nn.Conv2d(feature_dim, feature_dim, kernel_size, stride, pad, groups=groups, bias=False)),
			('bn', nn.BatchNorm2d(feature_dim)),
			('act', build_activation(self.act_func, inplace=True))
		]
		if self.use_se:
			depth_conv_modules.append(('se', SEModule(feature_dim)))
		self.depth_conv = nn.Sequential(OrderedDict(depth_conv_modules))

		self.point_linear = nn.Sequential(OrderedDict([
			('conv', nn.Conv2d(feature_dim, out_channels, 1, 1, 0, bias=False)),
			('bn', nn.BatchNorm2d(out_channels)),
		]))

	def forward(self, x):
		if self.inverted_bottleneck:
			x = self.inverted_bottleneck(x)
		x = self.depth_conv(x)
		x = self.point_linear(x)
		return x

	@property
	def module_str(self):
		if self.mid_channels is None:
			expand_ratio = self.expand_ratio
		else:
			expand_ratio = self.mid_channels // self.in_channels
		layer_str = '%dx%d_MBConv%d_%s' % (self.kernel_size, self.kernel_size, expand_ratio, self.act_func.upper())
		if self.use_se:
			layer_str = 'SE_' + layer_str
		layer_str += '_O%d' % self.out_channels
		if self.groups is not None:
			layer_str += '_G%d' % self.groups
		if isinstance(self.point_linear.bn, nn.GroupNorm):
			layer_str += '_GN%d' % self.point_linear.bn.num_groups
		elif isinstance(self.point_linear.bn, nn.BatchNorm2d):
			layer_str += '_BN'

		return layer_str

	@property
	def config(self):
		return {
			'name': MBConvLayer.__name__,
			'in_channels': self.in_channels,
			'out_channels': self.out_channels,
			'kernel_size': self.kernel_size,
			'stride': self.stride,
			'expand_ratio': self.expand_ratio,
			'mid_channels': self.mid_channels,
			'act_func': self.act_func,
			'use_se': self.use_se,
			'groups': self.groups,
		}

	@staticmethod
	def build_from_config(config):
		return MBConvLayer(**config)


class ResidualBlock(MyModule):

	def __init__(self, conv, shortcut, dropout_rate, dropblock, block_size):
		super(ResidualBlock, self).__init__()

		self.conv = conv
		self.shortcut = shortcut
		# hayeon
		self.num_batches_tracked = 0
		self.dropout_rate = dropout_rate
		self.dropblock = dropblock
		self.block_size = block_size
		self.DropBlock = DropBlock(block_size=self.block_size)
		
	def forward(self, x):
		# hayeon
		self.num_batches_tracked += 1

		if self.conv is None or isinstance(self.conv, ZeroLayer):
			res = x
		elif self.shortcut is None or isinstance(self.shortcut, ZeroLayer):
			res = self.conv(x)
		else:
			res = self.conv(x) + self.shortcut(x)

		# hayeon
		if self.dropout_rate > 0:
			if self.dropblock:
				feat_size = res.size()[2]
				keep_rate = max(1.0 - self.dropout_rate / (20*2000) * (self.num_batches_tracked), 1.0 - self.drop_rate)
				gamma = (1 - keep_rate) / self.block_size**2 * feat_size**2 / (feat_size - self.block_size + 1)**2
				res = self.DropBlock(res, gamma=gamma)
			else:
				res = F.dropout(res, p=self.dropout_rate, training=self.training, inplace=True)		
		return res

	@property
	def module_str(self):
		return '(%s, %s)' % (
			self.conv.module_str if self.conv is not None else None,
			self.shortcut.module_str if self.shortcut is not None else None
		)

	@property
	def config(self):
		return {
			'name': ResidualBlock.__name__,
			'conv': self.conv.config if self.conv is not None else None,
			'shortcut': self.shortcut.config if self.shortcut is not None else None,
		}

	@staticmethod
	def build_from_config(config):
		conv_config = config['conv'] if 'conv' in config else config['mobile_inverted_conv']
		conv = set_layer_from_config(conv_config)
		shortcut = set_layer_from_config(config['shortcut'])
		return ResidualBlock(conv, shortcut)

	@property
	def mobile_inverted_conv(self):
		return self.conv


class ResNetBottleneckBlock(MyModule):

	def __init__(self, in_channels, out_channels,
				 kernel_size=3, stride=1, expand_ratio=0.25, mid_channels=None, act_func='relu', groups=1,
				 downsample_mode='avgpool_conv'):
		super(ResNetBottleneckBlock, self).__init__()

		self.in_channels = in_channels
		self.out_channels = out_channels

		self.kernel_size = kernel_size
		self.stride = stride
		self.expand_ratio = expand_ratio
		self.mid_channels = mid_channels
		self.act_func = act_func
		self.groups = groups

		self.downsample_mode = downsample_mode

		if self.mid_channels is None:
			feature_dim = round(self.out_channels * self.expand_ratio)
		else:
			feature_dim = self.mid_channels

		feature_dim = make_divisible(feature_dim, MyNetwork.CHANNEL_DIVISIBLE)
		self.mid_channels = feature_dim

		# build modules
		self.conv1 = nn.Sequential(OrderedDict([
			('conv', nn.Conv2d(self.in_channels, feature_dim, 1, 1, 0, bias=False)),
			('bn', nn.BatchNorm2d(feature_dim)),
			('act', build_activation(self.act_func, inplace=True)),
		]))

		pad = get_same_padding(self.kernel_size)
		self.conv2 = nn.Sequential(OrderedDict([
			('conv', nn.Conv2d(feature_dim, feature_dim, kernel_size, stride, pad, groups=groups, bias=False)),
			('bn', nn.BatchNorm2d(feature_dim)),
			('act', build_activation(self.act_func, inplace=True))
		]))

		self.conv3 = nn.Sequential(OrderedDict([
			('conv', nn.Conv2d(feature_dim, self.out_channels, 1, 1, 0, bias=False)),
			('bn', nn.BatchNorm2d(self.out_channels)),
		]))

		if stride == 1 and in_channels == out_channels:
			self.downsample = IdentityLayer(in_channels, out_channels)
		elif self.downsample_mode == 'conv':
			self.downsample = nn.Sequential(OrderedDict([
				('conv', nn.Conv2d(in_channels, out_channels, 1, stride, 0, bias=False)),
				('bn', nn.BatchNorm2d(out_channels)),
			]))
		elif self.downsample_mode == 'avgpool_conv':
			self.downsample = nn.Sequential(OrderedDict([
				('avg_pool', nn.AvgPool2d(kernel_size=stride, stride=stride, padding=0, ceil_mode=True)),
				('conv', nn.Conv2d(in_channels, out_channels, 1, 1, 0, bias=False)),
				('bn', nn.BatchNorm2d(out_channels)),
			]))
		else:
			raise NotImplementedError

		self.final_act = build_activation(self.act_func, inplace=True)

	def forward(self, x):
		residual = self.downsample(x)

		x = self.conv1(x)
		x = self.conv2(x)
		x = self.conv3(x)

		x = x + residual
		x = self.final_act(x)
		return x

	@property
	def module_str(self):
		return '(%s, %s)' % (
			'%dx%d_BottleneckConv_%d->%d->%d_S%d_G%d' % (
				self.kernel_size, self.kernel_size, self.in_channels, self.mid_channels, self.out_channels,
				self.stride, self.groups
			),
			'Identity' if isinstance(self.downsample, IdentityLayer) else self.downsample_mode,
		)

	@property
	def config(self):
		return {
			'name': ResNetBottleneckBlock.__name__,
			'in_channels': self.in_channels,
			'out_channels': self.out_channels,
			'kernel_size': self.kernel_size,
			'stride': self.stride,
			'expand_ratio': self.expand_ratio,
			'mid_channels': self.mid_channels,
			'act_func': self.act_func,
			'groups': self.groups,
			'downsample_mode': self.downsample_mode,
		}

	@staticmethod
	def build_from_config(config):
		return ResNetBottleneckBlock(**config)
