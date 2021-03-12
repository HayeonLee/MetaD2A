import os
import sys
import json
import logging
import numpy as np
import copy
import torch
import torch.nn as nn
import random
import torch.optim as optim
from evaluator import OFAEvaluator
from torchprofile import profile_macs
from codebase.networks import NSGANetV2
from parser import get_parse
from eval_utils import get_dataset


args = get_parse()
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
device_list = [int(_) for _ in args.gpu.split(',')]
args.n_gpus = len(device_list)
args.device = torch.device("cuda:0")

if args.seed is None or args.seed < 0: args.seed = random.randint(1, 100000)
torch.cuda.manual_seed(args.seed)
torch.manual_seed(args.seed)
np.random.seed(args.seed)
random.seed(args.seed)


evaluator = OFAEvaluator(args,
                         model_path='../.torch/ofa_nets/ofa_mbv3_d234_e346_k357_w1.0')

args.save_path = os.path.join(args.save_path, f'evaluation/{args.data_name}')
if args.model_config.startswith('flops@'):
	args.save_path += f'-nsganetV2-{args.model_config}-{args.seed}'
else:
	args.save_path += f'-metaD2A-{args.bound}-{args.seed}'
if not os.path.exists(args.save_path):
	os.makedirs(args.save_path)

args.data_path = os.path.join(args.data_path, args.data_name)

log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                    format=log_format, datefmt='%m/%d %I:%M:%S %p')
fh = logging.FileHandler(os.path.join(args.save_path, 'log.txt'))
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)
if not torch.cuda.is_available():
	logging.info('no gpu self.args.device available')
	sys.exit(1)
logging.info("args = %s", args)



def set_architecture(n_cls):
	if args.model_config.startswith('flops@'):
		names = {'cifar10': 'CIFAR-10', 'cifar100': 'CIFAR-100',
		         'aircraft100': 'Aircraft', 'pets': 'Pets'}
		p = os.path.join('./searched-architectures/{}/net-{}/net.subnet'.
		                 format(names[args.data_name], args.model_config))
		g = json.load(open(p))
	else:
		g, acc = evaluator.get_architecture(args)
	
	subnet, config = evaluator.sample(g)
	net = NSGANetV2.build_from_config(subnet.config, drop_connect_rate=args.drop_path)
	net.load_state_dict(subnet.state_dict())
	
	NSGANetV2.reset_classifier(
		net, last_channel=net.classifier.in_features,
		n_classes=n_cls, dropout_rate=args.drop)
	# calculate #Paramaters and #FLOPS
	inputs = torch.randn(1, 3, args.img_size, args.img_size)
	flops = profile_macs(copy.deepcopy(net), inputs) / 1e6
	params = sum(p.numel() for p in net.parameters() if p.requires_grad) / 1e6
	net_name = "net_flops@{:.0f}".format(flops)
	logging.info('#params {:.2f}M, #flops {:.0f}M'.format(params, flops))
	OFAEvaluator.save_net_config(args.save_path, net, net_name + '.config')
	if args.n_gpus > 1:
		net = nn.DataParallel(net)  # data parallel in case more than 1 gpu available
	net = net.to(args.device)
	
	return net, net_name


def train(train_queue, net, criterion, optimizer):
	net.train()
	train_loss, correct, total = 0, 0, 0
	for step, (inputs, targets) in enumerate(train_queue):
		# upsample by bicubic to match imagenet training size
		inputs, targets = inputs.to(args.device), targets.to(args.device)
		optimizer.zero_grad()
		outputs = net(inputs)
		loss = criterion(outputs, targets)
		loss.backward()
		nn.utils.clip_grad_norm_(net.parameters(), args.grad_clip)
		optimizer.step()
		train_loss += loss.item()
		_, predicted = outputs.max(1)
		total += targets.size(0)
		correct += predicted.eq(targets).sum().item()
		if step % args.report_freq == 0:
			logging.info('train %03d %e %f', step, train_loss / total, 100. * correct / total)
	logging.info('train acc %f', 100. * correct / total)
	return train_loss / total, 100. * correct / total


def infer(valid_queue, net, criterion, early_stop=False):
	net.eval()
	test_loss, correct, total = 0, 0, 0
	with torch.no_grad():
		for step, (inputs, targets) in enumerate(valid_queue):
			inputs, targets = inputs.to(args.device), targets.to(args.device)
			outputs = net(inputs)
			loss = criterion(outputs, targets)
			test_loss += loss.item()
			_, predicted = outputs.max(1)
			total += targets.size(0)
			correct += predicted.eq(targets).sum().item()
			if step % args.report_freq == 0:
				logging.info('valid %03d %e %f', step, test_loss / total, 100. * correct / total)
			if early_stop and step == 10:
				break
	acc = 100. * correct / total
	logging.info('valid acc %f', 100. * correct / total)
	
	return test_loss / total, acc


def main():
	best_acc, top_checkpoints = 0, []
	
	train_queue, valid_queue, n_cls = get_dataset(args)
	net, net_name = set_architecture(n_cls)
	parameters = filter(lambda p: p.requires_grad, net.parameters())
	optimizer = optim.SGD(parameters, lr=args.lr, momentum=args.momentum,                  weight_decay=args.weight_decay)
	criterion = nn.CrossEntropyLoss().to(args.device)
	scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)
	
	for epoch in range(args.epochs):
		logging.info('epoch %d lr %e', epoch, scheduler.get_lr()[0])
		
		train(train_queue, net, criterion, optimizer)
		_, valid_acc = infer(valid_queue, net, criterion)
		# checkpoint saving
		
		if len(top_checkpoints) < args.topk:
			OFAEvaluator.save_net(args.save_path, net, net_name + '.ckpt{}'.format(epoch))
			top_checkpoints.append((os.path.join(args.save_path, net_name + '.ckpt{}'.format(epoch)), valid_acc))
		else:
			idx = np.argmin([x[1] for x in top_checkpoints])
			if valid_acc > top_checkpoints[idx][1]:
				OFAEvaluator.save_net(args.save_path, net, net_name + '.ckpt{}'.format(epoch))
				top_checkpoints.append((os.path.join(args.save_path, net_name + '.ckpt{}'.format(epoch)), valid_acc))
				# remove the idx
				os.remove(top_checkpoints[idx][0])
				top_checkpoints.pop(idx)
				print(top_checkpoints)
		if valid_acc > best_acc:
			OFAEvaluator.save_net(args.save_path, net, net_name + '.best')
			best_acc = valid_acc
		scheduler.step()
	


if __name__ == '__main__':
	main()
