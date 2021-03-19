import numpy as np
import torchvision.models as models
import torchvision.datasets as dset
import os
import torch
import argparse
import random
import torchvision.transforms as transforms
import os, sys
if sys.version_info[0] == 2:
	import cPickle as pickle
else:
	import pickle
from PIL import Image

parser = argparse.ArgumentParser("sota")
parser.add_argument('--gpu', type=str, default='0', help='set visible gpus')
parser.add_argument('--data-path', type=str, default='data', help='the path of save directory')
parser.add_argument('--dataset', type=str, default='cifar10', help='choose dataset')
parser.add_argument('--seed', type=int, default=-1, help='random seed')
args = parser.parse_args()

if args.seed is None or args.seed < 0: args.seed = random.randint(1, 100000)

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
np.random.seed(args.seed)
random.seed(args.seed)

# remove last fully-connected layer
model = models.resnet18(pretrained=True).eval()
feature_extractor = torch.nn.Sequential(*list(model.children())[:-1])


def get_transform(dataset):
	if args.dataset == 'mnist':
		mean, std = [0.1307, 0.1307, 0.1307], [0.3081, 0.3081, 0.3081]
	elif args.dataset == 'svhn':
		mean, std = [0.4376821, 0.4437697, 0.47280442], [0.19803012, 0.20101562, 0.19703614]
	elif args.dataset == 'cifar10':
		mean = [x / 255 for x in [125.3, 123.0, 113.9]]
		std = [x / 255 for x in [63.0, 62.1, 66.7]]
	elif args.dataset == 'cifar100':
		mean = [x / 255 for x in [129.3, 124.1, 112.4]]
		std = [x / 255 for x in [68.2, 65.4, 70.4]]
	elif args.dataset == 'imagenet32':
		mean = [x / 255 for x in [122.68, 116.66, 104.01]]
		std = [x / 255 for x in [66.22, 64.20, 67.86]]

	transform = transforms.Compose([
		transforms.Resize((32, 32)),
		transforms.ToTensor(),
		transforms.Normalize(mean, std),
	])
	if dataset == 'mnist':
		transform.transforms.append(transforms.Lambda(lambda x: x.repeat(3, 1, 1)))
	return transform


def process(dataset, n_classes):
	data_label = {i: [] for i in range(n_classes)}
	for x, y in dataset:
		data_label[y].append(x)
	for i in range(n_classes):
		data_label[i] = torch.stack(data_label[i])
	
	holder = {i: [] for i in range(n_classes)}
	for i in range(n_classes):
		with torch.no_grad():
			data = feature_extractor(data_label[i])
			holder[i].append(data.squeeze())
	return holder



class ImageNet32(object):
	train_list = [
		['train_data_batch_1', '27846dcaa50de8e21a7d1a35f30f0e91'],
		['train_data_batch_2', 'c7254a054e0e795c69120a5727050e3f'],
		['train_data_batch_3', '4333d3df2e5ffb114b05d2ffc19b1e87'],
		['train_data_batch_4', '1620cdf193304f4a92677b695d70d10f'],
		['train_data_batch_5', '348b3c2fdbb3940c4e9e834affd3b18d'],
		['train_data_batch_6', '6e765307c242a1b3d7d5ef9139b48945'],
		['train_data_batch_7', '564926d8cbf8fc4818ba23d2faac7564'],
		['train_data_batch_8', 'f4755871f718ccb653440b9dd0ebac66'],
		['train_data_batch_9', 'bb6dd660c38c58552125b1a92f86b5d4'],
		['train_data_batch_10', '8f03f34ac4b42271a294f91bf480f29b'],
	]
	valid_list = [
		['val_data', '3410e3017fdaefba8d5073aaa65e4bd6'],
	]
	
	def __init__(self, root, n_class, transform):
		self.transform = transform
		downloaded_list = self.train_list
		self.n_class = n_class
		self.data_label = {i: [] for i in range(n_class)}
		self.data = []
		self.targets = []
		
		for i, (file_name, checksum) in enumerate(downloaded_list):
			file_path = os.path.join(root, file_name)
			with open(file_path, 'rb') as f:
				if sys.version_info[0] == 2:
					entry = pickle.load(f)
				else:
					entry = pickle.load(f, encoding='latin1')
				for j, k in enumerate(entry['labels']):
					self.data_label[k - 1].append(entry['data'][j])
		
		for i in range(n_class):
			self.data_label[i] = np.vstack(self.data_label[i]).reshape(-1, 3, 32, 32)
			self.data_label[i] = self.data_label[i].transpose((0, 2, 3, 1))  # convert to HWC
	
	def get(self, use_num_cls, max_num=None):
		assert isinstance(use_num_cls, list) \
		       and len(use_num_cls) > 0 and len(use_num_cls) < self.n_class, \
			'invalid use_num_cls : {:}'.format(use_num_cls)
		new_data, new_targets = [], []
		for i in use_num_cls:
			new_data.append(self.data_label[i][:max_num] if max_num is not None else self.data_label[i])
			new_targets.extend([i] * max_num if max_num is not None
			                   else [i] * len(self.data_label[i]))
		self.data = np.concatenate(new_data)
		self.targets = new_targets
		
		imgs = []
		for img in self.data:
			img = Image.fromarray(img)
			img = self.transform(img)
			with torch.no_grad():
				imgs.append(feature_extractor(img.unsqueeze(0)).squeeze().unsqueeze(0))
		return torch.cat(imgs)


if __name__ == '__main__':
	ncls = {'mnist': 10, 'svhn': 10, 'cifar10': 10, 'cifar100': 100, 'imagenet32': 1000}
	transform = get_transform(args.dataset)
	if args.dataset == 'imagenet32':
		imgnet32 = ImageNet32(args.data, ncls[args.dataset], transform)
		data_label = {i: [] for i in range(1000)}
		for i in range(1000):
			m = imgnet32.get([i])
			data_label[i].append(m)
			if i % 10 == 0:
				print(f'Currently saving features of {i}-th class')
				torch.save(data_label, f'{args.save_path}/{args.dataset}bylabel.pt')
	else:
		if args.dataset == 'mnist':
			data = dset.MNIST(args.data_path, train=True, transform=transform, download=True)
		elif args.dataset == 'svhn':
			data = dset.SVHN(args.data_path, split='train', transform=transform, download=True)
		elif args.dataset == 'cifar10':
			data = dset.CIFAR10(args.data_path, train=True, transform=transform, download=True)
		elif args.dataset == 'cifar100':
			data = dset.CIFAR100(args.data_path, train=True, transform=transform, download=True)
		dataset = process(data, ncls[args.dataset])
		torch.save(dataset, f'{args.save_path}/{args.dataset}bylabel.pt')

