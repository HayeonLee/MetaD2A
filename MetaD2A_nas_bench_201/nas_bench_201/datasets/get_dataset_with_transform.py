##################################################
# Copyright (c) Xuanyi Dong [GitHub D-X-Y], 2019 #
# Modified by Hayeon Lee, Eunyoung Hyung 2021. 03.
##################################################
import os, sys, torch
import os.path as osp
import numpy as np
import torchvision.datasets as dset
import torchvision.transforms as transforms
from copy import deepcopy
# from PIL import Image
import random
import pdb
from .aircraft import FGVCAircraft
from .pets import PetDataset
from config_utils import load_config

Dataset2Class = {'cifar10': 10,
                 'cifar100': 100,
                 'mnist': 10,
                 'svhn': 10,
                 'aircraft': 30,
                 'pets': 37}


class CUTOUT(object):
  
  def __init__(self, length):
    self.length = length
  
  def __repr__(self):
    return ('{name}(length={length})'.format(name=self.__class__.__name__, **self.__dict__))
  
  def __call__(self, img):
    h, w = img.size(1), img.size(2)
    mask = np.ones((h, w), np.float32)
    y = np.random.randint(h)
    x = np.random.randint(w)
    
    y1 = np.clip(y - self.length // 2, 0, h)
    y2 = np.clip(y + self.length // 2, 0, h)
    x1 = np.clip(x - self.length // 2, 0, w)
    x2 = np.clip(x + self.length // 2, 0, w)
    
    mask[y1: y2, x1: x2] = 0.
    mask = torch.from_numpy(mask)
    mask = mask.expand_as(img)
    img *= mask
    return img


imagenet_pca = {
  'eigval': np.asarray([0.2175, 0.0188, 0.0045]),
  'eigvec': np.asarray([
    [-0.5675, 0.7192, 0.4009],
    [-0.5808, -0.0045, -0.8140],
    [-0.5836, -0.6948, 0.4203],
  ])
}


class Lighting(object):
  def __init__(self, alphastd,
               eigval=imagenet_pca['eigval'],
               eigvec=imagenet_pca['eigvec']):
    self.alphastd = alphastd
    assert eigval.shape == (3,)
    assert eigvec.shape == (3, 3)
    self.eigval = eigval
    self.eigvec = eigvec
  
  def __call__(self, img):
    if self.alphastd == 0.:
      return img
    rnd = np.random.randn(3) * self.alphastd
    rnd = rnd.astype('float32')
    v = rnd
    old_dtype = np.asarray(img).dtype
    v = v * self.eigval
    v = v.reshape((3, 1))
    inc = np.dot(self.eigvec, v).reshape((3,))
    img = np.add(img, inc)
    if old_dtype == np.uint8:
      img = np.clip(img, 0, 255)
    img = Image.fromarray(img.astype(old_dtype), 'RGB')
    return img
  
  def __repr__(self):
    return self.__class__.__name__ + '()'


def get_datasets(name, root, cutout, use_num_cls=None):
  if name == 'cifar10':
    mean = [x / 255 for x in [125.3, 123.0, 113.9]]
    std = [x / 255 for x in [63.0, 62.1, 66.7]]
  elif name == 'cifar100':
    mean = [x / 255 for x in [129.3, 124.1, 112.4]]
    std = [x / 255 for x in [68.2, 65.4, 70.4]]
  elif name.startswith('mnist'):
    mean, std = [0.1307, 0.1307, 0.1307], [0.3081, 0.3081, 0.3081]
  elif name.startswith('svhn'):
    mean, std = [0.4376821, 0.4437697, 0.47280442], [0.19803012, 0.20101562, 0.19703614]
  elif name.startswith('aircraft'):
    mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
  elif name.startswith('pets'):
    mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
  else:
    raise TypeError("Unknow dataset : {:}".format(name))
  
  # Data Argumentation
  if name == 'cifar10' or name == 'cifar100':
    lists = [transforms.RandomHorizontalFlip(), transforms.RandomCrop(32, padding=4), transforms.ToTensor(),
             transforms.Normalize(mean, std)]
    if cutout > 0: lists += [CUTOUT(cutout)]
    train_transform = transforms.Compose(lists)
    test_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])
    xshape = (1, 3, 32, 32)
  elif name.startswith('cub200'):
    train_transform = transforms.Compose([
      transforms.Resize((32, 32)),
      transforms.ToTensor(),
      transforms.Normalize(mean=mean, std=std)
    ])
    test_transform = transforms.Compose([
      transforms.Resize((32, 32)),
      transforms.ToTensor(),
      transforms.Normalize(mean=mean, std=std)
    ])
    xshape = (1, 3, 32, 32)
  elif name.startswith('mnist'):
    train_transform = transforms.Compose([
      transforms.Resize((32, 32)),
      transforms.ToTensor(),
      transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
      transforms.Normalize(mean, std),
    ])
    test_transform = transforms.Compose([
      transforms.Resize((32, 32)),
      transforms.ToTensor(),
      transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
      transforms.Normalize(mean, std)
    ])
    xshape = (1, 3, 32, 32)
  elif name.startswith('svhn'):
    train_transform = transforms.Compose([
      transforms.Resize((32, 32)),
      transforms.ToTensor(),
      transforms.Normalize(mean=mean, std=std)
    ])
    test_transform = transforms.Compose([
      transforms.Resize((32, 32)),
      transforms.ToTensor(),
      transforms.Normalize(mean=mean, std=std)
    ])
    xshape = (1, 3, 32, 32)
  elif name.startswith('aircraft'):
    train_transform = transforms.Compose([
      transforms.Resize((32, 32)),
      transforms.ToTensor(),
      transforms.Normalize(mean=mean, std=std)
    ])
    test_transform = transforms.Compose([
      transforms.Resize((32, 32)),
      transforms.ToTensor(),
      transforms.Normalize(mean=mean, std=std),
    ])
    xshape = (1, 3, 32, 32)
  elif name.startswith('pets'):
    train_transform = transforms.Compose([
      transforms.Resize((32, 32)),
      transforms.ToTensor(),
      transforms.Normalize(mean=mean, std=std)
    ])
    test_transform = transforms.Compose([
      transforms.Resize((32, 32)),
      transforms.ToTensor(),
      transforms.Normalize(mean=mean, std=std),
    ])
    xshape = (1, 3, 32, 32)
  else:
    raise TypeError("Unknow dataset : {:}".format(name))
  
  if name == 'cifar10':
    train_data = dset.CIFAR10(root, train=True, transform=train_transform, download=True)
    test_data = dset.CIFAR10(root, train=False, transform=test_transform, download=True)
    assert len(train_data) == 50000 and len(test_data) == 10000
  elif name == 'cifar100':
    train_data = dset.CIFAR100(root, train=True, transform=train_transform, download=True)
    test_data = dset.CIFAR100(root, train=False, transform=test_transform, download=True)
    assert len(train_data) == 50000 and len(test_data) == 10000
  elif name == 'mnist':
    train_data = dset.MNIST(root, train=True, transform=train_transform, download=True)
    test_data = dset.MNIST(root, train=False, transform=test_transform, download=True)
    assert len(train_data) == 60000 and len(test_data) == 10000
  elif name == 'svhn':
    train_data = dset.SVHN(root, split='train', transform=train_transform, download=True)
    test_data = dset.SVHN(root, split='test', transform=test_transform, download=True)
    assert len(train_data) == 73257 and len(test_data) == 26032
  elif name == 'aircraft':
    train_data = FGVCAircraft(root, class_type='manufacturer', split='trainval',
                              transform=train_transform, download=False)
    test_data = FGVCAircraft(root, class_type='manufacturer', split='test',
                             transform=test_transform, download=False)
    assert len(train_data) == 6667 and len(test_data) == 3333
  elif name == 'pets':
    train_data = PetDataset(root, train=True, num_cl=37,
                                       val_split=0.15,transforms=train_transform)
    test_data = PetDataset(root, train=False, num_cl=37,
                                       val_split=0.15,transforms=test_transform)
  else:
    raise TypeError("Unknow dataset : {:}".format(name))
  
  class_num = Dataset2Class[name] if use_num_cls is None else len(use_num_cls)
  return train_data, test_data, xshape, class_num


def get_nas_search_loaders(train_data, valid_data, dataset, config_root, batch_size, workers, num_cls=None):
  if isinstance(batch_size, (list, tuple)):
    batch, test_batch = batch_size
  else:
    batch, test_batch = batch_size, batch_size
  if dataset == 'cifar10':
    # split_Fpath = 'configs/nas-benchmark/cifar-split.txt'
    cifar_split = load_config('{:}/cifar-split.txt'.format(config_root), None, None)
    train_split, valid_split = cifar_split.train, cifar_split.valid  # search over the proposed training and validation set
    # logger.log('Load split file from {:}'.format(split_Fpath))      # they are two disjoint groups in the original CIFAR-10 training set
    # To split data
    xvalid_data = deepcopy(train_data)
    if hasattr(xvalid_data, 'transforms'):  # to avoid a print issue
      xvalid_data.transforms = valid_data.transform
    xvalid_data.transform = deepcopy(valid_data.transform)
    search_data = SearchDataset(dataset, train_data, train_split, valid_split)
    # data loader
    search_loader = torch.utils.data.DataLoader(search_data, batch_size=batch, shuffle=True, num_workers=workers,
                                                pin_memory=True)
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch,
                                               sampler=torch.utils.data.sampler.SubsetRandomSampler(train_split),
                                               num_workers=workers, pin_memory=True)
    valid_loader = torch.utils.data.DataLoader(xvalid_data, batch_size=test_batch,
                                               sampler=torch.utils.data.sampler.SubsetRandomSampler(valid_split),
                                               num_workers=workers, pin_memory=True)
  elif dataset == 'cifar100':
    cifar100_test_split = load_config('{:}/cifar100-test-split.txt'.format(config_root), None, None)
    search_train_data = train_data
    search_valid_data = deepcopy(valid_data)
    search_valid_data.transform = train_data.transform
    search_data = SearchDataset(dataset, [search_train_data, search_valid_data],
                                list(range(len(search_train_data))),
                                cifar100_test_split.xvalid)
    search_loader = torch.utils.data.DataLoader(search_data, batch_size=batch, shuffle=True, num_workers=workers,
                                                pin_memory=True)
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch, shuffle=True, num_workers=workers,
                                               pin_memory=True)
    valid_loader = torch.utils.data.DataLoader(valid_data, batch_size=test_batch,
                                               sampler=torch.utils.data.sampler.SubsetRandomSampler(
                                                 cifar100_test_split.xvalid), num_workers=workers, pin_memory=True)
  elif dataset in ['mnist', 'svhn', 'aircraft', 'pets']:
    if not os.path.exists('{:}/{}-test-split.txt'.format(config_root, dataset)):
      import json
      label_list = list(range(len(valid_data)))
      random.shuffle(label_list)
      strlist = [str(label_list[i]) for i in range(len(label_list))]
      split = {'xvalid': ["int", strlist[:len(valid_data) // 2]],
               'xtest': ["int", strlist[len(valid_data) // 2:]]}
      with open('{:}/{}-test-split.txt'.format(config_root, dataset), 'w') as f:
        f.write(json.dumps(split))
    test_split = load_config('{:}/{}-test-split.txt'.format(config_root, dataset), None, None)
    
    search_train_data = train_data
    search_valid_data = deepcopy(valid_data)
    search_valid_data.transform = train_data.transform
    search_data = SearchDataset(dataset, [search_train_data, search_valid_data],
                                list(range(len(search_train_data))), test_split.xvalid)
    search_loader = torch.utils.data.DataLoader(search_data, batch_size=batch, shuffle=True,
                                                num_workers=workers, pin_memory=True)
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch, shuffle=True,
                                               num_workers=workers, pin_memory=True)
    valid_loader = torch.utils.data.DataLoader(valid_data, batch_size=test_batch,
                                               sampler=torch.utils.data.sampler.SubsetRandomSampler(
                                                 test_split.xvalid), num_workers=workers, pin_memory=True)
  else:
    raise ValueError('invalid dataset : {:}'.format(dataset))
  return search_loader, train_loader, valid_loader

