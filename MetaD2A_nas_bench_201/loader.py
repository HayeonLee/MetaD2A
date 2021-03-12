###########################################################################################
# Copyright (c) Hayeon Lee, Eunyoung Hyung [GitHub MetaD2A], 2021
# Rapid Neural Architecture Search by Learning to Generate Graphs from Datasets, ICLR 2021
###########################################################################################
from __future__ import print_function
import os
import torch
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

    if is_pred:
      mtr_data_path = os.path.join(data_path, 'meta_train_tasks_predictor.pt')
      idx_path = os.path.join(data_path, 'meta_train_tasks_predictor_idx.pt')
    else:
      mtr_data_path = os.path.join(data_path, 'meta_train_tasks_generator.pt')
      idx_path = os.path.join(data_path, 'meta_train_tasks_generator_idx.pt')

    data = torch.load(mtr_data_path)
    self.acc = data['acc']
    self.task = data['task']
    self.graph = data['graph']

    random_idx_lst = torch.load(idx_path)
    self.idx_lst = {}
    self.idx_lst['valid'] = random_idx_lst[:400]
    self.idx_lst['train'] = random_idx_lst[400:]
    self.acc = torch.tensor(self.acc)
    self.mean = torch.mean(self.acc[self.idx_lst['train']]).item()
    self.std = torch.std(self.acc[self.idx_lst['train']]).item()
    self.task_lst = torch.load(os.path.join(
                                data_path, 'meta_train_task_lst.pt'))

  def set_mode(self, mode):
    self.mode = mode

  def __len__(self):
    return len(self.idx_lst[self.mode])

  def __getitem__(self, index):
    data = []
    ridx = self.idx_lst[self.mode]
    tidx = self.task[ridx[index]]
    classes = self.task_lst[tidx]
    graph = self.graph[ridx[index]]
    acc = self.acc[ridx[index]]
    for cls in classes:
      cx = self.x[cls-1][0]
      ridx = torch.randperm(len(cx))
      data.append(cx[ridx[:self.num_sample]])
    x = torch.cat(data)
    if self.acc_norm:
      acc = ((acc- self.mean) / self.std) / 100.0
    else:
      acc = acc / 100.0
    return x, graph, acc


class MetaTestDataset(Dataset):
  def __init__(self, data_path, data_name, num_sample, num_class=None):
    self.num_sample = num_sample
    self.data_name = data_name

    num_class_dict = {
      'cifar100': 100,
      'cifar10':  10,
      'mnist':    10,
      'svhn':     10,
      'aircraft': 30,
      'pets':     37
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
    x = torch.stack([item[0] for item in batch])
    graph = [item[1] for item in batch]
    acc = torch.stack([item[2] for item in batch])
    return [x, graph, acc]
