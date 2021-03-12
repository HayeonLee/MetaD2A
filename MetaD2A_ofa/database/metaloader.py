from torch.utils.data.sampler import Sampler
import os
import random
from PIL import Image
from collections import defaultdict
import torch
from torch.utils.data import Dataset, DataLoader
import glob


class RandCycleIter:
  '''
  Return data_list per class
  Shuffle the returning order after one epoch
  '''
  def __init__ (self, data, shuffle=True):
    self.data_list = list(data)
    self.length = len(self.data_list)
    self.i = self.length - 1
    self.shuffle = shuffle

  def __iter__ (self):
    return self

  def __next__ (self):
    self.i += 1

    if self.i == self.length:
      self.i = 0
      if self.shuffle:
        random.shuffle(self.data_list)

    return self.data_list[self.i]


class EpisodeSampler(Sampler):
  def __init__(self, max_way, query, ylst):
    self.max_way = max_way
    self.query = query
    self.ylst = ylst
    # self.n_epi = n_epi

    clswise_xidx = defaultdict(list)
    for i, y in enumerate(ylst):
      clswise_xidx[y].append(i)
    self.cws_xidx_iter = [RandCycleIter(cxidx, shuffle=True)
                          for cxidx in clswise_xidx.values()]
    self.n_cls = len(clswise_xidx)

    self.create_episode()


  def __iter__ (self):
    return self.get_index()

  def __len__ (self):
    return self.get_len()

  def create_episode(self):
    self.way = torch.randperm(int(self.max_way/10.0)-1)[0] * 10 + 10
    cls_lst = torch.sort(torch.randperm(self.max_way)[:self.way])[0]
    self.cls_itr = iter(cls_lst)
    self.cls_lst = cls_lst

  def get_len(self):
    return self.way * self.query

  def get_index(self):
    x_itr = self.cws_xidx_iter

    i, j = 0, 0
    while i < self.query * self.way:
      if j >= self.query:
        j = 0
      if j == 0:
        cls_idx = next(self.cls_itr).item()
        bb = [x_itr[cls_idx]] * self.query
        didx = next(zip(*bb))
      yield didx[j]
      # yield (didx[j], self.way)

      i += 1; j += 1


class MetaImageNetDataset(Dataset):
  def __init__(self, mode='val', 
        max_way=1000, query=10,
          dpath='/w14/dataset/ILSVRC2012', transform=None):
    self.dpath = dpath
    self.transform = transform
    self.mode = mode

    self.max_way = max_way
    self.query = query
    classes, class_to_idx = self._find_classes(dpath+'/'+mode)
    self.classes, self.class_to_idx = classes, class_to_idx
    # self.class_folder_lst = \
    #     glob.glob(dpath+'/'+mode+'/*')
    # ## sorting alphabetically
    # self.class_folder_lst = sorted(self.class_folder_lst)
    self.file_path_lst, self.ylst = [], []
    for cls in classes:
      xlst = glob.glob(dpath+'/'+mode+'/'+cls+'/*')
      self.file_path_lst += xlst[:self.query]
      y = class_to_idx[cls]
      self.ylst += [y] * len(xlst[:self.query])

    # for y, cls in enumerate(self.class_folder_lst):
    #   xlst = glob.glob(cls+'/*')
    #   self.file_path_lst += xlst[:self.query]
    #   self.ylst += [y] * len(xlst[:self.query])
    #   # self.file_path_lst += [xlst[_] for _ in
    #   #                torch.randperm(len(xlst))[:self.query]]
    #   # self.ylst += [cls.split('/')[-1]] * len(xlst)

    self.way_idx = 0
    self.x_idx = 0
    self.way = 2
    self.cls_lst = None


  def __len__(self):
    return self.way * self.query 

  def __getitem__(self, index):
    # if self.way != index[1]:
    #   self.way = index[1]
    # index = index[0]

    x = Image.open(
          self.file_path_lst[index]).convert('RGB')

    if self.transform is not None:
      x = self.transform(x)
    cls_name = self.ylst[index]
    y = self.cls_lst.index(cls_name)
    # y = self.way_idx
    # self.x_idx += 1
    # if self.x_idx == self.query:
    #   self.way_idx += 1
    #   self.x_idx = 0
    # if self.way_idx == self.way:
    #   self.way_idx = 0
    #   self.x_idx = 0
    return x, y #, cls_name # y # cls_name #y

  def _find_classes(self, dir: str):
      """
      Finds the class folders in a dataset.

      Args:
          dir (string): Root directory path.

      Returns:
          tuple: (classes, class_to_idx) where classes are relative to (dir), and class_to_idx is a dictionary.

      Ensures:
          No class is a subdirectory of another.
      """
      classes = [d.name for d in os.scandir(dir) if d.is_dir()]
      classes.sort()
      class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
      return classes, class_to_idx


class MetaDataLoader(DataLoader):
  def __init__(self, 
    dataset, sampler, batch_size, shuffle, num_workers):  
    super(MetaDataLoader, self).__init__(
                                dataset=dataset, 
                                sampler=sampler, 
                                batch_size=batch_size, 
                                shuffle=shuffle, 
                                num_workers=num_workers)


  def create_episode(self):
    self.sampler.create_episode()
    self.dataset.way = self.sampler.way
    self.dataset.cls_lst = self.sampler.cls_lst.tolist()


  def get_cls_idx(self):
    return self.sampler.cls_lst


def get_loader(mode='val', way=10, query=10, 
      n_epi=100, dpath='/w14/dataset/ILSVRC2012', 
        transform=None):
  trans = get_transforms(mode)
  dataset = MetaImageNetDataset(mode, way, query, dpath, trans)
  sampler = EpisodeSampler(
    way, query, n_epi, dataset.ylst)
  dataset.way = sampler.way
  dataset.cls_lst = sampler.cls_lst
  loader = MetaDataLoader(dataset=dataset,
                      sampler=sampler,
                      batch_size=10,
                      shuffle=False,
                      num_workers=4)
  return loader

# trloader = get_loader()

# trloader.create_episode()
# print(len(trloader))
# print(trloader.dataset.way)
# print(trloader.sampler.way)
# for i, episode in enumerate(trloader, start=1):
#   print(episode[2])
