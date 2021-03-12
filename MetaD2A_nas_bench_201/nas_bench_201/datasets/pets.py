###########################################################################################
# Copyright (c) Hayeon Lee, Eunyoung Hyung [GitHub MetaD2A], 2021
# Rapid Neural Architecture Search by Learning to Generate Graphs from Datasets, ICLR 2021
###########################################################################################
import torch
from glob import glob
from torch.utils.data.dataset import Dataset
import os
from PIL import Image


def load_image(filename):
  img = Image.open(filename)
  img = img.convert('RGB')
  return img

class PetDataset(Dataset):
  def __init__(self, root, train=True, num_cl=37, val_split=0.2, transforms=None):
    self.data = torch.load(os.path.join(root,'{}{}.pth'.format('train' if train else 'test',
                                                               int(100*(1-val_split)) if train else int(100*val_split))))
    self.len = len(self.data)
    self.transform = transforms
  def __getitem__(self, index):
    img, label = self.data[index]
    if self.transform:
      img = self.transform(img)
    return img, label
  def __len__(self):
    return self.len
    
if __name__ == '__main__':
  # Added
  import torchvision.transforms as transforms
  normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
  train_transform = transforms.Compose(
    [transforms.Resize(256), transforms.RandomRotation(45), transforms.CenterCrop(224),
     transforms.RandomHorizontalFlip(), transforms.ToTensor(), normalize])
  test_transform = transforms.Compose(
    [transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(), normalize])
  root = '/w14/dataset/MetaGen/pets'
  train_data, test_data = get_pets(root, num_cl=37, val_split=0.2,
                                   tr_transform=train_transform,
                                   te_transform=test_transform)
  import pdb;
  pdb.set_trace()
