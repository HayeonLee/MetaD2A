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
  def __init__(self, root, train=True, num_cl=37, val_split=0.15, transforms=None):
    pt_name = os.path.join(root, '{}{}.pth'.format('train' if train else 'test',
                                                                int(100 * (1 - val_split)) if train else int(
                                                                  100 * val_split)))
    if not os.path.exists(pt_name):
      filenames = glob(os.path.join(root, 'images') + '/*.jpg')
      classes = set()
  
      data = []
      labels = []
  
      for image in filenames:
        class_name = image.rsplit("/", 1)[1].rsplit('_', 1)[0]
        classes.add(class_name)
        img = load_image(image)
  
        data.append(img)
        labels.append(class_name)
  
      # convert classnames to indices
      class2idx = {cl: idx for idx, cl in enumerate(classes)}
      labels = torch.Tensor(list(map(lambda x: class2idx[x], labels))).long()
      data = list(zip(data, labels))
  
      class_values = [[] for x in range(num_cl)]
  
      # create arrays for each class type
      for d in data:
        class_values[d[1].item()].append(d)
  
      train_data = []
      val_data = []
  
      for class_dp in class_values:
        split_idx = int(len(class_dp) * (1 - val_split))
        train_data += class_dp[:split_idx]
        val_data += class_dp[split_idx:]
      torch.save(train_data, os.path.join(root, 'train{}.pth'.format(int(100 * (1 - val_split)))))
      torch.save(val_data, os.path.join(root, 'test{}.pth'.format(int(100 * val_split))))

    self.data = torch.load(pt_name)
    self.len = len(self.data)
    self.transform = transforms
  
  def __getitem__(self, index):
    img, label = self.data[index]
    
    if self.transform:
      img = self.transform(img)
    
    return img, label
  
  def __len__(self):
    return self.len

