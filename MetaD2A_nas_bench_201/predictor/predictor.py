###########################################################################################
# Copyright (c) Hayeon Lee, Eunyoung Hyung [GitHub MetaD2A], 2021
# Rapid Neural Architecture Search by Learning to Generate Graphs from Datasets, ICLR 2021
###########################################################################################
from __future__ import print_function
import torch
import os
import random
from tqdm import tqdm
import numpy as np
import time
import os
import shutil

from torch import nn, optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from scipy.stats import pearsonr

from utils import load_graph_config, decode_igraph_to_NAS_BENCH_201_string
from utils import Log, get_log
from utils import load_model, save_model, mean_confidence_interval

from loader import get_meta_train_loader, get_meta_test_loader, MetaTestDataset
from .predictor_model import PredictorModel
from nas_bench_201 import train_single_model


class Predictor:
  def __init__(self, args):
    self.args = args
    self.batch_size = args.batch_size
    self.data_path = args.data_path
    self.num_sample = args.num_sample
    self.max_epoch = args.max_epoch
    self.save_epoch = args.save_epoch
    self.model_path = args.model_path
    self.save_path = args.save_path
    self.model_name = args.model_name
    self.test = args.test
    self.device = torch.device("cuda:0")
    self.max_corr_dict = {'corr': -1, 'epoch': -1}
    self.train_arch = args.train_arch

    graph_config = load_graph_config(
                  args.graph_data_name, args.nvt, args.data_path)

    self.model = PredictorModel(args, graph_config)
    self.model.to(self.device)

    if self.test:
      self.data_name = args.data_name
      self.num_class = args.num_class
      self.load_epoch = args.load_epoch
      load_model(self.model, self.model_path, load_max_pt='ckpt_max_corr.pt')

    else:
      self.optimizer = optim.Adam(self.model.parameters(), lr=1e-4)
      self.scheduler = ReduceLROnPlateau(self.optimizer, 'min', 
                          factor=0.1, patience=10, verbose=True)
      self.mtrloader =  get_meta_train_loader(
        self.batch_size, self.data_path, self.num_sample, is_pred=True)
      
      self.acc_mean = self.mtrloader.dataset.mean
      self.acc_std = self.mtrloader.dataset.std

      self.mtrlog = Log(self.args, open(os.path.join(
        self.save_path, self.model_name, 'meta_train_predictor.log'), 'w'))
      self.mtrlog.print_args()

  def forward(self, x, arch):
    D_mu = self.model.set_encode(x.to(self.device))
    G_mu = self.model.graph_encode(arch)
    y_pred = self.model.predict(D_mu, G_mu)
    return y_pred

  def meta_train(self):
    sttime = time.time()
    for epoch in range(1, self.max_epoch + 1):
      self.mtrlog.ep_sttime = time.time()
      loss, corr = self.meta_train_epoch(epoch)
      self.scheduler.step(loss)
      self.mtrlog.print_pred_log(loss, corr, 'train', epoch)
      valoss, vacorr = self.meta_validation(epoch)
      if self.max_corr_dict['corr'] < vacorr:
        self.max_corr_dict['corr'] = vacorr
        self.max_corr_dict['epoch'] = epoch
        self.max_corr_dict['loss'] = valoss
        save_model(epoch, self.model, self.model_path, max_corr=True)

      self.mtrlog.print_pred_log(
        valoss, vacorr, 'valid', max_corr_dict=self.max_corr_dict)
      
      if epoch % self.save_epoch == 0:
        save_model(epoch, self.model, self.model_path)

    self.mtrlog.save_time_log()
    self.mtrlog.max_corr_log(self.max_corr_dict)

  def meta_train_epoch(self, epoch):
    self.model.to(self.device)
    self.model.train()

    self.mtrloader.dataset.set_mode('train')

    dlen = len(self.mtrloader.dataset)
    trloss = 0
    y_all, y_pred_all =[], []
    pbar = tqdm(self.mtrloader)

    for x, g, acc in pbar:
      self.optimizer.zero_grad()
      y_pred = self.forward(x, g)
      y = acc.to(self.device)
      loss = self.model.mseloss(y_pred, y.unsqueeze(-1))
      loss.backward()
      self.optimizer.step()

      y = y.tolist()
      y_pred = y_pred.squeeze().tolist()
      y_all += y
      y_pred_all += y_pred
      pbar.set_description(get_log(
        epoch, loss, y_pred, y, self.acc_std, self.acc_mean))
      trloss += float(loss)

    return trloss/dlen, pearsonr(np.array(y_all), 
                                  np.array(y_pred_all))[0]

  def meta_validation(self, epoch):
    self.model.to(self.device)
    self.model.eval()

    valoss = 0
    self.mtrloader.dataset.set_mode('valid')
    dlen = len(self.mtrloader.dataset)
    y_all, y_pred_all =[], []
    pbar = tqdm(self.mtrloader)

    with torch.no_grad():
      for x, g, acc in pbar:
        y_pred = self.forward(x, g)
        y = acc.to(self.device)
        loss = self.model.mseloss(y_pred, y.unsqueeze(-1))

        y = y.tolist()
        y_pred = y_pred.squeeze().tolist()
        y_all += y
        y_pred_all += y_pred
        pbar.set_description(get_log(
          epoch, loss, y_pred, y, self.acc_std, self.acc_mean, tag='val'))
        valoss += float(loss)

    return valoss/dlen, pearsonr(np.array(y_all), 
                                np.array(y_pred_all))[0]

  def meta_test(self):
    if self.data_name == 'all':
      for data_name in ['cifar10', 'cifar100', 'mnist', 'svhn', 'aircraft', 'pets']:
        self.meta_test_per_dataset(data_name)
    else:
      self.meta_test_per_dataset(self.data_name)

  def meta_test_per_dataset(self, data_name):
    self.nasbench201 = torch.load(
        os.path.join(self.data_path, 'nasbench201.pt'))
    self.test_dataset = MetaTestDataset(
      self.data_path, data_name, self.num_sample, self.num_class)

    meta_test_path = os.path.join(
      self.save_path, 'meta_test', data_name, 'best_arch')
    if not os.path.exists(meta_test_path):
      os.makedirs(meta_test_path)
    f_arch_str = open(
      os.path.join(meta_test_path, 'architecture.txt'), 'w')
    save_path = os.path.join(meta_test_path, 'accuracy.txt')
    f = open(save_path, 'w')
    arch_runs = []
    elasped_time = []

    if 'cifar' in data_name:
      N = 30
      runs = 10
      acc_runs = []
    else:
      N = 1
      runs = 1

    print(f'==> select top architectures for {data_name} by meta-predictor...')
    for run in range(1, runs + 1):
      print(f'==> run #{run}')
      gen_arch_str = self.load_generated_archs(data_name, run)
      gen_arch_igraph = self.get_items(
                  full_target=self.nasbench201['arch']['igraph'], 
                  full_source=self.nasbench201['arch']['str'], 
                  source=gen_arch_str)
      y_pred_all = []
      self.model.eval()
      self.model.to(self.device)

      sttime = time.time()
      with torch.no_grad():
        for i, arch_igraph in enumerate(gen_arch_igraph):
          x, g = self.collect_data(arch_igraph)
          y_pred = self.forward(x, g)
          y_pred = torch.mean(y_pred)
          y_pred_all.append(y_pred.cpu().detach().item())

      top_arch_lst = self.select_top_arch(
        data_name, torch.tensor(y_pred_all), gen_arch_str, N)
      arch_runs.append(top_arch_lst[0])
      elasped = time.time() - sttime
      elasped_time.append(elasped)

      if 'cifar' in data_name:
        acc = self.select_top_acc(data_name, top_arch_lst)
        acc_runs.append(acc)

    for run, arch_str in enumerate(arch_runs):
      f_arch_str.write(f'{arch_str}\n'); print(f'{arch_str}')

    time_path = os.path.join(
      self.save_path, 'meta_test', data_name, 'time.txt')
    with open(time_path, 'a') as f_time:
      msg = f'predictor average elasped time {np.mean(elasped_time):.2f}s'
      print(f'==> save time in {time_path}')
      f_time.write(msg+'\n'); print(msg)

    if self.train_arch:
      if not 'cifar' in data_name:
        acc_runs = self.train_single_arch(
                    data_name, arch_runs[0], meta_test_path)
      print(f'==> save results in {save_path}')
      for r, acc in enumerate(acc_runs):
        msg = f'run {r+1} {acc:.2f} (%)'
        f.write(msg+'\n'); print(msg)

      m, h = mean_confidence_interval(acc_runs)
      msg = f'Avg {m:.2f}+-{h.item():.2f} (%)'
      f.write(msg+'\n'); print(msg)

  def train_single_arch(self, data_name, arch_str, meta_test_path):
    seeds = (777, 888, 999)
    train_single_model(save_dir=meta_test_path, 
                       workers=8,
                       datasets=[data_name], 
                       xpaths=[f'{self.data_path}/raw-data/{data_name}'],
                       splits=[0], 
                       use_less=False,
                       seeds=seeds,
                       model_str=arch_str,
                       arch_config={'channel': 16, 'num_cells': 5})
    epoch = 49 if data_name == 'mnist' else 199
    test_acc_lst = []
    for seed in seeds:
      result = torch.load(os.path.join(meta_test_path, f'seed-0{seed}.pth'))
      test_acc_lst.append(result[data_name]['valid_acc1es'][f'x-test@{epoch}'])
    return test_acc_lst

  def select_top_arch_acc(
        self, data_name, y_pred_all, gen_arch_str, N):
    _, sorted_idx = torch.sort(y_pred_all, descending=True)
    gen_test_acc = self.get_items(
              full_target=self.nasbench201['test-acc'][data_name], 
              full_source=self.nasbench201['arch']['str'], 
              source=gen_arch_str)
    sorted_gen_test_acc = torch.tensor(gen_test_acc)[sorted_idx]
    sotred_gen_arch_str = [gen_arch_str[_] for _ in sorted_idx]    
    
    max_idx = torch.argmax(sorted_gen_test_acc[:N]).item()
    final_acc = sorted_gen_test_acc[:N][max_idx]
    final_str = sotred_gen_arch_str[:N][max_idx]
    return final_acc, final_str

  def select_top_arch(
        self, data_name, y_pred_all, gen_arch_str, N):
    _, sorted_idx = torch.sort(y_pred_all, descending=True)
    sotred_gen_arch_str = [gen_arch_str[_] for _ in sorted_idx]    
    final_str = sotred_gen_arch_str[:N]
    return final_str

  def select_top_acc(self, data_name, final_str):
    final_test_acc = self.get_items(
              full_target=self.nasbench201['test-acc'][data_name], 
              full_source=self.nasbench201['arch']['str'], 
              source=final_str)
    max_test_acc = max(final_test_acc)
    return max_test_acc

  def collect_data(self, arch_igraph):
    x_batch, g_batch = [], []
    for _ in range(10):
      x_batch.append(self.test_dataset[0])
      g_batch.append(arch_igraph)
    return torch.stack(x_batch).to(self.device), g_batch

  def get_items(self, full_target, full_source, source):
    return [full_target[full_source.index(_)] for _ in source]

  def load_generated_archs(self, data_name, run):
    mtest_path = os.path.join(
        self.save_path, 'meta_test', data_name, 'generated_arch')
    with open(os.path.join(mtest_path, f'run_{run}.txt'), 'r') as f:
      gen_arch_str = [_.split()[0] for _ in f.readlines()[1:]]
    return gen_arch_str
