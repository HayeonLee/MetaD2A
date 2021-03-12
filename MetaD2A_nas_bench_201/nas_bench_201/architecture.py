###############################################################
# NAS-Bench-201, ICLR 2020 (https://arxiv.org/abs/2001.00326) #
###############################################################
# Copyright (c) Xuanyi Dong [GitHub D-X-Y], 2019.08           #
###############################################################
import os, sys, time, torch, random, argparse
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True
from copy import deepcopy
from pathlib import Path

from config_utils import load_config
from procedures import save_checkpoint, copy_checkpoint
from procedures import get_machine_info
from datasets import get_datasets
from log_utils import Logger, AverageMeter, time_string, convert_secs2time
from models import CellStructure, CellArchitectures, get_search_spaces
from functions import evaluate_for_seed


def evaluate_all_datasets(arch, datasets, xpaths, splits, use_less, seed,
                          arch_config, workers, logger):
  machine_info, arch_config = get_machine_info(), deepcopy(arch_config)
  all_infos = {'info': machine_info}
  all_dataset_keys = []
  # look all the datasets
  for dataset, xpath, split in zip(datasets, xpaths, splits):
    # train valid data
    task = None
    train_data, valid_data, xshape, class_num = get_datasets(dataset, xpath, -1, task)
    
    # load the configuration    
    if dataset in ['mnist', 'svhn', 'aircraft', 'pets']:
      if use_less:
        config_path = 'nas_bench_201/configs/nas-benchmark/LESS.config'
      else:
        config_path = 'nas_bench_201/configs/nas-benchmark/{}.config'.format(dataset)
      
      p = 'nas_bench_201/configs/nas-benchmark/{:}-split.txt'.format(dataset)
      if not os.path.exists(p):
        import json
        label_list = list(range(len(train_data)))
        random.shuffle(label_list)
        strlist = [str(label_list[i]) for i in range(len(label_list))]
        splited = {'train': ["int", strlist[:len(train_data) // 2]],
                   'valid': ["int", strlist[len(train_data) // 2:]]}
        with open(p, 'w') as f:
          f.write(json.dumps(splited))
      split_info = load_config('nas_bench_201/configs/nas-benchmark/{:}-split.txt'.format(dataset), None, None)
    else:
      raise ValueError('invalid dataset : {:}'.format(dataset))
    
    config = load_config(config_path, {'class_num': class_num, 'xshape': xshape}, logger)
    # data loader
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=config.batch_size,
                                               shuffle=True, num_workers=workers, pin_memory=True)
    valid_loader = torch.utils.data.DataLoader(valid_data, batch_size=config.batch_size,
                                               shuffle=False, num_workers=workers, pin_memory=True)
    splits = load_config('nas_bench_201/configs/nas-benchmark/{}-test-split.txt'.format(dataset), None, None)
    ValLoaders = {'ori-test': valid_loader,
                  'x-valid': torch.utils.data.DataLoader(valid_data, batch_size=config.batch_size,
                                                         sampler=torch.utils.data.sampler.SubsetRandomSampler(
                                                           splits.xvalid),
                                                         num_workers=workers, pin_memory=True),
                  'x-test': torch.utils.data.DataLoader(valid_data, batch_size=config.batch_size,
                                                        sampler=torch.utils.data.sampler.SubsetRandomSampler(
                                                          splits.xtest),
                                                        num_workers=workers, pin_memory=True)
                    }
    dataset_key = '{:}'.format(dataset)
    if bool(split): dataset_key = dataset_key + '-valid'
    logger.log(
      'Evaluate ||||||| {:10s} ||||||| Train-Num={:}, Valid-Num={:}, Train-Loader-Num={:}, Valid-Loader-Num={:}, batch size={:}'.
        format(dataset_key, len(train_data), len(valid_data), len(train_loader), len(valid_loader), config.batch_size))
    logger.log('Evaluate ||||||| {:10s} ||||||| Config={:}'.format(dataset_key, config))
    for key, value in ValLoaders.items():
      logger.log('Evaluate ---->>>> {:10s} with {:} batchs'.format(key, len(value)))

    results = evaluate_for_seed(arch_config, config, arch, train_loader, ValLoaders, seed, logger)
    all_infos[dataset_key] = results
    all_dataset_keys.append(dataset_key)
  all_infos['all_dataset_keys'] = all_dataset_keys
  return all_infos


def train_single_model(save_dir, workers, datasets, xpaths, splits, use_less,
                       seeds, model_str, arch_config):
  assert torch.cuda.is_available(), 'CUDA is not available.'
  torch.backends.cudnn.enabled = True
  torch.backends.cudnn.deterministic = True
  torch.set_num_threads(workers)
  
  save_dir = Path(save_dir)
  logger = Logger(str(save_dir), 0, False)

  if model_str in CellArchitectures:
    arch = CellArchitectures[model_str]
    logger.log('The model string is found in pre-defined architecture dict : {:}'.format(model_str))
  else:
    try:
      arch = CellStructure.str2structure(model_str)
    except:
      raise ValueError('Invalid model string : {:}. It can not be found or parsed.'.format(model_str))
  
  assert arch.check_valid_op(get_search_spaces('cell', 'nas-bench-201')), '{:} has the invalid op.'.format(arch)
  # assert arch.check_valid_op(get_search_spaces('cell', 'full')), '{:} has the invalid op.'.format(arch)
  logger.log('Start train-evaluate {:}'.format(arch.tostr()))
  logger.log('arch_config : {:}'.format(arch_config))
  
  start_time, seed_time = time.time(), AverageMeter()
  for _is, seed in enumerate(seeds):
    logger.log(
      '\nThe {:02d}/{:02d}-th seed is {:} ----------------------<.>----------------------'.format(_is, len(seeds),
                                                                                                  seed))
    to_save_name = save_dir / 'seed-{:04d}.pth'.format(seed)
    if to_save_name.exists():
      logger.log('Find the existing file {:}, directly load!'.format(to_save_name))
      checkpoint = torch.load(to_save_name)
    else:
      logger.log('Does not find the existing file {:}, train and evaluate!'.format(to_save_name))
      checkpoint = evaluate_all_datasets(arch, datasets, xpaths, splits, use_less,
                                         seed, arch_config, workers, logger)
      torch.save(checkpoint, to_save_name)
    # log information
    logger.log('{:}'.format(checkpoint['info']))
    all_dataset_keys = checkpoint['all_dataset_keys']
    for dataset_key in all_dataset_keys:
      logger.log('\n{:} dataset : {:} {:}'.format('-' * 15, dataset_key, '-' * 15))
      dataset_info = checkpoint[dataset_key]
      # logger.log('Network ==>\n{:}'.format( dataset_info['net_string'] ))
      logger.log('Flops = {:} MB, Params = {:} MB'.format(dataset_info['flop'], dataset_info['param']))
      logger.log('config : {:}'.format(dataset_info['config']))
      logger.log('Training State (finish) = {:}'.format(dataset_info['finish-train']))
      last_epoch = dataset_info['total_epoch'] - 1
      train_acc1es, train_acc5es = dataset_info['train_acc1es'], dataset_info['train_acc5es']
      valid_acc1es, valid_acc5es = dataset_info['valid_acc1es'], dataset_info['valid_acc5es']
    # measure elapsed time
    seed_time.update(time.time() - start_time)
    start_time = time.time()
    need_time = 'Time Left: {:}'.format(convert_secs2time(seed_time.avg * (len(seeds) - _is - 1), True))
    logger.log(
      '\n<<<***>>> The {:02d}/{:02d}-th seed is {:} <finish> other procedures need {:}'.format(_is, len(seeds), seed,
                                                                                               need_time))
  logger.close()

