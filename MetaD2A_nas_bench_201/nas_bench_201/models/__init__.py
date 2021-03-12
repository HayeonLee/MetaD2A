##################################################
# Copyright (c) Xuanyi Dong [GitHub D-X-Y], 2019 #
##################################################
from os import path as osp
from typing import List, Text
import torch

__all__ = ['get_cell_based_tiny_net', 'get_search_spaces', \
           'CellStructure', 'CellArchitectures'
           ]

# useful modules
from config_utils import dict2config
from .SharedUtils import change_key
from .cell_searchs import CellStructure, CellArchitectures


# Cell-based NAS Models
def get_cell_based_tiny_net(config):
  if config.name == 'infer.tiny':
    from .cell_infers import TinyNetwork
    if hasattr(config, 'genotype'):
      genotype = config.genotype
    elif hasattr(config, 'arch_str'):
      genotype = CellStructure.str2structure(config.arch_str)
    else: raise ValueError('Can not find genotype from this config : {:}'.format(config))
    return TinyNetwork(config.C, config.N, genotype, config.num_classes)
  else:
    raise ValueError('invalid network name : {:}'.format(config.name))


# obtain the search space, i.e., a dict mapping the operation name into a python-function for this op
def get_search_spaces(xtype, name) -> List[Text]:
  if xtype == 'cell' or xtype == 'tss':  # The topology search space.
    from .cell_operations import SearchSpaceNames
    assert name in SearchSpaceNames, 'invalid name [{:}] in {:}'.format(name, SearchSpaceNames.keys())
    return SearchSpaceNames[name]
  elif xtype == 'sss':  # The size search space.
    if name == 'nas-bench-301':
      return {'candidates': [8, 16, 24, 32, 40, 48, 56, 64],
              'numbers': 5}
    else:
      raise ValueError('Invalid name : {:}'.format(name))
  else:
    raise ValueError('invalid search-space type is {:}'.format(xtype))
