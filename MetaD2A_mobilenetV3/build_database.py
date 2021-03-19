###########################################################################################
# Copyright (c) Hayeon Lee, Eunyoung Hyung [GitHub MetaD2A], 2021
# Rapid Neural Architecture Search by Learning to Generate Graphs from Datasets, ICLR 2021
###########################################################################################
import os
import random
import numpy as np
import torch
from parser import get_parser
from predictor import PredictorModel
from database import DatabaseOFA
from utils import load_graph_config

def main():
	args = get_parser()
	
	if args.gpu == 'all':
		device_list = range(torch.cuda.device_count())
		args.gpu = ','.join(str(_) for _ in device_list)
	else:
		device_list = [int(_) for _ in args.gpu.split(',')]
	os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
	args.device = torch.device("cuda:0")
	args.batch_size = args.batch_size * max(len(device_list), 1)

	torch.cuda.manual_seed(args.seed)
	torch.manual_seed(args.seed)
	np.random.seed(args.seed)
	random.seed(args.seed)
	
	args.model_path = os.path.join(args.save_path, args.model_name, 'model')
	
	if args.model_name == 'generator':
		graph_config = load_graph_config(
			args.graph_data_name, args.nvt, args.data_path)
		model = PredictorModel(args, graph_config)
		d = DatabaseOFA(args, model)
	else:
		d = DatabaseOFA(args)
		
	if args.collect:
		d.collect_db()
	else:
		assert args.index is not None
		assert args.imgnet is not None
		d.make_db()

if __name__ == '__main__':
	main()
