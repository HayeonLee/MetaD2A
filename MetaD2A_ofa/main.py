###########################################################################################
# Copyright (c) Hayeon Lee, Eunyoung Hyung [GitHub MetaD2A], 2021
# Rapid Neural Architecture Search by Learning to Generate Graphs from Datasets, ICLR 2021
###########################################################################################
import os
import random
import numpy as np
import torch
from parser import get_parser
from generator import Generator
from predictor import Predictor

def main():
	args = get_parser()
	os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
	args.device = torch.device("cuda:0")
	torch.cuda.manual_seed(args.seed)
	torch.manual_seed(args.seed)
	np.random.seed(args.seed)
	random.seed(args.seed)
	
	if not os.path.exists(args.save_path):
		os.makedirs(args.save_path)
	args.model_path = os.path.join(args.save_path, args.model_name, 'model')
	if not os.path.exists(args.model_path):
		os.makedirs(args.model_path)
	
	if args.model_name == 'generator':
		g = Generator(args)
		if args.test:
			args.model_path = os.path.join(args.save_path, 'predictor', 'model')
			hs = args.hs
			args.hs = 512
			p = Predictor(args)
			args.model_path = os.path.join(args.save_path, args.model_name, 'model')
			args.hs = hs
			g.meta_test(p)
		else:
			g.meta_train()
	elif args.model_name == 'predictor':
		p = Predictor(args)
		p.meta_train()
	else:
		raise ValueError('You should select generator|predictor|train_arch')


if __name__ == '__main__':
	main()
