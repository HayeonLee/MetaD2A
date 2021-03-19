import os
import torch
import time
import copy
import glob
from .imagenet import ImagenetDataProvider
from .imagenet_loader import ImagenetRunConfig
from .run_manager import RunManager
from ofa.model_zoo import ofa_net


class DatabaseOFA:
	def __init__(self, args, predictor=None):
		self.path = f'{args.data_path}/{args.model_name}'
		self.model_name = args.model_name
		self.index = args.index
		self.args = args
		self.predictor = predictor
		ImagenetDataProvider.DEFAULT_PATH = args.imgnet
		
		if not os.path.exists(self.path):
			os.makedirs(self.path)

	def make_db(self):
		self.ofa_network = ofa_net('ofa_mbv3_d234_e346_k357_w1.0', pretrained=True)
		self.run_config = ImagenetRunConfig(test_batch_size=self.args.batch_size,
		                                    n_worker=20)
		database = []
		st_time = time.time()
		f = open(f'{self.path}/txt_{self.index}.txt', 'w')
		for dn in range(10000):
			best_pp = -1
			best_info = None
			dls = None
			with torch.no_grad():
				if self.model_name == 'generator':
					for i in range(10):
						net_setting = self.ofa_network.sample_active_subnet()
						subnet = self.ofa_network.get_active_subnet(preserve_weight=True)
						if i == 0:
							run_manager = RunManager('.tmp/eval_subnet', self.args, subnet,
							                         self.run_config, init=False, pp=self.predictor)
							self.run_config.data_provider.assign_active_img_size(224)
							dls = {j: copy.deepcopy(run_manager.data_loader) for j in range(1, 10)}
						else:
							run_manager = RunManager('.tmp/eval_subnet', self.args, subnet,
							                         self.run_config,
							                         init=False, data_loader=dls[i], pp=self.predictor)
							run_manager.reset_running_statistics(net=subnet)
						
						loss, (top1, top5), pred_acc \
							= run_manager.validate(net=subnet, net_setting=net_setting)
						
						if best_pp < pred_acc:
							best_pp = pred_acc
							print('[%d] class=%d,\t loss=%.5f,\t top1=%.1f,\t top5=%.1f' % (
								dn, len(run_manager.cls_lst), loss, top1, top5))
							info_dict = {'loss': loss,
							             'top1': top1,
							             'top5': top5,
							             'net': net_setting,
							             'class': run_manager.cls_lst,
							             'params': run_manager.net_info['params'],
							             'flops': run_manager.net_info['flops'],
							             'test_transform': run_manager.test_transform
							             }
							best_info = info_dict
				elif self.model_name == 'predictor':
					net_setting = self.ofa_network.sample_active_subnet()
					subnet = self.ofa_network.get_active_subnet(preserve_weight=True)
					run_manager = RunManager('.tmp/eval_subnet', self.args, subnet, self.run_config, init=False)
					self.run_config.data_provider.assign_active_img_size(224)
					run_manager.reset_running_statistics(net=subnet)
					
					loss, (top1, top5), _ = run_manager.validate(net=subnet)
					print('[%d] class=%d,\t loss=%.5f,\t top1=%.1f,\t top5=%.1f' % (
						dn, len(run_manager.cls_lst), loss, top1, top5))
					best_info = {'loss': loss,
					             'top1': top1,
					             'top5': top5,
					             'net': net_setting,
					             'class': run_manager.cls_lst,
					             'params': run_manager.net_info['params'],
					             'flops': run_manager.net_info['flops'],
					             'test_transform': run_manager.test_transform
					             }
				database.append(best_info)
				if (len(database)) % 10 == 0:
					msg = f'{(time.time() - st_time) / 60.0:0.2f}(min) save {len(database)} database, {self.index} id'
					print(msg)
					f.write(msg + '\n')
					f.flush()
					torch.save(database, f'{self.path}/database_{self.index}.pt')
					
	def collect_db(self):
		if not os.path.exists(self.path + f'/processed'):
			os.makedirs(self.path + f'/processed')
			
		database = []
		dlst = glob.glob(self.path + '/*.pt')
		for filepath in dlst:
			database += torch.load(filepath)
		
		assert len(database) != 0
		
		print(f'The number of database: {len(database)}')
		torch.save(database, self.path + f'/processed/collected_database.pt')
