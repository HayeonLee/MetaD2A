from .imagenet import ImagenetDataProvider
from ofa_local.imagenet_classification.run_manager import RunConfig


__all__ = ['ImagenetRunConfig']


class ImagenetRunConfig(RunConfig):

	def __init__(self, n_epochs=150, init_lr=0.05, lr_schedule_type='cosine', lr_schedule_param=None,
	             dataset='imagenet', train_batch_size=256, test_batch_size=500, valid_size=None,
	             opt_type='sgd', opt_param=None, weight_decay=4e-5, label_smoothing=0.1, no_decay_keys=None,
	             mixup_alpha=None, model_init='he_fout', validation_frequency=1, print_frequency=10,
	             n_worker=32, resize_scale=0.08, distort_color='tf', image_size=224, **kwargs):
		super(ImagenetRunConfig, self).__init__(
			n_epochs, init_lr, lr_schedule_type, lr_schedule_param,
			dataset, train_batch_size, test_batch_size, valid_size,
			opt_type, opt_param, weight_decay, label_smoothing, no_decay_keys,
			mixup_alpha,
			model_init, validation_frequency, print_frequency
		)

		self.n_worker = n_worker
		self.resize_scale = resize_scale
		self.distort_color = distort_color
		self.image_size = image_size

	@property
	def data_provider(self):
		if self.__dict__.get('_data_provider', None) is None:
			if self.dataset == ImagenetDataProvider.name():
				DataProviderClass = ImagenetDataProvider
			else:
				raise NotImplementedError
			self.__dict__['_data_provider'] = DataProviderClass(
				train_batch_size=self.train_batch_size, test_batch_size=self.test_batch_size,
				valid_size=self.valid_size, n_worker=self.n_worker, resize_scale=self.resize_scale,
				distort_color=self.distort_color, image_size=self.image_size,
			)
		return self.__dict__['_data_provider']
