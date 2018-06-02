"""
Config file

Author: Clinton Wang, E-mail: `clintonjwang@gmail.com`, Github: `https://github.com/clintonjwang/voi-classifier`
"""

from os.path import *
from keras.callbacks import EarlyStopping

class Config:
	def __init__(self, dataset="etiology"):
		self.run_num = 2
		self.test_run_num = 2
		self.dims = [24,24,12]
		self.nb_channels = 3
		self.aug_factor = 10

		self.max_size = 350*350*100

		self.context_dims = [36,36,12]
		self.dual_img_inputs = False # whether to use both tight and gross image croppings for the network
		self.non_imaging_inputs = False # whether non-imaging inputs should be incorporated into the neural network
		self.probabilistic = False

		self.lesion_ratio = 0.7 # ratio of the lesion side length to the length of the cropped image

		self.pre_scale = .5 # normalizes images at augmentation time
		self.post_scale = 0. # normalizes images at train/test time

		# Information about the abnormality classes
		self.patient_sheetname = 'Patient Info'

		# Augmentation parameters
		self.intensity_scaling = [.05,.05]
		self.translate = [2,2,1]

		self.use_dataset(dataset)

		self.nb_classes = len(self.cls_names)
		self.phases = ["T1_20s", "T1_70s", "T1_3min"]

	def turn_on_clinical_features(self):
		self.non_imaging_inputs = True # whether non-imaging inputs should be incorporated into the neural network
		self.num_non_image_inputs = 3

	def use_dataset(self, dataset):
		if dataset == "lirads":
			self.base_dir = "E:\\LIRADS"
			self.coord_xls_path = 'Z:\\LIRADS\\Prototype1e.xlsx'
			self.test_num = 20
			self.full_img_dir = "Z:\\LIRADS\\full_imgs"

			self.cls_names = ['hcc', 'cholangio', 'colorectal', 'cyst', 'hemangioma', 'fnh']
			self.sheetnames = ['HCC', 'Cholangio', 'Colorectal', 'Cyst', 'Hemangioma', 'FNH']
			self.short_cls_names = ['HCC', 'ICC', 'CRC Met.', 'Cyst', 'Hemang.', 'FNH']
			self.dcm_dirs = ["Z:\\LIRADS\\DICOMs\\" + fn for fn in self.cls_names]
			self.simplify_map = {'hcc': 0, 'cyst': 1, 'hemangioma': 1, 'fnh': 1, 'cholangio': 2, 'colorectal': 2}
			
		elif dataset == "etiology":
			self.dims = [32,32,16]
			self.context_dims = [100,100,40]
			self.num_segs = 3
			self.loss_weights = [[2,5,100],[1,1,1]]
			self.state_dim = (*self.dims, 4)

			self.base_dir = "D:\\Etiology"
			self.coord_xls_path = "D:\\Etiology\\excel\\coords.xlsx"
			self.test_num = 5
			self.full_img_dir = join(self.base_dir, "imgs","full_imgs")

			self.cls_names = ['hbv', 'hcv', 'nonviral']
			self.sheetnames = ['HBV', 'HCV', 'Nonviral']
			self.short_cls_names = ['HBV', 'HCV', 'NV']
			self.dcm_dirs = ["D:\\Etiology\\Imaging"] * 3

		elif dataset == "radpath":
			self.base_dir = "D:\\Paula-project"
			self.coord_xls_path = "Z:\\Paula\\new coordinates_CW.xlsx"
			self.test_num = 5
			self.full_img_dir = join(self.base_dir, "full_imgs")

			self.cls_names = ['hcc', 'non-hcc']
			self.sheetnames = ['HCC', 'Non-HCC']
			self.short_cls_names = self.sheetnames
			self.dcm_dirs = ["Z:\\Paula\\Imaging"] * 2

		elif dataset == "lirads-expanded":
			self.base_dir = "E:\\LIRADS"
			self.coord_xls_path = 'Z:\\LIRADS\\Prototype1e.xlsx'
			self.test_num = 5
			self.full_img_dir = "Z:\\LIRADS\\full_imgs"

			self.cls_names = ['hcc', 'cholangio', 'colorectal', 'cyst', 'hemangioma', 'fnh', 'net', 'adenoma', 'abscess']
			self.sheetnames = ['HCC', 'Cholangio', 'Colorectal', 'Cyst', 'Hemangioma', 'FNH', 'NET', 'Adenoma', 'Abscess']
			self.short_cls_names = ['HCC', 'ICC', 'CRC Met.', 'Cyst', 'Hemang.', 'FNH', 'NET', 'Adenoma', 'Abscess']
			self.dcm_dirs = ["Z:\\LIRADS\\DICOMs\\" + fn for fn in self.cls_names]

		self.art_voi_path = join(self.base_dir, "excel\\voi_art_full.csv")
		self.ven_voi_path = join(self.base_dir, "excel\\voi_ven_full.csv")
		self.eq_voi_path = join(self.base_dir, "excel\\voi_eq_full.csv")
		self.dims_df_path = join(self.base_dir, "excel\\img_dims.csv")
		self.small_voi_path = join(self.base_dir, "excel\\small_vois_full.csv")
		self.run_stats_path = join(self.base_dir, "excel\\overnight_run.csv")
		self.patient_info_path = join(self.base_dir, "excel\\patient_info.csv")

		self.crops_dir = join(self.base_dir, "imgs\\rough_crops\\")
		self.unaug_dir = join(self.base_dir, "imgs\\unaug_imgs\\")
		self.aug_dir = join(self.base_dir, "imgs\\aug_imgs\\")
		self.replay_img_dir = join(self.base_dir, "imgs\\replay\\")
		self.model_dir = join(self.base_dir, "models")

class Hyperparams:
	def __init__(self):
		self.n = 4
		self.n_art = 0
		self.steps_per_epoch = 750
		self.epochs = 20
		self.run_2d = False
		self.f = [64,64,64,64,64]
		self.padding = ['same','same']
		self.dropout = [0.1,0.1]
		self.dense_units = 100
		self.dilation_rate = (1,1,1)
		#self.stride = (2,2,2)
		self.kernel_size = (3,3,2)
		self.pool_sizes = [(2,2,2),(2,2,2)]
		self.activation_type = 'relu'
		self.rcnn = True
		self.optimizer = 'adam'
		self.early_stopping = EarlyStopping(monitor='loss', min_delta=0.002, patience=3)
		self.skip_con = False
		#self.non_imaging_inputs = C.non_imaging_inputs

	def get_best_hyperparams(self):
		self.n = 4
		self.n_art = 0
		self.steps_per_epoch = 750
		self.epochs = 30
		self.f = [64,128,128]
		self.padding = ['same','valid']
		self.dropout = [0.1,0.1]
		self.dense_units = 100
		self.dilation_rate = (1,1,1)
		self.kernel_size = (3,3,2)
		self.pool_sizes = [(2,2,2),(2,2,1)]
		self.activation_type = 'relu'
		self.rcnn = False

	def get_capsnet_params(self):
		self.lr = 4
		self.lr_decay = 0
		self.n = 2
		self.steps_per_epoch = 750
		self.epochs = 30
		self.dense_layers = True
		self.dim_capsule = [8, 16]
		self.dense_units = 256 #512
		self.n_channels = 16 # 32
		self.rcnn = False

	def get_random_hyperparams(self):
		self.f = random.choice([[64,128,128], [64,64,128,128], [64,64,64,128], [64,64,128,128], [64,128,128,128], [64,128,128,128,128]])
		self.pool_sizes = [(2,2,2),(2,2,1)]
		self.padding = ['same','valid']
		self.dense_units = 100
		self.dilation_rate = (1,1,1)
		self.kernel_size = (3,3,2)
		self.pool_sizes = [(2,2,2),(2,2,1)]
		self.activation_type = 'relu'
		self.rcnn = False