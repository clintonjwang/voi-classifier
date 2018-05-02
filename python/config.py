"""
Config file

Author: Clinton Wang, E-mail: `clintonjwang@gmail.com`, Github: `https://github.com/clintonjwang/voi-classifier`
"""

#Run Numbers:
# 1 - classic
# 2 - classic but difficult
# 3 - atypical, maybe exclude
# 4 - atypical, exclude
# 5 - temporarily excluded
# 6 - inconsistent slices, missing phase, bad imaging quality, etc. see notes
from os.path import *

class Config:
	def __init__(self):
		self.run_num = 2
		self.test_run_num = 2
		self.dims = [24,24,12]
		self.nb_channels = 3
		self.aug_factor = 100
		self.train_frac = None#.9
		self.test_num = 10 # only used if train_frac is None

		self.context_dims = [36,36,12]
		self.dual_img_inputs = False # whether to use both tight and gross image croppings for the network
		self.non_imaging_inputs = False # whether non-imaging inputs should be incorporated into the neural network

		self.lesion_ratio = 0.7 # ratio of the lesion side length to the length of the cropped image

		self.pre_scale = .5 # normalizes images at augmentation time
		self.post_scale = 0. # normalizes images at train/test time

		self.base_dir = "C:\\Users\\Clinton\\Documents\\voi-classifier"
		self.art_voi_path = join(self.base_dir, "data\\voi_art_full.csv")
		self.ven_voi_path = join(self.base_dir, "data\\voi_ven_full.csv")
		self.eq_voi_path = join(self.base_dir, "data\\voi_eq_full.csv")
		self.dims_df_path = join(self.base_dir, "data\\img_dims.csv")
		self.int_df_path = join(self.base_dir, "data\\intensity.csv")
		self.small_voi_path = join(self.base_dir, "data\\small_vois_full.csv")
		self.run_stats_path = join(self.base_dir, "data\\overnight_run.csv")
		self.patient_info_path = join(self.base_dir, "data\\patient_info.csv")
		self.xls_name = 'Z:\\LIRADS\\Prototype1e.xlsx'

		# Information about the abnormality classes
		self.classes_to_include = ['hcc', 'cholangio', 'colorectal', 'cyst', 'hemangioma', 'fnh']
		self.nb_classes = len(self.classes_to_include)
		self.cls_names = ['hcc', 'cholangio', 'colorectal', 'cyst', 'hemangioma', 'fnh']#, 'net', 'adenoma', 'abscess']
		self.sheetnames = ['HCC', 'Cholangio', 'Colorectal', 'Cyst', 'Hemangioma', 'FNH', 'NET', 'Adenoma', 'Abscess']
		self.short_cls_names = ['HCC', 'ICC', 'CRC Met.', 'Cyst', 'Hemang.', 'FNH']#, 'NET', 'Adenoma', 'Abscess']
		self.patient_sheetname = 'Patient Info'
		self.img_dirs = ["Z:\\LIRADS\\DICOMs\\" + fn for fn in self.cls_names]
		self.simplify_map = {'hcc': 0, 'cyst': 1, 'hemangioma': 1, 'fnh': 1, 'cholangio': 2, 'colorectal': 2}

		self.full_img_dir = "Z:\\LIRADS\\full_imgs"
		self.output_img_dir = "Z:\\LIRADS\\OUTPUT\\1-14"
		self.crops_dir = "E:\\imgs\\unscaled_crops_full\\"
		self.aug_dir = "E:\\imgs\\aug_imgs_2412_full\\"
		self.orig_dir = "E:\\imgs\\orig_imgs_2412_full\\"
		self.artif_dir = "E:\\imgs\\artif_imgs_2412\\"
		self.model_dir = "E:\\models\\"

		# Augmentation parameters
		self.intensity_scaling = [.05,.05]
		self.translate = [2,2,1]
		self.use_paula_dataset()

	def turn_on_clinical_features(self):
		self.non_imaging_inputs = True # whether non-imaging inputs should be incorporated into the neural network
		self.num_non_image_inputs = 3

	def use_expanded_dataset(self):
		self.classes_to_include = ['hcc', 'cholangio', 'colorectal', 'cyst', 'hemangioma', 'fnh', 'net', 'adenoma', 'abscess']
		self.nb_classes = len(self.classes_to_include)
		self.cls_names = self.classes_to_include
		self.sheetnames = ['HCC', 'Cholangio', 'Colorectal', 'Cyst', 'Hemangioma', 'FNH', 'NET', 'Adenoma', 'Abscess']
		self.short_cls_names = ['HCC', 'ICC', 'CRC Met.', 'Cyst', 'Hemang.', 'FNH', 'NET', 'Adenoma', 'Abscess']
		
	def use_paula_dataset(self):
		self.test_num = 5

		self.classes_to_include = ['hcc', 'non-hcc']
		self.nb_classes = len(self.classes_to_include)
		self.cls_names = self.classes_to_include
		self.sheetnames = ['HCC', 'Non-HCC']
		self.short_cls_names = self.sheetnames
		self.simplify_map = {'hcc': 0, 'non-hcc': 1}
		self.img_dirs = ["Z:\\Paula\\Imaging", "Z:\\Paula\\Imaging"]

		self.base_dir = "D:\\Paula-project"
		self.art_voi_path = join(self.base_dir, "data\\voi_art_full.csv")
		self.ven_voi_path = join(self.base_dir, "data\\voi_ven_full.csv")
		self.eq_voi_path = join(self.base_dir, "data\\voi_eq_full.csv")
		self.dims_df_path = join(self.base_dir, "data\\img_dims.csv")
		self.int_df_path = join(self.base_dir, "data\\intensity.csv")
		self.small_voi_path = join(self.base_dir, "data\\small_vois_full.csv")
		self.run_stats_path = join(self.base_dir, "data\\overnight_run.csv")
		self.patient_info_path = join(self.base_dir, "data\\patient_info.csv")
		self.xls_name = "Z:\\Paula\\new coordinates_CW.xlsx"
		
		self.full_img_dir = join(self.base_dir, "full_imgs")
		self.output_img_dir = join(self.base_dir, "OUTPUT")
		self.crops_dir = join(self.base_dir, "imgs\\unscaled_crops_full\\")
		self.aug_dir = join(self.base_dir, "imgs\\aug_imgs_2412_full\\")
		self.orig_dir = join(self.base_dir, "imgs\\orig_imgs_2412_full\\")
		self.artif_dir = join(self.base_dir, "imgs\\artif_imgs_2412\\")
		self.model_dir = join(self.base_dir, "models\\")

	def use_artificial_samples(self):
		# Artificial sample parameters
		self.n_aug = 1500
		self.long_size_frac = [0.6, 0.85]
		self.max_side_ratio = 1.6
		self.noise_std = 0.05
		self.shade_std = 0.05

class Hyperparams:
	def __init__(self):
		from keras.callbacks import EarlyStopping
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