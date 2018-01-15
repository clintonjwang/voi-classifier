"""
Converts a nifti file to a numpy array.
Accepts either a single nifti file or a folder of niftis as the input argument.

Usage:
	python nii_to_np.py data/nifti/volume-01.nii
	python nii_to_np.py data/nifti/
	python nii_to_np.py data/nifti/ --dest data/numpy/
	python nii_to_np.py data/nifti/volume-01.nii --dest data/numpy/

Author: Clinton Wang, E-mail: `clintonjwang@gmail.com`, Github: `https://github.com/clintonjwang/voi-classifier`
"""

#Run Numbers:
# 1 - classic
# 2 - classic but difficult
# 3 - atypical, maybe exclude
# 4 - atypical, exclude
# 5 - temporarily excluded
# 6 - inconsistent slices, missing phase, bad imaging quality, etc. see notes

class Config:
	def __init__(self):
		import pyelastix
		self.run_num = 5
		self.dims = [24,24,12]
		self.nb_channels = 3
		self.reg_params = pyelastix.get_default_params(type="AFFINE")
		self.aug_factor = 100
		self.train_frac = None#.9
		self.test_num = 10 # only used if train_frac is None

		self.non_imaging_inputs = False # whether non-imaging inputs should be incorporated into the neural network
		self.num_non_image_inputs = 3

		self.lesion_ratio = 0.75 # ratio of the lesion side length to the length of the cropped image

		self.intensity_local_frac = .5 # normalizes all images locally by this fraction
		self.hard_scale = False # if True, normalizes all images within the tightly cropped region

		#self.model_path = "..\\data\\model.hdf5"
		self.art_voi_path = "..\\data\\voi_art_full.csv"
		self.ven_voi_path = "..\\data\\voi_ven_full.csv"
		self.eq_voi_path = "..\\data\\voi_eq_full.csv"
		self.dims_df_path = "..\\data\\img_dims.csv"
		self.int_df_path = "..\\data\\intensity.csv"
		self.small_voi_path = "..\\data\\small_vois_full.csv"
		self.run_stats_path = "..\\data\\overnight_run.csv"
		self.patient_info_path = "Z:\\patient_info.csv"
		self.xls_name = 'Z:\\Prototype1e.xlsx'

		# Information about the abnormality classes
		self.classes_to_include = ['hcc', 'cholangio', 'colorectal', 'cyst', 'hemangioma', 'fnh']
		self.sheetnames = ['HCC', 'Cholangio', 'Colorectal', 'Cyst', 'Hemangioma', 'FNH', 'Adenoma']
		self.patient_sheetname = 'Patient Info'
		self.cls_names = ['hcc', 'cholangio', 'colorectal', 'cyst', 'hemangioma', 'fnh', 'adenoma']
		self.img_dirs = self.cls_names
		self.simplify_map = {'hcc': 0, 'cyst': 1, 'hemangioma': 1, 'fnh': 1, 'cholangio': 2, 'colorectal': 2}

		self.full_img_dir = "Z:\\INPUT\\full_imgs_origdims"
		self.output_img_dir = "Z:\\OUTPUT\\1-14"
		self.crops_dir = "E:\\imgs\\unscaled_crops_full\\"
		self.aug_dir = "E:\\imgs\\aug_imgs_2412_full\\"
		self.orig_dir = "E:\\imgs\\orig_imgs_2412_full\\"
		self.artif_dir = "E:\\imgs\\artif_imgs_2412\\"
		self.model_dir = "E:\\models\\"

		# Augmentation parameters
		self.intensity_scaling = [.05,.05]
		self.translate = [2,2,1]

		# Artificial sample parameters
		self.long_size_frac = [0.6, 0.85]
		self.max_side_ratio = 1.6
		self.noise_std = 0.05
		self.shade_std = 0.05

class Hyperparams:
	def __init__(self):
		from keras.callbacks import EarlyStopping
		self.n = 1
		self.n_art = 0
		self.steps_per_epoch = 1500
		self.epochs = 15
		self.run_2d = False
		self.f = [64,64,128,128]
		self.padding = ['valid','valid']
		self.dropout = [0.1,0.1]
		self.dense_units = 128
		self.dilation_rate = (1,1,1)
		self.stride = (2,2,2)
		self.kernel_size = (3,3,2)
		self.pool_sizes = [(2,2,2),(1,1,1)]
		self.activation_type = 'relu'
		self.merge_layer = 0
		self.time_dist = True
		self.optimizer = 'adam'
		self.early_stopping = EarlyStopping(monitor='loss', min_delta=0.002, patience=3)
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
		self.merge_layer = 1
		self.time_dist = False

	def get_random_hyperparams(self):
		self.f = random.choice([[64,128,128], [64,64,128,128], [64,64,64,128], [64,64,128,128], [64,128,128,128], [64,128,128,128,128]])
		self.pool_sizes = [(2,2,2),(2,2,1)]
		self.padding = ['same','valid']
		self.dense_units = 100
		self.dilation_rate = (1,1,1)
		self.kernel_size = (3,3,2)
		self.pool_sizes = [(2,2,2),(2,2,1)]
		self.activation_type = 'relu'
		self.merge_layer = 1
		self.time_dist = False