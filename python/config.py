"""
Config file

Author: Clinton Wang, E-mail: `clintonjwang@gmail.com`, Github: `https://github.com/clintonjwang/voi-classifier`
"""

from os.path import *
from keras.callbacks import EarlyStopping
from keras.optimizers import Adam
import niftiutils.helper_fxns as hf

class Config:
	def __init__(self, dataset="radpath"):
		self.run_num = 5
		self.test_run_num = 2
		self.dims = [24,24,12]
		self.nb_channels = 3
		self.aug_factor = 100

		self.max_size = 350*350*100
		self.context_dims = [36,36,12]
		self.dual_img_inputs = False # whether to use both tight and gross image croppings for the network
		self.clinical_inputs = 0 # whether non-imaging inputs should be incorporated into the neural network

		self.lesion_ratio = 0.7 # ratio of the lesion side length to the length of the cropped image
		self.pre_scale = .5 # normalizes images at augmentation time
		self.post_scale = 0. # normalizes images at train/test time

		#optional
		self.focal_loss = 1.
		self.aleatoric = True
		self.aug_pred = False
		self.ensemble_num = 8
		self.ensemble_frac = .7 #train each submodel on this fraction of training data

		# Augmentation parameters
		self.intensity_scaling = [.05,.05]
		self.translate = [2,2,1]

		self.use_dataset(dataset)

		self.nb_classes = len(self.cls_names)
		self.phase_dirs = ["T1_20s", "T1_70s", "T1_3min"]

	def use_dataset(self, dataset):
		self.dataset = dataset
		if dataset.startswith("lirads") or dataset == "clinical":
			self.base_dir = "E:\\LIRADS"
			self.coord_xls_path = 'Z:\\LIRADS\\excel\\Prototype1e.xlsx'
			if dataset == "clinical":
				self.coord_xls_path = r"Z:\Paula\Clinical data project\coordinates + clinical variables.xlsx"
				self.clinical_inputs = 9 # whether non-imaging inputs should be incorporated into the neural network
			self.test_num = 10
			self.aug_factor = 100

			self.cls_names = ['hcc', 'cholangio', 'colorectal', 'cyst', 'hemangioma', 'fnh']
			self.sheetnames = ['HCC', 'Cholangio', 'Colorectal', 'Cyst', 'Hemangioma', 'FNH']
			self.short_cls_names = ['HCC', 'ICC', 'CRC Met.', 'Cyst', 'Hemang.', 'FNH']
			self.dcm_dirs = ["Z:\\LIRADS\\DICOMs\\" + fn for fn in self.cls_names]
			self.simplify_map = {'hcc': 0, 'cyst': 1, 'hemangioma': 1, 'fnh': 1, 'cholangio': 2, 'colorectal': 2}
			
			if dataset == "lirads-expanded":
				self.base_dir = "E:\\LIRADS"
				self.coord_xls_path = 'Z:\\LIRADS\\excel\\Prototype1e.xlsx'
				self.test_num = 5

				self.cls_names = ['hcc', 'cholangio', 'colorectal', 'cyst', 'hemangioma', 'fnh', 'net', 'adenoma', 'abscess']
				self.sheetnames = ['HCC', 'Cholangio', 'Colorectal', 'Cyst', 'Hemangioma', 'FNH', 'NET', 'Adenoma', 'Abscess']
				self.short_cls_names = ['HCC', 'ICC', 'CRC Met.', 'Cyst', 'Hemang.', 'FNH', 'NET', 'Adenoma', 'Abscess']
				self.dcm_dirs = ["Z:\\LIRADS\\DICOMs\\" + fn for fn in self.cls_names]

		elif dataset.startswith("etiology"):
			self.dims = [32,32,16]
			self.context_dims = [100,100,40]
			self.num_segs = 3
			self.loss_weights = [[5,20,400],[.1,.1,.1]]
			self.state_dim = (*self.dims, 4)

			self.sheetname = "Lesion Coordinates"
			self.base_dir = "D:\\Etiology"
			#self.coord_xls_path = "D:\\Etiology\\excel\\coords.xlsx"
			self.coord_xls_path = r"Z:\Sophie\Ethiologyproject\HCC Etiology Project Master Spreadsheet.xlsx"
			self.test_num = 5
			#self.full_img_dir = join(self.base_dir, "imgs", "full_imgs")

			self.cls_names = ['HBV', 'HCV', 'EtOH', 'NASH']
			self.dcm_dir = r"Z:\Sophie\Ethiologyproject\Additional MRIs"
			#self.dcm_dir = "D:\\Etiology\\Imaging"

		elif dataset == "radpath":
			self.base_dir = "D:\\Radpath"
			self.coord_xls_path = "Z:\\Paula\\Radpath\\new coordinates_CW.xlsx"
			self.test_num = 10

			self.cls_names = ['hcc', 'non-hcc']
			self.sheetnames = ['HCC', 'Non-HCC']
			self.short_cls_names = self.sheetnames
			self.dcm_dir = "Z:\\Paula\\Radpath\\Imaging"

		self.full_img_dir = "Z:\\LIRADS\\full_imgs"
		#self.patient_info_path = join(self.base_dir, "excel", "patient_data.csv")
		self.dim_cols = ["voxdim_x", "voxdim_y", "voxdim_z"]
		self.accnum_cols = ["MRN", "Sex", "AgeAtImaging", "Ethnicity"] + self.dim_cols

		self.voi_cols = [hf.flatten([[ph+ch+'1', ph+ch+'2'] for ch in ['x','y','z']]) for ph in ['a_','v_','e_','sm_']]
		self.art_cols, self.ven_cols, self.equ_cols, self.small_cols = self.voi_cols
		self.voi_cols = hf.flatten(self.voi_cols)

		# An accnum must be in accnum_df for it to be processed
		self.accnum_df_path = join("Z:\\LIRADS\\excel", "accnum_data.csv")
		# A lesion must be in lesion_df for it to be processed
		self.lesion_df_path = join(self.base_dir, "excel", "lesion_data.csv")
		self.run_stats_path = join(self.base_dir, "excel", "overnight_run.csv")
		self.label_df_path = join(self.base_dir, "excel", "lesion_labels.csv")

		self.crops_dir = join(self.base_dir, "imgs", "rough_crops")
		self.unaug_dir = join(self.base_dir, "imgs", "unaug_imgs")
		self.aug_dir = join(self.base_dir, "imgs", "aug_imgs")
		self.replay_img_dir = join(self.base_dir, "imgs", "replay")
		self.model_dir = join(self.base_dir, "models")

class Hyperparams:
	def __init__(self):
		self.n = 4
		self.cnn_type = 'vanilla'
		self.steps_per_epoch = 750
		self.epochs = 20
		self.f = [64,64,64,64,64]
		self.padding = ['same','same']
		self.dropout = 0.1
		self.depth = 22
		self.dense_units = 100
		self.kernel_size = (3,3,2)
		self.pool_sizes = [2,2]
		self.optimizer = Adam(lr=0.001)
		self.early_stopping = EarlyStopping(monitor='loss', min_delta=0.002, patience=5)
		self.skip_con = False
		self.mc_sampling = True

	def get_best_hyperparams(self, dataset):
		if dataset == 'radpath':
			self.n = 32
			self.steps_per_epoch = 100
			self.epochs = 30
			self.f = [64,80,80]
			self.padding = ['same','valid']
			self.dropout = 0.1
			self.dense_units = 100
			self.kernel_size = (3,3,2)
			self.pool_sizes = [2,2]
		else:
			self.n = 4
			self.steps_per_epoch = 750
			self.epochs = 30
			self.f = [64,128,128]
			self.padding = ['same','valid']
			self.dropout = 0.1
			self.dense_units = 100
			self.kernel_size = (3,3,2)
			self.pool_sizes = [2,(2,2,1)]

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
		raise ValueError()
