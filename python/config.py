"""
Config file

Author: Clinton Wang, E-mail: `clintonjwang@gmail.com`, Github: `https://github.com/clintonjwang/voi-classifier`
"""

from os.path import *
from keras.callbacks import EarlyStopping
from keras.optimizers import Adam
import niftiutils.helper_fxns as hf

class Config:
	def __init__(self, dataset="lirads"):
		self.run_num = 2
		self.test_run_num = 2
		self.dims = [32,32,16]
		self.nb_channels = 3
		self.aug_factor = 256

		self.max_size = 350*350*100
		self.context_dims = [36,36,12]
		self.dual_img_inputs = False # whether to use both tight and gross image croppings for the network
		self.clinical_inputs = 0 # whether non-imaging inputs should be incorporated into the neural network

		self.lesion_ratio = .7 # ratio of the lesion side length to the length of the cropped image
		self.pre_scale = .9 # normalizes images while saving augmented/unaugmented images
		self.post_scale = 0. # normalizes images at train/test time

		#optional
		self.focal_loss = 0.
		self.aleatoric = False #currently cannot be used with multiple inputs
		self.aug_pred = False
		self.ensemble_num = 0
		self.ensemble_frac = .7 #train each submodel on this fraction of training data

		# Augmentation parameters
		self.intensity_scaling = [.1,.01]

		self.use_dataset(dataset)

		self.nb_classes = len(self.cls_names)
		self.phase_dirs = ["T1_20s", "T1_70s", "T1_3min"]

	def use_dataset(self, dataset):
		self.dataset = dataset
		if dataset.startswith("lirads") or dataset == "clinical":
			self.base_dir = "/home/idealab/Documents/Clinton/LIRADS"
			self.coord_xls_path = '/mnt/LIRADS/excel/Prototype1e.xlsx'
			if dataset == "clinical":
				self.run_num = 4
				self.test_run_num = 4
				self.coord_xls_path = r"Z:\Paula\Clinical data project\3+4 POINT LESIONS ADDED coordinates + clinical variables.xlsx"
				self.clinical_inputs = 9 # whether non-imaging inputs should be incorporated into the neural network
			self.test_num = 10
			self.Z_reader = ['E103312835_1','12823036_0','12569915_0','E102093118_0','E102782525_0','12799652_0','E100894274_0','12874178_3','E100314676_0','12842070_0','13092836_2','12239783_0','12783467_0','13092966_0','E100962970_0','E100183257_1','E102634440_0','E106182827_0','12582632_0','E100121654_0','E100407633_0','E105310461_0','12788616_0','E101225606_0','12678910_1','E101083458_1','12324408_0','13031955_0','E101415263_0','E103192914_0','12888679_2','E106096969_0','E100192709_1','13112385_1','E100718398_0','12207268_0','E105244287_0','E102095465_0','E102613189_0','12961059_0','11907521_0','E105311123_0','12552705_0','E100610622_0','12975280_0','E105918926_0','E103020139_1','E101069048_1','E105427046_0','13028374_0','E100262351_0','12302576_0','12451831_0','E102929168_0','E100383453_0','E105344747_0','12569826_0','E100168661_0','12530153_0','E104697262_0']

			self.cls_names = ['hcc', 'cholangio', 'colorectal', 'cyst', 'hemangioma', 'fnh']
			self.sheetnames = ['HCC', 'Cholangio', 'Colorectal', 'Cyst', 'Hemangioma', 'FNH']
			self.short_cls_names = ['HCC', 'ICC', 'CRC Met.', 'Cyst', 'Hemang.', 'FNH']
			self.dcm_dirs = [join("/mnt/LIRADS/DICOMs", cls) for cls in self.cls_names]
			self.simplify_map = {'hcc': 0, 'cyst': 1, 'hemangioma': 1, 'fnh': 1, 'cholangio': 2, 'colorectal': 2}
			
			if dataset == "lirads-expanded":
				self.base_dir = "E:\\LIRADS"
				self.coord_xls_path = 'Z:\\LIRADS\\excel\\Prototype1e.xlsx'
				self.test_num = 5

				self.cls_names = ['hcc', 'cholangio', 'colorectal', 'cyst', 'hemangioma', 'fnh', 'net', 'adenoma', 'abscess']
				self.sheetnames = ['HCC', 'Cholangio', 'Colorectal', 'Cyst', 'Hemangioma', 'FNH', 'NET', 'Adenoma', 'Abscess']
				self.short_cls_names = ['HCC', 'ICC', 'CRC Met.', 'Cyst', 'Hemang.', 'FNH', 'NET', 'Adenoma', 'Abscess']
				self.dcm_dirs = ["Z:\\LIRADS\\DICOMs\\" + cls for cls in self.cls_names]

		elif dataset.startswith("etiology"):
			self.dims = [32,32,16]
			self.context_dims = [100,100,40]
			self.num_segs = 3
			self.loss_weights = [[5,20,400],[.1,.1,.1]]
			self.state_dim = (*self.dims, 4)

			self.sheetname = "Lesion Coordinates"
			self.base_dir = "D:\\Etiology"
			#self.coord_xls_path = "D:\\Etiology\\excel\\coords.xlsx"
			self.coord_xls_path = r"Z:\Sophie\Etiologyproject\HCC Etiology Project Master Spreadsheet.xlsx"
			self.test_num = 5
			#self.full_img_dir = join(self.base_dir, "imgs", "full_imgs")

			self.cls_names = ['HBV', 'HCV', 'EtOH', 'NASH']
			self.dcm_dir = r"Z:\Sophie\Etiologyproject\Additional MRIs"
			#self.dcm_dir = "D:\\Etiology\\Imaging"

		elif dataset == "radpath":
			self.base_dir = "D:\\Radpath"
			self.coord_xls_path = "Z:\\Paula\\Radpath\\new coordinates_CW.xlsx"
			self.test_num = 5
			self.dims = [32,32,16]
			self.Z_reader = ['E106405787_0','E106329048_0','E106158268_0','E106120112_0','E106097391_0','E106097366_0','E106004664_0','E105906532_0','E105799828_0','E105492224_0','E105344790_0','E105333398_0','E105326292_0','E105310461_0','E105160323_0','E105152299_0','E105124678_0','E105110150_0','E105095742_0','E105066561_0','E104833037_0','E104587275_0','E104270981_0','E104201087_0','E104140436_0','E104099161_0','E104082888_0','E103678771_0','E103570649_0','E103314435_0','E103306623_0','E103301795_0','E103020139_1','E102929168_0','E102634440_0','E102589834_0','E102424706_0','E102388865_0','E102256903_0','E102130844_0','E102088195_1','E102031795_0','E102027289_0','E101949001_0','E101895019_0','E101892543_0','E101880575_0','E101805234_0','E101790015_0','E101784996_0','E101779513_0','E101773506_0','E101686218_0','E101523098_3','E101449797_0','E101442376_0','E101396972_0','E101290891_0','E101158768_1','E101068962_0']

			self.cls_names = ['hcc', 'non-hcc']
			self.sheetnames = ['HCC', 'Non-HCC']
			self.short_cls_names = self.sheetnames
			self.dcm_dir = "Z:\\Paula\\Radpath\\Imaging"

		self.full_img_dir = "/mnt/LIRADS/full_imgs"
		#self.patient_info_path = join(self.base_dir, "excel", "patient_data.csv")
		self.dim_cols = ["voxdim_x", "voxdim_y", "voxdim_z"]
		self.accnum_cols = ["MRN", "Sex", "AgeAtImaging", "Ethnicity"] + self.dim_cols + ["downsample"]

		self.voi_cols = [hf.flatten([[ph+ch+'1', ph+ch+'2'] for ch in ['x','y','z']]) for ph in ['a_','v_','e_']]
		self.art_cols, self.ven_cols, self.equ_cols = self.voi_cols
		self.pad_cols = ['pad_x','pad_y','pad_z']
		self.voi_cols = hf.flatten(self.voi_cols) + self.pad_cols

		# An accnum must be in accnum_df for it to be processed
		self.accnum_df_path = join("/mnt/LIRADS/excel", "accnum_data.csv")
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
	def __init__(self, dataset=None):
		self.n = 4
		self.cnn_type = 'vanilla'
		self.steps_per_epoch = 50
		self.epochs = 3
		self.f = [64,64,64,64]
		self.padding = ['same','same']
		self.dropout = 0.1
		self.depth = 6
		self.dense_units = 100
		self.kernel_size = (3,3,2)
		self.pool_sizes = [2,2]
		self.optimizer = Adam(lr=0.0001)
		self.early_stopping = EarlyStopping(monitor='loss', min_delta=0.001, patience=25)
		self.skip_con = False
		self.mc_sampling = False #currently cannot be used with multiple inputs

		if dataset is not None:
			self.get_best_hyperparams(dataset)

	def get_best_hyperparams(self, dataset):
		if dataset == 'radpath':
			self.n = 8
			self.epochs = 30
			self.steps_per_epoch = 150
			self.f = [64,100,100,100]
			self.dense_units = 100
			self.padding = ['valid','valid']
			self.dropout = .2
			self.kernel_size = (3,3,2)
			self.pool_sizes = [(2,2,1),(2,2,1)]
		elif dataset == 'lirads':
			self.n = 4 #5
			self.epochs = 256
			self.steps_per_epoch = 256
			self.dropout = .2
			self.dense_units = 128
			self.padding = ['same','valid']
			self.f = [64,128,128,128]
			self.kernel_size = (3,3,2)
			self.pool_sizes = [2,2]
		else:
			self.kernel_size = (3,3,2)
			self.n = 4 #5
			self.epochs = 30
			self.steps_per_epoch = 300
			self.dropout = .1
			self.dense_units = 100
			self.padding = ['same','valid'] #['same','valid']
			self.f = [64,100,100]
			self.pool_sizes = [2,2] #[2,2]

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
