import pyelastix

#Run Numbers:
# 1 - classic
# 2 - classic but hard
# 3 - atypical, maybe exclude
# 4 - atypical, exclude
# 5 - not yet included
# 6 - inconsistent slices, missing phase, bad imaging quality, etc. see notes

class Config:
	def __init__(self):
		self.run_num = 2
		self.dims = [24,24,12]# [24,24,12]
		self.nb_channels = 3
		self.params = pyelastix.get_default_params(type="AFFINE")
		self.classes_to_include = ['hcc', 'cholangio', 'colorectal', 'cyst', 'hemangioma', 'fnh']
		self.aug_factor = 50#{"cyst": 65, "hcc": 65, "hemangioma": 75, "fnh": 70, "adenoma": 70, "colorectal": 70, "cholangio": 70}
		self.train_frac = .8
		#self.train_frac = {"cyst": .8, "fnh": .8, "hcc": .8, "hemangioma": .8, 'adenoma': .8, "colorectal": .8, "cholangio": .8}
		self.padding = 0.75
		self.intensity_scaling = [.05,.05]
		self.intensity_local_frac = .5

		self.model_path = "..\\data\\model.hdf5"
		self.art_voi_path = "..\\data\\voi_art.csv"
		self.ven_voi_path = "..\\data\\voi_ven.csv"
		self.eq_voi_path = "..\\data\\voi_eq.csv"
		self.dims_df_path = "..\\data\\img_dims.csv"
		self.int_df_path = "..\\data\\intensity.csv"
		self.small_voi_path = "..\\data\\small_vois.csv"
		self.run_stats_path = "..\\data\\overnight_run.csv"
		self.patient_info_path = "Z:\\patient_info.csv"

		self.xls_name = 'Z:\\Prototype1d.xlsx'
		self.cls_names = ['hcc', 'hcc', 'cyst', 'hemangioma', 'fnh', 'cholangio', 'colorectal', 'adenoma']
		self.sheetnames = ['OPTN 5A', 'OPTN 5B', 'Cyst', 'Hemangioma', 'FNH', 'Cholangio', 'Colorectal', 'Adenoma']
		self.img_dirs = ['OPTN5A', 'optn5b', 'simple_cysts', 'hemangioma', 'fnh', 'cholangio', 'colorectal', 'adenoma']

		self.full_img_dir = "Z:\\INPUT\\full_imgs_origdims"
		self.output_img_dir = "Z:\\OUTPUT\\12-15"
		self.vois_dir = "Z:\\OUTPUT\\small-vois\\12-11\\"
		self.crops_dir = "E:\\imgs\\unscaled_crops_hardscaled\\"
		self.artif_dir = "E:\\imgs\\artif_imgs_2412\\"
		self.aug_dir = "E:\\imgs\\aug_imgs_2412_cropint\\"
		self.orig_dir = "E:\\imgs\\orig_imgs_2412_cropint\\"

		# Artificial sample parameters
		self.long_size_frac = [0.6, 0.85]
		self.max_side_ratio = 1.6
		self.noise_std = 0.05
		self.shade_std = 0.05

		self.hard_scale = False
