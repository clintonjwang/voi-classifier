import pyelastix

class Config:
	def __init__(self):
		self.run_num = 2
		self.dims = [36,36,12] #[48,48,12]
		self.nb_channels = 3
		self.params = pyelastix.get_default_params(type="AFFINE")
		self.classes_to_include = ['hcc', 'cholangio', 'colorectal', 'cyst', 'hemangioma', 'fnh']
		self.aug_factor = 50#{"cyst": 65, "hcc": 65, "hemangioma": 75, "fnh": 70, "adenoma": 70, "colorectal": 70, "cholangio": 70}
		self.train_frac = .8
		#self.train_frac = {"cyst": .8, "fnh": .8, "hcc": .8, "hemangioma": .8, 'adenoma': .8, "colorectal": .8, "cholangio": .8}
		self.padding = 0.85
		self.intensity_scaling = [.05,.05]

		self.model_path = "model.hdf5"		
		self.art_voi_path = "voi_art.csv"
		self.ven_voi_path = "voi_ven.csv"
		self.eq_voi_path = "eq_ven.csv"
		self.dims_df_path = "img_dims.csv"
		self.int_df_path = "intensity.csv"
		self.small_voi_path = "small_vois.csv"
		self.run_stats_path = "overnight_run.csv"
		self.patient_info_path = "Z:\\INPUT\\patient_info.csv"

		self.full_img_dir = "Z:\\INPUT\\full_imgs_origdims"
		self.output_img_dir = "Z:\\OUTPUT\\12-08-3d"
		self.vois_dir = "Z:\\OUTPUT\\12-11-vois_art-int\\"
		self.crops_dir = "E:\\imgs\\unscaled_crops\\"
		self.artif_dir = "E:\\imgs\\artif_imgs\\"
		self.aug_dir = "E:\\imgs\\aug_imgs\\"
		self.orig_dir = "E:\\imgs\\orig_imgs\\"

		# Artificial sample parameters
		self.long_size_frac = [0.6, 0.85]
		self.max_side_ratio = 1.6
		self.noise_std = 0.05
		self.shade_std = 0.05
