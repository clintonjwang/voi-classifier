class Config:
	def __init__(self):
		self.run_num = 2
		self.dims = [24,24,12]
		self.nb_channels = 3
		self.nb_classes = 4
		self.vox_dims = [1.25, 1.25, 2.5]
		self.aug_factor = 50#{"cyst": 65, "hcc": 65, "hemangioma": 75, "fnh": 70, "adenoma": 70, "colorectal": 70, "cholangio": 70}
		self.train_frac = {"cyst": .8, "fnh": .8, "hcc": .8, "hemangioma": .8, 'adenoma': .8, "colorectal": .8, "cholangio": .8}

		self.model_path = "model.hdf5"
		self.full_img_dir = "Z:\\full_imgs_origdims"
		#self.full_img_dir = "Z:\\full_imgs_x125_z25"
		self.art_voi_path = "voi_art.csv"
		self.ven_voi_path = "voi_ven.csv"
		self.eq_voi_path = "eq_ven.csv"
		self.dims_df_path = "img_dims.csv"
		self.int_df_path = "intensity.csv"
		self.aug_dir = "Z:\\aug_imgs\\"
		self.orig_dir = "Z:\\orig_imgs\\"
		self.crops_dir = "Z:\\unscaled_training_data\\"