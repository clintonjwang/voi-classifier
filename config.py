class Config:
	def __init__(self):
		self.run_num = 2
		self.dims = [24,24,12]
		self.nb_channels = 3
		self.nb_classes = 3
		self.vox_dims = [1.5, 1.5, 4]
		self.aug_factor = {"cyst": 45, "hcc": 45, "hemangioma": 55, "fnh": 50, "adenoma": 50, "colorectal": 50, "cholangio": 50}

		self.model_path = "model.hdf5"
		self.full_img_dir = "full_imgs3phase"
		self.art_voi_path = "voi_art.csv"
		self.ven_voi_path = "voi_ven.csv"
		self.eq_voi_path = "eq_ven.csv"
		self.dims_df_path = "img_dims.csv"
		self.aug_dir = "aug_imgs\\"
		self.orig_dir = "orig_imgs\\"