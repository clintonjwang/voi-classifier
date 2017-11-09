class Config:
	def __init__(self):
		self.dims = [24,24,12]
		self.nb_channels = 2
		self.nb_classes = 3
		self.aug_factor = {"cyst": 50, "hcc": 50, "hemangioma": 50, "fnh": 50, "adenoma": 50, "colorectal": 50}

		self.model_path = "model.hdf5"
		self.full_img_dir = "3cls2phase"
		self.art_voi_path = "voi_art.csv"
		self.ven_voi_path = "voi_ven.csv"
		self.aug_dir = "aug_imgs\\"
		self.orig_dir = "orig_imgs\\"