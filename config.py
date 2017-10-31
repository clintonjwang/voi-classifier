class Config:
	def __init__(self):
		self.dims = [24,24,12]
		self.nb_channels = 1
		self.nb_classes = 5
		self.aug_factor = {"cyst": 100, "hcc": 100, "hemangioma": 100, "fnh": 100, "adenoma": 100}

		self.model_path = "model.hdf5"
		self.aug_dir = "aug_imgs\\"
		self.orig_dir = "orig_imgs\\"