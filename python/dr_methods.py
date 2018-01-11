import config
import datetime
import helper_fxns as hf
import numpy as np
import os
import pandas as pd
import random
import time

###########################
### BULK METHODS ACROSS ALL CLASSES
###########################

def plot_check(cls, num, C, accnum=None):
	"""Plot the unscaled, cropped or augmented versions of a lesion.
	Lesion selected at random from cls if accnum is None."""

	if accnum==None:
		fn = random.choice(os.listdir(C.crops_dir + cls))
		accnum = fn[:fn.find('.')]
		print(accnum)
		
	if num==1:
		img = np.load(C.crops_dir + cls + "\\" + accnum + ".npy")
	elif num==2:
		img = np.load(C.orig_dir + cls + "\\" + accnum + ".npy")
	else:
		img = np.load(C.aug_dir + cls + "\\" + accnum + "_" + str(random.randint(0,C.aug_factor-1)) + ".npy")
	hf.plot_section_auto(img, normalize=True)

	return img

###########################
### METHODS FOR EXTRACTING VOIS FROM THE SPREADSHEET
###########################

def load_vois_all(C=None, classes=None):
	"""Load all the vois for all classes and save them into the csvs specified by config.
	If classes is specified, can limit the classes to do this for"""

	if C is None:
		C = config.Config()
		
	base_dir = "Z:"

	for cls in C.classes_to_include:
		if not os.path.exists(C.full_img_dir + "\\" + cls):
			os.makedirs(C.full_img_dir + "\\" + cls)

	dims_df = pd.read_csv(C.dims_df_path)
	voi_df_art = pd.DataFrame(columns = ["Filename", "x1", "x2", "y1", "y2", "z1", "z2", "cls",
									 "flipz", "real_dx", "real_dy", "real_dz", "id", "lesion_num"])
	voi_df_ven = pd.DataFrame(columns = ["id", "x1", "x2", "y1", "y2", "z1", "z2"]) #voi_df_ven only contains entries where manually specified
	voi_df_eq = pd.DataFrame(columns = ["id", "x1", "x2", "y1", "y2", "z1", "z2"]) #voi_df_ven only contains entries where manually specified
	voi_dfs = [voi_df_art, voi_df_ven, voi_df_eq]

	for i in range(7):
		if classes is not None and cls_names[i] not in classes:
			continue
		voi_dfs = load_vois_batch(C.cls_names[i], voi_dfs, dims_df)

def load_vois_batch(cls, voi_dfs=None, dims_df=None, C=None, verbose=False, acc_nums=None, overwrite=True):
	"""Load all vois belonging to a class based on the contents of the spreadsheet."""
	
	s = time.time()
	if C is None:
		C = config.Config()
	if dims_df is None:
		dims_df = pd.read_csv(C.dims_df_path)

	sheetname = C.sheetnames[C.cls_names.index(cls)]

	if voi_dfs is None:
		voi_df_art = pd.read_csv(C.art_voi_path)
		voi_df_ven = pd.read_csv(C.ven_voi_path)
		voi_df_eq = pd.read_csv(C.eq_voi_path)
	else:
		voi_df_art, voi_df_ven, voi_df_eq = voi_dfs

	src_data_df = pd.read_excel(C.xls_name, sheetname)
	src_data_df = preprocess_df(src_data_df, C)
	
	if acc_nums is None:
		print("\nLoading VOIs from sheet", sheetname)
		acc_nums = list(set(src_data_df['Patient E Number'].dropna().astype(str).tolist()))
	else:
		print("\nLoading VOIs for", acc_nums)
		if overwrite:
			voi_df_art, voi_df_ven, voi_df_eq = remove_vois(voi_df_art, voi_df_ven, voi_df_eq, acc_nums, cls)

	for cnt, acc_num in enumerate(acc_nums):
		voi_df_art, voi_df_ven, voi_df_eq = load_vois(cls, acc_num, src_data_df, dims_df, voi_df_art, voi_df_ven, voi_df_eq, C)

		if verbose:
			print(acc_num, "%d out of %d acc_nums loaded" % (cnt+1, len(acc_nums)))
		else:
			print(".", end="")
			
	print("Overall time: %s" % str(time.time() - s))

	voi_df_art.to_csv(C.art_voi_path, index=False)
	voi_df_ven.to_csv(C.ven_voi_path, index=False)
	voi_df_eq.to_csv(C.eq_voi_path, index=False)

	return voi_df_art, voi_df_ven, voi_df_eq

def load_vois(cls, acc_num, df=None, dims_df=None, voi_df_art=None, voi_df_ven=None, voi_df_eq=None, C=None):
	"""Load all vois belonging to an acc_num.
	"""
	if C is None:
		C = config.Config()

	if voi_df_art is None:
		voi_df_art = pd.read_csv(C.art_voi_path)
		voi_df_ven = pd.read_csv(C.ven_voi_path)
		voi_df_eq = pd.read_csv(C.eq_voi_path)
	if dims_df is None:
		dims_df = pd.read_csv(C.dims_df_path)

	if df is None:
		index = C.cls_names.index(cls)
		df = pd.read_excel(C.xls_name, C.sheetnames[index])
		df = preprocess_df(df, C)

	df_subset = df.loc[df['Patient E Number'].astype(str) == acc_num]
	img = np.load(C.full_img_dir + "\\" + cls + "\\" + str(acc_num) + ".npy")

	for _, row in df_subset.iterrows():
		x = (int(row['x1']), int(row['x2']))
		y = (int(row['y1']), int(row['y2']))
		z = (int(row['z1']), int(row['z2']))
		
		try:
			cur_dims = dims_df[dims_df["AccNum"] == acc_num].iloc[0].values[1:]
		except NameError:
			raise ValueError("dims_df not yet loaded for", acc_num)
		
		y = (img.shape[1]-y[1], img.shape[1]-y[0]) # flip y
		if row['Flipped'] != "Yes":
			z = (img.shape[2]-z[1], img.shape[2]-z[0]) # flip z
		if row['Flipped'] == "Xflip":
			x = (img.shape[0]-x[1], img.shape[0]-x[0]) # flip y
		
		voi_df_art, art_id = _add_voi_row(voi_df_art, acc_num, x,y,z, vox_dims=cur_dims,
									 cls=cls, flipz=(row['Flipped'] == "Yes"), return_id = True)

		if "Image type2" in row.keys() and row['Image type2'] == "VP-T1":
			x = (int(row['x3']), int(row['x4']))
			y = (int(row['y3']), int(row['y4']))
			z = (int(row['z3']), int(row['z4']))
			
			y = (img.shape[1]-y[1], img.shape[1]-y[0]) # flip y
			if row['Flipped'] != "Yes":
				z = (img.shape[2]-z[1], img.shape[2]-z[0]) # flip z
				
			voi_df_ven = _add_voi_row(voi_df_ven, art_id, x,y,z)
			
		if "Image type3" in row.keys() and row['Image type3'] in ["EQ-T1", "DP-T1"]:
			x = (int(row['x5']), int(row['x6']))
			y = (int(row['y5']), int(row['y6']))
			z = (int(row['z5']), int(row['z6']))
			
			y = (img.shape[1]-y[1], img.shape[1]-y[0]) # flip y
			if row['Flipped'] != "Yes":
				z = (img.shape[2]-z[1], img.shape[2]-z[0]) # flip z
				
			voi_df_eq = _add_voi_row(voi_df_eq, art_id, x,y,z)

	return voi_df_art, voi_df_ven, voi_df_eq

def _add_voi_row(voi_df, acc_num, x, y, z, vox_dims=None, cls=None, flipz=None, return_id=False):
	"""Append voi info to the dataframe voi_df. Overwrite any previous entries."""
	
	if return_id:
		try:
			lesion_num = max(voi_df[voi_df["Filename"] == str(acc_num) + ".npy"]["lesion_num"]) + 1
		except ValueError:
			lesion_num = 0
			
		row_id = str(acc_num)+'_'+str(lesion_num)
	else:
		row_id = acc_num
	
	voi_df = voi_df[voi_df["id"] != row_id]
	
	if len(voi_df) == 0:
		i = 0
	else:
		i = voi_df.index[-1]+1
		
	if return_id:
		dx = (x[1] - x[0])*vox_dims[0]
		dy = (y[1] - y[0])*vox_dims[1]
		dz = (z[1] - z[0])*vox_dims[2]
		
		voi_df.loc[i] = [str(acc_num) + ".npy", x[0], x[1], y[0], y[1], z[0], z[1], cls, flipz, dx, dy, dz, row_id, lesion_num]
		return voi_df, row_id
		
	else:
		voi_df.loc[i] = [row_id, x[0], x[1], y[0], y[1], z[0], z[1]]
		return voi_df

###########################
### METHODS FOR LOADING DICOMS
###########################

def _load_imgs(img_dir, cls, sheetname, dims_df, C=None, verbose=False, acc_nums=None):
	"""Load images stored in folder cls and excel spreadsheet specified by C with name sheetname.
	Saves images to C.full_img_dir and saves vois to the global vois variable.
	Scales images and VOIs so that each voxel is 1.5 x 1.5 x 4 cm
	"""
	
	if C is None:
		C = config.Config()
	s = time.time()
	df = pd.read_excel(C.xls_name, sheetname)
	df = preprocess_df(df, C)

	if acc_nums is None:
		print("\nLoading DCMs of type", sheetname)
		acc_nums = list(set(df['Patient E Number'].dropna().astype(str).tolist()))
		canskip = True
	else:
		print("\nLoading DCM for", acc_nums)
		canskip = False

	for cnt, acc_num in enumerate(acc_nums):
		if canskip and os.path.exists(C.full_img_dir + "\\" + cls + "\\" + str(acc_num) + ".npy"):
			print(acc_num, "has already been saved. Skipping.")
			continue
			
		df_subset = df.loc[df['Patient E Number'].astype(str) == acc_num]
		if len(df_subset) == 0:
			print(acc_num, "not properly marked in the spreadsheet", C.xls_name, "/", sheetname)
			continue

		subdir = img_dir+"\\"+acc_num
		art, cur_dims = hf.dcm_load(subdir+"\\T1_AP", flip_x=True)
		ven, _ = hf.dcm_load(subdir+"\\T1_VP", flip_x=True)

		# register phases if venous was not specified separately
		if "Image type2" not in df_subset.columns or df_subset.iloc[0]["Image type2"] != "VP-T1":
			ven, _ = hf.reg_imgs(moving=ven, fixed=art, params=C.reg_params, rescale_only=False)
			
		dims_df = add_to_dims_df(dims_df, acc_num, cur_dims)

		if C.nb_channels == 3:
			eq, _ = hf.dcm_load(subdir+"\\T1_EQ", flip_x=True)

			if "Image type3" not in df_subset.columns or df_subset.iloc[0]["Image type3"] != "EQ-T1":
				eq, _ = hf.reg_imgs(moving=eq, fixed=art, params=C.reg_params, rescale_only=False)
			img = np.transpose(np.stack((art, ven, eq)), (1,2,3,0))
		else:
			img = np.transpose(np.stack((art, ven)), (1,2,3,0))
			
		np.save(C.full_img_dir + "\\" + cls + "\\" + str(acc_num), img)

		if verbose:
			print(acc_num, "%d out of %d acc_nums loaded" % (cnt+1, len(acc_nums)))
		else:
			print(".", end="")
			
	print("Overall time: %s" % str(time.time() - s))
	
	return dims_df

def add_to_dims_df(dims_df, acc_num, cur_dims):
	"""Append scale info to the dataframe dims_df. Overwrite any previous entries."""
	
	dims_df = dims_df[dims_df["AccNum"] != acc_num]
	
	if len(dims_df) == 0:
		i = 0
	else:
		i = dims_df.index[-1] + 1
		
	dims_df.loc[i] = [acc_num] + list(cur_dims)
	
	return dims_df

def reload_imgs(acc_nums, cls, C=None, update_intensities=False):
	"""Save partially cropped (unscaled) images and update dims_df and intensity_df."""

	if C is None:
		C = config.Config()
	dims_df = pd.read_csv(C.dims_df_path)
	index = C.cls_names.index(cls)

	if cls=="hcc":
		try:
			dims_df = _load_imgs("Z:\\" + C.img_dirs[index], cls, C.sheetnames[index], dims_df, C, acc_nums=acc_nums)
		except:
			pass
		try:
			dims_df = _load_imgs("Z:\\optn5b", cls, C.sheetnames[index], dims_df, C, acc_nums=acc_nums)
		except:
			pass
	else:
		dims_df = _load_imgs("Z:\\" + C.img_dirs[index], cls, C.sheetnames[index], dims_df, C, acc_nums=acc_nums)
	
	if update_intensities:
		for acc_num in acc_nums:
			intensity_df = _get_intensities(acc_num=acc_num, cls=cls)
			intensity_df.to_csv(C.int_df_path, index=False)

	dims_df.to_csv(C.dims_df_path, index=False)

###########################
### INTENSITY SCALING METHODS
###########################

def _get_intensities(C=None, acc_num=None, cls=None):
	"""Return a dataframe with the normalizing intensities of each image's channels.
	Can be done across all classes, across specific classes or for a specific acc_num."""

	if C is None:
		C = config.Config()
	if acc_num is not None:
		intensity_df = pd.read_csv(C.int_df_path)
		img = np.load(C.full_img_dir + "\\" + cls + "\\" + acc_num + ".npy")
		intensity_df = add_intensity_df(intensity_df, img, acc_num)

	elif cls is not None:
		intensity_df = pd.read_csv(C.int_df_path)
		for fn in os.listdir(C.full_img_dir + "\\" + cls):
			img = np.load(C.full_img_dir + "\\" + cls + "\\" + fn)
			intensity_df = add_intensity_df(intensity_df, img, fn[:-4])

	else:
		intensity_df = pd.DataFrame(columns = ["AccNum", "art_int", "ven_int", "eq_int"])
		for cls in C.classes_to_include:
			for fn in os.listdir(C.full_img_dir + "\\" + cls):
				img = np.load(C.full_img_dir + "\\" + cls + "\\" + fn)
				intensity_df = add_intensity_df(intensity_df, img, fn[:-4])
				
	intensity_df.to_csv(C.int_df_path, index=False)

	return intensity_df

def add_intensity_df(intensity_df, img, acc_num):
	"""Append scale info to the dataframe dims_df. Overwrite any previous entries."""
	
	intensity_df = intensity_df[intensity_df["AccNum"] != acc_num]
	
	if len(intensity_df) == 0:
		i = 0
	else:
		i = intensity_df.index[-1] + 1
		
	intensity_df.loc[i] = [acc_num, _get_scaling_intensity(img[:,:,:,0]),
						   _get_scaling_intensity(img[:,:,:,1]),
						   _get_scaling_intensity(img[:,:,:,2])]
	
	return intensity_df

def _get_scaling_intensity(img):
	"""Return intensity value to normalize img and all its transforms to. img should be 3D with no channels."""

	"""temp_img = img[img.shape[0]//5:img.shape[0]*3//5,
				   img.shape[1]//5:img.shape[1]*3//5,
				   img.shape[2]//5:img.shape[2]*4//5]
	temp_img = temp_img[temp_img > np.mean(temp_img)*2/3]
	hist = np.histogram(temp_img, bins=15)
	a = list(hist[0])
	max_value = max(a)
	max_index = a.index(max_value)
	ret = (hist[1][max_index] + hist[1][max_index+1]) / 2"""
	#temp_img = img[img.shape[0]//5:img.shape[0]//2,
	#               img.shape[1]//5:img.shape[1]//2,
	#               img.shape[2]//5:img.shape[2]*4//5]

	return np.amax(img)

###########################
### PATIENT INFO
###########################

def load_patient_info(img_dir, cls, sheetname, patient_info_df, C, verbose=False, acc_nums=None):
	"""TBD
	"""
	
	s = time.time()
	print("\nLoading DCMs of type", sheetname)
	df = pd.read_excel(C.xls_name, sheetname)
	df = preprocess_df(df, C)

	if acc_nums is None:
		acc_nums = list(set(df['Patient E Number'].dropna().astype(str).tolist()))

	if len(patient_info_df) == 0:
		i = 0
	else:
		i = patient_info_df.index[-1]+1

	for cnt, acc_num in enumerate(acc_nums):
		df_subset = df.loc[df['Patient E Number'].astype(str) == acc_num]
		subdir = img_dir+"\\"+acc_num
		fn = subdir+"\\T1_AP\\metadata.xml"

		try:
			f = open(fn, 'r')
		except:
			skip = True
			foldernames = [x for x in os.listdir(subdir) if 'T1' in x or 'post' in x or 'post' in x]
			for folder in foldernames:
				fn = subdir + "\\" + folder + "\\metadata.xml"
				if os.path.exists(fn):
					f = open(fn, 'r')
					skip = False
					break
			if skip:
				print(acc_num, end=",")
				continue

		txt = ''.join(f.readlines())
		result = {}
		mrn_tag = '<DicomAttribute tag="00100020" vr="LO" keyword="PatientID">'
		birthdate_tag = '<DicomAttribute tag="00100030" vr="DA" keyword="PatientsBirthDate">'
		curdate_tag = 'DicomAttribute tag="00080021" vr="DA" keyword="SeriesDate">'
		sex_tag = '<DicomAttribute tag="00100040" vr="CS" keyword="PatientsSex">'
		search_terms = [mrn_tag, birthdate_tag, curdate_tag, sex_tag]

		for search_term in search_terms:
			result[search_term] = hf.get_dcm_header_value(txt, search_term)

		mrn = result[mrn_tag]
		try:
			imgdate = datetime.datetime.strptime(result[curdate_tag], "%Y%m%d").date()
		except:
			print(acc_num + "\\" + folder, end=",")
		birthdate = datetime.datetime.strptime(result[birthdate_tag], "%Y%m%d").date()

		if imgdate.month > birthdate.month or (imgdate.month > birthdate.month and imgdate.day >= birthdate.day):
			age = imgdate.year - birthdate.year
		else:
			age = imgdate.year - birthdate.year - 1

		sex = result[sex_tag]

		patient_info_df.loc[cnt+i] = [mrn, sex, acc_num, age, cls]

		if verbose:
			print(acc_num, "%d out of %d acc_nums loaded" % (cnt+1, len(acc_nums)))
		#elif cnt % 4 == 0:
		#	print(".", end="")
			
	print("Overall time: %s" % str(time.time() - s))
	return patient_info_df

###########################
### FILE I/O
###########################

def preprocess_df(df, C):
	"""Select only rows for this run. Collect acc_nums and voi coordinates."""
	
	df = df[df['Run'] <= C.run_num].dropna(subset=["x1"])
	
	return df.drop(set(df.columns).difference(['Patient E Number', 
		  'x1', 'x2', 'y1', 'y2', 'z1', 'z2', 'Image type', 'Flipped',
		  'x3', 'x4', 'y3', 'y4', 'z3', 'z4', 'Image type2',
		  'x5', 'x6', 'y5', 'y6', 'z5', 'z6', 'Image type3']), axis=1)

def add_deltas(voi_df):
	"""No longer in use"""
	voi_df = voi_df.astype({"x1": int, "x2": int, "y1": int, "y2": int, "z1": int, "z2": int})
	voi_df['dx'] = voi_df.apply(lambda row: row['x2'] - row['x1'], axis=1)
	voi_df['dy'] = voi_df.apply(lambda row: row['y2'] - row['y1'], axis=1)
	voi_df['dz'] = voi_df.apply(lambda row: row['z2'] - row['z1'], axis=1)
	
	return voi_df

def delete_imgs(acc_nums, cls, C, sheetname=None):
	"""Delete images. No reason to use."""

	if sheetname is not None:
		df = pd.read_excel(C.xls_name, sheetname)
		df = preprocess_df(df, C)
		acc_nums = list(set(df['Patient E Number'].dropna().astype(str).tolist()))
	
	for acc_num in acc_nums:
		fn = C.full_img_dir + "\\" + cls + "\\" + str(acc_num) + ".npy"
		if os.path.exists(fn):
			os.remove(fn)

def check_folders(img_dir, sheetname, C):
	df = pd.read_excel(C.xls_name, sheetname)
	df = preprocess_df(df, C)
	acc_nums = list(set(df['Patient E Number'].dropna().astype(str).tolist()))

	for cnt, acc_num in enumerate(acc_nums):
		df_subset = df.loc[df['Patient E Number'].astype(str) == acc_num]

		if not os.path.exists(img_dir + "\\" +acc_num + "\\T1_AP"):
			print(img_dir + "\\" +acc_num + "\\T1_AP is missing")
		if not os.path.exists(img_dir + "\\" +acc_num + "\\T1_VP"):
			print(img_dir + "\\" +acc_num + "\\T1_VP is missing")
		if not os.path.exists(img_dir + "\\" +acc_num + "\\T1_EQ"):
			print(img_dir + "\\" +acc_num + "\\T1_EQ is missing")

def remove_vois(voi_df_art, voi_df_ven, voi_df_eq, acc_nums, cls):
	"""Remove voi from the voi csvs"""
	for acc_num in acc_nums:
		ids_to_delete = list(voi_df_art[(voi_df_art["Filename"] == acc_num+".npy") & (voi_df_art["cls"] == cls)]["id"].values)
		voi_df_ven = voi_df_ven[~voi_df_ven["id"].isin(ids_to_delete)]
		voi_df_eq = voi_df_eq[~voi_df_eq["id"].isin(ids_to_delete)]
		voi_df_art = voi_df_art[~voi_df_art["id"].isin(ids_to_delete)]

	return voi_df_art, voi_df_ven, voi_df_eq

def _scale_vois(x, y, z, pre_reg_scale, field=None, post_reg_scale=None):
	"""Scale vois. Unused"""
	
	scale = pre_reg_scale
	x = (round(x[0]*scale[0]), round(x[1]*scale[0]))
	y = (round(y[0]*scale[1]), round(y[1]*scale[1]))
	z = (round(z[0]*scale[2]), round(z[1]*scale[2]))
	
	if field is not None:
		xvoi_distortions = field[0][x[0]:x[1]+1, y[0]:y[1]+1, z[0]:z[1]+1]
		yvoi_distortions = field[1][x[0]:x[1]+1, y[0]:y[1]+1, z[0]:z[1]+1]
		zvoi_distortions = field[2][x[0]:x[1]+1, y[0]:y[1]+1, z[0]:z[1]+1]

		x = (x[0] + int(np.amin(xvoi_distortions[0,:,:])), x[1] + int(np.amax(xvoi_distortions[-1,:,:])))
		y = (y[0] + int(np.amin(yvoi_distortions[:,0,:])), y[1] + int(np.amax(yvoi_distortions[:,-1,:])))
		z = (z[0] + int(np.amin(zvoi_distortions[:,:,0])), z[1] + int(np.amax(zvoi_distortions[:,:,-1])))
	
		scale = post_reg_scale
		x = (round(x[0]*scale[0]), round(x[1]*scale[0]))
		y = (round(y[0]*scale[1]), round(y[1]*scale[1]))
		z = (round(z[0]*scale[2]), round(z[1]*scale[2]))
	
	return x, y, z
