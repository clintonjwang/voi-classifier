"""
Contains methods for converting dcm files to npy files cropped at the regions surrounding
Converts a nifti file to a numpy array.
Accepts either a single nifti file or a folder of niftis as the input argument.

Usage:
	python cnn_builder.py

Author: Clinton Wang, E-mail: `clintonjwang@gmail.com`, Github: `https://github.com/clintonjwang/voi-classifier`
"""

import argparse
import config
import datetime
import helper_fxns as hf
import numpy as np
import os
import pandas as pd
import random
import time


def autofill_cls_arg(func):
	"""Decorator that autofills the first argument with the classes
	specified by C.classes_to_include if it is not included."""

	def wrapper(*args, **kwargs):
		if len(args) == 0:
			C = config.Config()
			for cls in C.classes_to_include:
				result = func(cls, *args[1:], **kwargs)
		else:
			result = func(*args, **kwargs)
		return result

	return wrapper

###########################
### QC methods
###########################

def plot_check(num, lesion_num=None, normalize=[-1,0]):
	"""Plot the unscaled, cropped or augmented versions of a lesion.
	Lesion selected at random from cls if lesion_num is None."""

	C = config.Config()

	small_voi_df = pd.read_csv(C.small_voi_path)
	cls = small_voi_df.loc[small_voi_df["id"] == lesion_num, "cls"].values[0]

	if lesion_num==None:
		fn = random.choice(os.listdir(C.crops_dir + cls))
		lesion_num = fn[:fn.find('.')]
		print(lesion_num)
		
	if num==1:
		img = np.load(C.crops_dir + cls + "\\" + lesion_num + ".npy")
	elif num==2:
		img = np.load(C.orig_dir + cls + "\\" + lesion_num + ".npy")
	else:
		img = np.load(C.aug_dir + cls + "\\" + lesion_num + "_" + str(random.randint(0,C.aug_factor-1)) + ".npy")
	hf.plot_section_auto(img, normalize=normalize)

	return img

@autofill_cls_arg
def report_missing_folders(cls=None):
	"""Checks to see if any image phases are missing from the DICOM directories"""

	C = config.Config()
	i = C.cls_names.index(cls)
	sheetname = C.sheetnames[i]
	img_dir = "Z:\\"+C.img_dirs[i]

	df = pd.read_excel(C.xls_name, sheetname)
	df = _filter_voi_df(df, C)
	acc_nums = list(set(df['Patient E Number'].dropna().astype(str).tolist()))

	for cnt, acc_num in enumerate(acc_nums):
		df_subset = df.loc[df['Patient E Number'].astype(str) == acc_num]
		subfolder = img_dir + "\\" + acc_num

		if not os.path.exists(subfolder + "\\T1_AP"):
			print(subfolder + "\\T1_AP is missing")
		if not os.path.exists(subfolder + "\\T1_VP"):
			print(subfolder + "\\T1_VP is missing")
		if not os.path.exists(subfolder + "\\T1_EQ"):
			print(subfolder + "\\T1_EQ is missing")

###########################
### METHODS FOR EXTRACTING VOIS FROM THE SPREADSHEET
###########################

@autofill_cls_arg
def dcm2npy_batch(cls=None, acc_nums=None, update_intensities=False, overwrite=True):
	"""Converts dcms to full-size npy, update dims_df and (optionally) update intensity_df."""

	C = config.Config()

	try:
		dims_df = pd.read_csv(C.dims_df_path)
	except FileNotFoundError:
		dims_df = pd.DataFrame(columns = ["AccNum", "x", "y", "z"])

	i = C.cls_names.index(cls)
	sheetname = C.sheetnames[i]
	img_dir = "Z:\\"+C.img_dirs[i]

	src_data_df = pd.read_excel(C.xls_name, sheetname)
	src_data_df = _filter_voi_df(src_data_df, C)

	if not os.path.exists(os.path.join(C.full_img_dir, cls)):
		os.makedirs(os.path.join(C.full_img_dir, cls))

	if acc_nums is None:
		acc_nums = list(set(src_data_df['Patient E Number'].dropna().astype(str).tolist()))

	for cnt, acc_num in enumerate(acc_nums):
		dims_df = _dcm2npy(load_dir=os.path.join(img_dir, acc_num),
			save_path=os.path.join(C.full_img_dir, cls, str(acc_num) + ".npy"), dims_df=dims_df,
			info=src_data_df.loc[src_data_df['Patient E Number'].astype(str) == acc_num], overwrite=overwrite)
		
		if update_intensities:
			_get_intensities(cls=cls, acc_num=acc_num)
	
	dims_df.to_csv(C.dims_df_path, index=False)

@autofill_cls_arg
def load_vois_batch(cls=None, acc_nums=None, overwrite=True):
	"""Updates the voi_dfs based on the raw spreadsheet.
	dcm2npy_batch() must be run first to produce full size npy images."""

	C = config.Config()

	try:
		voi_df_art = pd.read_csv(C.art_voi_path)
		voi_df_ven = pd.read_csv(C.ven_voi_path)
		voi_df_eq = pd.read_csv(C.eq_voi_path)
	except FileNotFoundError:
		voi_df_art = pd.DataFrame(columns = ["Filename", "x1", "x2", "y1", "y2", "z1", "z2", "cls",
										 "flipz", "real_dx", "real_dy", "real_dz", "id", "lesion_num"])
		voi_df_ven = pd.DataFrame(columns = ["id", "x1", "x2", "y1", "y2", "z1", "z2"]) #voi_df_ven only contains entries where manually specified
		voi_df_eq = pd.DataFrame(columns = ["id", "x1", "x2", "y1", "y2", "z1", "z2"]) #voi_df_ven only contains entries where manually specified

	dims_df = pd.read_csv(C.dims_df_path)
	sheetname = C.sheetnames[C.cls_names.index(cls)]
	src_data_df = pd.read_excel(C.xls_name, sheetname)
	src_data_df = _filter_voi_df(src_data_df, C)
	
	if acc_nums is None:
		acc_nums = list(set(src_data_df['Patient E Number'].dropna().astype(str).tolist()))

	if overwrite:
		voi_df_art, voi_df_ven, voi_df_eq = _remove_accnums_from_vois(voi_df_art, voi_df_ven, voi_df_eq, acc_nums, cls)

	for cnt, acc_num in enumerate(acc_nums):
		voi_df_art, voi_df_ven, voi_df_eq = _load_vois(cls, acc_num, src_data_df, voi_df_art, voi_df_ven, voi_df_eq)

	voi_df_art.to_csv(C.art_voi_path, index=False)
	voi_df_ven.to_csv(C.ven_voi_path, index=False)
	voi_df_eq.to_csv(C.eq_voi_path, index=False)

@autofill_cls_arg
def load_patient_info(cls=None, acc_nums=None, save_path=None, verbose=False):
	"""Loads patient demographic info from metadata files downloaded alongside the dcms."""

	def get_patient_info(netadata_txt):
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
		except ValueError:
			print(acc_num + "\\" + folder, end=",")
		birthdate = datetime.datetime.strptime(result[birthdate_tag], "%Y%m%d").date()

		if imgdate.month > birthdate.month or (imgdate.month > birthdate.month and imgdate.day >= birthdate.day):
			age = imgdate.year - birthdate.year
		else:
			age = imgdate.year - birthdate.year - 1

		sex = result[sex_tag]

		return [mrn, sex, acc_num, age, cls]

	C = config.Config()

	i = C.cls_names.index(cls)
	sheetname = C.sheetnames[i]
	img_dir = "Z:\\"+C.img_dirs[i]

	print("\nLoading DCMs of type", sheetname)
	df = pd.read_excel(C.xls_name, sheetname)
	df = _filter_voi_df(df, C)

	if acc_nums is None:
		acc_nums = list(set(df['Patient E Number'].dropna().astype(str).tolist()))

	patient_info_df = pd.DataFrame(columns = ["MRN", "Sex", "AccNum", "AgeAtImaging", "cls"])

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
		except FileNotFoundError:
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

		patient_info_df.loc[cnt+i] = get_patient_info(''.join(f.readlines()))

		if verbose:
			print(acc_num, "%d out of %d acc_nums loaded" % (cnt+1, len(acc_nums)))
		#elif cnt % 4 == 0:
		#	print(".", end="")
			
	patient_info_df.to_csv(save_path, index=False)

###########################
### Subroutines
###########################

@autofill_cls_arg
def _get_intensities(cls=None, acc_num=None):
	"""Return a dataframe with the normalizing intensities of each image's channels.
	Can be done across all classes, across specific classes or for a specific acc_num."""
	C = config.Config()
	
	try:
		intensity_df = pd.read_csv(C.int_df_path)
	except FileNotFoundError:
		intensity_df = pd.DataFrame(columns = ["AccNum", "art_int", "ven_int", "eq_int"])

	if acc_num is not None:
		img = np.load(C.full_img_dir + "\\" + cls + "\\" + acc_num + ".npy")
		intensity_df = _add_intensity_df(intensity_df, img, acc_num)

	else:
		intensity_df = pd.read_csv(C.int_df_path)
		for fn in os.listdir(C.full_img_dir + "\\" + cls):
			img = np.load(C.full_img_dir + "\\" + cls + "\\" + fn)
			intensity_df = _add_intensity_df(intensity_df, img, fn[:-4])
				
	intensity_df.to_csv(C.int_df_path, index=False)

def _add_to_dims_df(dims_df, acc_num, cur_dims):
	"""Append scale info to the dataframe dims_df. Overwrite any previous entries."""
	
	dims_df = dims_df[dims_df["AccNum"] != acc_num]
	
	if len(dims_df) == 0:
		i = 0
	else:
		i = dims_df.index[-1] + 1
		
	dims_df.loc[i] = [acc_num] + list(cur_dims)
	
	return dims_df

def _add_intensity_df(intensity_df, img, acc_num):
	"""Append scale info to the dataframe dims_df. Overwrite any previous entries."""
	
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

	intensity_df = intensity_df[intensity_df["AccNum"] != acc_num]
	
	if len(intensity_df) == 0:
		i = 0
	else:
		i = intensity_df.index[-1] + 1
		
	intensity_df.loc[i] = [acc_num, _get_scaling_intensity(img[:,:,:,0]),
						   _get_scaling_intensity(img[:,:,:,1]),
						   _get_scaling_intensity(img[:,:,:,2])]
	
	return intensity_df

def _remove_accnums_from_vois(voi_df_art, voi_df_ven, voi_df_eq, acc_nums, cls):
	"""Remove voi from the voi csvs"""
	for acc_num in acc_nums:
		ids_to_delete = list(voi_df_art[(voi_df_art["Filename"] == acc_num+".npy") & (voi_df_art["cls"] == cls)]["id"].values)
		voi_df_ven = voi_df_ven[~voi_df_ven["id"].isin(ids_to_delete)]
		voi_df_eq = voi_df_eq[~voi_df_eq["id"].isin(ids_to_delete)]
		voi_df_art = voi_df_art[~voi_df_art["id"].isin(ids_to_delete)]

	return voi_df_art, voi_df_ven, voi_df_eq

def _filter_voi_df(df, filters):
	"""Select only rows for this run. Collect acc_nums and voi coordinates."""
	
	df = df[df['Run'] <= filters.run_num].dropna(subset=["x1"])
	
	return df.drop(set(df.columns).difference(['Patient E Number', 
		  'x1', 'x2', 'y1', 'y2', 'z1', 'z2', 'Image type', 'Flipped',
		  'x3', 'x4', 'y3', 'y4', 'z3', 'z4', 'Image type2',
		  'x5', 'x6', 'y5', 'y6', 'z5', 'z6', 'Image type3']), axis=1)

def _dcm2npy(load_dir, save_path, dims_df, info=None, flip_x=True, overwrite=True):
	"""Assumes save_path's folder has already been created."""

	acc_num = info.iloc[0]["Patient E Number"]

	if os.path.exists(save_path) and not overwrite:
		print(acc_num, "has already been saved. Skipping.")
		return dims_df

	if len(info) == 0:
		print(acc_num, "not properly marked in the spreadsheet")
		return dims_df

	art, cur_dims = hf.dcm_load(os.path.join(load_dir, "T1_AP"), flip_x=flip_x)
	
	# register phases if venous was not specified separately
	ven, _ = hf.dcm_load(os.path.join(load_dir, "T1_VP"), flip_x=flip_x)
	if "Image type2" not in info.columns or info.iloc[0]["Image type2"] != "VP-T1":
		ven, _ = hf.reg_imgs(moving=ven, fixed=art, params=C.reg_params)

	eq, _ = hf.dcm_load(os.path.join(load_dir, "T1_EQ"), flip_x=flip_x)
	if "Image type3" not in info.columns or info.iloc[0]["Image type3"] != "EQ-T1":
		eq, _ = hf.reg_imgs(moving=eq, fixed=art, params=C.reg_params)

	img = np.transpose(np.stack((art, ven, eq)), (1,2,3,0))
	
	np.save(save_path, img)

	dims_df = _add_to_dims_df(dims_df, acc_num, cur_dims)

	return dims_df

def _load_vois(cls, acc_num, df=None, voi_df_art=None, voi_df_ven=None, voi_df_eq=None):
	"""Load all vois belonging to an acc_num.
	"""
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

	C = config.Config()

	dims_df = pd.read_csv(C.dims_df_path)

	if voi_df_art is None:
		voi_df_art = pd.read_csv(C.art_voi_path)
		voi_df_ven = pd.read_csv(C.ven_voi_path)
		voi_df_eq = pd.read_csv(C.eq_voi_path)

	if df is None:
		index = C.cls_names.index(cls)
		df = pd.read_excel(C.xls_name, C.sheetnames[index])
		df = _filter_voi_df(df, C)

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

if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='Load a nifti file and visualize its cross-section.')
	parser.add_argument('load_path', help='nifti file to load, or folder containing niftis')
	parser.add_argument('--dest', default=".", help='directory to save numpy array(s)')
	args = parser.parse_args()

	s = time.time()
	dcm2npy_batch()
	print("Time to load dcms: %s" % str(time.time() - s))

	s = time.time()
	load_vois_batch()
	print("Time to load cropped lesions: %s" % str(time.time() - s))

	print("Finished!")