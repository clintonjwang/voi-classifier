"""
Contains methods for converting dcm files to npy files cropped at the regions surrounding
Converts a dcm file to a numpy array.
Accepts either a single nifti file or a folder of niftis as the input argument.

Assumes that all the DICOM images for a study are stored in subfolders...
The study subfolder should be stored in a subfolder named after the class to which it belongs.
If a single study has multiple classes, it needs to be copied to each class's subfolder.

base_directory/cyst/E123456789/ax_haste/
> 0.dcm
> 1.dcm
> ...
> metadata.xml

Usage:
	python dr_methods.py
	python dr_methods.py --cls hcc
	python dr_methods.py -v -c cyst
	python dr_methods.py -ovc hemangioma

Author: Clinton Wang, E-mail: `clintonjwang@gmail.com`, Github: `https://github.com/clintonjwang/voi-classifier`
"""

import argparse
import config
import datetime
import niftiutils.helper_fxns as hf
import numpy as np
import os
import pandas as pd
import random
import time

def autofill_cls_arg(func):
	"""Decorator that autofills the first argument with the classes
	specified by C.classes_to_include if it is not included."""

	def wrapper(*args, **kwargs):
		if (len(args) == 0 or args[0] is None) and ('cls' not in kwargs or kwargs['cls'] is None):
			C = config.Config()
			kwargs.pop('cls', None)
			for cls in C.classes_to_include:
				result = func(cls, *args[1:], **kwargs)
		else:
			result = func(*args, **kwargs)
		return result

	return wrapper

###########################
### QC methods
###########################

def plot_check(num, lesion_id=None, cls=None, normalize=[-1,0]):
	"""Plot the unscaled, cropped or augmented versions of a lesion.
	Lesion selected at random from cls if lesion_id is None.
	Either lesion_id or cls must be specified.
	If accession number is put instead of lesion_id, picks the first lesion."""

	C = config.Config()

	if lesion_id is None:
		fn = random.choice(os.listdir(os.path.join(C.crops_dir, cls)))
		lesion_id = fn[:fn.find('.')]
		print(lesion_id)
	elif cls is None:
		small_voi_df = pd.read_csv(C.small_voi_path)
		try:
			cls = small_voi_df.loc[small_voi_df["id"] == lesion_id, "cls"].values[0]
		except:
			lesion_id += "_0"
			cls = small_voi_df.loc[small_voi_df["id"] == lesion_id, "cls"].values[0]
		
	if num == 0:
		img = np.load(os.path.join(C.full_img_dir, cls, lesion_id[:lesion_id.find('_')] + ".npy"))
	elif num == 1:
		img = np.load(os.path.join(C.crops_dir, cls, lesion_id + ".npy"))
	elif num == 2:
		img = np.load(os.path.join(C.orig_dir, cls, lesion_id + ".npy"))
	elif num == 3:
		img = np.load(os.path.join(C.aug_dir, cls, lesion_id + "_" + str(random.randint(0,C.aug_factor-1)) + ".npy"))
	else:
		raise ValueError(num + " should be 0 (uncropped), 1 (gross cropping), 2 (unaugmented) or 3 (augmented)")
	hf.plot_section_auto(img, normalize=normalize)

	return img

@autofill_cls_arg
def check_dims_df(cls=None):
	"""Checks to see if dims_df is missing any accession numbers."""

	C = config.Config()
	i = C.cls_names.index(cls)
	sheetname = C.sheetnames[i]
	df = pd.read_excel(C.xls_name, sheetname)
	df = _filter_voi_df(df, C)
	acc_nums = set(df['Patient E Number'].astype(str).tolist())
	dims_df = pd.read_csv(C.dims_df_path)
	missing = acc_nums.difference(dims_df["AccNum"])
	if len(missing) > 0:
		print(cls, missing)

@autofill_cls_arg
def report_missing_folders(cls=None):
	"""Checks to see if any image phases are missing from the DICOM directories"""

	C = config.Config()
	i = C.cls_names.index(cls)

	df = pd.read_excel(C.xls_name, C.sheetnames[i])
	df = _filter_voi_df(df, C)
	acc_nums = list(set(df['Patient E Number'].tolist()))

	for cnt, acc_num in enumerate(acc_nums):
		df_subset = df.loc[df['Patient E Number'].astype(str) == acc_num]
		subfolder = C.img_dirs[i] + "\\" + acc_num

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
def dcm2npy_batch(cls=None, acc_nums=None, update_intensities=False, overwrite=True, verbose=False):
	"""Converts dcms to full-size npy, update dims_df and (optionally) update intensity_df."""

	C = config.Config()

	try:
		dims_df = pd.read_csv(C.dims_df_path)
	except FileNotFoundError:
		dims_df = pd.DataFrame(columns = ["AccNum", "x", "y", "z"])

	i = C.cls_names.index(cls)

	src_data_df = pd.read_excel(C.xls_name, C.sheetnames[i])
	src_data_df = _filter_voi_df(src_data_df, C)

	if not os.path.exists(os.path.join(C.full_img_dir, cls)):
		os.makedirs(os.path.join(C.full_img_dir, cls))

	if acc_nums is None:
		acc_nums = list(set(src_data_df['Patient E Number'].values))
	else:
		acc_nums = set(acc_nums).intersection(src_data_df['Patient E Number'].values)

	for cnt, acc_num in enumerate(acc_nums):
		dims_df = _dcm2npy(load_dir=os.path.join(C.img_dirs[i], acc_num),
			save_path=os.path.join(C.full_img_dir, cls, str(acc_num) + ".npy"), dims_df=dims_df,
			info=src_data_df.loc[src_data_df['Patient E Number'].astype(str) == acc_num],
			overwrite=overwrite, verbose=verbose)
		
		if update_intensities:
			_get_intensities(cls=cls, acc_num=acc_num)

		if verbose:
			print("%d out of %d accession numbers loaded" % (cnt+1, len(acc_nums)))
		elif cnt % 5 == 2:
			print(".", end="")
	
	dims_df.to_csv(C.dims_df_path, index=False)

@autofill_cls_arg
def load_vois_batch(cls=None, acc_nums=None, overwrite=True, verbose=False):
	"""Updates the voi_dfs based on the raw spreadsheet.
	dcm2npy_batch() must be run first to produce full size npy images."""

	def write_voi_dfs(*args):
		C = config.Config()

		if len(args) == 1:
			voi_df_art, voi_df_ven, voi_df_eq = args[0]
		else:
			voi_df_art, voi_df_ven, voi_df_eq = args

		voi_df_art.to_csv(C.art_voi_path)
		voi_df_ven.to_csv(C.ven_voi_path)
		voi_df_eq.to_csv(C.eq_voi_path)

	C = config.Config()

	dims_df = pd.read_csv(C.dims_df_path)

	i = C.cls_names.index(cls)
	src_data_df = pd.read_excel(C.xls_name, C.sheetnames[i])
	src_data_df = _filter_voi_df(src_data_df, C)
	if acc_nums is None:
		acc_nums = list(set(src_data_df['Patient E Number'].values))
	else:
		acc_nums = set(acc_nums).intersection(src_data_df['Patient E Number'].values)
	
	voi_df_art, voi_df_ven, voi_df_eq = get_voi_dfs()

	if overwrite:
		voi_df_art, voi_df_ven, voi_df_eq = _remove_accnums_from_vois(voi_df_art, voi_df_ven, voi_df_eq, acc_nums, cls)
	else:
		acc_nums = set(acc_nums).difference(voi_df_art[voi_df_art["cls"] == cls]["acc_num"].values)

	voi_dfs = voi_df_art, voi_df_ven, voi_df_eq
	for cnt, acc_num in enumerate(acc_nums):
		voi_dfs = _load_vois(cls, acc_num, voi_dfs)

		if cnt % 10 == 2:
			print(".", end="")
			write_voi_dfs(voi_dfs)

	write_voi_dfs(voi_dfs)

@autofill_cls_arg
def load_patient_info(cls=None, acc_nums=None, overwrite=False, verbose=False):
	"""Loads patient demographic info from metadata files downloaded alongside the dcms."""

	def get_patient_info(metadata_txt):
		result = {}
		mrn_tag = '<DicomAttribute tag="00100020" vr="LO" keyword="PatientID">'
		birthdate_tag = '<DicomAttribute tag="00100030" vr="DA" keyword="PatientsBirthDate">'
		curdate_tag = 'DicomAttribute tag="00080021" vr="DA" keyword="SeriesDate">'
		sex_tag = '<DicomAttribute tag="00100040" vr="CS" keyword="PatientsSex">'
		ethnic_tag = '<DicomAttribute tag="00102160" vr="SH" keyword="EthnicGroup">'
		search_terms = [mrn_tag, birthdate_tag, curdate_tag, sex_tag, ethnic_tag]

		for search_term in search_terms:
			result[search_term] = hf.get_dcm_header_value(metadata_txt, search_term)

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
		ethnicity = result[ethnic_tag]

		if ethnicity.upper() == 'W':
			ethnicity = "White"
		elif ethnicity.upper() == 'B':
			ethnicity = "Black"
		elif ethnicity.upper() == 'H':
			ethnicity = "Hisp"
		elif ethnicity.upper() == 'O':
			ethnicity = "Other"
		elif ethnicity in ['U', 'P', "Pt Refused"] or len(ethnicity) > 12:
			ethnicity = "Unknown"

		return [mrn, sex, acc_num, age, ethnicity, cls]

	C = config.Config()

	i = C.cls_names.index(cls)
	df = pd.read_excel(C.xls_name, C.sheetnames[i])
	df = _filter_voi_df(df, C)

	if acc_nums is None:
		acc_nums = set(df['Patient E Number'].astype(str).values)

	try:
		patient_info_df = pd.read_csv(C.patient_info_path)
	except FileNotFoundError:
		patient_info_df = pd.DataFrame(columns = ["MRN", "Sex", "AccNum", "AgeAtImaging", "Ethnicity", "cls"])

	if not overwrite:
		acc_nums = acc_nums.difference(patient_info_df[patient_info_df["cls"] == cls]["AccNum"].values)

	i = len(patient_info_df)
	print(cls)
	for cnt, acc_num in enumerate(acc_nums):
		df_subset = df.loc[df['Patient E Number'].astype(str) == acc_num]
		subdir = os.path.join(C.img_dirs[i], acc_num)
		fn = subdir+"\\T1_AP\\metadata.xml"

		try:
			f = open(fn, 'r')
		except FileNotFoundError:
			missing_metadata = True
			foldernames = [x for x in os.listdir(subdir) if 'T1' in x or 'post' in x or 'post' in x]
			for folder in foldernames:
				fn = subdir + "\\" + folder + "\\metadata.xml"
				if os.path.exists(fn):
					f = open(fn, 'r')
					missing_metadata = False
					break
			if missing_metadata:
				print(acc_num, end=",")
				continue

		patient_info_df.loc[cnt+i] = get_patient_info(''.join(f.readlines()))

		if cnt % 20 == 2:
			patient_info_df.to_csv(C.patient_info_path, index=False)

	patient_info_df.to_csv(C.patient_info_path, index=False)

###########################
### Public Subroutines
###########################

def get_voi_dfs():
	C = config.Config()

	try:
		voi_df_art = pd.read_csv(C.art_voi_path, index_col=0)
		voi_df_ven = pd.read_csv(C.ven_voi_path, index_col=0)
		voi_df_eq = pd.read_csv(C.eq_voi_path, index_col=0)
		voi_df_art["acc_num"] = voi_df_art["acc_num"].astype(str)
	except FileNotFoundError:
		voi_df_art = pd.DataFrame(columns = ["acc_num", "x1", "x2", "y1", "y2", "z1", "z2", "cls",
										 "real_dx", "real_dy", "real_dz", "run_num"])
		voi_df_ven = pd.DataFrame(columns = ["x1", "x2", "y1", "y2", "z1", "z2"]) #voi_df_ven only contains entries where manually specified
		voi_df_eq = pd.DataFrame(columns = ["x1", "x2", "y1", "y2", "z1", "z2"]) #voi_df_ven only contains entries where manually specified

	return voi_df_art, voi_df_ven, voi_df_eq

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
	ids_to_delete = list(voi_df_art[(voi_df_art["acc_num"].isin(acc_nums)) & (voi_df_art["cls"] == cls)].index)
	voi_df_ven = voi_df_ven[~voi_df_ven.index.isin(ids_to_delete)]
	voi_df_eq = voi_df_eq[~voi_df_eq.index.isin(ids_to_delete)]
	voi_df_art = voi_df_art[~voi_df_art.index.isin(ids_to_delete)]

	return voi_df_art, voi_df_ven, voi_df_eq

def _filter_voi_df(df, filters):
	"""Select only rows for this run. Collect acc_nums and voi coordinates."""
	
	df = df[df['Run'] <= filters.run_num].dropna(subset=["x1"])
	df['Patient E Number'] = df['Patient E Number'].astype(str)
	
	return df.drop(set(df.columns).difference(['Patient E Number', 
		  'x1', 'x2', 'y1', 'y2', 'z1', 'z2', 'Flipped',
		  'x3', 'x4', 'y3', 'y4', 'z3', 'z4',
		  'x5', 'x6', 'y5', 'y6', 'z5', 'z6', 'Run']), axis=1)

def _dcm2npy(load_dir, save_path, dims_df, info=None, flip_x=True, overwrite=True, verbose=False):
	"""Assumes save_path's folder has already been created."""

	C = config.Config()
	acc_num = info.iloc[0]["Patient E Number"]

	if os.path.exists(save_path) and not overwrite:
		if verbose:
			print(acc_num, "has already been saved. Skipping.")
		return dims_df

	if len(info) == 0:
		print(acc_num, "not properly marked in the spreadsheet")
		return dims_df

	try:
		art, cur_dims = hf.dcm_load(os.path.join(load_dir, "T1_AP"), flip_x=flip_x)
	except ValueError:
		raise ValueError(load_dir + " cannot be loaded")
	
	# register phases if venous was not specified separately
	ven, _ = hf.dcm_load(os.path.join(load_dir, "T1_VP"), flip_x=flip_x)
	if not np.isnan(info['x3']):
		ven, _ = hf.reg_elastix(moving=ven, fixed=art)

	eq, _ = hf.dcm_load(os.path.join(load_dir, "T1_EQ"), flip_x=flip_x)
	if not np.isnan(row['x5']):
		eq, _ = hf.reg_elastix(moving=eq, fixed=art)

	img = np.transpose(np.stack((art, ven, eq)), (1,2,3,0))
	
	np.save(save_path, img)

	dims_df = _add_to_dims_df(dims_df, acc_num, cur_dims)

	return dims_df

def _load_vois(cls, acc_num, voi_dfs=None):
	"""Load all vois belonging to an acc_num. Does not overwrite entries."""

	def _add_voi_row(voi_df, x, y, z, acc_num=None, cls=None, run_num=-1, vox_dims=None, index=None):
		"""Append voi info to the dataframe voi_df.
		If an index is passed, will overwrite any existing entry for that index.
		Otherwise, will create a new row."""
		
		if index is None:
			lesion_ids = voi_df[voi_df["acc_num"] == acc_num].index
			if len(lesion_ids) > 0:
			    index = acc_num + "_" + str(max([int(lesion_id[lesion_id.find('_')+1:]) for lesion_id in lesion_ids]) + 1)
			else:
			    index = acc_num + "_0"

			real_dx = (x[1] - x[0])*vox_dims[0]
			real_dy = (y[1] - y[0])*vox_dims[1]
			real_dz = (z[1] - z[0])*vox_dims[2]

			voi_df.loc[index] = [str(acc_num), x[0], x[1], y[0], y[1], z[0], z[1], cls, real_dx, real_dy, real_dz, run_num]
			return voi_df, index
			
		else:
			voi_df.loc[index] = [x[0], x[1], y[0], y[1], z[0], z[1]]
			return voi_df

	C = config.Config()

	dims_df = pd.read_csv(C.dims_df_path)

	if voi_dfs is None:
		voi_df_art, voi_df_ven, voi_df_eq = get_voi_dfs()
	else:
		voi_df_art, voi_df_ven, voi_df_eq = voi_dfs

	src_data_df = pd.read_excel(C.xls_name, C.sheetnames[C.cls_names.index(cls)])
	src_data_df = _filter_voi_df(src_data_df, C)

	df_subset = src_data_df.loc[src_data_df['Patient E Number'].astype(str) == acc_num]
	img = np.load(os.path.join(C.full_img_dir, cls, str(acc_num) + ".npy"))

	for _, row in df_subset.iterrows():
		x = (int(row['x1']), int(row['x2']))
		y = (int(row['y1']), int(row['y2']))
		z = (int(row['z1']), int(row['z2']))
		
		try:
			cur_dims = dims_df[dims_df["AccNum"] == acc_num].iloc[0].values[1:]
		except:
			raise ValueError("dims_df not yet loaded for", acc_num)
		
		y = (img.shape[1]-y[1], img.shape[1]-y[0]) # flip y
		if row['Flipped'] != "Yes":
			z = (img.shape[2]-z[1], img.shape[2]-z[0]) # flip z
		
		voi_df_art, lesion_id = _add_voi_row(voi_df_art, x,y,z, vox_dims=cur_dims,
						acc_num=acc_num, cls=cls, run_num=int(row["Run"]))

		if not np.isnan(row['x3']):
			x = (int(row['x3']), int(row['x4']))
			y = (int(row['y3']), int(row['y4']))
			z = (int(row['z3']), int(row['z4']))
			
			y = (img.shape[1]-y[1], img.shape[1]-y[0]) # flip y
			if row['Flipped'] != "Yes":
				z = (img.shape[2]-z[1], img.shape[2]-z[0]) # flip z
				
			voi_df_ven = _add_voi_row(voi_df_ven, x,y,z, index=lesion_id)
			
		if not np.isnan(row['x5']):
			x = (int(row['x5']), int(row['x6']))
			y = (int(row['y5']), int(row['y6']))
			z = (int(row['z5']), int(row['z6']))
			
			y = (img.shape[1]-y[1], img.shape[1]-y[0]) # flip y
			if row['Flipped'] != "Yes":
				z = (img.shape[2]-z[1], img.shape[2]-z[0]) # flip z
				
			voi_df_eq = _add_voi_row(voi_df_eq, x,y,z, index=lesion_id)

	return voi_df_art, voi_df_ven, voi_df_eq

if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='Convert DICOMs to npy files and transfer voi coordinates from excel to csv.')
	parser.add_argument('-c', '--cls', help='limit to a specific class')
	parser.add_argument('-v', '--verbose', action='store_true', help='verbosity')
	parser.add_argument('-o', '--overwrite', action='store_true', help='overwrite')
	args = parser.parse_args()

	s = time.time()
	dcm2npy_batch(cls=args.cls, verbose=args.verbose, overwrite=args.overwrite)
	print("Time to convert dcm to npy: %s" % str(time.time() - s))

	s = time.time()
	load_vois_batch(cls=args.cls, verbose=args.verbose, overwrite=args.overwrite)
	print("Time to load voi coordinates: %s" % str(time.time() - s))

	print("Finished!")