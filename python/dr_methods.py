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
import importlib
import niftiutils.helper_fxns as hf
import niftiutils.masks as masks
import niftiutils.registration as reg
import numpy as np
import os
from os.path import *
import pandas as pd
import random
import time
import glob

def autofill_cls_arg(func):
	"""Decorator that autofills the first argument with the classes
	specified by C if it is not included."""

	def wrapper(*args, **kwargs):
		if (len(args) == 0 or args[0] is None) and ('cls' not in kwargs or kwargs['cls'] is None):
			C = config.Config()
			kwargs.pop('cls', None)
			for cls in C.cls_names:
				result = func(cls, *args[1:], **kwargs)
		else:
			result = func(*args, **kwargs)
		return result

	return wrapper

###########################
### QC methods
###########################

@autofill_cls_arg
def check_dims_df(cls=None):
	"""Checks to see if dims_df is missing any accession numbers."""

	C = config.Config()
	df = get_coords_df(cls)
	accnums = set(df['acc #'].astype(str).tolist())
	dims_df = pd.read_csv(C.dims_df_path)
	missing = accnums.difference(dims_df["acc_num"])
	if len(missing) > 0:
		print(cls, missing)

@autofill_cls_arg
def report_missing_folders(cls=None):
	"""Checks to see if any image phases are missing from the DICOM directories"""

	import importlib
	importlib.reload(config)
	C = config.Config()
	df = get_coords_df(cls)
	accnums = list(set(df['acc #'].tolist()))

	for cnt, accnum in enumerate(accnums):
		df_subset = df.loc[df['acc #'].astype(str) == accnum]
		subfolder = C.img_dirs[i] + "\\" + accnum

		#if not exists(subfolder + "\\T1_multiphase"):
		for ph in C.phases:
			if not exists(join(subfolder, ph)):
				print(subfolder, "is missing")
				break

###########################
### METHODS FOR EXTRACTING VOIS FROM THE SPREADSHEET
###########################

def build_coords_df(accnum_xls_path, overwrite=False):
	"""Builds all coords from scratch, without the ability to add or change individual coords"""
	importlib.reload(masks)
	C = config.Config("etiology")
	input_df = pd.read_excel(accnum_xls_path,
				 sheetname="Prelim Analysis Patients", index_col=0, parse_cols="A,J")

	accnum_dict = {category: list(input_df[input_df["Category"] == category].index.astype(str)) for category in C.sheetnames}

	writer = pd.ExcelWriter(C.coord_xls_path)
	dfs = {}
	if exists(C.coord_xls_path):
		for category in C.sheetnames:
			dfs[category] = pd.read_excel(C.coord_xls_path, sheet_name=category, index_col=0)
			dfs[category].to_excel(writer, category)
	del dfs

	for category in C.sheetnames:
		print(category)
		if exists(C.coord_xls_path):
			coords_df = pd.read_excel(C.coord_xls_path, sheet_name=category, index_col=0)
			if not overwrite:
				accnum_dict[category] = list(set(accnum_dict[category]).difference(coords_df['acc #'].values.astype(str)))
		else:
			coords_df = pd.DataFrame(columns=['acc #', 'Run', 'Flipped', 
				  'x1', 'y1', 'z1', 'x2', 'y2', 'z2'])

		#print(accnum_dict[category])
		for ix,accnum in enumerate(accnum_dict[category]):
			load_dir = join(C.dcm_dirs[0], accnum)
			if not exists(join(load_dir, 'Segs', 'tumor_20s_0.ids')):
				masks.off2ids(join(load_dir, 'Segs', 'tumor_20s.off'))

			try:
				art,D = hf.dcm_load(join(load_dir, C.phases[0]))
			except:
				raise ValueError(load_dir)
			#ven,_ = hf.dcm_load(join(load_dir, C.phases[1]))
			#equ,_ = hf.dcm_load(join(load_dir, C.phases[2]))
			#ven,d = reg.reg_elastix(ven, art)
			#equ,d = reg.reg_elastix(equ, art)

			#I = np.stack([art, ven, equ],-1)
			#save_path = join(r"D:\Multiphase-color", accnum)
			#hf.save_tricolor_dcm(save_path, imgs=I)

			for fn in glob.glob(join(load_dir, 'Segs', 'tumor_20s_*.ids')):
				try:
					_,coords = masks.crop_img_to_mask_vicinity([art,D], fn[:-4], return_crops=True)
				except:
					raise ValueError(fn)
				lesion_id = accnum + fn[fn.rfind('_'):-4]
				coords_df.loc[lesion_id] = [accnum, "1", ""] + coords[0] + coords[1]
				#	M = masks.get_mask(fn, D, img.shape)
				#	M = hf.crop_nonzero(M, C)[0]

			if ix % 2 == 1:
				print('.', end='')
				coords_df.to_excel(writer, sheet_name=category)
				writer.save()

		coords_df.to_excel(writer, sheet_name=category)
		writer.save()

@autofill_cls_arg
def dcm2npy_batch(cls=None, accnums=None, overwrite=False, verbose=False):
	"""Converts dcms to full-size npy, update dims_df. Requires coords_df."""

	import importlib
	importlib.reload(config)
	C = config.Config()

	if exists(C.dims_df_path):
		dims_df = pd.read_csv(C.dims_df_path, index_col=0)
	else:
		dims_df = pd.DataFrame(columns = ["x", "y", "z"])

	src_data_df = get_coords_df(cls)

	if not exists(join(C.full_img_dir, cls)):
		os.makedirs(join(C.full_img_dir, cls))

	if accnums is None:
		accnums = list(set(src_data_df['acc #'].values))
	else:
		accnums = set(accnums).intersection(src_data_df['acc #'].values)

	cls_num = C.cls_names.index(cls)

	for cnt, accnum in enumerate(accnums):
		info = src_data_df.loc[src_data_df['acc #'].astype(str) == accnum]

		if len(info) == 0:
			print(accnum, "not properly marked in the spreadsheet.")
			continue

		save_path = join(C.full_img_dir, cls, str(accnum) + ".npy")
		if exists(save_path) and not overwrite:
			continue

		load_dir = join(C.dcm_dirs[cls_num], accnum)
		try:
			art, D = hf.dcm_load(join(load_dir, C.phases[0]))
			ven = hf.dcm_load(join(load_dir, C.phases[1]))[0]
			eq = hf.dcm_load(join(load_dir, C.phases[2]))[0]
		except ValueError:
			raise ValueError(load_dir + " cannot be loaded")

		if 'x3' in info and np.all([np.isnan(x) for x in info['x3'].values]):
			ven, _ = reg.reg_elastix(moving=ven, fixed=art)
		if 'x5' in info and np.all([np.isnan(x) for x in info['x5'].values]):
			eq, _ = reg.reg_elastix(moving=eq, fixed=art)

		try:
			img = np.stack((art, ven, eq), -1)
		except ValueError:
			raise ValueError(accnum + " has a bad header")
		
		np.save(save_path, img)

		if verbose:
			print("%d out of %d accession numbers loaded" % (cnt+1, len(accnums)))
		elif cnt % 5 == 2:
			print(".", end="")

		dims_df.loc[accnum] = list(D)
		dims_df.to_csv(C.dims_df_path)

@autofill_cls_arg
def load_vois_batch(cls=None, accnums=None, overwrite=False):
	"""Updates the voi_dfs based on the raw spreadsheet.
	dcm2npy_batch() must be run first to produce full size npy images."""

	C = config.Config()

	src_data_df = get_coords_df(cls)
	if accnums is None:
		accnums = list(set(src_data_df['acc #'].values))
	else:
		accnums = set(accnums).intersection(src_data_df['acc #'].values)
	
	voi_df_art, voi_df_ven, voi_df_eq = get_voi_dfs()

	if overwrite:
		voi_df_art, voi_df_ven, voi_df_eq = _remove_accnums_from_vois(voi_df_art, voi_df_ven, voi_df_eq, accnums, cls)
	else:
		accnums = set(accnums).difference(voi_df_art[voi_df_art["cls"] == cls]["acc_num"].values)

	voi_dfs = voi_df_art, voi_df_ven, voi_df_eq
	for cnt, accnum in enumerate(accnums):
		voi_dfs = _load_vois(cls, accnum, voi_dfs)

		if cnt % 10 == 2:
			print(".", end="")
			write_voi_dfs(voi_dfs)

	write_voi_dfs(voi_dfs)

def write_voi_dfs(*args):
	C = config.Config()

	if len(args) == 1:
		voi_df_art, voi_df_ven, voi_df_eq = args[0]
	else:
		voi_df_art, voi_df_ven, voi_df_eq = args

	voi_df_art.to_csv(C.art_voi_path)
	voi_df_ven.to_csv(C.ven_voi_path)
	voi_df_eq.to_csv(C.eq_voi_path)
	
@autofill_cls_arg
def load_patient_info(cls=None, accnums=None, overwrite=False, verbose=False):
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
			print(accnum + "\\" + folder, end=",")
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

		return [mrn, sex, accnum, age, ethnicity, cls]

	C = config.Config()
	df = get_coords_df(cls)

	if accnums is None:
		accnums = set(df['acc #'].astype(str).values)

	try:
		patient_info_df = pd.read_csv(C.patient_info_path)
	except FileNotFoundError:
		patient_info_df = pd.DataFrame(columns = ["MRN", "Sex", "acc_num", "AgeAtImaging", "Ethnicity", "cls"])

	if not overwrite:
		accnums = set(accnums).difference(patient_info_df[patient_info_df["cls"] == cls]["acc_num"].values)

	length = len(patient_info_df)
	print(cls)
	for cnt, accnum in enumerate(accnums):
		df_subset = df.loc[df['acc #'].astype(str) == accnum]
		subdir = join(C.img_dirs[i], accnum)
		fn = join(subdir, "T1_AP", "metadata.xml")

		try:
			f = open(fn, 'r')
		except FileNotFoundError:
			missing_metadata = True
			foldernames = [x for x in os.listdir(subdir) if 'T1' in x or 'post' in x or 'post' in x]
			for folder in foldernames:
				fn = join(subdir, folder, "metadata.xml")
				if exists(fn):
					f = open(fn, 'r')
					missing_metadata = False
					break
			if missing_metadata:
				print(accnum, end=",")
				continue

		patient_info_df.loc[cnt+length] = get_patient_info(''.join(f.readlines()))

		if cnt % 20 == 2:
			patient_info_df.to_csv(C.patient_info_path, index=False)

	patient_info_df.to_csv(C.patient_info_path, index=False)

@autofill_cls_arg
def mask2voi(cls=None):
	C = config.Config()

	src_data_df = get_coords_df(cls)
	if accnums is None:
		accnums = list(set(src_data_df['acc #'].values))
	else:
		accnums = set(accnums).intersection(src_data_df['acc #'].values)
	
	voi_df_art, voi_df_ven, voi_df_eq = get_voi_dfs()

	if overwrite:
		voi_df_art, voi_df_ven, voi_df_eq = _remove_accnums_from_vois(voi_df_art, voi_df_ven, voi_df_eq, accnums, cls)
	else:
		accnums = set(accnums).difference(voi_df_art[voi_df_art["cls"] == cls]["acc_num"].values)

	voi_dfs = voi_df_art, voi_df_ven, voi_df_eq
	for cnt, accnum in enumerate(accnums):
		voi_dfs = _load_vois(cls, accnum, voi_dfs)

		if cnt % 10 == 5:
			print(".", end="")
			write_voi_dfs(voi_dfs)

	write_voi_dfs(voi_dfs)

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

def get_coords_df(cls):
	C = config.Config()
	df = pd.read_excel(C.coord_xls_path, C.sheetnames[C.cls_names.index(cls)])
	df = df[df['Run'] <= C.run_num].dropna(subset=["x1"])
	df['acc #'] = df['acc #'].astype(str)
	
	return df.drop(set(df.columns).difference(['acc #', 'Run', 'Flipped', 
		  'x1', 'x2', 'y1', 'y2', 'z1', 'z2',
		  'x3', 'x4', 'y3', 'y4', 'z3', 'z4',
		  'x5', 'x6', 'y5', 'y6', 'z5', 'z6']), axis=1)

def _remove_accnums_from_vois(voi_df_art, voi_df_ven, voi_df_eq, accnums, cls=None):
	"""Remove voi from the voi csvs"""
	if cls is None:
		ids_to_delete = list(voi_df_art[voi_df_art["acc_num"].isin(accnums)].index)
	else:
		ids_to_delete = list(voi_df_art[(voi_df_art["acc_num"].isin(accnums)) & (voi_df_art["cls"] == cls)].index)
	voi_df_ven = voi_df_ven[~voi_df_ven.index.isin(ids_to_delete)]
	voi_df_eq = voi_df_eq[~voi_df_eq.index.isin(ids_to_delete)]
	voi_df_art = voi_df_art[~voi_df_art.index.isin(ids_to_delete)]

	return voi_df_art, voi_df_ven, voi_df_eq


def _load_vois(cls, accnum, voi_dfs=None):
	"""Load all vois belonging to an accnum. Does not overwrite entries."""

	def _add_voi_row(voi_df, x, y, z, accnum=None, cls=None, run_num=-1, vox_dims=None, index=None):
		"""Append voi info to the dataframe voi_df.
		If an index is passed, will overwrite any existing entry for that index.
		Otherwise, will create a new row."""
		
		if index is None:
			lesion_ids = voi_df[voi_df["acc_num"] == accnum].index
			if len(lesion_ids) > 0:
				lesion_nums = [int(lesion_id[lesion_id.find('_')+1:]) for lesion_id in lesion_ids]
				for num in range(len(lesion_nums)+1):
					if num not in lesion_nums:
						new_num = num
				index = accnum + "_" + str(new_num)
			else:
				index = accnum + "_0"

			real_dx = abs(x[1] - x[0])*vox_dims[0]
			real_dy = abs(y[1] - y[0])*vox_dims[1]
			real_dz = abs(z[1] - z[0])*vox_dims[2]

			voi_df.loc[index] = [str(accnum), x[0], x[1], y[0], y[1], z[0], z[1], cls, real_dx, real_dy, real_dz, run_num]
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

	src_data_df = get_coords_df(cls)

	df_subset = src_data_df.loc[src_data_df['acc #'].astype(str) == accnum]
	img = np.load(join(C.full_img_dir, cls, str(accnum) + ".npy"))

	for _, row in df_subset.iterrows():
		x = (int(row['x1']), int(row['x2']))
		y = (int(row['y1']), int(row['y2']))
		z = (int(row['z1']), int(row['z2']))
		
		try:
			cur_dims = dims_df[dims_df["acc_num"] == accnum].iloc[0].values[1:]
		except:
			raise ValueError("dims_df not yet loaded for", accnum)
		
		y = (img.shape[1]-y[1], img.shape[1]-y[0]) # flip y
		if row['Flipped'] != "Yes":
			z = (img.shape[2]-z[1], img.shape[2]-z[0]) # flip z
		
		voi_df_art, lesion_id = _add_voi_row(voi_df_art, x,y,z, vox_dims=cur_dims,
						accnum=accnum, cls=cls, run_num=int(row["Run"]))

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