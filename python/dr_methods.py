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
import datetime
import glob
import importlib
import os
import random
import time
from os.path import *

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import config
import seg_methods as sm
import niftiutils.helper_fxns as hf
import niftiutils.transforms as tr
import niftiutils.masks as masks
import niftiutils.registration as reg
import niftiutils.visualization as vis

importlib.reload(reg)
importlib.reload(masks)
importlib.reload(config)
C = config.Config()

def autofill_cls_arg(func):
	"""Decorator that autofills the first argument with the classes
	specified by C if it is not included."""

	def wrapper(*args, **kwargs):
		if (len(args) == 0 or args[0] is None) and ('cls' not in kwargs or kwargs['cls'] is None):
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
def check_accnum_df(cls=None):
	"""Checks to see if accnum_df is missing any accession numbers."""
	df = get_coords_df(cls)
	accnums = set(df['acc #'].tolist())
	accnum_df = get_accnum_df()
	missing = accnums.difference(accnum_df.index)
	if len(missing) > 0:
		print(cls, missing)

@autofill_cls_arg
def missing_dcms(cls=None):
	"""Checks to see if any image phases are missing from the DICOM directories"""
	df = get_coords_df(cls)
	accnums = list(set(df['acc #'].tolist()))

	for cnt, accnum in enumerate(accnums):
		df_subset = df.loc[df['acc #'].astype(str) == accnum]
		if hasattr(C,'dcm_dirs'):
			subfolder = join(C.dcm_dirs[i], accnum)
		else:
			subfolder = join(C.dcm_dir, accnum)

		#if not exists(subfolder + "\\T1_multiphase"):
		for ph in C.phases:
			if not exists(join(subfolder, ph)):
				print(subfolder, "is missing")
				break

def tricolorize(accnums=None, save_path="D:\\Etiology\\imgs\\tricolor"):
	if not exists(save_path):
		os.makedirs(save_path)

	if accnums is None:
		accnums = [x[:-4] for x in os.listdir(C.full_img_dir) if not x.endswith("seg.npy")]
	for accnum in accnums:
		I = np.load(join(C.full_img_dir, accnum+".npy"))
		hf.save_tricolor_dcm(join(save_path, accnum), imgs=I)

def tumor_cont(accnums=None, save_path="D:\\Etiology\\imgs\\contours"):
	if not exists(save_path):
		os.makedirs(save_path)

	if accnums is None:
		accnums = [x[:-4] for x in os.listdir(C.full_img_dir) if not x.endswith("seg.npy")]
	for accnum in accnums:
		if not exists(join(C.full_img_dir, accnum+"_tumorseg.npy")):
			continue
		I = np.load(join(C.full_img_dir, accnum+".npy"))[...,0]
		M = np.load(join(C.full_img_dir, accnum+"_tumorseg.npy"))
		M = masks.get_largest_mask(M)
		
		try:
			#Z = masks.crop_vicinity(I,M, padding=.1, add_mask_cont=True)
			I,crops = masks.crop_vicinity(I,M, padding=.1, return_crops=True)
			M = hf.crop_nonzero(M, crops)[0]
			sl = I.shape[2]//2
			Z = vis.create_contour_img(I[...,sl], M[...,sl])

			fig = plt.imshow(np.transpose(Z, (1,0,2)))
			fig.axes.get_xaxis().set_visible(False)
			fig.axes.get_yaxis().set_visible(False)
			plt.savefig(join(save_path, accnum+".png"), dpi=150)
		except:
			print(accnum)

###########################
### METHODS FOR EXTRACTING VOIS FROM THE SPREADSHEET
###########################

def off2ids_batch(accnum_xls_path=None, accnum_dict=None):
	# only for segs
	if accnum_dict is None:
		input_df = pd.read_excel(accnum_xls_path, "Prelim Analysis Patients", index_col=0, parse_cols="A,J")
		accnum_dict = {category: list(input_df[input_df["Category"] == category].index.astype(str)) for category in C.sheetnames}

	for category in C.sheetnames:
		print(category)
		if category not in accnum_dict:
			continue
		for ix,accnum in enumerate(accnum_dict[category]):
			print('.',end='')
			load_dir = join(C.dcm_dir, accnum)
			for fn in glob.glob(join(load_dir, 'Segs', 'tumor_20s_*')):
				os.remove(fn)
			masks.off2ids(join(load_dir, 'Segs', 'tumor_20s.off'), R=[2,2,3])
			masks.off2ids(join(load_dir, 'Segs', 'liver.off'), R=[3,3,4], num_foci=1)

def build_coords_df(accnum_xls_path):
	"""Builds all coords from scratch, without the ability to add or change individual coords"""
	input_df = pd.read_excel(accnum_xls_path, "Prelim Analysis Patients", index_col=0, parse_cols="A,J")

	accnum_dict = {category: list(input_df[input_df["Category"] == category].index.astype(str)) for category in C.sheetnames}

	writer = pd.ExcelWriter(C.coord_xls_path)

	for category in C.sheetnames:
		print(category)
		#if exists(C.coord_xls_path):
		#	coords_df = pd.read_excel(C.coord_xls_path, sheet_name=category, index_col=0)
		#	if not overwrite:
		#		accnum_dict[category] = list(set(accnum_dict[category]).difference(coords_df['acc #'].values.astype(str)))
		#else:
		coords_df = pd.DataFrame(columns=['acc #', 'Run', 'Flipped', 
			  'x1', 'y1', 'z1', 'x2', 'y2', 'z2'])

		for ix,accnum in enumerate(accnum_dict[category]):
			"""load_dir = join(C.dcm_dirs[0], accnum)
			if not exists(join(load_dir, 'Segs', 'tumor_20s_0.ids')):
				masks.off2ids(join(load_dir, 'Segs', 'tumor_20s.off'))

			try:
				art,D = hf.nii_load(join(load_dir, "nii_dir", "20s.nii.gz"))
			except:
				raise ValueError(load_dir)
			#ven,_ = hf.dcm_load(join(load_dir, C.phases[1]))
			#equ,_ = hf.dcm_load(join(load_dir, C.phases[2]))

			for fn in glob.glob(join(load_dir, 'Segs', 'tumor_20s_*.ids')):
				try:
					_,coords = masks.crop_img_to_mask_vicinity([art,D], fn[:-4], return_crops=True)
				except:
					raise ValueError(fn)
				lesion_id = accnum + fn[fn.rfind('_'):-4]
				coords_df.loc[lesion_id] = [accnum, "1", ""] + coords[0] + coords[1]
				#	M = masks.get_mask(fn, D, img.shape)
				#	M = hf.crop_nonzero(M, C)[0]"""
			coords_df.loc[accnum+"_0"] = [accnum, "1", ""] + [0]*6

			print('.', end='')
			if ix % 5 == 2:
				coords_df.to_excel(writer, sheet_name=category)
				writer.save()

		coords_df.to_excel(writer, sheet_name=category)
		writer.save()

@autofill_cls_arg
def dcm2npy(cls=None, accnums=None, overwrite=False, exec_reg=False, save_seg=False, downsample=1):
	"""Converts dcms to full-size npy, update accnum_df. Requires coords_df."""
	
	src_data_df = get_coords_df(cls)
	accnum_df = get_accnum_df()
	if accnums is None:
		accnums = list(set(src_data_df['acc #'].values))
	else:
		accnums = set(accnums).intersection(src_data_df['acc #'].values)

	if hasattr(C,'dcm_dirs'):
		root = C.dcm_dirs[C.cls_names.index(cls)]
	else:
		root = C.dcm_dir

	for cnt, accnum in enumerate(accnums):
		load_dir = join(root, accnum)
		save_path = join(C.full_img_dir, accnum + ".npy")

		if not exists(join(load_dir, C.phase_dirs[0])):
			load_dir = join("Z:\\LIRADS\\DICOMs\\hcc", accnum)
			if not exists(join(load_dir, C.phase_dirs[0])):
				continue
		if not overwrite and exists(save_path) and accnum in accnum_df.index and \
				not np.isnan(accnum_df.loc[accnum, "voxdim_x"]):
			continue

		try:
			if exists(join(load_dir, "nii_dir", "20s.nii.gz")):
				art,D = hf.load_img(join(load_dir, "nii_dir", "20s.nii.gz"))
				ven,_ = hf.load_img(join(load_dir, "nii_dir", "70s.nii.gz"))
				eq,_ = hf.load_img(join(load_dir, "nii_dir", "3min.nii.gz"))
			else:
				art,D = hf.load_img(join(load_dir, C.phase_dirs[0]))
				ven,_ = hf.load_img(join(load_dir, C.phase_dirs[1]))
				eq,_ = hf.load_img(join(load_dir, C.phase_dirs[2]))

			if exec_reg:
				art, ven, eq, slice_shift = reg.crop_reg(art, ven, eq)#, "bspline", num_iter=30)
			else:
				slice_shift = 0

			img = np.stack((art, ven, eq), -1)

			if np.product(art.shape) > C.max_size:
				downsample = 2

			if downsample != 1:
				img = tr.scale3d(img, [1/downsample, 1/downsample, 1])
				D = [D[0]*downsample, D[1]*downsample, D[2]]
		except:
			raise ValueError(accnum)

		np.save(join(C.full_img_dir, accnum+".npy"), img)

		if save_seg:
			sm.save_segs([accnum], downsample, slice_shift, art.shape[-1])

		if cnt % 3 == 2:
			print(".", end="")
		accnum_df.loc[accnum] = get_patient_row(accnum, cls) + list(D)
		accnum_df.to_csv(C.accnum_df_path)

@autofill_cls_arg
def load_vois(cls=None, accnums=None, overwrite=False, save_seg=False):
	"""Updates the voi_dfs based on the raw spreadsheet.
	dcm2npy() must be run first to produce full size npy images."""

	src_data_df = get_coords_df(cls)
	lesion_df = get_lesion_df()
	if accnums is None:
		accnums = set(src_data_df['acc #'].values)
	else:
		accnums = set(accnums).intersection(src_data_df['acc #'].values)
	
	if overwrite:
		lesion_df = lesion_df[~((lesion_df["accnum"].isin(accnums)) & (lesion_df["cls"] == cls))]
	else:
		accnums = set(accnums).difference(lesion_df[lesion_df["cls"] == cls]["accnum"].values)

	for cnt, accnum in enumerate(accnums):
		df_subset = src_data_df[src_data_df['acc #'] == accnum]

		if save_seg:
			load_dir = join(C.dcm_dir, accnum)
			I,_ = hf.nii_load(join(load_dir, "nii_dir", "20s.nii.gz"))
			
			downsample = 1
			if np.product(I.shape) > C.max_size:
				downsample = 2
				for i in ['x','y']:
					for j in ['1','2']:
						df_subset[i+j] = df_subset[i+j] / downsample
			sm.save_segs([accnum], downsample)

		for _, row in df_subset.iterrows():
			x,y,z = [[int(row[ch+'1']), int(row[ch+'2'])] for ch in ['x','y','z']]
			#if row['Flipped'] != "Yes":
			#	z = [img.shape[2]-z[1], img.shape[2]-z[0]] # flip z

			lesion_ids = lesion_df[lesion_df["accnum"] == accnum].index
			if len(lesion_ids) > 0:
				lesion_nums = [int(lid[lid.find('_')+1:]) for lid in lesion_ids]
				for num in range(len(lesion_nums)+1):
					if num not in lesion_nums:
						new_num = num
			else:
				new_num = 0

			l_id = accnum + "_" + str(new_num)
			lesion_df.loc[l_id, ["accnum", "cls", "run_num"] + C.art_cols] = \
						[accnum, cls, int(row["Run"])]+x+y+z

			if 'x3' in row and not np.isnan(row['x3']):
				x,y,z = [[int(row[ch+'3']), int(row[ch+'4'])] for ch in ['x','y','z']]
				#if row['Flipped'] != "Yes":
				#	z = [img.shape[2]-z[1], img.shape[2]-z[0]] # flip z
				lesion_df.loc[l_id, C.ven_cols] = x+y+z
				
			if 'x5' in row and not np.isnan(row['x5']):
				x,y,z = [[int(row[ch+'5']), int(row[ch+'6'])] for ch in ['x','y','z']]
				#if row['Flipped'] != "Yes":
				#	z = [img.shape[2]-z[1], img.shape[2]-z[0]] # flip z
				lesion_df.loc[l_id, C.equ_cols] = x+y+z

		print(".", end="")
		if cnt % 5 == 2:
			lesion_df.to_csv(C.lesion_df_path)
	lesion_df.to_csv(C.lesion_df_path)

def get_patient_row(accnum, cls=None):
	if hasattr(C,'dcm_dirs'):
		subdir = join(C.dcm_dirs[C.cls_names.index(cls)], accnum)
	else:
		subdir = join(C.dcm_dir, accnum)
	fn = join(subdir, "T1_AP", "metadata.xml")
	if exists(fn):
		f = open(fn, 'r')
	else:
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
			return [np.nan]*4

	return read_metadata(''.join(f.readlines()))

def read_metadata(metadata_txt):
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
		print(mrn, end=",")
	birthdate = datetime.datetime.strptime(result[birthdate_tag], "%Y%m%d").date()

	if imgdate.month > birthdate.month or (imgdate.month > birthdate.month and imgdate.day >= birthdate.day):
		age = imgdate.year - birthdate.year
	else:
		age = imgdate.year - birthdate.year - 1

	sex = result[sex_tag]
	ethnicity = result[ethnic_tag]
	ethnicity = ethnicity.strip().upper()

	if ethnicity in ['W','WHITE']:
		ethnicity = "White"
	elif ethnicity in ['B','BLACK']:
		ethnicity = "Black"
	elif ethnicity == 'H':
		ethnicity = "Hispanic"
	elif ethnicity == 'P':
		ethnicity = "Pacific Islander"
	elif ethnicity in ['O','OTHER']:
		ethnicity = "Other"
	elif ethnicity in ['U', "PT REFUSED"] or len(ethnicity) > 20:
		ethnicity = "Unknown"
	else:
		raise ValueError(ethnicity)

	return [mrn, sex, age, ethnicity]

@autofill_cls_arg
def load_patient_info(cls=None, accnums=None, overwrite=False, verbose=False):
	"""Loads patient demographic info from metadata files downloaded alongside the dcms."""

	df = get_coords_df(cls)
	if accnums is None:
		accnums = set(df['acc #'].values)

	accnum_df = get_accnum_df()
	if not overwrite:
		accnums = set(accnums).difference(accnum_df.index.values)

	for cnt, accnum in enumerate(accnums):
		accnum_df.loc[accnum, C.accnum_cols[:4]] = get_patient_row(accnum, cls)

		if cnt % 20 == 2:
			accnum_df.to_csv(C.accnum_df_path)
	accnum_df.to_csv(C.mrn_df_path)

def load_clinical_vars():
	#only for Paula clinical
	xls_path=r"Z:\Paula\Clinical data project\coordinates + clinical variables.xlsx"
	train_path="E:\\LIRADS\\excel\\clinical_data_train.xlsx"
	test_path="E:\\LIRADS\\excel\\clinical_data_test.xlsx"

	def isnumber(x):
		try:
			float(x)
			return True
		except:
			return False

	DFs = {}
	cols = ['age', 'gender', 'AST', 'ALT', 'ALP', 'albumin', 'TBIL', 'PT', 'INR']
	for ix,cls in enumerate(C.sheetnames):
		df = pd.read_excel(xls_path, sheet_name=cls, index_col=2)
		df = df[~df.index.duplicated(keep='first')]
		df = df[['age ', 'gender ', 'AST <34', 'ALT <34',
		   'alk.Phosphatase 30-130', 'Albumin: 3.5-5.0', 'Total Bilirubin <1.2',
		   'Prothrombin  9.9-12.3', 'INR 0.8-1.15']]
		df.columns = cols
		DFs[C.cls_names[ix]] = df

	big_df = pd.concat([DFs[cls] for cls in DFs])
	big_df = big_df[big_df.applymap(isnumber)]

	fill_df = []
	for cls in DFs:
		df = DFs[cls]
		df.loc[df['gender'].astype(str) == 'M', 'gender'] = 0
		df.loc[df['gender'].astype(str) == 'F', 'gender'] = 1
		df = df[df.applymap(isnumber)]
		for col in ['age', 'AST', 'ALT', 'ALP', 'albumin', 'TBIL', 'PT', 'INR']:
			df[col] = (df[col].astype(float) - big_df[col].median()) / \
					(np.nanpercentile(big_df[col].astype(float).values, 80.) - np.nanpercentile(big_df[col].astype(float).values, 20.))
		DFs[cls] = df

	big_df = pd.concat([DFs[cls] for cls in DFs])
	big_df = big_df[~big_df.index.duplicated(keep='first')]
	big_df.fillna(0).to_excel(test_path)

	for cls in DFs:
		df = DFs[cls]
		for col in ['age', 'AST', 'ALT', 'ALP', 'albumin', 'TBIL', 'PT', 'INR']:
			df[col].fillna(np.nanmedian(df[col]), inplace=True)
		DFs[cls] = df

	fill_df = pd.concat([DFs[cls] for cls in DFs])
	fill_df = fill_df[~fill_df.index.duplicated(keep='first')]
	fill_df.to_excel(train_path)

###########################
### Build/retrieve dataframes
###########################

def get_lesion_df():
	if exists(C.lesion_df_path):
		lesion_df = pd.read_csv(C.lesion_df_path, index_col=0)
		lesion_df["accnum"] = lesion_df["accnum"].astype(str)
		lesion_df[C.art_cols] = lesion_df[C.art_cols].astype(int)
	else:
		lesion_df = pd.DataFrame(columns = ["accnum", "cls", "run_num"] + C.voi_cols)

	return lesion_df

def get_accnum_df():
	if exists(C.accnum_df_path):
		accnum_df = pd.read_csv(C.accnum_df_path, index_col=0)
		accnum_df.index = accnum_df.index.map(str)
	else:
		accnum_df = pd.DataFrame(columns=C.accnum_cols)

	return accnum_df

def get_coords_df(cls=None):
	if hasattr(C,'sheetnames'):
		df = pd.read_excel(C.coord_xls_path, C.sheetnames[C.cls_names.index(cls)])
	else:
		df = pd.read_excel(C.coord_xls_path, C.sheetname)
		if cls is not None:
			df = df[df["cls"] == cls]
	df = df[df['Run'] <= C.run_num].dropna(subset=["x1"])
	df['acc #'] = df['acc #'].astype(str)
	
	return df.drop(set(df.columns).difference(['acc #', 'Run', 'Flipped', 
		  'x1', 'x2', 'y1', 'y2', 'z1', 'z2',
		  'x3', 'x4', 'y3', 'y4', 'z3', 'z4',
		  'x5', 'x6', 'y5', 'y6', 'z5', 'z6']), axis=1)

if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='Convert DICOMs to npy files and transfer voi coordinates from excel to csv.')
	parser.add_argument('-c', '--cls', help='limit to a specific class')
	parser.add_argument('-v', '--verbose', action='store_true', help='verbosity')
	parser.add_argument('-o', '--overwrite', action='store_true', help='overwrite')
	args = parser.parse_args()

	s = time.time()
	dcm2npy(cls=args.cls, verbose=args.verbose, overwrite=args.overwrite)
	print("Time to convert dcm to npy: %s" % str(time.time() - s))

	s = time.time()
	load_vois_batch(cls=args.cls, verbose=args.verbose, overwrite=args.overwrite)
	print("Time to load voi coordinates: %s" % str(time.time() - s))

	print("Finished!")