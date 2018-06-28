"""
Converts a nifti file to a numpy array.
Accepts either a single nifti file or a folder of niftis as the input argument.

Usage:
	python voi_methods.py
	python voi_methods.py --cls hcc
	python voi_methods.py -v -c cyst
	python voi_methods.py -ovc hemangioma

Author: Clinton Wang, E-mail: `clintonjwang@gmail.com`, Github: `https://github.com/clintonjwang/voi-classifier`
"""

import argparse
import copy
import glob
import importlib
import math
import multiprocessing
import os
import random
import time
from os.path import *

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.misc import imsave
from skimage.transform import rescale, resize

import config
import dr_methods as drm
import niftiutils.helper_fxns as hf
import niftiutils.masks as masks
import niftiutils.transforms as tr
import niftiutils.visualization as vis

importlib.reload(drm)
importlib.reload(tr)
importlib.reload(config)
C = config.Config()

#####################################
### QC methods
#####################################

def plot_check(num, lesion_id=None, normalize=[-1,1]):
	"""Plot the unscaled, cropped or augmented versions of a lesion.
	Lesion selected at random from cls if lesion_id is None.
	Either lesion_id or cls must be specified.
	If accession number is put instead of lesion_id, picks the first lesion."""
	if lesion_id.find('_') == -1:
		lesion_id += '_0'
		
	if num == 0:
		img = np.load(join(C.full_img_dir, lesion_id[:lesion_id.find('_')] + ".npy"))
	elif num == 1:
		img = np.load(join(C.crops_dir, lesion_id + ".npy"))
	elif num == 2:
		img = np.load(join(C.unaug_dir, lesion_id + ".npy"))
	elif num == 3:
		img = np.load(join(C.aug_dir, lesion_id + "_" + str(random.randint(0,C.aug_factor-1)) + ".npy"))
	else:
		raise ValueError(num + " should be 0 (uncropped), 1 (gross cropping), 2 (unaugmented) or 3 (augmented)")
	vis.draw_slices(img, normalize=normalize)

	return img

def remove_lesion_id(lesion_id):
	for fn in glob.glob(join(C.crops_dir, lesion_id+"*")):
		os.remove(fn)
	for fn in glob.glob(join(C.unaug_dir, lesion_id+"*")):
		os.remove(fn)
	for fn in glob.glob(join(C.aug_dir, lesion_id+"*")):
		os.remove(fn)

@drm.autofill_cls_arg
def xref_dirs_with_excel(cls=None, fix_inplace=True):
	"""Make sure the image directories have all the images expected based on the config settings
	(aug factor and run number) and VOI spreadsheet contents.
	If fix_inplace is True, reload any mismatched accnums."""
	lesion_df = pd.read_csv(C.small_voi_path)

	print("Checking", cls)
	bad_accnums = []

	index = C.cls_names.index(cls)
	accnum_df = C.sheetnames[index]
	df = pd.read_excel(C.coord_xls_path, C.sheetnames[index])
	lesion_df = drm.get_lesion_df()
	xls_accnums = list(df[df['Run'] <= C.run_num]['acc #'].astype(str))
	unique, counts = np.unique(xls_accnums, return_counts=True)
	xls_set = set(unique)
	xls_cnts = dict(zip(unique, counts))

	# Check for loaded images
	for accnum in xls_set:
		if not exists(C.full_img_dir + "\\" + cls + "\\" + accnum + ".npy"):
			print(accnum, "is contained in the spreadsheet but has no loaded image in", C.full_img_dir)

			bad_accnums.append(accnum)

	# Check for lesion_df
	accnums = list(lesion_df[lesion_df["cls"] == cls]["accnum"])
	unique, counts = np.unique(accnums, return_counts=True)
	diff = xls_set.difference(unique)
	if len(diff) > 0:
		print(diff, "are contained in the spreadsheet but not in lesion_df.")
		bad_accnums += list(diff)

	diff = set(unique).difference(xls_set)
	if len(diff) > 0:
		print(diff, "are contained in lesion_df but not the spreadsheet.")
		bad_accnums += list(diff)

	overlap = xls_set.intersection(unique)
	voi_cnts = dict(zip(unique, counts))
	for accnum in overlap:
		if voi_cnts[accnum] != xls_cnts[accnum]:
			print("Mismatch in number of lesions in the spreadsheet vs lesion_df for", accnum)
			bad_accnums.append(accnum)

	# Check rough cropped lesions
	accnums = [fn[:fn.find("_")] for fn in os.listdir(C.crops_dir + "\\" + cls)]
	unique, counts = np.unique(accnums, return_counts=True)
	diff = xls_set.difference(unique)
	if len(diff) > 0:
		print(diff, "are contained in the spreadsheet but not in", C.crops_dir)
		bad_accnums += list(diff)
	diff = set(unique).difference(xls_set)
	if len(diff) > 0:
		print(diff, "are contained in", C.crops_dir, "but not the spreadsheet.")
		bad_accnums += list(diff)

	overlap = xls_set.intersection(unique)
	voi_cnts = dict(zip(unique, counts))
	for accnum in overlap:
		if voi_cnts[accnum] != xls_cnts[accnum]:
			print("Mismatch in number of lesions in the spreadsheet vs", C.crops_dir, "for", accnum)
			bad_accnums.append(accnum)

	# Check unaugmented lesions
	accnums = [fn[:fn.find("_")] for fn in os.listdir(C.unaug_dir)]
	unique, counts = np.unique(accnums, return_counts=True)
	diff = xls_set.difference(unique)
	if len(diff) > 0:
		print(diff, "are contained in the spreadsheet but not in", C.unaug_dir)
		bad_accnums += list(diff)
	diff = set(unique).difference(xls_set)
	if len(diff) > 0:
		print(diff, "are contained in", C.unaug_dir, "but not the spreadsheet.")
		bad_accnums += list(diff)

	overlap = xls_set.intersection(unique)
	voi_cnts = dict(zip(unique, counts))
	for accnum in overlap:
		if voi_cnts[accnum] != xls_cnts[accnum]:
			print("Mismatch in number of lesions in the spreadsheet vs", C.unaug_dir, "for", accnum)
			bad_accnums.append(accnum)


	# Check augmented lesions
	lesion_ids_folder = set([fn[:fn.rfind("_")] for fn in os.listdir(C.aug_dir)])
	lesion_ids_df = set(lesion_df[lesion_df["cls"] == cls].index)
	diff = lesion_ids_df.difference(lesion_ids_folder)
	if len(diff) > 0:
		print(diff, "are contained in lesion_df but not in", C.aug_dir)
		bad_accnums += [x[:x.find('_')] for x in diff]
	diff = lesion_ids_folder.difference(lesion_ids_df)
	if len(diff) > 0:
		print(diff, "are contained in", C.aug_dir, "but not in lesion_df.")
		bad_accnums += [x[:x.find('_')] for x in diff]

	# Fix lesions
	if fix_inplace and len(bad_accnums) > 0:
		print("Reloading", set(bad_accnums))
		for accnum in set(bad_accnums):
			reset_accnum(accnum)

def reset_accnum(accnum):
	"""Reset an accession number (only assumes dcm2npy has been called)"""
	accnum = str(accnum)

	lesion_df = pd.read_csv(C.small_voi_path, index_col=0)
	lesion_df["accnum"] = lesion_df["accnum"].astype(str)
	lesion_df = lesion_df[lesion_df["accnum"] != accnum]
	lesion_df.to_csv(C.small_voi_path)

	for cls in C.cls_names:
		for base_dir in [C.crops_dir, C.unaug_dir, C.aug_dir]:
			for fn in glob.glob(join(base_dir, cls, accnum+"*")):
				os.remove(fn)

	lesion_df = drm.get_lesion_df()
	lesion_df = drm._remove_accnums_from_vois(voi_df_art, voi_df_ven, voi_df_eq, [accnum])
	lesion_df.to_csv(C.lesion_df_path)

	for cls in C.cls_names:
		reload_accnum(cls, accnums=[accnum], augment=True)

def load_accnum(cls=None, accnums=None, augment=True):
	#Reloads cropped, scaled and augmented images. Updates voi_dfs and small_vois accordingly.
	#May fail if the accnum already exists - should call reset_accnum instead
	for base_dir in [C.crops_dir, C.unaug_dir, C.aug_dir]:
		if not exists(join(base_dir, cls)):
			os.makedirs(join(base_dir, cls))

	drm.load_vois_batch(cls, accnums, overwrite=True)
	extract_vois(cls, accnums, overwrite=True)
	save_unaugmented_set(cls, accnums, overwrite=True)
	if augment:
		save_augmented_set(cls, accnums, overwrite=True)

@drm.autofill_cls_arg
def save_vois_as_imgs(cls=None, lesion_ids=None, save_dir=None, normalize=None, rescale_factor=3, fn_prefix="", fn_suffix=None, separate_by_cls=True):
	"""Save all voi images as jpg."""
	if separate_by_cls:
		save_dir = join(save_dir, cls)
		if fn_suffix is None:
			fn_suffix = ""
	if not exists(save_dir):
		os.makedirs(save_dir)

	lesion_df = drm.get_lesion_df()
	if lesion_ids is not None:
		fns = [lesion_id+".npy" for lesion_id in lesion_ids if lesion_id+".npy" in os.listdir(C.unaug_dir) \
				and lesion_df.loc[lesion_id,"cls"] == cls]
	else:
		fns = [fn for fn in os.listdir(C.unaug_dir) if fn[:-4] in lesion_df.index and \
				lesion_df.loc[fn[:-4],"cls"] == cls]

	for fn in fns:
		img = np.load(join(C.unaug_dir, fn))

		img_slice = img[:,:, img.shape[2]//2].astype(float)
		img_slice = vis.normalize_img(img_slice, normalize)
			
		ch1 = np.transpose(img_slice[:,::-1,0], (1,0))
		ch2 = np.transpose(img_slice[:,::-1,1], (1,0))
		
		if C.nb_channels == 2:
			ret = np.empty([ch1.shape[0]*C.nb_channels, ch1.shape[1]])
			ret[:ch1.shape[0],:] = ch1
			ret[ch1.shape[0]:,:] = ch2
			
		elif C.nb_channels == 3:
			ch3 = np.transpose(img_slice[:,::-1,2], (1,0))

			ret = np.empty([ch1.shape[0]*C.nb_channels, ch1.shape[1]])
			ret[:ch1.shape[0],:] = ch1
			ret[ch1.shape[0]:ch1.shape[0]*2,:] = ch2
			ret[ch1.shape[0]*2:,:] = ch3
		
		if fn_suffix is None:
			suffix = " (%s)" % cls
		else:
			suffix = fn_suffix

		imsave("%s\\%s%s%s.png" % (save_dir, fn_prefix, fn[:-4], suffix), rescale(ret, rescale_factor, mode='constant'))

@drm.autofill_cls_arg
def save_segs_as_imgs(cls=None, lesion_ids=None, save_dir="D:\\Etiology\\screenshots", normalize=None, rescale_factor=3, fn_prefix="", fn_suffix=None, separate_by_cls=True):
	"""Save all voi images as jpg."""
	if separate_by_cls:
		save_dir = join(save_dir, cls)
		if fn_suffix is None:
			fn_suffix = ""
	if not exists(save_dir):
		os.makedirs(save_dir)
			
	src_data_df = drm.get_coords_df(cls)
	accnums = src_data_df["acc #"].values
	lesion_ids = [x[:-4] for x in os.listdir(C.crops_dir) if x[:x.find('_')] in accnums]

	for fn in lesion_ids:
		img = np.load(join(C.unaug_dir, fn+".npy"))
		img_slice = img[:,:, img.shape[2]//2].astype(float)
		img_slice = vis.normalize_img(img_slice, normalize)
			
		ch1 = np.transpose(img_slice[:,:,0], (1,0))
		ch2 = np.transpose(img_slice[:,:,1], (1,0))
		ch3 = np.transpose(img_slice[:,:,2], (1,0))

		ret = np.empty([ch1.shape[0]*C.nb_channels, ch1.shape[1]])
		ret[:ch1.shape[0],:] = ch1
		ret[ch1.shape[0]:ch1.shape[0]*2,:] = ch2
		ret[ch1.shape[0]*2:,:] = ch3
		
		if fn_suffix is None:
			suffix = " (%s)" % cls
		else:
			suffix = fn_suffix

		imsave("%s\\%s%s%s.png" % (save_dir, fn_prefix, fn, suffix), rescale(ret, rescale_factor, mode='constant'))

@drm.autofill_cls_arg
def save_imgs_with_bbox(cls=None, lesion_ids=None, save_dir=None, normalize=None, fixed_width=100, fn_prefix="", fn_suffix=None, separate_by_cls=True):
	"""Save images of grossly cropped lesions with a bounding box around the tighter crop.
	If fixed_width is None, the images are not scaled.
	Otherwise, the images are made square with the given width in pixels."""

	lesion_df = pd.read_csv(C.small_voi_path)

	if save_dir is None:
		save_dir = C.output_img_dir
	if separate_by_cls:
		save_dir = join(save_dir, cls)
		if fn_suffix is None:
			fn_suffix = ""

	if not exists(save_dir):
		os.makedirs(save_dir)
		
	if lesion_ids is None:
		lesion_ids = [x[:-4] for x in os.listdir(C.crops_dir)]
	else:
		lesion_ids = [lesion_id for lesion_id in lesion_ids if lesion_id+".npy" in os.listdir(C.crops_dir)]

	for lesion_id in lesion_ids:
		#cls = lesion_df.loc.loc[lesion_id, "cls"].values[0]
		img = np.load(join(C.crops_dir, lesion_id + ".npy"))
		img_slice = img[:,:, img.shape[2]//2].astype(float)
		#for ch in range(img_slice.shape[-1]):
		#	img_slice[:, :, ch] *= 255/np.amax(img_slice[:, :, ch])
		if normalize is not None:
			img_slice[0,0,:]=normalize[0]
			img_slice[0,-1,:]=normalize[1]

		img_slice = np.stack([img_slice, img_slice, img_slice], axis=2)
		
		img_slice = _draw_bbox(img_slice, padded_coords(lesion_df, lesion_id))
			
		ch1 = np.transpose(img_slice[:,::-1,:,0], (1,0,2))
		ch2 = np.transpose(img_slice[:,::-1,:,1], (1,0,2))
		
		if C.nb_channels == 2:
			ret = np.empty([ch1.shape[0]*2, ch1.shape[1], 3])
			ret[:ch1.shape[0],:,:] = ch1
			ret[ch1.shape[0]:,:,:] = ch2
			
		elif C.nb_channels == 3:
			ch3 = np.transpose(img_slice[:,::-1,:,2], (1,0,2))

			ret = np.empty([ch1.shape[0]*3, ch1.shape[1], 3])
			ret[:ch1.shape[0],:,:] = ch1
			ret[ch1.shape[0]:ch1.shape[0]*2,:,:] = ch2
			ret[ch1.shape[0]*2:,:,:] = ch3
			
		else:
			raise ValueError("Invalid num channels")
		
		if fn_suffix is None:
			suffix = " (%s)" % cls
		else:
			suffix = fn_suffix

		if fixed_width is not None:
			imsave("%s\\%s%s%s.png" % (save_dir, fn_prefix, lesion_id, suffix), resize(ret, [fixed_width*3, fixed_width]))
		else:
			imsave("%s\\%s%s%s.png" % (save_dir, fn_prefix, lesion_id, suffix), ret)

def semiauto_label_lesions(lesion_ids=None):
	print("1: Arterial enhancement")
	print("2: Washout")
	print("3: Capsule")
	print("4: Bad image")

	if exists(C.label_lesion_df):
		df = pd.read_csv(C.label_lesion_df, index_col=0)
	else:
		df = pd.DataFrame(columns=["Arterial enhancement", "Washout", "Capsule", "Bad image"])

	if lesion_ids is None:
		lesion_df = drm.get_lesion_df()
		lesion_ids = lesion_df.index

	for lesion_id in lesion_ids:
		plot_check(2, lesion_id, None)
		print(lesion_id)
		ins = input()
		row = np.zeros(4)
		if ins == "q":
			return
		if '0' in ins:
			continue
		if '4' in ins:
			row[-1] = 1
		if '1' in ins:
			row[0] = 1
		if '2' in ins:
			row[0] = 2
		if '3' in ins:
			row[0] = 3
		df.loc[lesion_id] = row

	df.to_csv(C.label_lesion_df)


#####################################
### Data Creation
#####################################

def extract_seg_vois(accnums=None):
	if not exists(C.crops_dir):
		os.makedirs(C.crops_dir)
	if accnums is None:
		accnums = [x[:-4] for x in os.listdir(C.full_img_dir) if not x.endswith("seg.npy")]

	accnum_df = pd.read_csv(C.accnum_df_path, index_col=0)
	accnum_df.index = accnum_df.index.map(str)

	for accnum in accnums:
		if not exists(join(C.full_img_dir, accnum+"_tumorseg.npy")):
			continue
		try:
			D = np.product(accnum_df.loc[accnum, C.dim_cols].values)
			I = np.load(join(C.full_img_dir, accnum+".npy"))
			M_all = np.load(join(C.full_img_dir, accnum+"_tumorseg.npy"))
			M_all = masks.split_mask(M_all)
			for m_ix, M in enumerate(M_all):
				V = M.sum()*D
				if V < 1000 or V > 1000000: #1cc to 1000cc
					continue
				crop_img = masks.crop_vicinity(I,M, padding=.2, min_pad=10)
				crop_img = tr.normalize_intensity(crop_img, max_intensity=1, min_intensity=-1)
				np.save(join(C.crops_dir, accnum+"_%d.npy"%m_ix), crop_img)
		except:
			print(accnum)

def save_seg_set(accnums=None, overwrite=True, unaug=True, aug=True, num_cores=None):
	if not exists(C.unaug_dir):
		os.makedirs(C.unaug_dir)
	if not exists(C.aug_dir):
		os.makedirs(C.aug_dir)

	if accnums is None:
		lesion_ids = [x[:-4] for x in os.listdir(C.crops_dir)]
	else:
		lesion_ids = [x[:-4] for x in os.listdir(C.crops_dir) if x[:x.find('_')] in accnums]

	if unaug:
		for ix, lesion_id in enumerate(lesion_ids):
			if not overwrite and exists(join(C.unaug_dir, lesion_id+".npy")):
				continue
			I = np.load(join(C.crops_dir, lesion_id+".npy"))
			I = tr.rescale_img(I, C.dims)
			np.save(join(C.unaug_dir, lesion_id), I)

	if aug:
		if num_cores is None:
			num_cores = multiprocessing.cpu_count() - 1
		if num_cores > 1:
			Parallel(n_jobs=num_cores)(delayed(_save_augmented_img)(lesion_id, overwrite=overwrite) for lesion_id in lesion_ids)
		else:
			for lesion_id in lesion_ids:
				_save_augmented_img(lesion_id, overwrite=overwrite)

@drm.autofill_cls_arg
def extract_vois(cls=None, accnums=None, lesion_ids=None, overwrite=False):
	"""Produces grossly cropped but unscaled versions of the images.
	This intermediate step makes debugging and visualization easier, and augmentation faster.
	Rotation, scaling, etc. for augmentation can be done directly on these images.
	drm.load_vois_batch() must be run first to populate the lesion_df.
	Overwrites any existing images and lesion_df entries without checking."""
	if not exists(C.crops_dir):
		os.makedirs(C.crops_dir)

	lesion_df = drm.get_lesion_df()
	if lesion_ids is None:
		if accnums is None:
			lesion_ids = lesion_df[lesion_df["cls"] == cls].index
		else:
			lesion_ids = lesion_df[(lesion_df["accnum"].isin(accnums)) & (lesion_df["cls"] == cls)].index

	for cnt, l_id in enumerate(lesion_ids):
		if not overwrite and not np.isnan(lesion_df.loc[l_id, "sm_x1"]):
			continue

		img_path = join(C.full_img_dir, lesion_df.loc[l_id, "accnum"]+".npy")
		if not exists(img_path):
			continue

		img = np.load(img_path)
		img = tr.normalize_intensity(img, max_I=1, min_I=-1)
		crop_img, coords = _extract_voi(img, lesion_df.loc[l_id])
		lesion_df.loc[l_id, C.small_cols] = coords

		np.save(join(C.crops_dir, l_id), crop_img)

		print(".", end="")
		if cnt % 20 == 2:
			lesion_df.to_csv(C.lesion_df_path)
	
	lesion_df.to_csv(C.lesion_df_path)

@drm.autofill_cls_arg
def save_unaugmented_set(cls=None, accnums=None, lesion_ids=None, custom_vois=None, lesion_ratio=None, overwrite=True):
	"""Save unaugmented lesion images. Overwrites without checking."""

	lesion_df = drm.get_lesion_df()

	if not exists(C.unaug_dir):
		os.makedirs(C.unaug_dir)

	if lesion_ids is None:
		if accnums is None:
			lesion_ids = [x[:-4] for x in os.listdir(C.crops_dir) if x[:-4] in lesion_df[lesion_df["cls"] == cls].index]
		else:
			lesion_ids = [x[:-4] for x in os.listdir(C.crops_dir) if x[:x.find('_')] in accnums and \
					x[:-4] in lesion_df[lesion_df["cls"] == cls].index]

	for ix, lesion_id in enumerate(lesion_ids):
		if not overwrite and exists(join(C.unaug_dir, lesion_id)):
			continue
		img = np.load(join(C.crops_dir, lesion_id + ".npy"))
		try:
			if C.pre_scale > 0:
				img = tr.normalize_intensity(img, max_I=1, min_I=-1, frac=C.pre_scale)
		except:
			os.remove(join(C.crops_dir, lesion_id + ".npy"))
		img = tr.rescale_img(img, C.dims)
		#if custom_vois is None:
		#	unaug_img = _resize_img(join(C.crops_dir, lesion_id + ".npy"),
		#				lesion_df.loc[lesion_id, C.small_cols])
		#else:
		#	unaug_img = _resize_img(join(C.crops_dir, lesion_id + ".npy"), custom_vois[ix], lesion_ratio)

		np.save(join(C.unaug_dir, lesion_id), img)

@drm.autofill_cls_arg
def save_augmented_set(cls=None, accnums=None, num_cores=None, overwrite=True):
	"""Augment all images in cls using CPU parallelization.
	Overwrite can be an int, in which case it will create
	augmented samples enumerated starting at that number."""

	lesion_df = drm.get_lesion_df()

	if not exists(C.aug_dir):
		os.makedirs(C.aug_dir)

	if accnums is not None:
		lesion_ids = [x[:-4] for x in os.listdir(C.crops_dir) if x[:x.find('_')] in accnums]
		for lesion_id in lesion_ids:
			_save_augmented_img(lesion_id, lesion_df.loc[lesion_id, C.small_cols], overwrite=overwrite)

	else:
		lesion_ids = [x[:-4] for x in os.listdir(C.crops_dir) if x[:-4] in lesion_df.index]

		t = time.time()
		if num_cores is None:
			num_cores = multiprocessing.cpu_count() - 1

		if num_cores > 1:
			Parallel(n_jobs=num_cores)(delayed(_save_augmented_img)(lesion_id,
				lesion_df.loc[lesion_id, C.small_cols],
				overwrite=overwrite) for lesion_id in lesion_ids)
		else:
			for lesion_id in lesion_ids:
				_save_augmented_img(lesion_id, lesion_df.loc[lesion_id, C.small_cols], overwrite=overwrite)

		print(cls, time.time()-t)


#####################################
### Public Subroutines
#####################################

def padded_coords(lesion_df, lesion_id):
	"""Ideally should place the voi in the same place in the image"""
	return lesion_df.loc[lesion_id]

def get_scale_ratios(voi, lesion_ratio=None):
	"""Based on a voi in an image and the final dimensions desired,
	determine how much the image needs to be scaled so that the voi fits
	in the desired dimensions, with optional padding.

	lesion_ratio of .8 means that the width of the lesion is 80% of the image width.
	lesion_ratio of 1 means no padding is added."""

	if lesion_ratio is None:
		lesion_ratio = C.lesion_ratio

	dx = [voi[ix+1] - voi[ix] for ix in [0,2,4]]
	scale_ratios = [C.dims[ix]/dx[ix] * lesion_ratio for ix in range(3)]

	return scale_ratios

#####################################
### Subroutines
#####################################

def _draw_bbox(img_slice, voi):
	"""Draw a colored box around the voi of an image slice showing how it would be cropped."""

	scale_ratios = get_scale_ratios(voi)
	
	crop = [img_slice.shape[i] - round(C.dims[i]/scale_ratios[i]) for i in range(2)]
	
	x1 = crop[0]//2
	x2 = -crop[0]//2
	y1 = crop[1]//2
	y2 = -crop[1]//2

	img_slice[x1:x2, y2, 2, :] = 1
	img_slice[x1:x2, y2, :2, :] = 1

	img_slice[x1:x2, y1, 2, :] = 1
	img_slice[x1:x2, y1, :2, :] = 1

	img_slice[x1, y1:y2, 2, :] = 1
	img_slice[x1, y1:y2, :2, :] = 1

	img_slice[x2, y1:y2, 2, :] = 1
	img_slice[x2, y1:y2, :2, :] = 1

	dx = int(5/4*x1 - img_slice.shape[0]/4)
	dy = int(5/4*y1 - img_slice.shape[1]/4)
	
	return img_slice[dx:-dx,dy:-dy,:,:]

def _augment_img(img, num_samples, voi=None, save_name=None, overwrite=True):
	"""For rescaling an img to final_dims while scaling to make sure the image contains the voi."""
	if type(overwrite) == int:
		start=overwrite
	else:
		start=0
	
	if voi is not None:
		buffer1 = C.lesion_ratio-.1
		buffer2 = C.lesion_ratio+.1
		scale_ratios = [1]*3#get_scale_ratios(voi, lesion_ratio=1)
	else:
		scale_ratios = [1]*3
		buffer1 = 1.
		buffer2 = 1.2

	aug_imgs = []
	
	for img_num in range(start, num_samples):		
		flip = [random.choice([-1, 1]) for _ in range(3)]

		if voi is None:
			aug_img = img[::flip[0], ::flip[1], ::flip[2], ...]

			if np.min(img.shape[:-1]) > 8:
				sl = [slice(random.randint(0,2), random.randint(-3,-1)) for _ in range(3)]
				aug_img = aug_img[sl]

			aug_img = tr.rescale_img(aug_img, C.dims)
			for axis in range(3):
				aug_img = tr.rotate(aug_img, random.choice([0, 90, 180, 270]), axis=axis)

		else:
			scales = [random.uniform(scale_ratios[ix]*buffer1, scale_ratios[ix]*buffer2) for ix in range(3)]
			trans = [random.randint(-C.translate[ix], C.translate[ix]) for ix in range(3)]

			aug_img = img
			for axis in range(3):
				aug_img = tr.rotate(aug_img, random.randint(0, 359), axis=axis)
			
			aug_img = tr.rescale_img(aug_img, scales)
			crops = np.clip([aug_img.shape[i] - C.dims[i] for i in range(3)], 0, 50)
			#for i in range(3):
			#	assert crops[i]>=

			#aug_img = tr.offset_phases(aug_img, max_offset=2, max_z_offset=1)
			sl = [slice(crops[ix]//2 *flip[ix] + trans[ix], -crops[ix]//2 *flip[ix] + trans[ix], flip[ix]) for ix in range(3)]
			aug_img = aug_img[sl]
			if np.min(aug_img.shape) == 0:
				return
			aug_img = tr.rescale_img(aug_img, C.dims)
		
		for ix in range(3):
			aug_img[...,ix] = aug_img[...,ix] * random.gauss(1,C.intensity_scaling[0]) + random.gauss(0,C.intensity_scaling[1])

		np.save(save_name + "_" + str(img_num), aug_img)

def _save_augmented_img(lesion_id, voi=None, overwrite=True):
	"""Written in a way to allow partial overwriting"""
	if lesion_id.find('.') != -1:
		lesion_id = lesion_id[:-4]
	if not overwrite and exists(join(C.aug_dir, lesion_id + "_0.npy")):
		return

	img = np.load(join(C.crops_dir, lesion_id + ".npy"))
	if C.pre_scale > 0:
		img = tr.normalize_intensity(img, max_I=1, min_I=-1, frac=C.pre_scale)
	_augment_img(img, C.aug_factor, voi=voi, save_name=join(C.aug_dir, lesion_id), overwrite=overwrite)

def _resize_img(img_path, voi, lesion_ratio=None):
	"""For rescaling an img to final_dims while scaling to make sure the image contains the voi.
	Do not reuse img
	"""
	scale_ratios = get_scale_ratios(voi, lesion_ratio=lesion_ratio)

	img = np.load(img_path)
	if C.pre_scale > 0:
		img = tr.normalize_intensity(img, max_I=1, min_I=-1, frac=C.pre_scale)
	img = tr.rescale_img(img, scale_ratios)

	crop = [img.shape[i] - C.dims[i] for i in range(3)]

	for i in range(3):
		assert crop[i]>=0
	
	img = img[crop[0]//2:-crop[0]//2, crop[1]//2:-crop[1]//2, crop[2]//2:-crop[2]//2, :]

	return img

def _extract_voi(img, voi, pad_lims=[5,40]):
	"""Input: image, a voi to center on, and the min dims of the unaugmented img.
	Outputs loosely cropped voi-centered image and coords of the voi within this loosely cropped image.
	"""
	x1 = [voi['a_x1'], img.shape[1]-voi['a_y2'], img.shape[2]-voi['a_z2']]
	x2 = [voi['a_x2'], img.shape[1]-voi['a_y1'], img.shape[2]-voi['a_z1']]
	
	# align phases
	if not np.isnan(voi["v_x1"]):
		dx = np.array([((voi["v_"+char+"1"] + voi["v_"+char+"2"]) - (x1[ix]+x2[ix])) / 2 for ix,char in enumerate(['x','y','z'])], int)
		pad = int(np.abs(dx).max()+1)
		sl = [slice(pad+offset,-pad+offset) for offset in dx]
		img[...,1] = np.pad(img[...,1], pad, 'constant')[sl]

	if not np.isnan(voi["e_x1"]):
		dx = np.array([((voi["e_"+char+"1"] + voi["e_"+char+"2"]) - (x1[ix]+x2[ix])) // 2 for ix,char in enumerate(['x','y','z'])], int)
		pad = int(np.abs(dx).max()+1)
		sl = [slice(pad+offset,-pad+offset) for offset in dx]
		img[...,2] = np.pad(img[...,2], pad, 'constant')[sl]

	dx = np.array([x2[ix] - x1[ix] for ix in range(3)])
	assert np.all(np.greater(x2,x1)), str(voi["accnum"])
	pad = np.clip(dx * (math.sqrt(2) - 1) / 2, *pad_lims).astype(int) #min padding needed for rotation
	x1_ = [max(x1[ix]-pad[ix], 0) for ix in range(3)]
	x2_ = [min(x2[ix]+pad[ix], img.shape[ix]-1) for ix in range(3)]
	sl = [slice(x1_[ix], x2_[ix]) for ix in range(3)]
		
	return img[sl], np.array(list(zip(x1_, x2_))).flatten()

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