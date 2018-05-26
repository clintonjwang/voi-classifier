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
import ast
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
import niftiutils.transforms as tr
import niftiutils.visualization as vis

#####################################
### QC methods
#####################################

def plot_check(num, lesion_id=None, cls=None, normalize=[-.8,.5]):
	"""Plot the unscaled, cropped or augmented versions of a lesion.
	Lesion selected at random from cls if lesion_id is None.
	Either lesion_id or cls must be specified.
	If accession number is put instead of lesion_id, picks the first lesion."""

	C = config.Config()

	if lesion_id.find('_') == -1:
		lesion_id += '_0'

	cls = pd.read_csv(C.small_voi_path, index_col=0).loc[lesion_id, "cls"].values[0]
		
	if num == 0:
		img = np.load(join(C.full_img_dir, lesion_id[:lesion_id.find('_')] + ".npy"))
	elif num == 1:
		img = np.load(join(C.crops_dir, cls, lesion_id + ".npy"))
	elif num == 2:
		img = np.load(join(C.unaug_dir, cls, lesion_id + ".npy"))
	elif num == 3:
		img = np.load(join(C.aug_dir, cls, lesion_id + "_" + str(random.randint(0,C.aug_factor-1)) + ".npy"))
	else:
		raise ValueError(num + " should be 0 (uncropped), 1 (gross cropping), 2 (unaugmented) or 3 (augmented)")
	vis.draw_slices(img, normalize=normalize)

	return img

@drm.autofill_cls_arg
def xref_dirs_with_excel(cls=None, fix_inplace=True):
	"""Make sure the image directories have all the images expected based on the config settings
	(aug factor and run number) and VOI spreadsheet contents.
	If fix_inplace is True, reload any mismatched accnums."""
	importlib.reload(drm)
	C = config.Config()

	small_voi_df = pd.read_csv(C.small_voi_path)

	print("Checking", cls)
	bad_accnums = []

	index = C.cls_names.index(cls)
	accnum_df = C.sheetnames[index]
	df = pd.read_excel(C.coord_xls_path, C.sheetnames[index])
	voi_df = drm.get_voi_dfs()[0]
	xls_accnums = list(df[df['Run'] <= C.run_num]['acc #'].astype(str))
	unique, counts = np.unique(xls_accnums, return_counts=True)
	xls_set = set(unique)
	xls_cnts = dict(zip(unique, counts))

	# Check for loaded images
	for accnum in xls_set:
		if not exists(C.full_img_dir + "\\" + cls + "\\" + accnum + ".npy"):
			print(accnum, "is contained in the spreadsheet but has no loaded image in", C.full_img_dir)

			bad_accnums.append(accnum)

	# Check for small_voi_df
	accnums = list(small_voi_df[small_voi_df["cls"] == cls]["accnum"])
	unique, counts = np.unique(accnums, return_counts=True)
	diff = xls_set.difference(unique)
	if len(diff) > 0:
		print(diff, "are contained in the spreadsheet but not in small_voi_df.")
		bad_accnums += list(diff)

	diff = set(unique).difference(xls_set)
	if len(diff) > 0:
		print(diff, "are contained in small_voi_df but not the spreadsheet.")
		bad_accnums += list(diff)

	overlap = xls_set.intersection(unique)
	voi_cnts = dict(zip(unique, counts))
	for accnum in overlap:
		if voi_cnts[accnum] != xls_cnts[accnum]:
			print("Mismatch in number of lesions in the spreadsheet vs small_voi_df for", accnum)
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
	accnums = [fn[:fn.find("_")] for fn in os.listdir(join(C.unaug_dir, cls))]
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
	lesion_ids_folder = set([fn[:fn.rfind("_")] for fn in os.listdir(join(C.aug_dir, cls))])
	lesion_ids_df = set(voi_df[voi_df["cls"] == cls].index)
	diff = lesion_ids_df.difference(lesion_ids_folder)
	if len(diff) > 0:
		print(diff, "are contained in voi_df but not in", C.aug_dir)
		bad_accnums += [x[:x.find('_')] for x in diff]
	diff = lesion_ids_folder.difference(lesion_ids_df)
	if len(diff) > 0:
		print(diff, "are contained in", C.aug_dir, "but not in voi_df.")
		bad_accnums += [x[:x.find('_')] for x in diff]

	# Fix lesions
	if fix_inplace and len(bad_accnums) > 0:
		print("Reloading", set(bad_accnums))
		for accnum in set(bad_accnums):
			reset_accnum(accnum)

def reset_accnum(accnum):
	"""Reset an accession number (only assumes dcm2npy has been called)"""

	importlib.reload(config)
	accnum = str(accnum)

	C = config.Config()
	small_voi_df = pd.read_csv(C.small_voi_path, index_col=0)
	small_voi_df["accnum"] = small_voi_df["accnum"].astype(str)
	small_voi_df = small_voi_df[small_voi_df["accnum"] != accnum]
	small_voi_df.to_csv(C.small_voi_path)

	for cls in C.cls_names:
		for base_dir in [C.crops_dir, C.unaug_dir, C.aug_dir]:
			for fn in glob.glob(join(base_dir, cls, accnum+"*")):
				os.remove(fn)

	voi_df_art, voi_df_ven, voi_df_eq = drm.get_voi_dfs()
	voi_df_art, voi_df_ven, voi_df_eq = drm._remove_accnums_from_vois(voi_df_art, voi_df_ven, voi_df_eq, [accnum])
	voi_dfs = voi_df_art, voi_df_ven, voi_df_eq
	drm.write_voi_dfs(voi_dfs)

	for cls in C.cls_names:
		reload_accnum(cls, accnums=[accnum], augment=True)

def load_accnum(cls=None, accnums=None, augment=True):
	#Reloads cropped, scaled and augmented images. Updates voi_dfs and small_vois accordingly.
	#May fail if the accnum already exists - should call reset_accnum instead
	importlib.reload(drm)
	C = config.Config()

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
	importlib.reload(hf)
	C = config.Config()

	if save_dir is None:
		save_dir = C.output_img_dir
	if separate_by_cls:
		save_dir = join(save_dir, cls)
		if fn_suffix is None:
			fn_suffix = ""
	if not exists(save_dir):
		os.makedirs(save_dir)

	if lesion_ids is not None:
		fns = [lesion_id+".npy" for lesion_id in lesion_ids if lesion_id+".npy" in os.listdir(join(C.unaug_dir, cls))]
	else:
		fns = os.listdir(join(C.unaug_dir, cls))

	for fn in fns:
		img = np.load(C.unaug_dir + cls + "\\" + fn)

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
def save_imgs_with_bbox(cls=None, lesion_ids=None, save_dir=None, normalize=None, fixed_width=100, fn_prefix="", fn_suffix=None, separate_by_cls=True):
	"""Save images of grossly cropped lesions with a bounding box around the tighter crop.
	If fixed_width is None, the images are not scaled.
	Otherwise, the images are made square with the given width in pixels."""

	C = config.Config()

	small_voi_df = pd.read_csv(C.small_voi_path)

	if save_dir is None:
		save_dir = C.output_img_dir
	if separate_by_cls:
		save_dir = join(save_dir, cls)
		if fn_suffix is None:
			fn_suffix = ""

	if not exists(save_dir):
		os.makedirs(save_dir)
		
	if lesion_ids is None:
		lesion_ids = [x[:-4] for x in os.listdir(join(C.crops_dir, cls))]
	else:
		lesion_ids = [lesion_id for lesion_id in lesion_ids if lesion_id+".npy" in os.listdir(join(C.crops_dir, cls))]

	#voi_df = drm.get_voi_dfs()[0]
	#lesion_ids = set(lesion_ids).intersection(voi_df[voi_df["run_num"] > 2].index)
	
	for lesion_id in lesion_ids:
		#cls = small_voi_df.loc.loc[lesion_id, "cls"].values[0]
		img = np.load(join(C.crops_dir, cls, lesion_id + ".npy"))
		img_slice = img[:,:, img.shape[2]//2].astype(float)
		#for ch in range(img_slice.shape[-1]):
		#	img_slice[:, :, ch] *= 255/np.amax(img_slice[:, :, ch])
		if normalize is not None:
			img_slice[0,0,:]=normalize[0]
			img_slice[0,-1,:]=normalize[1]

		img_slice = np.stack([img_slice, img_slice, img_slice], axis=2)
		
		img_slice = _draw_bbox(img_slice, padded_coords(small_voi_df, lesion_id))
			
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

def padded_coords(small_voi_df, lesion_id):
	return _get_voi_coords(small_voi_df.loc[lesion_id])

#####################################
### Data Creation
#####################################

@drm.autofill_cls_arg
def extract_vois(cls=None, accnums=None, overwrite=False):
	"""Produces grossly cropped but unscaled versions of the images.
	This intermediate step makes debugging and visualization easier, and augmentation faster.
	Rotation, scaling, etc. for augmentation can be done directly on these images.
	drm.load_vois_batch() must be run first to populate the voi_dfs.
	Overwrites any existing images and voi_df entries without checking."""

	C = config.Config()
	voi_df_art, voi_df_ven, voi_df_eq = drm.get_voi_dfs()

	if exists(C.small_voi_path):
		small_voi_df = pd.read_csv(C.small_voi_path, index_col=0)
		small_voi_df["accnum"] = small_voi_df["accnum"].astype(str)
	else:
		small_voi_df = pd.DataFrame(columns=["accnum", "cls", "x1", "x2", "y1", "y2", "z1", "z2"])

	if not exists(join(C.crops_dir, cls)):
		os.makedirs(join(C.crops_dir, cls))

	if accnums is None:
		accnums = [x[:-4] for x in os.listdir(C.full_img_dir)]
	if not overwrite:
		accnums = list(set(accnums).difference(small_voi_df["accnum"]))

	for img_num, accnum in enumerate(accnums):
		art_vois = voi_df_art[(voi_df_art["accnum"] == accnum) & (voi_df_art["cls"] == cls)]
		if len(art_vois) == 0:
			continue

		img = np.load(join(C.full_img_dir, accnum+".npy"))
		if overwrite:
			small_voi_df = small_voi_df[~((small_voi_df["accnum"] == accnum) & (small_voi_df["cls"] == cls))]

		# iterate over each voi in that image
		for lesion_id, voi_row in art_vois.iterrows():
			ven_voi = voi_df_ven.loc[lesion_id] if lesion_id in voi_df_ven.index else None
			eq_voi = voi_df_eq.loc[lesion_id] if lesion_id in voi_df_eq.index else None

			try:
				cropped_img, coords = _extract_voi(img, copy.deepcopy(voi_row), C.dims, ven_voi=ven_voi, eq_voi=eq_voi)
				cropped_img = tr.normalize_intensity(cropped_img, max_intensity=1, min_intensity=-1)
			except:
				raise ValueError(lesion_id)

			np.save(join(C.crops_dir, cls, lesion_id), cropped_img)
			small_voi_df.loc[lesion_id] = [accnum, cls] + coords

		if img_num % 20 == 0:
			print(".", end="")
			small_voi_df.to_csv(C.small_voi_path)
	
	small_voi_df.to_csv(C.small_voi_path)

@drm.autofill_cls_arg
def save_unaugmented_set(cls=None, accnums=None, lesion_ids=None, custom_vois=None, lesion_ratio=None, overwrite=True):
	"""Save unaugmented lesion images. Overwrites without checking."""

	C = config.Config()
	small_voi_df = pd.read_csv(C.small_voi_path, index_col=0)

	if not exists(join(C.unaug_dir, cls)):
		os.makedirs(join(C.unaug_dir, cls))

	if lesion_ids is None:
		if accnums is None:
			lesion_ids = [x[:-4] for x in os.listdir(C.crops_dir + cls) if x[:-4] in small_voi_df.index]
		else:
			lesion_ids = [x[:-4] for x in os.listdir(C.crops_dir + cls) if x[:x.find('_')] in accnums]

	for ix, lesion_id in enumerate(lesion_ids):
		if not overwrite and exists(join(C.unaug_dir, cls, lesion_id)):
			continue
		if custom_vois is None:
			try:
				unaug_img = _resize_img(join(C.crops_dir, cls, lesion_id + ".npy"),
							_get_voi_coords(small_voi_df.loc[lesion_id]))
			except:
				continue
		else:
			unaug_img = _resize_img(join(C.crops_dir, cls, lesion_id + ".npy"), custom_vois[ix], lesion_ratio)

		np.save(join(C.unaug_dir, cls, lesion_id), unaug_img)

@drm.autofill_cls_arg
def save_augmented_set(cls=None, accnums=None, num_cores=None, overwrite=True):
	"""Augment all images in cls using CPU parallelization.
	Overwrite can be an int, in which case it will create
	augmented samples enumerated starting at that number."""

	C = config.Config()
	small_voi_df = pd.read_csv(C.small_voi_path, index_col=0)

	if not exists(join(C.aug_dir, cls)):
		os.makedirs(join(C.aug_dir, cls))

	if accnums is not None:
		lesion_ids = [x[:-4] for x in os.listdir(C.crops_dir + cls) if x[:x.find('_')] in accnums]
		for lesion_id in lesion_ids:
			_save_augmented_img(lesion_id, cls, _get_voi_coords(small_voi_df.loc[lesion_id]), overwrite=overwrite)

	else:
		lesion_ids = [x[:-4] for x in os.listdir(C.crops_dir + cls) if x[:-4] in small_voi_df.index]

		t = time.time()
		if num_cores is None:
			num_cores = multiprocessing.cpu_count() - 1

		if num_cores > 1:
			Parallel(n_jobs=num_cores)(delayed(_save_augmented_img)(lesion_id, cls,
				_get_voi_coords(small_voi_df.loc[lesion_id]),
				overwrite=overwrite) for lesion_id in lesion_ids)
		else:
			for lesion_id in lesion_ids:
				_save_augmented_img(lesion_id, cls, _get_voi_coords(small_voi_df.loc[lesion_id]), overwrite=overwrite)

		print(cls, time.time()-t)


#####################################
### Public Subroutines
#####################################

def get_scale_ratios(voi, final_dims=None, lesion_ratio=None):
	"""Based on a voi in an image and the final dimensions desired,
	determine how much the image needs to be scaled so that the voi fits
	in the desired dimensions, with optional padding.

	lesion_ratio of .8 means that the width of the lesion is 80% of the image width.
	lesion_ratio of 1 means no padding is added."""

	C = config.Config()

	if final_dims is None:
		final_dims = C.dims
	if lesion_ratio is None:
		lesion_ratio = C.lesion_ratio

	x1 = voi[0]
	x2 = voi[1]
	y1 = voi[2]
	y2 = voi[3]
	z1 = voi[4]
	z2 = voi[5]
	dx = x2 - x1
	dy = y2 - y1
	dz = z2 - z1
	
	scale_ratios = [final_dims[0]/dx * lesion_ratio, final_dims[1]/dy * lesion_ratio, final_dims[2]/dz * lesion_ratio]

	return scale_ratios

#####################################
### Subroutines
#####################################

def _draw_bbox(img_slice, voi):
	"""Draw a colored box around the voi of an image slice showing how it would be cropped."""
	C = config.Config()

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

def _augment_img(img, voi, num_samples, add_reflections=False, save_name=None, overwrite=True):
	"""For rescaling an img to final_dims while scaling to make sure the image contains the voi.
	add_reflections and save_name cannot be used simultaneously"""
	C = config.Config()
	if type(overwrite) == int:
		start=overwrite
	else:
		start=0

	final_dims = C.dims
	
	buffer1 = C.lesion_ratio-.1
	buffer2 = C.lesion_ratio+.1
	scale_ratios = get_scale_ratios(voi, lesion_ratio=1)

	aug_imgs = []
	
	for img_num in range(start, num_samples):
		scales = [random.uniform(scale_ratios[0]*buffer1, scale_ratios[0]*buffer2),
				 random.uniform(scale_ratios[1]*buffer1, scale_ratios[1]*buffer2),
				 random.uniform(scale_ratios[2]*buffer1, scale_ratios[2]*buffer2)]
		
		angle = random.randint(0, 359)
		
		trans = [random.randint(-C.translate[0], C.translate[0]),
				 random.randint(-C.translate[1], C.translate[1]),
				 random.randint(-C.translate[2], C.translate[2])]
		
		flip = [random.choice([-1, 1]), random.choice([-1, 1]), random.choice([-1, 1])]

		temp_img = tr.scale3d(img, scales)
		temp_img = tr.rotate(temp_img, angle)

		crops = [temp_img.shape[i] - final_dims[i] for i in range(3)]
	
		for i in range(3):
			assert crops[i]>=0

		#temp_img = add_noise(temp_img)

		temp_img = tr.offset_phases(temp_img, max_offset=2, max_z_offset=1)
		temp_img = temp_img[crops[0]//2 *flip[0] + trans[0] : -crops[0]//2 *flip[0] + trans[0] : flip[0],
							crops[1]//2 *flip[1] + trans[1] : -crops[1]//2 *flip[1] + trans[1] : flip[1],
							crops[2]//2 *flip[2] + trans[2] : -crops[2]//2 *flip[2] + trans[2] : flip[2], :]
		
		temp_img[:,:,:,0] = temp_img[:,:,:,0] * random.gauss(1,C.intensity_scaling[0]) + random.gauss(0,C.intensity_scaling[1])
		temp_img[:,:,:,1] = temp_img[:,:,:,1] * random.gauss(1,C.intensity_scaling[0]) + random.gauss(0,C.intensity_scaling[1])
		temp_img[:,:,:,2] = temp_img[:,:,:,2] * random.gauss(1,C.intensity_scaling[0]) + random.gauss(0,C.intensity_scaling[1])

		if save_name is None:
			aug_imgs.append(temp_img)
		else:
			np.save(save_name + "_" + str(img_num), temp_img)
		
		if add_reflections:
			aug_imgs.append(tr.generate_reflected_img(temp_img))
	
	return aug_imgs

def _save_augmented_img(lesion_id, cls, voi_coords, overwrite=True):
	"""Written in a way to allow partial overwriting"""
	C = config.Config()

	if lesion_id.find('.') != -1:
		lesion_id = lesion_id[:-4]

	if not overwrite and exists(join(C.aug_dir, cls, lesion_id + "_0.npy")):
		return

	img = np.load(join(C.crops_dir, cls, lesion_id + ".npy"))
	if C.pre_scale > 0:
		img = tr.normalize_intensity(img, 1., -1., fraction=C.pre_scale)
	_augment_img(img, voi_coords, num_samples=C.aug_factor, save_name=join(C.aug_dir, cls, lesion_id), overwrite=overwrite)

def _get_voi_coords(small_voi_df_row):
	return small_voi_df_row[["x1","x2","y1","y2","z1","z2"]].values

def _resize_img(img_path, voi, lesion_ratio=None):
	"""For rescaling an img to final_dims while scaling to make sure the image contains the voi.
	Do not reuse img
	"""
	C = config.Config()
	scale_ratios = get_scale_ratios(voi, lesion_ratio=lesion_ratio)

	img = np.load(img_path)
	img = tr.scale3d(img, scale_ratios)

	crop = [img.shape[i] - C.dims[i] for i in range(3)]

	for i in range(3):
		assert crop[i]>=0
	
	img = img[crop[0]//2:-crop[0]//2, crop[1]//2:-crop[1]//2, crop[2]//2:-crop[2]//2, :]

	if C.pre_scale > 0:
		img = tr.normalize_intensity(img, 1., -1., fraction=C.pre_scale)

	return img

def _extract_voi(img, voi, min_dims, ven_voi=None, eq_voi=None):
	"""Input: image, a voi to center on, and the min dims of the unaugmented img.
	Outputs loosely cropped voi-centered image and coords of the voi within this loosely cropped image.
	"""
	
	def _align_phases(img, voi, ch_voi, ch):
		"""Align phases based on centers along each axis"""

		img_ch = copy.deepcopy(img[...,ch])
		dx = ((ch_voi["x1"] + ch_voi["x2"]) - (voi["x1"] + voi["x2"])) // 2
		dy = ((ch_voi["y1"] + ch_voi["y2"]) - (voi["y1"] + voi["y2"])) // 2
		dz = ((ch_voi["z1"] + ch_voi["z2"]) - (voi["z1"] + voi["z2"])) // 2
		
		pad = int(max(abs(dx), abs(dy), abs(dz)))+1
		img_ch = np.pad(img_ch, pad, 'constant')[pad+dx:-pad+dx, pad+dy:-pad+dy, pad+dz:-pad+dz]
		
		if ch == 1:
			return np.stack([img[...,0], img_ch, img[...,2]], -1)
		elif ch == 2:
			return np.stack([img[...,0], img[...,1], img_ch], -1)

	img = copy.deepcopy(img)
	
	x1 = voi['x1']
	x2 = voi['x2']
	y1 = img.shape[1]-voi['y2']
	y2 = img.shape[1]-voi['y1']
	z1 = voi['z1']
	z2 = voi['z2']
	dx = x2 - x1
	dy = y2 - y1
	dz = z2 - z1
	assert dx > 0 and dy > 0 and dz > 0, "Bad voi for " + str(voi["accnum"])
	
	# align all phases
	if ven_voi is not None:
		img = _align_phases(img, voi, ven_voi, 1)
	if eq_voi is not None:
		img = _align_phases(img, voi, eq_voi, 2)

	#padding around lesion
	def func(i,x):
		tmp = max(min_dims[i], x) * 2*math.sqrt(2) - x
		#if tmp > 150:
		#	raise ValueError(voi["accnum"] + " has a large tumor. Lower resolution needed.")
		return max(tmp, 50)
	xpad = func(0,dx)#max(min_dims[0], dx) * 2*math.sqrt(2) - dx
	ypad = func(1,dy)#max(min_dims[1], dy) * 2*math.sqrt(2) - dy
	zpad = func(2,dz)#max(min_dims[2], dz) * 2*math.sqrt(2) - dz
	
	#padding in case voi is too close to edge
	side_padding = math.ceil(max(xpad, ypad, zpad) / 2)
	pad_img = []
	for ch in range(img.shape[-1]):
		pad_img.append(np.pad(img[...,ch], side_padding, 'constant'))
	pad_img = np.stack(pad_img, -1)
	
	#choice of ceil/floor needed to make total padding amount correct
	x1 += side_padding - math.floor(xpad/2)
	x2 += side_padding + math.ceil(xpad/2)
	y1 += side_padding - math.floor(ypad/2)
	y2 += side_padding + math.ceil(ypad/2)
	z1 += side_padding - math.floor(zpad/2)
	z2 += side_padding + math.ceil(zpad/2)
	
	new_voi = [xpad//2, dx + xpad//2,
			   ypad//2, dy + ypad//2,
			   zpad//2, dz + zpad//2]
	
	for i in new_voi:
		assert i>=0
		
	return pad_img[x1:x2, y1:y2, z1:z2], [int(x) for x in new_voi]

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