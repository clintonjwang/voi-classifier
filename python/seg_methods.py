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
import niftiutils.helper_fxns as hf
import niftiutils.masks as masks
import niftiutils.transforms as tr
import niftiutils.visualization as vis

#####################################
### QC methods
#####################################

def plot_check(accnum=None, normalize=None, cropped=False):
	C = config.Config()
	cls = pd.read_csv(C.small_voi_path, index_col=0).loc[lesion_id, "cls"].values[0]
	img = np.load(join(C.full_img_dir, cls, lesion_id[:lesion_id.find('_')] + ".npy"))
	vis.draw_slices(img, normalize=normalize)

	return img

def xref_dirs_with_excel(cls=None, fix_inplace=True):
	pass

def reset_accnum(accnum):
	pass

def load_accnum(cls=None, accnums=None, augment=True):
	pass

def make_pngs(cls=None, lesion_ids=None, save_dir=None, normalize=None, fixed_width=100, fn_prefix="", fn_suffix=None, separate_by_cls=True):
	accnums = [basename(fn[:-4]) for fn in glob.glob(join(C.full_img_dir, "*.npy")) if not fn.endswith("_seg.npy")]

	for accnum in accnums:
		img = np.load(join(C.full_img_dir, accnum+".npy"))
		M = np.load(join(C.full_img_dir, accnum+"_seg.npy"))

def make_dcms(cls=None, lesion_ids=None, save_dir=None):
	pass

#####################################
### Data Creation
#####################################

def save_segs(accnums=None):
	"""Save all segmentations as numpy."""
	importlib.reload(hf)
	C = config.Config()

	if accnums is None:
		accnums = [basename(fn[:-4]) for fn in glob.glob(join(C.full_img_dir, "*.npy")) if \
				not fn.endswith("seg.npy")]

	for accnum in accnums:
		load_dir = join(C.dcm_dirs[0], accnum)
		if not exists(join(load_dir, "nii_dir", "20s.nii.gz")):
			continue

		M_paths = glob.glob(join(load_dir, 'Segs', 'tumor_20s_*.ids'))
		if len(M_paths) == 0:
			continue
		M = masks.get_mask(M_paths[0], img_path=join(load_dir, "nii_dir", "20s.nii.gz"))
		for path in M_paths[1:]:
			try:
				M += masks.get_mask(path, img_path=join(load_dir, "nii_dir", "20s.nii.gz"))
			except:
				os.remove(path)
				os.remove(path[:-4]+".ics")
		if np.product(M.shape) > C.max_size:
			M = tr.scale3d(M, [.5]*3) > .5
		np.save(join(C.full_img_dir, accnum+"_tumorseg.npy"), M)

		M_path = join(load_dir, 'Segs', 'liver.ids')
		if not exists(M_path):
			continue
		M = masks.get_mask(M_path, img_path=join(load_dir, "nii_dir", "20s.nii.gz"))
		if np.product(M.shape) > C.max_size:
			M = tr.scale3d(M, [.5]*3) > .5
		np.save(join(C.full_img_dir, accnum+"_liverseg.npy"), M)

def crop_seg(accnum, coords):
	"""Output the ground truth segmentation for an accnum."""
	importlib.reload(hf)
	C = config.Config()
	M = np.load(join(C.full_img_dir, accnum+"_seg.npy"))
	sl = [slice(coords[i], coords[i+1]) for i in [0,2,4]]
	return M[sl]

def transform_masks():
	pass

def augmented_seg(accnum, coords):
	"""Use 3D affine matrices"""
	pass


#####################################
### Subroutines
#####################################

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

def _extract_voi(img, voi, min_dims):
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
