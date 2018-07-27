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
import os
import random
import time
from os.path import *

import numpy as np
import pandas as pd
from scipy.misc import imsave
from skimage.transform import rescale, resize

import config
import niftiutils.helper_fxns as hf
import niftiutils.masks as masks
import niftiutils.transforms as tr
import niftiutils.visualization as vis

importlib.reload(hf)
C = config.Config()

#####################################
### QC methods
#####################################

def plot_check(accnum=None, normalize=None, cropped=False):
	C = config.Config()
	cls = pd.read_csv(C.small_voi_path, index_col=0).loc[lesion_id, "cls"].values[0]
	img = np.load(join(C.full_img_dir, cls, lesion_id[:lesion_id.find('_')] + ".npy"))
	vis.draw_slices(img, normalize=normalize)

	return img

def make_pngs(cls=None, lesion_ids=None, save_dir=None, normalize=None, fixed_width=100, fn_prefix="", fn_suffix=None, separate_by_cls=True):
	accnums = [basename(fn[:-4]) for fn in glob.glob(join(C.full_img_dir, "*.npy")) if not fn.endswith("_seg.npy")]

	for accnum in accnums:
		img = np.load(join(C.full_img_dir, accnum+".npy"))
		M = np.load(join(C.full_img_dir, accnum+"_seg.npy"))

#####################################
### Data Creation
#####################################

def save_segs(accnums=None, downsample=None, slice_shift=0, target_depth=None):
	"""Save all segmentations as numpy."""
	#input_df = pd.read_excel(accnum_xls_path,
	#			 sheetname="Prelim Analysis Patients", index_col=0, parse_cols="A,J")
	#accnums = np.array([list(input_df[input_df["Category"] == category].index.astype(str)) for category in C.sheetnames]).flatten()

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
		z = target_depth if target_depth is not None else M.shape[-1]
		for path in M_paths[1:]:
			#try:
			M += masks.get_mask(path, img_path=join(load_dir, "nii_dir", "20s.nii.gz"))
			#except:
			#	os.remove(path)
			#	os.remove(path[:-4]+".ics")
		M = M[...,slice_shift:slice_shift + z]
		if downsample is not None:
			M = tr.scale3d(M, [1/downsample, 1/downsample, 1]) > .5
		np.save(join(C.full_img_dir, accnum+"_tumorseg.npy"), M)

		M_path = join(load_dir, 'Segs', 'liver.ids')
		if not exists(M_path):
			continue
		M = masks.get_mask(M_path, img_path=join(load_dir, "nii_dir", "20s.nii.gz"))
		M = M[...,slice_shift:slice_shift + z]
		if downsample is not None:
			M = tr.scale3d(M, [1/downsample, 1/downsample, 1]) > .5
		np.save(join(C.full_img_dir, accnum+"_liverseg.npy"), M)

def crop_seg(accnum, coords):
	"""Output the ground truth segmentation for an accnum."""
	importlib.reload(hf)
	C = config.Config()
	M = np.load(join(C.full_img_dir, accnum+"_seg.npy"))
	sl = [slice(coords[i], coords[i+1]) for i in [0,2,4]]
	return M[sl]
