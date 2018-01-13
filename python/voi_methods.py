"""
Converts a nifti file to a numpy array.
Accepts either a single nifti file or a folder of niftis as the input argument.

Usage:
	python cnn_builder.py

Author: Clinton Wang, E-mail: `clintonjwang@gmail.com`, Github: `https://github.com/clintonjwang/voi-classifier`
"""

import ast
import config
import copy
import dr_methods as drm
import helper_fxns as hf
import math
import numpy as np
import os
import pandas as pd
import random
import time
import transforms as tr
from joblib import Parallel, delayed
import multiprocessing
from scipy.misc import imsave
from skimage.transform import rescale, resize

#####################################
### QC methods
#####################################

@drm.autofill_cls_arg
def xref_dirs_with_excel(cls=None, fix_inplace=True):
	"""Make sure the image directories have all the images expected based on the config settings
	(aug factor and run number) and VOI spreadsheet contents.
	If fix_inplace is True, reload any mismatched acc_nums."""

	C = config.Config()

	small_voi_df = pd.read_csv(C.small_voi_path)

	print("Checking", cls)
	bad_acc_nums = []

	index = C.cls_names.index(cls)
	acc_num_df = C.sheetnames[index]
	df = pd.read_excel(C.xls_name, C.sheetnames[index])
	voi_df = pd.read_csv(C.art_voi_path)
	xls_accnums = list(df[df['Run'] <= C.run_num]['Patient E Number'].astype(str))
	unique, counts = np.unique(xls_accnums, return_counts=True)
	xls_set = set(unique)
	xls_cnts = dict(zip(unique, counts))

	# Check for loaded images
	for acc_num in xls_set:
		if not os.path.exists(C.full_img_dir + "\\" + cls + "\\" + acc_num + ".npy"):
			print(acc_num, "is contained in the spreadsheet but has no loaded image in", C.full_img_dir)

			bad_acc_nums.append(acc_num)

	# Check for small_voi_df
	acc_nums = list(small_voi_df[small_voi_df["cls"] == cls]["acc_num"])
	unique, counts = np.unique(acc_nums, return_counts=True)
	diff = xls_set.difference(unique)
	if len(diff) > 0:
		print(diff, "are contained in the spreadsheet but not in small_voi_df.")
		bad_acc_nums += list(diff)

	diff = set(unique).difference(xls_set)
	if len(diff) > 0:
		print(diff, "are contained in small_voi_df but not the spreadsheet.")
		bad_acc_nums += list(diff)

	overlap = xls_set.intersection(unique)
	voi_cnts = dict(zip(unique, counts))
	for acc_num in overlap:
		if voi_cnts[acc_num] != xls_cnts[acc_num]:
			print("Mismatch in number of lesions in the spreadsheet vs small_voi_df for", acc_num)
			bad_acc_nums.append(acc_num)

	# Check rough cropped lesions
	acc_nums = [fn[:fn.find("_")] for fn in os.listdir(C.crops_dir + "\\" + cls)]
	unique, counts = np.unique(acc_nums, return_counts=True)
	diff = xls_set.difference(unique)
	if len(diff) > 0:
		print(diff, "are contained in the spreadsheet but not in", C.crops_dir)
		bad_acc_nums += list(diff)
	diff = set(unique).difference(xls_set)
	if len(diff) > 0:
		print(diff, "are contained in", C.crops_dir, "but not the spreadsheet.")
		bad_acc_nums += list(diff)

	overlap = xls_set.intersection(unique)
	voi_cnts = dict(zip(unique, counts))
	for acc_num in overlap:
		if voi_cnts[acc_num] != xls_cnts[acc_num]:
			print("Mismatch in number of lesions in the spreadsheet vs", C.crops_dir, "for", acc_num)
			bad_acc_nums.append(acc_num)

	# Check unaugmented lesions
	acc_nums = [fn[:fn.find("_")] for fn in os.listdir(C.orig_dir + "\\" + cls)]
	unique, counts = np.unique(acc_nums, return_counts=True)
	diff = xls_set.difference(unique)
	if len(diff) > 0:
		print(diff, "are contained in the spreadsheet but not in", C.orig_dir)
		bad_acc_nums += list(diff)
	diff = set(unique).difference(xls_set)
	if len(diff) > 0:
		print(diff, "are contained in", C.orig_dir, "but not the spreadsheet.")
		bad_acc_nums += list(diff)

	overlap = xls_set.intersection(unique)
	voi_cnts = dict(zip(unique, counts))
	for acc_num in overlap:
		if voi_cnts[acc_num] != xls_cnts[acc_num]:
			print("Mismatch in number of lesions in the spreadsheet vs", C.orig_dir, "for", acc_num)
			bad_acc_nums.append(acc_num)


	# Check augmented lesions
	lesion_ids_folder = set([fn[:fn.rfind("_")] for fn in os.listdir(C.aug_dir + "\\" + cls)])
	lesion_ids_df = set(voi_df.loc[voi_df["cls"] == cls,"id"].values)
	diff = lesion_ids_df.difference(lesion_ids_folder)
	if len(diff) > 0:
		print(diff, "are contained in voi_df but not in", C.aug_dir)
		bad_acc_nums += [x[:x.find('_')] for x in diff]
	diff = lesion_ids_folder.difference(lesion_ids_df)
	if len(diff) > 0:
		print(diff, "are contained in", C.aug_dir, "but not in voi_df.")
		bad_acc_nums += [x[:x.find('_')] for x in diff]

	# Fix lesions
	if fix_inplace:
		print("Reloading", set(bad_acc_nums))
		for acc_num in set(bad_acc_nums):
			reload_accnum(acc_num, cls)

def reload_accnum(acc_num, cls, augment=True, overwrite=True):
	"""Reloads cropped, scaled and augmented images. Updates voi_dfs and small_vois accordingly."""
	drm.load_vois_batch(cls, acc_nums=[acc_num], overwrite=overwrite)

	if overwrite:
		remove_lesion_from_folders(cls, acc_num)

	extract_vois(cls, [acc_num])
	save_unaugment_set(cls, acc_num)
	if augment:
		parallel_augment(cls, acc_num)


@drm.autofill_cls_arg
def save_vois_as_imgs(cls=None, lesion_ids=None, save_dir=None, normalize=[-1,1], rescale_factor=3):
	"""Save all voi images as jpg."""
	C = config.Config()

	if save_dir is None:
		save_dir = os.path.join(C.vois_dir, cls)
	if not os.path.exists(save_dir):
		os.makedirs(save_dir)

	if lesion_ids is not None:
		fns = [acc_num+".npy" for acc_num in lesion_ids]
	else:
		fns = os.listdir(os.path.join(C.orig_dir, cls))

	for fn in fns:
		img = np.load(C.orig_dir + cls + "\\" + fn)

		img_slice = img[:,:, img.shape[2]//2, :].astype(float)

		if normalize is not None:
			img_slice[0,0,:]=normalize[0]
			img_slice[0,-1,:]=normalize[1]
			
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
			fn_suffix = " (%s)" % cls

		imsave("%s\\%s%s.png" % (save_dir, fn[:-4], fn_suffix), rescale(ret, rescale_factor, mode='constant'))

@drm.autofill_cls_arg
def save_imgs_with_bbox(cls=None, lesion_ids=None, fn_suffix=None, save_dir=None, normalize=None, fixed_width=100):

	C = config.Config()

	small_voi_df = pd.read_csv(C.small_voi_path)

	if not os.path.exists(save_dir):
		os.makedirs(save_dir)
		
	if lesion_ids is not None:
		fns = [lesion_id+".npy" for lesion_id in lesion_ids]
	else:
		fns = os.listdir(os.path.join(C.crops_dir, cls))
	
	for fn in fns:
		img_fn = fn[:fn.find('_')] + ".npy"
		cls = small_voi_df.loc[small_voi_df["id"] == fn[:-4], "cls"].values[0]

		img = np.load(C.crops_dir + cls + "\\" + fn)
		img_slice = img[:,:, img.shape[2]//2, :].astype(float)
		#for ch in range(img_slice.shape[-1]):
		#	img_slice[:, :, ch] *= 255/np.amax(img_slice[:, :, ch])
		if normalize is not None:
			img_slice[0,0,:]=normalize[0]
			img_slice[0,-1,:]=normalize[1]

		img_slice = np.stack([img_slice, img_slice, img_slice], axis=2)
		
		img_slice = _draw_bbox(img_slice, vm._get_coords(small_voi_df[small_voi_df["id"] == fn[:-4]]))
			
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
			fn_suffix = " (%s)" % cls

		if fixed_width is not None:
			imsave("%s\\%s%s.png" % (save_dir, fn[:-4], fn_suffix), resize(ret, [fixed_width*3, fixed_width]))
		else:
			imsave("%s\\%s%s.png" % (save_dir, fn[:-4], fn_suffix), ret)

def remove_lesion_from_folders(cls=None, acc_num=None, lesion_id=None, include_augment=True):
	"""Can either specify both cls and acc_num or just lesion_id"""

	C = config.Config()

	if lesion_id is None:
		for fn in os.listdir(C.crops_dir + cls):
			if fn.startswith(acc_num):
				os.remove(C.crops_dir + cls + "\\" + fn)
		for fn in os.listdir(C.orig_dir + cls):
			if fn.startswith(acc_num):
				os.remove(C.orig_dir + cls + "\\" + fn)
		if include_augment:
			for fn in os.listdir(C.aug_dir + cls):
				if fn.startswith(acc_num):
					os.remove(C.aug_dir + cls + "\\" + fn)
	else:
		small_voi_df = pd.read_csv(C.small_voi_path)
		cls = small_voi_df.loc[small_voi_df["id"] == lesion_id, "cls"].values[0]

		os.remove(os.path.join(C.crops_dir, cls, lesion_id+".npy"))
		os.remove(os.path.join(C.orig_dir, cls, lesion_id+".npy"))
		if include_augment:
			for fn in os.listdir(C.aug_dir + cls):
				if fn.startswith(lesion_id):
					os.remove(C.aug_dir + cls + "\\" + fn)

#####################################
### Data Creation
#####################################

@drm.autofill_cls_arg
def extract_vois(cls=None, acc_nums=None, debug=False, overwrite_df=False):
	"""Produces grossly cropped but unscaled versions of the images.
	This intermediate step makes debugging and visualization much easier.
	drm.load_vois_batch() must be run first to populate the voi_dfs."""

	C = config.Config()
	voi_df_art = pd.read_csv(C.art_voi_path)
	voi_df_ven = pd.read_csv(C.ven_voi_path)
	voi_df_eq = pd.read_csv(C.eq_voi_path)
	#intensity_df = pd.read_csv(C.int_df_path)

	try:
		small_voi_df = pd.read_csv(C.small_voi_path)
		if overwrite_df:
			small_voi_df = small_voi_df[small_voi_df["cls"] != cls]
	except FileNotFoundError:
		small_voi_df = pd.DataFrame(columns=["id", "acc_num", "cls", "coords"])

	if acc_nums is None:
		acc_nums = [x[:-4] for x in os.listdir(C.full_img_dir + "\\" + cls)]

	for img_num, acc_num in enumerate(acc_nums):
		try:
			img = np.load(C.full_img_dir+"\\"+cls+"\\"+acc_num+".npy")
		except Exception as e:
			if debug:
				raise ValueError(e)
			continue

		art_vois = voi_df_art[(voi_df_art["Filename"] == acc_num+".npy") & (voi_df_art["cls"] == cls)]

		small_voi_df = _rm_lesion_from_voi_df(small_voi_df, acc_num, cls)

		# iterate over each voi in that image
		for voi_row in art_vois.iterrows():
			ven_voi = voi_df_ven[voi_df_ven["id"] == voi_row[1]["id"]]
			eq_voi = voi_df_eq[voi_df_eq["id"] == voi_row[1]["id"]]

			cropped_img, small_voi = _extract_voi(img, copy.deepcopy(voi_row[1]), C.dims, ven_voi=ven_voi, eq_voi=eq_voi)
			cropped_img = scale_intensity(cropped_img, 1, max_int=2, keep_min=True) - 1
			#cropped_img = _scale_intensity_df(cropped_img, intensity_df[intensity_df["acc_num"] == img_fn[:img_fn.find('.')]])

			lesion_id = str(voi_row[1]["id"])
			np.save(os.path.join(C.crops_dir, cls, lesion_id), cropped_img)

			small_voi_df = _add_small_voi(small_voi_df, lesion_id, cls, small_voi)

		if img_num % 20 == 0:
			print(".", end="")
	
	small_voi_df.to_csv(C.small_voi_path, index=False)

@drm.autofill_cls_arg
def save_unaugment_set(cls=None, acc_num=None):
	"""Save unaugmented lesion images"""
	C = config.Config()

	small_voi_df = pd.read_csv(C.small_voi_path)

	if acc_num is None:
		acc_num_subset = os.listdir(C.crops_dir + cls)
	else:
		acc_num_subset = [x for x in os.listdir(C.crops_dir + cls) if x.startswith(acc_num)]

	for fn in acc_num_subset:
		img = np.load(C.crops_dir + cls + "\\" + fn)

		try:
			unaug_img = _resize_img(img, _get_coords(small_voi_df[small_voi_df["id"] == fn[:-4]]))
		except Exception as e:
			if len(classes) == 1:
				raise ValueError(e)
			print(cls, fn)
			continue
		np.save(C.orig_dir + cls + "\\" + fn, unaug_img)

@drm.autofill_cls_arg
def parallel_augment(cls=None, acc_num=None, num_cores=None, overwrite=True):
	"""Augment all images in cls using CPU parallelization"""
	C = config.Config()
	small_voi_df = pd.read_csv(C.small_voi_path)

	if acc_num is not None:
		fn_subset = [x for x in os.listdir(C.crops_dir + cls) if x.startswith(acc_num)]
		for fn in fn_subset:
			_save_augmented_img(fn, cls, _get_coords(small_voi_df[small_voi_df["id"] == fn[:-4]]), C)

	else:
		t = time.time()
		if num_cores is None:
			num_cores = multiprocessing.cpu_count()

		if num_cores > 1:
			Parallel(n_jobs=num_cores)(delayed(_save_augmented_img)(fn, cls,
				_get_coords(small_voi_df[small_voi_df["id"] == fn[:-4]]),
				C, overwrite=overwrite) for fn in os.listdir(C.crops_dir + cls))
		else:
			for fn in os.listdir(C.crops_dir + cls):
				_save_augmented_img(fn, cls, small_vois[fn[:-4]], C, overwrite=overwrite)

		print(cls, time.time()-t)


#####################################
### Public Subroutines
#####################################

def get_scale_ratios(voi, final_dims=None, lesion_ratio=None):
	"""Based on a voi in an image and the final dimensions desired,
	determine how much the image needs to be scaled so that the voi fits
	in the desired dimensions, with optional padding.
	Padding of 1 means no padding is added. <1 adds padding"""

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

def scale_intensity(img, fraction=.5, max_int=2, keep_min=False):
	"""Scales each channel intensity separately.
	Assumes original is within a -1 to 1 scale.
	When fraction is 1, force max to be 1 and min to be -1.
	When fraction is 0, rescale within a -1 to 1 scale."""

	img = img.astype(float)
	fraction = min(max(fraction, 0), 1)

	for ch in range(img.shape[3]):
		ch_max = np.amax(img[:,:,:,ch])
		ch_min = np.amin(img[:,:,:,ch])
		target_max = max_int * fraction + ch_max * (1-fraction)
		if keep_min:
			target_min = ch_min
		else:
			target_min = -max_int * fraction + ch_min * (1-fraction)
		img[:,:,:,ch] = img[:,:,:,ch] - ch_min
		img[:,:,:,ch] = img[:,:,:,ch] * (target_max - target_min) / (ch_max - ch_min) + target_min

	return img

#####################################
### Subroutines
#####################################

def _augment_img(img, voi, num_samples, translate=[2,2,1], add_reflections=False, save_name=None, intensity_scaling=[.05,.05], overwrite=True):
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

		temp_img = tr.scale3d(img, scales)
		temp_img = tr.rotate(temp_img, angle)
		
		if translate is not None:
			trans = [random.randint(-translate[0], translate[0]),
					 random.randint(-translate[1], translate[1]),
					 random.randint(-translate[2], translate[2])]
		else:
			trans = [0,0,0]
		
		flip = [random.choice([-1, 1]), random.choice([-1, 1]), random.choice([-1, 1])]

		crops = [temp_img.shape[i] - final_dims[i] for i in range(3)]
	
		for i in range(3):
			assert crops[i]>=0

		#temp_img = add_noise(temp_img)

		temp_img = tr.offset_phases(temp_img, max_offset=2, max_z_offset=1)
		temp_img = temp_img[crops[0]//2 *flip[0] + trans[0] : -crops[0]//2 *flip[0] + trans[0] : flip[0],
							crops[1]//2 *flip[1] + trans[1] : -crops[1]//2 *flip[1] + trans[1] : flip[1],
							crops[2]//2 *flip[2] + trans[2] : -crops[2]//2 *flip[2] + trans[2] : flip[2], :]
		
		#temp_img = scale_intensity(temp_img, C.intensity_local_frac, max_int=1.5)#random.gauss(C.intensity_local_frac, 0.1))
		temp_img[:,:,:,0] = temp_img[:,:,:,0] * random.gauss(1,intensity_scaling[0]) + random.gauss(0,intensity_scaling[1])
		temp_img[:,:,:,1] = temp_img[:,:,:,1] * random.gauss(1,intensity_scaling[0]) + random.gauss(0,intensity_scaling[1])
		temp_img[:,:,:,2] = temp_img[:,:,:,2] * random.gauss(1,intensity_scaling[0]) + random.gauss(0,intensity_scaling[1])

		if save_name is None:
			aug_imgs.append(temp_img)
		else:
			np.save(save_name + "_" + str(img_num), temp_img)
		
		if add_reflections:
			aug_imgs.append(tr.generate_reflected_img(temp_img))
	
	return aug_imgs

def _save_augmented_img(fn, cls, voi_coords, C, overwrite=True):
	"""Written in a way to allow partial overwriting"""
	if not overwrite and os.path.exists(C.aug_dir + cls + "\\" + fn[:-4] + "_0.npy"):
		return

	img = np.load(C.crops_dir + cls + "\\" + fn)
	_augment_img(img, voi_coords, num_samples=C.aug_factor, translate=[2,2,1],
			save_name=C.aug_dir + cls + "\\" + fn[:-4], intensity_scaling=C.intensity_scaling, overwrite=overwrite)

def _scale_intensity_df(img, intensity_row, min_int=1):
	"""Rescale intensities in img such that the max intensity of the original image has a value of 1. Min intensity is -1."""
	try:
		img = img.astype(float)
		img[:,:,:,0] = (img[:,:,:,0] * 2 / float(intensity_row["art_int"])) - min_int
		img[:,:,:,1] = (img[:,:,:,1] * 2 / float(intensity_row["ven_int"])) - min_int
		img[:,:,:,2] = (img[:,:,:,2] * 2 / float(intensity_row["eq_int"])) - min_int
	except:
		raise ValueError("intensity_row is probably missing")

	return img

def _get_coords(small_voi_df_row):
	try:
		return ast.literal_eval(small_voi_df_row["coords"].values[0])
	except:
		return small_voi_df_row["coords"].values[0]

def _resize_img(img, voi, C=None):
	"""For rescaling an img to final_dims while scaling to make sure the image contains the voi.
	Do not reuse img
	"""
	if C is None:
		C = config.Config()

	scale_ratios = get_scale_ratios(voi)

	img = tr.scale3d(img, scale_ratios)

	crop = [img.shape[i] - C.dims[i] for i in range(3)]

	for i in range(3):
		assert crop[i]>=0
	
	img = img[crop[0]//2:-crop[0]//2, crop[1]//2:-crop[1]//2, crop[2]//2:-crop[2]//2, :]

	return img

def _add_small_voi(small_voi_df, lesion_id, cls, coords):
	"""Add a row to small_voi_df."""

	if len(small_voi_df) == 0:
		i = 0
	else:
		i = small_voi_df.index[-1]+1
	
	small_voi_df.loc[i] = [lesion_id, lesion_id[:lesion_id.find("_")], cls, coords]

	return small_voi_df

def _rm_lesion_from_voi_df(small_voi_df, acc_num, cls=None):
	"""Remove any elements in small_voi_df with the given acc_num. Limit to cls if specified."""

	if cls is not None:
		small_voi_df = small_voi_df[~((small_voi_df["acc_num"] == acc_num) & (small_voi_df["cls"] == cls))]
	else:
		small_voi_df = small_voi_df[~(small_voi_df["acc_num"] == acc_num)]

	return small_voi_df

def _extract_voi(img, voi, min_dims, ven_voi=[], eq_voi=[]):
	"""Input: image, a voi to center on, and the min dims of the unaugmented img.
	Outputs loosely cropped voi-centered image and coords of the voi within this loosely cropped image.
	"""
	
	def _align_phases(img, voi, ven_voi, ch):
		"""Align phases based on centers along each axis"""

		temp_ven = copy.deepcopy(img[:,:,:,ch])
		dx = ((ven_voi["x1"] + ven_voi["x2"]) - (voi["x1"] + voi["x2"])) // 2
		dy = ((ven_voi["y1"] + ven_voi["y2"]) - (voi["y1"] + voi["y2"])) // 2
		dz = ((ven_voi["z1"] + ven_voi["z2"]) - (voi["z1"] + voi["z2"])) // 2
		
		pad = int(max(abs(dx), abs(dy), abs(dz)))+1
		temp_ven = np.pad(temp_ven, pad, 'constant')[pad+dx:-pad+dx, pad+dy:-pad+dy, pad+dz:-pad+dz]
		
		if ch == 1:
			return np.stack([img[:,:,:,0], temp_ven, img[:,:,:,2]], axis=3)
		elif ch == 2:
			return np.stack([img[:,:,:,0], img[:,:,:,1], temp_ven], axis=3)

	temp_img = copy.deepcopy(img)
	
	x1 = voi['x1']
	x2 = voi['x2']
	y1 = voi['y1']
	y2 = voi['y2']
	z1 = voi['z1']
	z2 = voi['z2']
	dx = x2 - x1
	dy = y2 - y1
	dz = z2 - z1
	assert dx > 0, "Bad voi for " + str(voi["id"])
	assert dy > 0, "Bad voi for " + str(voi["id"])
	assert dz > 0, "Bad voi for " + str(voi["id"])
	
	# align all phases
	if len(ven_voi) > 0:
		ven_voi = ven_voi.iloc[0]
		temp_img = _align_phases(temp_img, voi, ven_voi, 1)
		
	if len(eq_voi) > 0:
		eq_voi = eq_voi.iloc[0]
		temp_img = _align_phases(temp_img, voi, eq_voi, 2)

	#padding around lesion
	xpad = max(min_dims[0], dx) * 2*math.sqrt(2) - dx
	ypad = max(min_dims[1], dy) * 2*math.sqrt(2) - dy
	zpad = max(min_dims[2], dz) * 2*math.sqrt(2) - dz
	
	#padding in case voi is too close to edge
	side_padding = math.ceil(max(xpad, ypad, zpad) / 2)
	pad_img = []
	for ch in range(temp_img.shape[-1]):
		pad_img.append(np.pad(temp_img[:,:,:,ch], side_padding, 'constant'))
	pad_img = np.stack(pad_img, axis=3)
	
	assert xpad > 0
	assert ypad > 0
	assert zpad > 0
	
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
		
	return pad_img[x1:x2, y1:y2, z1:z2, :], [int(x) for x in new_voi]

if __name__ == '__main__':
	import doctest
	doctest.testmod()