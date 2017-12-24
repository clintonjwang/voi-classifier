import copy
import csv
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
from skimage.transform import rescale


def xref_dirs_with_excel(C, classes=None):
	"""Make sure the image directories have all the images expected based on the config settings
	(aug factor and run number) and VOI spreadsheet contents."""

	small_voi_df = pd.read_csv(C.small_voi_path)

	if classes is None:
		classes = C.classes_to_include

	for cls in classes:
		print(cls)

		index = C.cls_names.index(cls)
		acc_num_df = C.sheetnames[index]
		df = pd.read_excel(C.xls_name, C.sheetnames[index])
		xls_accnums = list(df[df['Run'] <= C.run_num]['Patient E Number'].astype(str))
		unique, counts = np.unique(xls_accnums, return_counts=True)
		xls_set = set(unique)
		xls_cnts = dict(zip(unique, counts))

		# Check for loaded images
		for acc_num in xls_set:
			if not os.path.exists(C.full_img_dir + "\\" + cls + "\\" + acc_num + ".npy"):
				print(acc_num, "is contained in the spreadsheet but has no loaded image in", C.full_img_dir)

		# Check for small_voi_df
		acc_nums = list(small_voi_df[small_voi_df["cls"] == cls]["acc_num"])
		unique, counts = np.unique(acc_nums, return_counts=True)
		diff = xls_set.difference(unique)
		if len(diff) > 0:
			print(diff, "are contained in the spreadsheet but not in small_voi_df.")
		diff = set(unique).difference(xls_set)
		if len(diff) > 0:
			print(diff, "are contained in small_voi_df but not the spreadsheet.")

		overlap = xls_set.intersection(unique)
		voi_cnts = dict(zip(unique, counts))
		for acc_num in overlap:
			if voi_cnts[acc_num] != xls_cnts[acc_num]:
				print("Mismatch in number of lesions in the spreadsheet vs small_voi_df for", acc_num)

		# Check rough cropped lesions
		acc_nums = [fn[:fn.find("_")] for fn in os.listdir(C.crops_dir + "\\" + cls)]
		unique, counts = np.unique(acc_nums, return_counts=True)
		diff = xls_set.difference(unique)
		if len(diff) > 0:
			print(diff, "are contained in the spreadsheet but not in", C.crops_dir)
		diff = set(unique).difference(xls_set)
		if len(diff) > 0:
			print(diff, "are contained in", C.crops_dir, "but not the spreadsheet.")

		overlap = xls_set.intersection(unique)
		voi_cnts = dict(zip(unique, counts))
		for acc_num in overlap:
			if voi_cnts[acc_num] != xls_cnts[acc_num]:
				print("Mismatch in number of lesions in the spreadsheet vs", C.crops_dir, "for", acc_num)

		# Check cropped lesions
		acc_nums = [fn[:fn.find("_")] for fn in os.listdir(C.orig_dir + "\\" + cls)]
		unique, counts = np.unique(acc_nums, return_counts=True)
		diff = xls_set.difference(unique)
		if len(diff) > 0:
			print(diff, "are contained in the spreadsheet but not in", C.orig_dir)
		diff = set(unique).difference(xls_set)
		if len(diff) > 0:
			print(diff, "are contained in", C.orig_dir, "but not the spreadsheet.")

		overlap = xls_set.intersection(unique)
		voi_cnts = dict(zip(unique, counts))
		for acc_num in overlap:
			if voi_cnts[acc_num] != xls_cnts[acc_num]:
				print("Mismatch in number of lesions in the spreadsheet vs", C.orig_dir, "for", acc_num)

#####################################
### METHODS FOR DATA AUGMENTATION
#####################################

def parallel_augment(cls, small_vois, C, num_cores=None, overwrite=True):
	"""Augment all images in cls using CPU parallelization"""
	if num_cores is None:
		num_cores = multiprocessing.cpu_count()

	if num_cores > 1:
		Parallel(n_jobs=num_cores)(delayed(save_augmented_img)(fn, cls, small_vois[fn[:-4]], C, overwrite=overwrite) for fn in os.listdir(C.crops_dir + cls))
	else:
		for fn in os.listdir(C.crops_dir + cls):
			save_augmented_img(fn, cls, small_vois[fn[:-4]], C, overwrite=overwrite)

def save_augmented_img(fn, cls, voi_coords, C, overwrite=True):
	if not overwrite and os.path.exists(C.aug_dir + cls + "\\" + fn[:-4] + "_0.npy"):
		return

	img = np.load(C.crops_dir + cls + "\\" + fn)
	augment_img(img, C, voi_coords, num_samples=C.aug_factor, translate=[2,2,1],
			save_name=C.aug_dir + cls + "\\" + fn[:-4], intensity_scaling=C.intensity_scaling, overwrite=overwrite)

def augment_img(img, C, voi, num_samples, translate=None, add_reflections=False, save_name=None, intensity_scaling=[.05,.05], overwrite=True):
	"""For rescaling an img to final_dims while scaling to make sure the image contains the voi.
	add_reflections and save_name cannot be used simultaneously"""
	if type(overwrite) == int:
		start=overwrite
	else:
		start=0

	final_dims = C.dims
	x1 = voi[0]
	x2 = voi[1]
	y1 = voi[2]
	y2 = voi[3]
	z1 = voi[4]
	z2 = voi[5]
	dx = x2 - x1
	dy = y2 - y1
	dz = z2 - z1
	
	buffer1 = C.padding-.1
	buffer2 = C.padding+.1
	scale_ratios = [final_dims[0]/dx, final_dims[1]/dy, final_dims[2]/dz]

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

#####################################
### METHODS FOR OUTPUTTING IMAGES
#####################################

def save_vois_as_imgs(cls, C, num_ch=3, normalize=True, rescale_factor=3, acc_nums=None):
	"""Save all voi images as jpg."""

	os.listdir(C.crops_dir + cls)
	if acc_nums is not None:
		fns = [acc_num+".npy" for acc_num in acc_nums]
	else:
		fns = os.listdir(C.orig_dir + cls)
	save_dir = C.vois_dir + cls
	if not os.path.exists(save_dir):
		os.makedirs(save_dir)

	for fn in fns:
		img = np.load(C.orig_dir + cls + "\\" + fn)

		img_slice = img[:,:, img.shape[2]//2, :].astype(float)

		if normalize:
			img_slice[0,0,:]=-.7
			img_slice[0,-1,:]=.7
			
		ch1 = np.transpose(img_slice[:,::-1,0], (1,0))
		ch2 = np.transpose(img_slice[:,::-1,1], (1,0))
		
		if num_ch == 2:
			ret = np.empty([ch1.shape[0]*2, ch1.shape[1]])
			ret[:ch1.shape[0],:] = ch1
			ret[ch1.shape[0]:,:] = ch2
			
		elif num_ch == 3:
			ch3 = np.transpose(img_slice[:,::-1,2], (1,0))

			ret = np.empty([ch1.shape[0]*3, ch1.shape[1]])
			ret[:ch1.shape[0],:] = ch1
			ret[ch1.shape[0]:ch1.shape[0]*2,:] = ch2
			ret[ch1.shape[0]*2:,:] = ch3

		imsave("%s\\%s (%s).png" % (save_dir, fn[:fn.find('.')], cls), rescale(ret, rescale_factor, mode='constant'))

#####################################
### METHODS FOR IMAGE CROPPING / MANIPULATION
#####################################

def resize_img(img, C, voi):
	"""For rescaling an img to final_dims while scaling to make sure the image contains the voi.
	Do not reuse img
	"""
	final_dims = C.dims
	
	x1 = voi[0]
	x2 = voi[1]
	y1 = voi[2]
	y2 = voi[3]
	z1 = voi[4]
	z2 = voi[5]
	dx = x2 - x1
	dy = y2 - y1
	dz = z2 - z1
	
	padding = C.padding
	#consider making the padding bigger for small lesions
	scale_ratios = [final_dims[0]/dx * padding, final_dims[1]/dy * padding, final_dims[2]/dz * padding]
	
	img = tr.scale3d(img, scale_ratios)
	#if scale_ratio < 0.9: #need to shrink original image to fit
	#    img = tr.scale3d(img, [scale_ratio]*3)
	#elif scale_ratio > 1.4: #need to enlarge original image
	#    img = tr.scale3d(img, [scale_ratio]*3)
	
	crop = [img.shape[i] - final_dims[i] for i in range(3)]

	for i in range(3):
		assert crop[i]>=0
	
	img = img[crop[0]//2:-crop[0]//2, crop[1]//2:-crop[1]//2, crop[2]//2:-crop[2]//2, :]
	#img = scale_intensity(img, C.intensity_local_frac)

	return img

def align_phases(img, voi, ven_voi, ch):
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

def extract_vois(small_voi_df, C, voi_df_art, voi_df_ven, voi_df_eq, intensity_df=None, classes=None, acc_nums=None, debug=False):
	"""Retrieve grossly cropped but unscaled versions of the images."""
	
	t = time.time()

	if classes is None:
		classes = C.classes_to_include

	# iterate over image series
	for cls in classes:
		if acc_nums is None:
			iter_acc_nums = [x[:-4] for x in os.listdir(C.full_img_dir + "\\" + cls)]
		else:
			iter_acc_nums = acc_nums

		for img_num, acc_num in enumerate(iter_acc_nums):
			try:
				img = np.load(C.full_img_dir+"\\"+cls+"\\"+acc_num+".npy")
			except Exception as e:
				if debug:
					raise ValueError(e)
				continue

			art_vois = voi_df_art[(voi_df_art["Filename"] == acc_num+".npy") & (voi_df_art["cls"] == cls)]

			small_voi_df = _filter_small_vois(small_voi_df, acc_num, cls)

			# iterate over each voi in that image
			for voi in art_vois.iterrows():
				ven_voi = voi_df_ven[voi_df_ven["id"] == voi[1]["id"]]
				eq_voi = voi_df_eq[voi_df_eq["id"] == voi[1]["id"]]

				cropped_img, small_voi = _extract_voi(img, copy.deepcopy(voi[1]), C.dims, ven_voi=ven_voi, eq_voi=eq_voi)
				cropped_img = scale_intensity(cropped_img, 1, max_int=2, keep_min=True) - 1
				#cropped_img = rescale_int(cropped_img, intensity_df[intensity_df["acc_num"] == img_fn[:img_fn.find('.')]])

				fn = acc_num + "_" + str(voi[1]["lesion_num"])
				np.save(C.crops_dir + cls + "\\" + fn, cropped_img)

				small_voi_df = _add_small_voi(small_voi_df, acc_num, cls, small_voi)

			if img_num % 20 == 0:
				print(".", end="")
	print("")
	print(time.time()-t)
	
	return small_voi_df

def _add_small_voi(small_voi_df, acc_num, cls, coords):
	"""Add a row to small_voi_df."""

	if len(small_voi_df) == 0:
		i = 0
	else:
		i = small_voi_df.index[-1]+1
	
	small_voi_df.loc[i] = [acc_num, cls, coords]

	return small_voi_df

def _filter_small_vois(small_voi_df, acc_num, cls=None):
	"""Remove any elements in small_voi_df with the given acc_num. Limit to cls if specified."""

	if cls is not None:
		small_voi_df = small_voi_df[~((small_voi_df["acc_num"] == acc_num) & (small_voi_df["cls"] == cls))]
	else:
		small_voi_df = small_voi_df[~(small_voi_df["acc_num"] == acc_num)]

	return small_voi_df

"""def _extract_vois_orig(small_vois, C, voi_df_art, voi_df_ven, voi_df_eq, intensity_df):	
	t = time.time()

	# iterate over image series
	for cls in C.classes_to_include:
		for img_num, img_fn in enumerate(os.listdir(C.full_img_dir + "\\" + cls)):
			img = np.load(C.full_img_dir+"\\"+cls+"\\"+img_fn)
			art_vois = voi_df_art[(voi_df_art["Filename"] == img_fn) & (voi_df_art["cls"] == cls)]

			# iterate over each voi in that image
			for voi in art_vois.iterrows():
				ven_voi = voi_df_ven[voi_df_ven["id"] == voi[1]["id"]]
				eq_voi = voi_df_eq[voi_df_eq["id"] == voi[1]["id"]]

				cropped_img, small_voi = extract_voi(img, copy.deepcopy(voi[1]), C.dims, ven_voi=ven_voi, eq_voi=eq_voi)
				cropped_img = scale_intensity(cropped_img, 1, max_int=2, keep_min=True) - 1
				#cropped_img = rescale_int(cropped_img, intensity_df[intensity_df["acc_num"] == img_fn[:img_fn.find('.')]])

				fn = img_fn[:-4] + "_" + str(voi[1]["lesion_num"])
				np.save(C.crops_dir + cls + "\\" + fn, cropped_img)
				small_vois[fn] = small_voi

			if img_num % 20 == 0:
				print(".", end="")
	print("")
	print(time.time()-t)
	
	return small_vois
"""

def _extract_voi(img, voi, min_dims, ven_voi=[], eq_voi=[]):
	"""Input: image, a voi to center on, and the min dims of the unaugmented img.
	Outputs loosely cropped voi-centered image and coords of the voi within this loosely cropped image.
	"""
	
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
		temp_img = align_phases(temp_img, voi, ven_voi, 1)
		
	if len(eq_voi) > 0:
		eq_voi = eq_voi.iloc[0]
		temp_img = align_phases(temp_img, voi, eq_voi, 2)

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

#####################################
### SINGLE ACC_NUM METHODS
#####################################

def save_lesion_imgs(C, classes=None, acc_num=None):
	"""Save unaugmented lesion images"""

	small_voi_df = pd.read_csv(C.small_voi_path)
	if classes is None:
		classes = C.classes_to_include

	t = time.time()
	for cls in classes:
		if acc_num is None:
			acc_num_subset = os.listdir(C.crops_dir + cls)
		else:
			acc_num_subset = [x for x in os.listdir(C.crops_dir + cls) if x.startswith(acc_num)]

		for fn in acc_num_subset:
			img = np.load(C.crops_dir + cls + "\\" + fn)
			try:
				unaug_img = resize_img(img, C, small_voi_df.loc[(small_voi_df["acc_num"] == \
					fn[:fn.find("_")]) & (small_voi_df["cls"] == cls), "coords"])
			except Exception as e:
				print(cls, e)
				continue
			np.save(C.orig_dir + cls + "\\" + fn, unaug_img)
	print(time.time()-t)

def reload_accnum(acc_num, cls, C, augment=True, overwrite=True):
	"""Reloads cropped, scaled and augmented images. Updates voi_dfs and small_vois accordingly.
	small_vois needs to include class.
	Should be updated to use other methods."""

	# Update VOIs
	voi_df_art = pd.read_csv(C.art_voi_path)
	voi_df_ven = pd.read_csv(C.ven_voi_path)
	voi_df_eq = pd.read_csv(C.eq_voi_path)
	voi_dfs = [voi_df_art, voi_df_ven, voi_df_eq]
	dims_df = pd.read_csv(C.dims_df_path)

	#try:
	voi_dfs = drm.load_vois_batch(cls, C.sheetnames[C.cls_names.index(cls)], voi_dfs, dims_df, C, acc_nums=[acc_num], overwrite=overwrite)
	#except Exception as e:
	#	print(acc_num, "is not loaded or included.")
	#	remove_acc_num(acc_num, cls, C)
	#	return

	voi_df_art, voi_df_ven, voi_df_eq = voi_dfs
	voi_df_art.to_csv(C.art_voi_path, index=False)
	voi_df_ven.to_csv(C.ven_voi_path, index=False)
	voi_df_eq.to_csv(C.eq_voi_path, index=False)

	# Update small_vois / cropped image
	intensity_df = pd.read_csv(C.int_df_path)
	small_voi_df = pd.read_csv(C.small_voi_path)

	img_fn = acc_num + ".npy"
	img = np.load(C.full_img_dir+"\\"+cls+"\\"+img_fn)
	art_vois = voi_df_art[(voi_df_art["Filename"] == img_fn) & (voi_df_art["cls"] == cls)]

	if overwrite:
		for fn in os.listdir(C.crops_dir + cls):
			if fn.startswith(acc_num):
				os.remove(C.crops_dir + cls + "\\" + fn)
		for fn in os.listdir(C.orig_dir + cls):
			if fn.startswith(acc_num):
				os.remove(C.orig_dir + cls + "\\" + fn)
		if augment:
			for fn in os.listdir(C.aug_dir + cls):
				if fn.startswith(acc_num):
					os.remove(C.aug_dir + cls + "\\" + fn)

	small_voi_df = _filter_small_vois(small_voi_df, img_fn[:-4], cls)

	for voi in art_vois.iterrows():
		ven_voi = voi_df_ven[voi_df_ven["id"] == voi[1]["id"]]
		eq_voi = voi_df_eq[voi_df_eq["id"] == voi[1]["id"]]

		cropped_img, small_voi = extract_voi(img, copy.deepcopy(voi[1]), C.dims, ven_voi=ven_voi, eq_voi=eq_voi)
		cropped_img = scale_intensity(cropped_img, 1, max_int=2, keep_min=True) - 1
		#cropped_img = rescale_int(cropped_img, intensity_df[intensity_df["acc_num"] == img_fn[:img_fn.find('.')]])

		fn = img_fn[:-4] + "_" + str(voi[1]["lesion_num"])
		np.save(C.crops_dir + cls + "\\" + fn, cropped_img)

		small_voi_df = _add_small_voi(small_voi_df, img_fn[:-4], cls, small_voi)

		# Update scaled and augmented images
		unaug_img = resize_img(copy.deepcopy(cropped_img), C, small_voi)
		np.save(C.orig_dir + cls + "\\" + fn, unaug_img)
		if augment:
			augment_img(cropped_img, C, small_voi, num_samples=C.aug_factor, translate=[2,2,1], save_name=C.aug_dir + cls + "\\" + fn)

	small_voi_df.to_csv(C.small_voi_path, index=False)

def remove_acc_num(acc_num, cls, C):
	"""Remove acc_num from processed image folders (still)"""

	small_voi_df = pd.read_csv(C.small_voi_path)

	small_voi_df = _filter_small_vois(small_voi_df, acc_num, cls)

	with open(C.small_voi_path, 'w', newline='') as csv_file:
		writer = csv.writer(csv_file)
		for key, value in small_vois.items():
			writer.writerow([key, value])

	for fn in os.listdir(C.crops_dir + cls):
		if fn.startswith(acc_num):
			os.remove(C.crops_dir + cls + "\\" + fn)
	for fn in os.listdir(C.orig_dir + cls):
		if fn.startswith(acc_num):
			os.remove(C.orig_dir + cls + "\\" + fn)
	for fn in os.listdir(C.aug_dir + cls):
		if fn.startswith(acc_num):
			os.remove(C.aug_dir + cls + "\\" + fn)

###########################
### INTENSITY SCALING METHODS
###########################

def rescale_int(img, intensity_row, min_int=1):
	"""Rescale intensities in img such that the max intensity of the original image has a value of 1. Min intensity is -1."""
	try:
		img = img.astype(float)
		img[:,:,:,0] = (img[:,:,:,0] * 2 / float(intensity_row["art_int"])) - min_int
		img[:,:,:,1] = (img[:,:,:,1] * 2 / float(intensity_row["ven_int"])) - min_int
		img[:,:,:,2] = (img[:,:,:,2] * 2 / float(intensity_row["eq_int"])) - min_int
	except:
		raise ValueError("intensity_row is probably missing")

	return img

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

if __name__ == '__main__':
	import doctest
	doctest.testmod()