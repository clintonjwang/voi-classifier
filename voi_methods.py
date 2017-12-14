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


###########################
### BULK METHODS ACROSS ALL CLASSES
###########################

def parallel_augment(cls, small_vois, C, num_cores=None):
	"""Augment all images in cls using CPU parallelization"""
	if num_cores is None:
		num_cores = multiprocessing.cpu_count()

	if num_cores > 1:
		Parallel(n_jobs=num_cores)(delayed(save_augmented_img)(fn, cls, small_vois[fn[:-4]], C) for fn in os.listdir(C.crops_dir + cls))
	else:
		for fn in os.listdir(C.crops_dir + cls):
			save_augmented_img(fn, cls, small_vois[fn[:-4]], C)

def save_augmented_img(fn, cls, voi_coords, C):
	img = np.load(C.crops_dir + cls + "\\" + fn)
	augment_img(img, C.dims, voi_coords, num_samples=C.aug_factor, translate=[2,2,1], save_name=C.aug_dir + cls + "\\" + fn[:-4], intensity_scaling=C.intensity_scaling)

def save_all_vois(cls, C, num_ch=3, normalize=True, rescale_factor=3, acc_nums=None):
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
			img_slice[0,0,:]=-1
			img_slice[0,-1,:]=.8
			
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

def extract_vois(small_vois, C, voi_df_art, voi_df_ven, voi_df_eq, intensity_df):
	"""Call extract_voi on all images in C.full_img_dir"""
	
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
				cropped_img = rescale_int(cropped_img, intensity_df[intensity_df["AccNum"] == img_fn[:img_fn.find('.')]])

				fn = img_fn[:-4] + "_" + str(voi[1]["lesion_num"])
				np.save(C.crops_dir + cls + "\\" + fn, cropped_img)
				small_vois[fn] = small_voi

			if img_num % 20 == 0:
				print(".", end="")
	print("")
	print(time.time()-t)
	
	return small_vois


###########################
### SINGLE ACC_NUM METHODS
###########################

def reload_accnum(accnum, cls, C, augment=True):
	"""Reloads cropped, scaled and augmented images. Updates voi_dfs and small_vois accordingly."""

	# Update VOIs
	xls_name = 'Z:\\Prototype1d.xlsx'
	cls_names = ['hcc', 'hcc', 'cyst', 'hemangioma', 'fnh', 'cholangio', 'colorectal', 'adenoma']
	sheetnames = ['OPTN 5A', 'OPTN 5B', 'Cyst', 'Hemangioma', 'FNH', 'Cholangio', 'Colorectal', 'Adenoma']
	img_dirs = ['OPTN5A', 'optn5b', 'simple_cysts', 'hemangioma', 'fnh', 'cholangio', 'colorectal', 'adenoma']
	voi_df_art = pd.read_csv(C.art_voi_path)
	voi_df_ven = pd.read_csv(C.ven_voi_path)
	voi_df_eq = pd.read_csv(C.eq_voi_path)
	voi_dfs = [voi_df_art, voi_df_ven, voi_df_eq]
	dims_df = pd.read_csv(C.dims_df_path)

	try:
		if cls=="5a":
			voi_dfs = drm.load_vois("hcc", xls_name, sheetnames[0], voi_dfs, dims_df, C, acc_nums=[accnum])
			cls = "hcc"
		elif cls=="5b":
			voi_dfs = drm.load_vois("hcc", xls_name, sheetnames[1], voi_dfs, dims_df, C, acc_nums=[accnum])
			cls = "hcc"
		elif cls=="hcc":
			raise ValueError("Specify 5a or 5b")
		else:
			voi_dfs = drm.load_vois(cls, xls_name, sheetnames[cls_names.index(cls)], voi_dfs, dims_df, C, acc_nums=[accnum])
	except:
		print(accnum, "is not loaded or included.")
		remove_accnum(accnum, cls, C)
		return

	voi_df_art, voi_df_ven, voi_df_eq = voi_dfs
	voi_df_art.to_csv(C.art_voi_path, index=False)
	voi_df_ven.to_csv(C.ven_voi_path, index=False)
	voi_df_eq.to_csv(C.eq_voi_path, index=False)


	# Update small_vois / cropped image
	intensity_df = pd.read_csv(C.int_df_path)
	with open(C.small_voi_path, 'r') as csv_file:
		reader = csv.reader(csv_file)
		small_vois = dict(reader)
	for key in small_vois:
		if key[:key.find('_')] != accnum:
			small_vois[key] = [int(x) for x in small_vois[key][1:-1].split(', ')]

	img_fn = accnum + ".npy"
	img = np.load(C.full_img_dir+"\\"+cls+"\\"+img_fn)
	art_vois = voi_df_art[(voi_df_art["Filename"] == img_fn) & (voi_df_art["cls"] == cls)]


	for fn in os.listdir(C.crops_dir + cls):
		if fn.startswith(accnum):
			os.remove(C.crops_dir + cls + "\\" + fn)
	for fn in os.listdir(C.orig_dir + cls):
		if fn.startswith(accnum):
			os.remove(C.orig_dir + cls + "\\" + fn)

	if augment:
		for fn in os.listdir(C.aug_dir + cls):
			if fn.startswith(accnum):
				os.remove(C.aug_dir + cls + "\\" + fn)


	for voi in art_vois.iterrows():
		ven_voi = voi_df_ven[voi_df_ven["id"] == voi[1]["id"]]
		eq_voi = voi_df_eq[voi_df_eq["id"] == voi[1]["id"]]

		cropped_img, small_voi = extract_voi(img, copy.deepcopy(voi[1]), C.dims, ven_voi=ven_voi, eq_voi=eq_voi)
		cropped_img = rescale_int(cropped_img, intensity_df[intensity_df["AccNum"] == img_fn[:img_fn.find('.')]])

		fn = img_fn[:-4] + "_" + str(voi[1]["lesion_num"])
		np.save(C.crops_dir + cls + "\\" + fn, cropped_img)
		small_vois[fn] = small_voi

		# Update scaled and augmented images
		unaug_img = resize_img(copy.deepcopy(cropped_img), C.dims, small_voi)
		np.save(C.orig_dir + cls + "\\" + fn, unaug_img)
		if augment:
			augment_img(cropped_img, C.dims, small_voi, num_samples=C.aug_factor, translate=[2,2,1], save_name=C.aug_dir + cls + "\\" + fn)

	with open(C.small_voi_path, 'w', newline='') as csv_file:
		writer = csv.writer(csv_file)
		for key, value in small_vois.items():
			writer.writerow([key, value])

def remove_accnum(accnum, cls, C):
	with open(C.small_voi_path, 'r') as csv_file:
		reader = csv.reader(csv_file)
		small_vois = dict(reader)
	for key in small_vois:
		if key[:key.find('_')] != accnum:
			small_vois[key] = [int(x) for x in small_vois[key][1:-1].split(', ')]

	with open(C.small_voi_path, 'w', newline='') as csv_file:
		writer = csv.writer(csv_file)
		for key, value in small_vois.items():
			writer.writerow([key, value])

	for fn in os.listdir(C.crops_dir + cls):
		if fn.startswith(accnum):
			os.remove(C.crops_dir + cls + "\\" + fn)
	for fn in os.listdir(C.orig_dir + cls):
		if fn.startswith(accnum):
			os.remove(C.orig_dir + cls + "\\" + fn)
	for fn in os.listdir(C.aug_dir + cls):
		if fn.startswith(accnum):
			os.remove(C.aug_dir + cls + "\\" + fn)

def extract_voi(img, voi, min_dims, ven_voi=[], eq_voi=[]):
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
	assert dx > 0
	assert dy > 0
	assert dz > 0
	
	# align all phases
	if len(ven_voi) > 0:
		ven_voi = ven_voi.iloc[0]
		temp_img = hf.align(temp_img, voi, ven_voi, 1)
		
	if len(eq_voi) > 0:
		eq_voi = eq_voi.iloc[0]
		temp_img = hf.align(temp_img, voi, eq_voi, 2)

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
	
	#x1 = voi['x1'] + side_padding - math.ceil(xpad/2)
	
	new_voi = [xpad//2, dx + xpad//2,
			   ypad//2, dy + ypad//2,
			   zpad//2, dz + zpad//2]
	
	for i in new_voi:
		assert i>=0
		
	return pad_img[x1:x2, y1:y2, z1:z2, :], [int(x) for x in new_voi]

def augment_img(img, final_dims, voi, num_samples, translate=None, add_reflections=False, save_name=None, intensity_scaling=[.05,.05]):
	"""For rescaling an img to final_dims while scaling to make sure the image contains the voi.
	add_reflections and save_name cannot be used simultaneously"""

	x1 = voi[0]
	x2 = voi[1]
	y1 = voi[2]
	y2 = voi[3]
	z1 = voi[4]
	z2 = voi[5]
	dx = x2 - x1
	dy = y2 - y1
	dz = z2 - z1
	
	buffer1 = 0.7
	buffer2 = 0.9
	scale_ratios = [final_dims[0]/dx, final_dims[1]/dy, final_dims[2]/dz]

	#img = adjust_cropped_intensity(img)
	aug_imgs = []
	
	for img_num in range(num_samples):
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
		
		temp_img = adjust_cropped_intensity(temp_img)
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

def rescale_int(img, intensity_row, min_int=1):
	"""Rescale intensities in img by the """
	try:
		img = img.astype(float)
		img[:,:,:,0] = (img[:,:,:,0] * 3 / float(intensity_row["art_int"])) - min_int
		img[:,:,:,1] = (img[:,:,:,1] * 3 / float(intensity_row["ven_int"])) - min_int
		img[:,:,:,2] = (img[:,:,:,2] * 3 / float(intensity_row["eq_int"])) - min_int
	except:
		raise ValueError("intensity_row is probably missing")

	return img

def adjust_cropped_intensity(img):
	"""Force max to be 1 and min to be -1, scaling each channel separately"""

	img = img.astype(float)

	img[:,:,:,0] = img[:,:,:,0] - np.amin(img[:,:,:,0])
	img[:,:,:,0] = img[:,:,:,0] * 2 / np.amax(img[:,:,:,0]) - 1

	img[:,:,:,1] = img[:,:,:,1] - np.amin(img[:,:,:,1])
	img[:,:,:,1] = img[:,:,:,1] * 2 / np.amax(img[:,:,:,1]) - 1

	img[:,:,:,2] = img[:,:,:,2] - np.amin(img[:,:,:,2])
	img[:,:,:,2] = img[:,:,:,2] * 2 / np.amax(img[:,:,:,2]) - 1

	return img

def resize_img(img, final_dims, voi):
	"""For rescaling an img to final_dims while scaling to make sure the image contains the voi.
	Do not reuse img
	"""
	
	x1 = voi[0]
	x2 = voi[1]
	y1 = voi[2]
	y2 = voi[3]
	z1 = voi[4]
	z2 = voi[5]
	dx = x2 - x1
	dy = y2 - y1
	dz = z2 - z1
	
	padding = 0.85
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
	img = adjust_cropped_intensity(img)

	return img

if __name__ == '__main__':
	print("This is not meant to be called as a script.")