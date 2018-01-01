import copy
import numpy as np
import os
from scipy.misc import imsave
from skimage.transform import rescale

###########################
### FOR TRAINING
###########################

def separate_phases(X, non_imaging_inputs=False):
	"""Assumes X[0] contains imaging and X[1] contains dimension data.
	Reformats such that X[0:2] has 3 phases and X[3] contains dimension data.
	Image data still is 5D (nb_samples, 3D, 1 channel)."""
	
	if non_imaging_inputs:
		dim_data = copy.deepcopy(X[1])
		img_data = X[0]
		X[1] = np.expand_dims(X[0][:,:,:,:,1], axis=4)
		X += [np.expand_dims(X[0][:,:,:,:,2], axis=4)]
		X += [dim_data]
		X[0] = np.expand_dims(X[0][:,:,:,:,0], axis=4)
	
	else:
		X = np.array(X)
		X = [np.expand_dims(X[:,:,:,:,0], axis=4), np.expand_dims(X[:,:,:,:,1], axis=4), np.expand_dims(X[:,:,:,:,2], axis=4)]

	return X

def separate_phases_2d(X):
	"""Assumes X[0] contains imaging and X[1] contains dimension data.
	Reformats such that X[0:2] has 3 phases and X[3] contains dimension data.
	Image data still is 5D (nb_samples, 3D, 1 channel)."""
	
	if non_imaging_inputs:
		dim_data = copy.deepcopy(X[1])
		img_data = X[0]
		if len(X[0].shape)==5:
			X[0] = X[0][:,:,:,X[0].shape[3]//2,:]

		X[1] = np.expand_dims(X[0][:,:,:,1], axis=3)
		X += [np.expand_dims(X[0][:,:,:,2], axis=3)]
		X += [dim_data]
		X[0] = np.expand_dims(X[0][:,:,:,0], axis=3)

	else:
		X = np.array(X)
		X = [np.expand_dims(X[:,:,:,0], axis=4), np.expand_dims(X[:,:,:,1], axis=4), np.expand_dims(X[:,:,:,2], axis=4)]
	
	return X

def collect_unaug_data(C, voi_df, verbose=False):
	"""Return dictionary pointing to X (img data) and Z (filenames) and dictionary storing number of samples of each class."""
	orig_data_dict = {}
	num_samples = {}
	import voi_methods as vm

	for cls in C.classes_to_include:
		if verbose:
			print("\n"+cls)

		x = np.empty((10000, C.dims[0], C.dims[1], C.dims[2], C.nb_channels))
		x2 = np.empty((10000, 2))
		z = []

		for index, img_fn in enumerate(os.listdir(C.orig_dir+cls)):
			try:
				x[index] = np.load(C.orig_dir+cls+"\\"+img_fn)
				if C.hard_scale:
					x[index] = vm.scale_intensity(x[index], 1, max_int=2)#, keep_min=True)
			except:
				raise ValueError(C.orig_dir+cls+"\\"+img_fn + " not found")
			z.append(img_fn)
			
			row = voi_df[(voi_df["Filename"] == img_fn[:img_fn.find('_')] + ".npy") &
						 (voi_df["lesion_num"] == int(img_fn[img_fn.find('_')+1:-4]))]
			
			try:
				skip=False
				x2[index] = [(float(row["real_dx"]) * float(row["real_dy"]) * float(row["real_dz"])) ** (1/3) / 50,
							max(float(row["real_dx"]), float(row["real_dy"])) / float(row["real_dz"])]
			except TypeError:
				print(img_fn[:img_fn.find('_')], end=",")
				skip=True
				continue
				#raise ValueError(img_fn + " is probably missing a voi_df entry.")

		if not skip:
			x.resize((index+1, C.dims[0], C.dims[1], C.dims[2], C.nb_channels)) #shrink first dimension to fit
			x2.resize((index+1, 2)) #shrink first dimension to fit
			orig_data_dict[cls] = [x,x2,np.array(z)]
			num_samples[cls] = index + 1
		
	return orig_data_dict, num_samples


###########################
### FOR OUTPUTTING IMAGES AFTER TRAINING
###########################

def save_output(Z, y_pred, y_true, voi_df_art, small_voi_df, cls_mapping, C, save_dir=None):
	"""Parent method; saves all imgs in """
	if save_dir is None:
		save_dir = C.output_img_dir

	for cls in cls_mapping:
		if not os.path.exists(save_dir + "\\correct\\" + cls):
			os.makedirs(save_dir + "\\correct\\" + cls)
		if not os.path.exists(save_dir + "\\incorrect\\" + cls):
			os.makedirs(save_dir + "\\incorrect\\" + cls)

	for i in range(len(Z)):
		if y_pred[i] != y_true[i]:
			plot_multich_with_bbox(Z[i], cls_mapping[y_pred[i]], voi_df_art, small_voi_df, save_dir=save_dir + "\\incorrect\\" + cls_mapping[y_true[i]], C=C)
		else:
			plot_multich_with_bbox(Z[i], cls_mapping[y_pred[i]], voi_df_art, small_voi_df, save_dir=save_dir + "\\correct\\" + cls_mapping[y_true[i]], C=C)

def plot_multich_with_bbox(fn, pred_class, voi_df_art, small_voi_df, num_ch=3, save_dir=None, C=None):
	"""Plot"""

	normalize = True

	if not os.path.exists(save_dir):
		os.makedirs(save_dir)
		
	img_fn = fn[:fn.find('_')] + ".npy"
	voi = voi_df_art[(voi_df_art["Filename"] == img_fn) &
					 (voi_df_art["lesion_num"] == int(fn[fn.find('_')+1:fn.rfind('.')]))].iloc[0]
	
	img = np.load(C.crops_dir + voi["cls"] + "\\" + fn)
	img_slice = img[:,:, img.shape[2]//2, :].astype(float)
	#for ch in range(img_slice.shape[-1]):
	#	img_slice[:, :, ch] *= 255/np.amax(img_slice[:, :, ch])
	if normalize:
		img_slice[0,0,:]=-1
		img_slice[0,-1,:]=.8

	img_slice = np.stack([img_slice, img_slice, img_slice], axis=2)
	
	img_slice = draw_bbox(img_slice, C, small_voi_df.loc[small_voi_df["acc_num"] == img_fn[:-4], "coords"])
		
	ch1 = np.transpose(img_slice[:,::-1,:,0], (1,0,2))
	ch2 = np.transpose(img_slice[:,::-1,:,1], (1,0,2))
	
	if num_ch == 2:
		ret = np.empty([ch1.shape[0]*2, ch1.shape[1], 3])
		ret[:ch1.shape[0],:,:] = ch1
		ret[ch1.shape[0]:,:,:] = ch2
		
	elif num_ch == 3:
		ch3 = np.transpose(img_slice[:,::-1,:,2], (1,0,2))

		ret = np.empty([ch1.shape[0]*3, ch1.shape[1], 3])
		ret[:ch1.shape[0],:,:] = ch1
		ret[ch1.shape[0]:ch1.shape[0]*2,:,:] = ch2
		ret[ch1.shape[0]*2:,:,:] = ch3
		
	else:
		raise ValueError("Invalid num channels")
		
	imsave("%s\\large-%s (pred %s).png" % (save_dir, fn[:-4], pred_class), ret)


	rescale_factor = 3
	cls = voi["cls"]
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

	imsave("%s\\small-%s (pred %s).png" % (save_dir, fn[:fn.find('.')], pred_class), rescale(ret, rescale_factor, mode='constant'))

def condense_cm(y_true, y_pred, cls_mapping):
	simplify_map = {'hcc': 0, 'cyst': 1, 'hemangioma': 1, 'fnh': 1, 'cholangio': 2, 'colorectal': 2}
	y_true_simp = np.array([simplify_map[cls_mapping[y]] for y in y_true])
	y_pred_simp = np.array([simplify_map[cls_mapping[y]] for y in y_pred])
	
	return y_true_simp, y_pred_simp, ['hcc', 'benign', 'malignant non-hcc']

def draw_bbox(img_slice, C, voi):
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
	
	buffer = C.padding
	scale_ratios = [final_dims[0]/dx * buffer, final_dims[1]/dy * buffer, final_dims[2]/dz * buffer]
	
	crop = [img_slice.shape[i] - round(final_dims[i]/scale_ratios[i]) for i in range(2)]
	
	for i in range(2):
		assert crop[i]>=0
		
	x1 = crop[0]//2
	x2 = -crop[0]//2
	y1 = crop[1]//2
	y2 = -crop[1]//2

	img_slice[x1:x2, y2, 2, :] = 1
	img_slice[x1:x2, y2, :2, :] = -1

	img_slice[x1:x2, y1, 2, :] = 1
	img_slice[x1:x2, y1, :2, :] = -1

	img_slice[x1, y1:y2, 2, :] = 1
	img_slice[x1, y1:y2, :2, :] = -1

	img_slice[x2, y1:y2, 2, :] = 1
	img_slice[x2, y1:y2, :2, :] = -1
	
	return img_slice
