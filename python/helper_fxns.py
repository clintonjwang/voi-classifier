from custom_mods.convert_dicom import dicom_series_to_nifti
from custom_mods.convert_siemens import dicom_to_nifti
import copy
import dicom
import math
import matplotlib
import matplotlib.pyplot as plt
import nibabel
import numpy as np
import os
import pyelastix
import requests
import random
import transforms as tr

###########################
### IMAGE LOADING
###########################

def dcm_load(path2series):
	"""
	Load a dcm series as a 3D array along with its dimensions.
	
	returns as a tuple:
	- the normalized (0-255) image
	- the spacing between pixels in cm
	"""

	try:
		tmp_fn = "tmp.nii"
		dicom_series_to_nifti(path2series, tmp_fn)
		#dicom_to_nifti(path2series, tmp_fn)

		ret = ni_load(tmp_fn)

	except Exception as e:
		print(path2series, e)
		return None

	try:
		os.remove(tmp_fn)
	except:
		print("Cannot delete %s" % tmp_fn)

	return ret

def ni_load(filename):
	"""
	Load a nifti image as a 3D array along with its dimensions.
	
	returns as a tuple:
	- the normalized (0-255) image
	- the spacing between pixels in cm
	"""
	
	img = nibabel.load(filename)
	img = nibabel.as_closest_canonical(img) # make sure it is in the correct orientation

	dims = img.header['pixdim'][1:4]
	dim_units = img.header['xyzt_units']
	
	img = np.asarray(img.dataobj).astype(dtype='float64')
	img = img[::-1,:,:]
	img =  (255 * (img / np.amax(img))).astype(dtype='uint8')
	
	if dim_units == 2: #or np.sum(img) * dims[0] * dims[1] * dims[2] > 10000:
		dims = [d/10 for d in dims]
	
	return img, dims

def get_spect_series(path, just_header=False):
    import convert_dicom
    import tempfile
    import shutil
    import dicom2nifti.settings as settings
    import dicom2nifti.common as common
    temp_directory = tempfile.mkdtemp()
    dicom_directory = os.path.join(temp_directory, 'dicom')
    shutil.copytree(path, dicom_directory)

    if convert_dicom.is_compressed(dicom_directory):
        if settings.gdcmconv_path is None and convert_dicom._which('gdcmconv') is None and convert_dicom._which('gdcmconv.exe') is None:
            raise ConversionError('GDCMCONV_NOT_FOUND')

        convert_dicom.logger.info('Decompressing dicom files in %s' % dicom_directory)
        for root, _, files in os.walk(dicom_directory):
            for dicom_file in files:
                if common.is_dicom_file(os.path.join(root, dicom_file)):
                    convert_dicom.decompress_dicom(os.path.join(root, dicom_file))

    dicom_input = common.read_dicom_directory(dicom_directory)
    
    if just_header:
        return dicom_input[0]
    
    rows = dicom_input[0][('0028', '0010')].value
    cols = dicom_input[0][('0028', '0011')].value
    ch = dicom_input[0][('0028', '0002')].value
    frames = dicom_input[0][('0028', '0008')].value
    bytelen = dicom_input[0][('0028', '0101')].value//8

    if len(dicom_input) == 1:
        ls = list(dicom_input[0][('7fe0', '0010')].value)
        img = [ls[x]+ls[x+1]*256 for x in range(0,len(ls),2)]
        img = np.reshape(img,(frames,rows,cols))
        canon_img = np.transpose(img, (2,1,0))[:,::-1,:]
    else:
        sl_list = []

        for sl in dicom_input:
            arr = sl[('7fe0', '0010')].value
            sl_list.append(np.reshape(list(arr),(rows,cols,ch)))
        img = np.array(sl_list)
        canon_img = np.transpose(img, (2,1,0,3))[:,::-1,:,:]

    return canon_img

def get_dcm_header_value(txt, search_term):
	"""Gets value corresponding to a dicom tag
	search_term should be formatted like '<DicomAttribute tag="00100020" vr="LO" keyword="PatientID">'
	"""

	index = txt.find(search_term) + len(search_term+'\n      <Value number="1">')
	if index == -1:
		raise ValueError(search_term, "not found")
	return txt[index:index + txt[index:].find("</Value>")]

###########################
### IMAGE PREPROCESSING
###########################

def apply_mask(img, mask_file):
	"""Apply the mask in mask_file to img and return the masked image."""

	with open(mask_file, 'rb') as f:
		mask = f.read()
		mask = np.fromstring(mask, dtype='uint8')
		mask = np.array(mask).reshape((img.shape[2], img.shape[1], img.shape[0]))
		#mask = np.reshape(mask, img.shape, order='F')
		mask = np.transpose(mask, (2,1,0))
		#mask = mask[:,::-1,:]
		
	img[:,:,:,0][mask == 0] = 0
	#img[mask <= 0] = 0

	return img

def create_diff(art_img, pre_img):
	diff = art_img - pre_img # Calculate the subtracted image
	diff[diff < 0] = 0  # The pre-contrast should never be more than the arterial. Clamp negative values to zero.

	#draw_fig(diff, 'whole')
	
	return diff

def rescale(img, target_dims, cur_dims):
	vox_scale = [float(cur_dims[i]/target_dims[i]) for i in range(3)]
	img = tr.scale3d(img, vox_scale)
	
	return img, vox_scale

def normalize(img, desired_max=255):
	img = img.astype(float)
	i_min = np.amin(img)
	return (img - i_min) / (np.amax(img) - i_min) * desired_max

def reg_imgs(moving, fixed, params, rescale_only=False):	
	fshape = fixed.shape
	mshape = moving.shape
	scale = [fshape[i]/mshape[i] for i in range(3)]
	moving = tr.scale3d(moving, scale)
	
	assert moving.shape == fixed.shape, ("Shapes not aligned in reg_imgs", moving.shape, fixed.shape)
	
	if not rescale_only:
		moving = np.ascontiguousarray(moving).astype('float32')
		fixed = np.ascontiguousarray(fixed).astype('float32')

		moving, _ = pyelastix.register(moving, fixed, params, verbose=0)
		
	return moving, scale

###########################
### IMAGE FEATURES
###########################

def get_vol(img, dims, dim_units):
	return np.sum(img>0) * np.prod(dims)

def get_hist(img):
	"""Returns histogram in array and graphical forms."""
	h = plt.hist(flatten(img, times=2))
	plt.title("Histogram")
	plt.xlabel("Value")
	plt.ylabel("Frequency")
	
	#mean_intensity = np.mean(diff[img > 0])
	#std_intensity = np.std(diff[img > 0])

	return h, plt.gcf()
	

#########################
### PLOTTING
#########################

def _plot_without_axes(img, cmap):
	"""Gets rid of axes in a figure"""

	fig = plt.imshow(img, cmap=cmap)
	fig.axes.get_xaxis().set_visible(False)
	fig.axes.get_yaxis().set_visible(False)

def plot_section_auto(orig_img, normalize=False, frac=None):
	"""Only accepts 3D images or 4D images with at least 3 channels.
	If 3D, outputs slices at 1/4, 1/2 and 3/4.
	If 4D, outputs middle slice for the first 3 channels.
	If 4D, can specify an optional frac argument to also output a slice at a different fraction."""

	if len(orig_img.shape) == 4:
		img = copy.deepcopy(orig_img)
		if normalize:
			if np.amin(img) < -.2:
				img[0,0,:,:]=-.7
			else:
				img[0,0,:,:]=0
			img[0,-1,:,:]=.7

		if frac is None:
			plt.subplot(131)
			_plot_without_axes(np.transpose(img[:, ::-1, img.shape[2]//2, 0], (1,0)), cmap='gray')
			plt.subplot(132)
			_plot_without_axes(np.transpose(img[:, ::-1, img.shape[2]//2, 1], (1,0)), cmap='gray')
			plt.subplot(133)
			_plot_without_axes(np.transpose(img[:, ::-1, img.shape[2]//2, 2], (1,0)), cmap='gray')
		else:
			plt.subplot(231)
			_plot_without_axes(np.transpose(img[:, ::-1, img.shape[2]//2, 0], (1,0)), cmap='gray')
			plt.subplot(232)
			_plot_without_axes(np.transpose(img[:, ::-1, img.shape[2]//2, 1], (1,0)), cmap='gray')
			plt.subplot(233)
			_plot_without_axes(np.transpose(img[:, ::-1, img.shape[2]//2, 2], (1,0)), cmap='gray')

			plt.subplot(234)
			_plot_without_axes(np.transpose(img[:, ::-1, int(img.shape[2]*frac), 0], (1,0)), cmap='gray')
			plt.subplot(235)
			_plot_without_axes(np.transpose(img[:, ::-1, int(img.shape[2]*frac), 1], (1,0)), cmap='gray')
			plt.subplot(236)
			_plot_without_axes(np.transpose(img[:, ::-1, int(img.shape[2]*frac), 2], (1,0)), cmap='gray')

	else:
		if normalize:
			img[0,0,:]=-1
			img[0,-1,:]=.8

		plt.subplot(131)
		_plot_without_axes(np.transpose(img[:, ::-1, img.shape[2]//4], (1,0)), cmap='gray')
		plt.subplot(132)
		_plot_without_axes(np.transpose(img[:, ::-1, img.shape[2]//2], (1,0)), cmap='gray')
		plt.subplot(133)
		_plot_without_axes(np.transpose(img[:, ::-1, img.shape[2]*3//4], (1,0)), cmap='gray')

	plt.subplots_adjust(wspace=0, hspace=0)

def plot_slice_flips(img, df, pad=30, flipz="both"):
	"""Function to plot an image slice given a VOI, to test whether the z axis is flipped."""

	if flipz=="both":
		plt.subplot(121)
		plt.imshow(np.transpose(img[df['x1']-pad:df['x2']+pad,
								df['y2']+pad:df['y1']-pad:-1,
								(df['z1']+df['z2'])//2, 0], (1,0)), cmap='gray')
		
		plt.subplot(122)
		plt.imshow(np.transpose(img[df['x1']-pad:df['x2']+pad,
									df['y2']+pad:df['y1']-pad:-1,
									img.shape[2]-(df['z1']+df['z2'])//2, 0], (1,0)), cmap='gray')

def plot_section_xyz(img, x,y,z, pad=30):
	plt.subplot(121)
	plt.imshow(np.transpose(img[x[0]-pad:x[1]+pad, y[1]+pad:y[0]-pad:-1, (z[0]+z[1])//2,0], (1,0)), cmap='gray')

def flatten(l, times=1):
	for _ in range(times):
		l = [item for sublist in l for item in sublist]
	return l

def draw_slice(img, filename, slice=None):
	"""Draw a slice of an image of type np array and save it to disk. Unused."""
	cnorm = matplotlib.colors.Normalize(vmin=0, vmax=np.amax(img))
	
	if slice is None and len(img.shape)>2:
		slice=img.shape[2]//2

	w = 20
	h = int(float(img.shape[1]) / img.shape[0] * w)
	img_slice = img[:,:,slice]

	img_slice = np.rot90(img_slice)

	fig = plt.figure(frameon=False)
	fig.set_size_inches(w,h)
	ax = plt.Axes(fig, [0., 0., 1., 1.])
	ax.set_axis_off()
	fig.add_axes(ax)
	ax.imshow(img_slice, interpolation='bilinear', norm=cnorm, cmap=plt.cm.gray, aspect='auto')

	filename += '.png'
	plt.savefig(filename)
	print('Slice saved as %s' % filename)
	fig.set_size_inches(w//3,h//3)
	plt.show()
