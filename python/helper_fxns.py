from dicom2nifti.convert_dicom import dicom_series_to_nifti
from dicom2nifti.convert_siemens import dicom_to_nifti
import copy
import dicom
import math
import matplotlib
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import os
import pyelastix
import random
import re
import requests
import shutil
import SimpleITK as sitk
import subprocess
import transforms as tr

###########################
### IMAGE LOADING
###########################

def dcm_load(path2series, flip_x=False, flip_y=False):
	"""
	Load a dcm series as a 3D array along with its dimensions.
	
	returns as a tuple:
	- the normalized (0-255) image
	- the spacing between pixels in cm
	"""

	try:
		tmp_fn = "tmp.nii.gz"
		dicom_series_to_nifti(path2series, tmp_fn)

		ret = ni_load(tmp_fn, flip_x, flip_y)

	except Exception as e:
		print(path2series, e)
		return None

	try:
		os.remove(tmp_fn)
	except:
		print("Cannot delete %s" % tmp_fn)

	return ret

def ni_load(filename, flip_x=False, flip_y=False, normalize=False, binary=False):
	"""
	Load a nifti image as a 3D array along with its dimensions.
	
	returns as a tuple:
	- the normalized (0-255) image
	- the spacing between pixels in cm
	"""
	
	img = nib.load(filename)
	img = nib.as_closest_canonical(img) # make sure it is in the correct orientation

	dims = img.header['pixdim'][1:4]
	dim_units = img.header['xyzt_units']
	
	img = np.asarray(img.dataobj).astype(dtype='float64')
	if normalize:
		img = 255 * (img / np.amax(img))
	if binary:
		img = 255 * (img / np.amax(img))
		img = img.astype('uint8')
	
	if dim_units == 2: #or np.sum(img) * dims[0] * dims[1] * dims[2] > 10000:
		dims = [d/10 for d in dims]
	if len(img.shape) == 4:
		img = img[::(-1)**flip_x,::(-1)**flip_y,:,:]
	else:
		img = img[::(-1)**flip_x,::(-1)**flip_y,:]

	return img, dims

def get_spect_series(path, just_header=False):
	import convert_dicom
	import tempfile
	import shutil
	import dicom2nifti.settings as settings
	import dicom2nifti.common as common
	import dicom2nifti.compressed_dicom as compressed_dicom
	temp_directory = tempfile.mkdtemp()
	dicom_directory = os.path.join(temp_directory, 'dicom')
	shutil.copytree(path, dicom_directory)

	if convert_dicom.is_compressed(dicom_directory):
		if settings.gdcmconv_path is None and convert_dicom._which('gdcmconv') is None and convert_dicom._which('gdcmconv.exe') is None:
			raise ConversionError('GDCMCONV_NOT_FOUND')

		convert_dicom.logger.info('Decompressing dicom files in %s' % dicom_directory)
		for root, _, files in os.walk(dicom_directory):
			for dicom_file in files:
				if compressed_dicom.is_dicom_file(os.path.join(root, dicom_file)):
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

def save_nii(img, dest, dims=(1,1,1), flip_x=False, flip_y=False):
	affine = np.eye(4)
	for i in range(3):
		affine[i,i] = dims[i]
	if len(img.shape) == 4:
		nii = nib.Nifti1Image(img[::(-1)**flip_x,::(-1)**flip_y,:,:], affine)
	else:
		nii = nib.Nifti1Image(img[::(-1)**flip_x,::(-1)**flip_y,:], affine)
	nib.save(nii, dest)

###########################
### MASKS
###########################

def create_threshold_mask(img, mask_filename, threshold, template_mask_fn=None):
	"""Create and save a mask of orig_img based on a threshold value."""
	mask = np.zeros(img.shape)
	mask[img > threshold] = 255
	mask = mask.astype('uint8')
	save_mask(mask, mask_filename, template_mask_fn)

def get_mask(mask_file, dims):
	"""Apply the mask in mask_file to img and return the masked image."""
	with open(mask_file, 'rb') as f:
		mask = f.read()
		mask = np.fromstring(mask, dtype='uint8')
		mask = np.array(mask).reshape((dims[::-1]))
		mask = np.transpose(mask, (2,1,0))

	return mask

def save_mask(orig_mask, filename, template_mask_fn=None):
	"""Assumes mask is an np.ndarray."""
	mask = copy.deepcopy(orig_mask)
	mask[mask != 0] = 255
	mask = np.transpose(mask, (2,1,0))
	mask = np.ascontiguousarray(mask).astype('uint8')
	with open(filename, 'wb') as f:
		f.write(mask)

	if template_mask_fn is not None:
		if not template_mask_fn.endswith('.ics'):
			template_mask_fn = template_mask_fn[:template_mask_fn.find('.')] + ".ics"
		shutil.copy(template_mask_fn, filename[:filename.find('.')] + ".ics")

	return True

def apply_mask(orig_img, mask_file):
	"""Apply the mask in mask_file to img and return the masked image."""
	img = copy.deepcopy(orig_img)
	mask = get_mask(mask_file, img.shape)

	if len(img.shape) == 4:
		for ch in img.shape[3]:
			img[:,:,:,ch][mask == 0] = 0
	else:
		img[mask == 0] = 0
	#img[mask <= 0] = 0

	return img

def rescale_mask(mask_file, orig_dims, dims):
	"""Apply the mask in mask_file to img and return the masked image."""

	return img


###########################
### IMAGE PREPROCESSING
###########################

def rescale(img, target_dims, cur_dims=None):
	if cur_dims is not None:
		vox_scale = [float(cur_dims[i]/target_dims[i]) for i in range(3)]
	else:
		vox_scale = [float(target_dims[i]/img.shape[i]) for i in range(3)]
	
	return tr.scale3d(img, vox_scale), vox_scale

def normalize(img):
	#t2[mrn] = t2[mrn] * 255/np.amax(t2[mrn])
	i_min = np.amin(img)
	return (img - i_min) / (np.amax(img) - i_min) * 255

###########################
### REGISTRATION
###########################

def reg_bis(fixed_img_path, moving_img_path, out_transform_path="default", out_img_path="default",
	path_to_bis="C:\\yale\\bioimagesuite35\\bin\\", overwrite=True):
	"""BioImageSuite. Shutil required because BIS cannot output to other drives,
	and because the output image argument is broken."""

	temp_img_path = ".\\temp_out_img.nii"
	temp_xform_path = ".\\temp_out_xform"
	
	if out_transform_path == "default":
		out_transform_path = add_to_filename(moving_img_path, "-xform")

	if out_img_path == "default":
		out_img_path = add_to_filename(moving_img_path, "-reg")

	if (not overwrite) and os.path.exists(out_img_path):
		print(out_img_path, "already exists. Skipping registration.")
		return None
	
	cmd = ''.join([path_to_bis, "bis_linearintensityregister.bat -inp ", fixed_img_path,
			  " -inp2 ", moving_img_path, " -out ", temp_xform_path]).replace("\\","/")

	subprocess.run(cmd.split())
	if out_img_path is not None:
		shutil.copy(temp_img_path, out_img_path)
	if out_transform_path is not None:
		shutil.copy(temp_xform_path, out_transform_path)
	os.remove(temp_img_path)
	os.remove(temp_xform_path)

	return out_img_path, out_transform_path

def reg_elastix(*args):
	reg_imgs(args)

def reg_imgs(moving, fixed, params, rescale_only=False):
	reg_img = copy.deepcopy(moving)
	try:
		reg_img = np.ascontiguousarray(moving).astype('float32')
		fixed = np.ascontiguousarray(fixed).astype('float32')

		reg_img, field = pyelastix.register(reg_img, fixed, params, verbose=0)

	except Exception as e:
		print(e)
		fshape = fixed.shape
		mshape = moving.shape
		field = [fshape[i]/mshape[i] for i in range(3)]
		reg_img = tr.scale3d(moving, field)
		
		#assert moving.shape == fixed.shape, ("Shapes not aligned in reg_imgs", moving.shape, fixed.shape)

		
	return reg_img, field

def reg_sitk(fixed_path, moving_path, out_transform_path, out_img_path, verbose=False, reg_type="demons"):
	"""Assumes fixed and moving images are the same dimensions"""

	fixed = sitk.ReadImage(fixed_path, sitk.sitkFloat32)
	moving = sitk.ReadImage(moving_path, sitk.sitkFloat32)

	if reg_type == "demons":
		matcher = sitk.HistogramMatchingImageFilter()
		matcher.SetNumberOfHistogramLevels(1024)
		matcher.SetNumberOfMatchPoints(7)
		matcher.ThresholdAtMeanIntensityOn()
		moving = matcher.Execute(moving,fixed)
		
		R = sitk.DemonsRegistrationFilter()
		R.SetNumberOfIterations( 50 )
		R.SetStandardDeviations( 1.0 )

		if verbose:
			def command_iteration(filter):
				print("{0:3} = {1:10.5f}".format(filter.GetElapsedIterations(), filter.GetMetric()))
			R.AddCommand( sitk.sitkIterationEvent, lambda: command_iteration(R) )
	
		displacementField = R.Execute( fixed, moving )
		outTx = sitk.DisplacementFieldTransform( displacementField )

	else:
		R = sitk.ImageRegistrationMethod()

		if reg_type == 'sgd-ms':
			R.SetMetricAsMeanSquares()
			R.SetOptimizerAsRegularStepGradientDescent(4.0, .01, 200 )
			R.SetInitialTransform(sitk.TranslationTransform(fixed.GetDimension()))

		elif reg_type == 'gdls':
			fixed = sitk.Normalize(fixed)
			fixed = sitk.DiscreteGaussian(fixed, 2.0)
			moving = sitk.Normalize(moving)
			moving = sitk.DiscreteGaussian(moving, 2.0)

			R.SetMetricAsJointHistogramMutualInformation()
			R.SetOptimizerAsGradientDescentLineSearch(learningRate=1.0,
										  numberOfIterations=200,
										  convergenceMinimumValue=1e-5,
										  convergenceWindowSize=5)
			R.SetInitialTransform(sitk.TranslationTransform(fixed.GetDimension()))

		elif reg_type == 'sgd-corr':
			#doesn't work
			R.SetMetricAsCorrelation()
			R.SetOptimizerAsRegularStepGradientDescent(learningRate=2.0,
										   minStep=1e-4,
										   numberOfIterations=500,
										   gradientMagnitudeTolerance=1e-8 )
			R.SetOptimizerScalesFromIndexShift()
			tx = sitk.CenteredTransformInitializer(fixed, moving, sitk.Similarity2DTransform())
			R.SetInitialTransform(tx)


		elif reg_type == 'sgd-mi':
			numberOfBins = 24
			samplingPercentage = 0.10

			R.SetMetricAsMattesMutualInformation(numberOfBins)
			R.SetMetricSamplingPercentage(samplingPercentage,sitk.sitkWallClock)
			R.SetMetricSamplingStrategy(R.RANDOM)
			R.SetOptimizerAsRegularStepGradientDescent(1.0,.001,200)
			R.SetInitialTransform(sitk.TranslationTransform(fixed.GetDimension()))


		elif reg_type == 'bspline-corr':
			transformDomainMeshSize=[8]*moving.GetDimension()
			tx = sitk.BSplineTransformInitializer(fixed, transformDomainMeshSize )
			
			R.SetMetricAsCorrelation()

			R.SetOptimizerAsLBFGSB(gradientConvergenceTolerance=1e-5,
								   numberOfIterations=100,
								   maximumNumberOfCorrections=5,
								   maximumNumberOfFunctionEvaluations=1000,
								   costFunctionConvergenceFactor=1e+7)
			R.SetInitialTransform(tx, True)

		elif reg_type == 'bspline-mi':
			transformDomainMeshSize=[10]*moving.GetDimension()
			tx = sitk.BSplineTransformInitializer(fixed, transformDomainMeshSize )
			
			R.SetMetricAsMattesMutualInformation(50)
			R.SetOptimizerAsGradientDescentLineSearch(5.0, 100,
													  convergenceMinimumValue=1e-4,
													  convergenceWindowSize=5)
			R.SetOptimizerScalesFromPhysicalShift( )
			R.SetInitialTransform(tx)

		elif reg_type == 'disp':
			initialTx = sitk.CenteredTransformInitializer(fixed, moving, sitk.AffineTransform(fixed.GetDimension()))

			R = sitk.ImageRegistrationMethod()

			R.SetShrinkFactorsPerLevel([3,2,1])
			R.SetSmoothingSigmasPerLevel([2,1,1])

			R.SetMetricAsJointHistogramMutualInformation(20)
			R.MetricUseFixedImageGradientFilterOff()
			R.MetricUseFixedImageGradientFilterOff()


			R.SetOptimizerAsGradientDescent(learningRate=1.0,
											numberOfIterations=100,
											estimateLearningRate = R.EachIteration)
			R.SetOptimizerScalesFromPhysicalShift()

			R.SetInitialTransform(initialTx,inPlace=True)

		elif reg_type == 'exhaust':
			R = sitk.ImageRegistrationMethod()

			R.SetMetricAsMattesMutualInformation(numberOfHistogramBins = 50)

			sample_per_axis=12
			if fixed.GetDimension() == 2:
				tx = sitk.Euler2DTransform()
				# Set the number of samples (radius) in each dimension, with a
				# default step size of 1.0
				R.SetOptimizerAsExhaustive([sample_per_axis//2,0,0])
				# Utilize the scale to set the step size for each dimension
				R.SetOptimizerScales([2.0*pi/sample_per_axis, 1.0,1.0])
			elif fixed.GetDimension() == 3:
				tx = sitk.Euler3DTransform()
				R.SetOptimizerAsExhaustive([sample_per_axis//2,sample_per_axis//2,sample_per_axis//4,0,0,0])
				R.SetOptimizerScales([2.0*pi/sample_per_axis,2.0*pi/sample_per_axis,2.0*pi/sample_per_axis,1.0,1.0,1.0])

			# Initialize the transform with a translation and the center of
			# rotation from the moments of intensity.
			tx = sitk.CenteredTransformInitializer(fixed, moving, tx)

			R.SetInitialTransform(tx)


		R.SetInterpolator(sitk.sitkLinear)

		if reg_type == 'bspline-mi':
			R.SetShrinkFactorsPerLevel([6,2,1])
			R.SetSmoothingSigmasPerLevel([6,2,1])


		if verbose:
			def command_iteration(method) :
				print("{0:3} = {1:10.5f} : {2}".format(method.GetOptimizerIteration(),
											   method.GetMetricValue(),
											   method.GetOptimizerPosition()))
			R.AddCommand( sitk.sitkIterationEvent, lambda: command_iteration(R) )

		outTx = R.Execute(fixed, moving)

		if reg_type == 'disp':
			R.SetMovingInitialTransform(outTx)
			R.SetInitialTransform(displacementTx, inPlace=True)

			R.SetMetricAsANTSNeighborhoodCorrelation(4)
			R.MetricUseFixedImageGradientFilterOff()
			R.MetricUseFixedImageGradientFilterOff()


			R.SetShrinkFactorsPerLevel([3,2,1])
			R.SetSmoothingSigmasPerLevel([2,1,1])

			R.SetOptimizerScalesFromPhysicalShift()
			R.SetOptimizerAsGradientDescent(learningRate=1,
											numberOfIterations=300,
											estimateLearningRate=R.EachIteration)

			outTx.AddTransform( R.Execute(fixed, moving) )

	if verbose:
		print("-------")
		print(outTx)
		print("Optimizer stop condition: {0}".format(R.GetOptimizerStopConditionDescription()))
		print(" Iteration: {0}".format(R.GetOptimizerIteration()))
		print(" Metric value: {0}".format(R.GetMetricValue()))

	resampler = sitk.ResampleImageFilter()
	resampler.SetReferenceImage(fixed)
	resampler.SetInterpolator(sitk.sitkLinear)
	resampler.SetDefaultPixelValue(0)
	resampler.SetTransform(outTx)
	out_img = resampler.Execute(moving)
	sitk.WriteTransform(outTx, out_transform_path)
	sitk.WriteImage(out_img, out_img_path)

def transform(moving_img_path, transform_path, fixed_img_path, out_img_path="default",
	path_to_bis="C:\\yale\\bioimagesuite35\\bin\\"):
	"""Transforms based on existing transform. fixed_img_path is to define final dimensions."""
	
	temp_img_path = ".\\temp_out_img.nii"
	temp_xform_path = ".\\temp_out_xform"

	if out_img_path == "default":
		out_img_path = add_to_filename(moving_img_path, "-reg")
	
	cmd = ''.join([path_to_bis, "bis_linearintensityregister.bat -inp ", fixed_img_path,
			  " -inp2 ", moving_img_path, " -out ", temp_xform_path,
			  " -useinitial ", transform_path, " -iterations 0"]).replace("\\","/")

	subprocess.run(cmd.split())
	shutil.copy(temp_img_path, out_img_path)
	os.remove(temp_img_path)
	os.remove(temp_xform_path)

	return out_img_path

def transform_sitk(moving_path, transform_path, target_path=None):
	"""Transforms without scaling image"""
	
	if target_path is None:
		target_path = moving_path
	
	moving = sitk.ReadImage(moving_path, sitk.sitkFloat32)
	tx = sitk.ReadTransform(transform_path)
	moving_reg = sitk.Resample(moving, tx)
	sitk.WriteImage(moving_reg, target_path)


###########################
### IMAGE FEATURES
###########################

def get_vol(img, dims, dim_units):
	return np.sum(img>0) * np.prod(dims)

def get_hist(img, bins=None, plot_fig=True):
	"""Returns histogram in array and graphical forms."""
	h = plt.hist(flatten(img, times=len(img.shape)-1), bins=bins)
	plt.title("Histogram")
	plt.xlabel("Value")
	plt.ylabel("Frequency")
	
	#mean_intensity = np.mean(diff[img > 0])
	#std_intensity = np.std(diff[img > 0])

	return h, plt.gcf()
	

#########################
### UTILITY
#########################

def _plot_without_axes(img, cmap):
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
			img[0,0,:,:]=-.7
			img[0,-1,:,:]=.7

		if frac is None:
			plt.subplot(131)
			_plot_without_axes(np.transpose(img[:, :, img.shape[2]//2, 0], (1,0)), cmap='gray')
			plt.subplot(132)
			_plot_without_axes(np.transpose(img[:, :, img.shape[2]//2, 1], (1,0)), cmap='gray')
			plt.subplot(133)
			_plot_without_axes(np.transpose(img[:, :, img.shape[2]//2, 2], (1,0)), cmap='gray')
		else:
			plt.subplot(231)
			_plot_without_axes(np.transpose(img[:, :, img.shape[2]//2, 0], (1,0)), cmap='gray')
			plt.subplot(232)
			_plot_without_axes(np.transpose(img[:, :, img.shape[2]//2, 1], (1,0)), cmap='gray')
			plt.subplot(233)
			_plot_without_axes(np.transpose(img[:, :, img.shape[2]//2, 2], (1,0)), cmap='gray')

			plt.subplot(234)
			_plot_without_axes(np.transpose(img[:, :, int(img.shape[2]*frac), 0], (1,0)), cmap='gray')
			plt.subplot(235)
			_plot_without_axes(np.transpose(img[:, :, int(img.shape[2]*frac), 1], (1,0)), cmap='gray')
			plt.subplot(236)
			_plot_without_axes(np.transpose(img[:, :, int(img.shape[2]*frac), 2], (1,0)), cmap='gray')

	else:
		img = copy.deepcopy(orig_img)
		if normalize:
			img[0,0,:]=-1
			img[0,-1,:]=.8
		if frac is not None:
			print("frac does nothing for 3D images.")

		plt.subplot(131)
		_plot_without_axes(np.transpose(img[:, :, img.shape[2]//4], (1,0)), cmap='gray')
		plt.subplot(132)
		_plot_without_axes(np.transpose(img[:, :, img.shape[2]//2], (1,0)), cmap='gray')
		plt.subplot(133)
		_plot_without_axes(np.transpose(img[:, :, img.shape[2]*3//4], (1,0)), cmap='gray')

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
	"""Draw a slice of an image of type np array and save it to disk."""
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
	
###########################
### MISC
###########################

def add_to_filename(fn, addition):
	x = fn.find(".")
	return fn[:x] + addition + fn[x:]

def str_to_lists(raw, dtype=float):
	bigstr = str(raw)
	bigstr = re.sub(r'(\d)\s+(\d)', r'\1,\2', bigstr)
	bigstr = re.sub(r'(\d)\s+(\d)', r'\1,\2', bigstr)
	bigstr = re.sub(r'\]\s*\[', r';', bigstr)
	bigstr = bigstr.replace('[', '')
	bigstr = bigstr.replace(']', '')
	bigstr = bigstr.replace(' ', '')
	ret = [[dtype(x) for x in sublist.split(',')] for sublist in bigstr.split(';')]

	return ret