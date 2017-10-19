import numpy as np
import cv2
import imutils

def scale3d(img, scale):
	[scalex, scaley, scalez] = scale

	if len(img.shape) == 4:
		inter = np.zeros([round(img.shape[0] * scalex), round(img.shape[1] * scaley), img.shape[2], img.shape[3]])
		scaled = np.zeros([round(img.shape[0] * scalex), round(img.shape[1] * scaley), round(img.shape[2] * scalez), img.shape[3]])
		for ch in range(img.shape[-1]):
			for s in range(img.shape[2]):
				inter[:,:,s,ch] = cv2.resize(img[:,:,s,ch], (0,0), fx=scaley, fy=scalex)
			for s in range(inter.shape[0]):
				scaled[s,:,:,ch] = cv2.resize(inter[s,:,:,ch], (0,0), fx=scalez, fy=1)

	elif len(img.shape) == 3:
		inter = np.zeros([round(img.shape[0] * scalex), round(img.shape[1] * scaley), img.shape[2]])
		scaled = np.zeros([round(img.shape[0] * scalex), round(img.shape[1] * scaley), round(img.shape[2] * scalez)])
		for s in range(img.shape[2]):
			inter[:,:,s] = cv2.resize(img[:,:,s], (0,0), fx=scaley, fy=scalex)
		for s in range(inter.shape[0]):
			scaled[s,:,:] = cv2.resize(inter[s,:,:], (0,0), fx=scalez, fy=1)

	else:
		return None

	return scaled


def scalex(img, scale):
	scaled = np.zeros([round(img.shape[0] * scale)] + list(img.shape[1:]))

	#can replace rotate with rotate_bound to expand image
	if len(img.shape) == 4:
		for ch in range(img.shape[-1]):
			for s in range(img.shape[2]):
				scaled[:,:,s,ch] = cv2.resize(img[:,:,s,ch], (0,0), fx=1, fy=scale)

	elif len(img.shape) == 3:
		for s in range(img.shape[2]):
			scaled[:,:,s] = cv2.resize(img[:,:,s], (0,0), fx=1, fy=scale)

	else:
		return None

	return scaled

def scaley(img, scale):
	scaled = np.zeros([img.shape[0]] + [round(img.shape[1] * scale)] + list(img.shape[2:]))

	#can replace rotate with rotate_bound to expand image
	if len(img.shape) == 4:
		for ch in range(img.shape[-1]):
			for s in range(img.shape[2]):
				scaled[:,:,s,ch] = cv2.resize(img[:,:,s,ch], (0,0), fx=scale, fy=1)

	elif len(img.shape) == 3:
		for s in range(img.shape[2]):
			scaled[:,:,s] = cv2.resize(img[:,:,s], (0,0), fx=scale, fy=1)

	else:
		return None

	return scaled

def scalez(img, scale):
	#can replace rotate with rotate_bound to expand image
	if len(img.shape) == 4:
		scaled = np.zeros(list(img.shape[:2]) + [round(img.shape[2] * scale)] + [img.shape[3]])
		for ch in range(img.shape[-1]):
			for s in range(img.shape[0]):
				scaled[s,:,:,ch] = cv2.resize(img[s,:,:,ch], (0,0), fx=scale, fy=1)

	elif len(img.shape) == 3:
		scaled = np.zeros(list(img.shape[:2]) + [round(img.shape[2] * scale)])
		for s in range(img.shape[0]):
			scaled[s,:,:] = cv2.resize(img[s,:,:], (0,0), fx=scale, fy=1)

	else:
		return None

	return scaled


def rotate(img, angle):
	rotated = np.zeros(img.shape)

	#can replace rotate with rotate_bound to expand image
	for ch in range(img.shape[-1]):
		for s in range(img.shape[-2]):
			rotated[:,:,s,ch] = imutils.rotate(img[:,:,s,ch], angle)
	return rotated


def add_noise(image, noise_typ="gauss"):
	if noise_typ == "gauss":
		row,col,ch= image.shape
		mean = 0
		var = 0.1
		sigma = var**0.5
		gauss = np.random.normal(mean,sigma,(row,col,ch))
		gauss = gauss.reshape(row,col,ch)
		noisy = image + gauss
		return noisy
		
	elif noise_typ == "s&p":
		row,col,ch = image.shape
		s_vs_p = 0.5
		amount = 0.004
		out = np.copy(image)
		# Salt mode
		num_salt = np.ceil(amount * image.size * s_vs_p)
		coords = [np.random.randint(0, i - 1, int(num_salt))
			  for i in image.shape]
		out[coords] = 1

		# Pepper mode
		num_pepper = np.ceil(amount* image.size * (1. - s_vs_p))
		coords = [np.random.randint(0, i - 1, int(num_pepper))
			  for i in image.shape]
		out[coords] = 0
		return out

	elif noise_typ == "poisson":
		vals = len(np.unique(image))
		vals = 2 ** np.ceil(np.log2(vals))
		noisy = np.random.poisson(image * vals) / float(vals)
		return noisy

	elif noise_typ =="speckle":
		row,col,ch = image.shape
		gauss = np.random.randn(row,col,ch)
		gauss = gauss.reshape(row,col,ch)		
		noisy = image + image * gauss
		return noisy

	return None