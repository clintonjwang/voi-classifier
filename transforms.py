import cv2
import imutils
import math
import numpy as np
import random

def scale3d(img, scale):
	[scalex, scaley, scalez] = scale

	if len(img.shape) == 4:
		inter = np.zeros([int(round(img.shape[0] * scalex)), int(round(img.shape[1] * scaley)), img.shape[2], img.shape[3]])
		scaled = np.zeros([int(round(img.shape[0] * scalex)), int(round(img.shape[1] * scaley)),
			int(round(img.shape[2] * scalez)), img.shape[3]])
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

def generate_reflected_img(img):
	"""Randomly generate an image by reflecting half of img across one of its axes"""
	choice = random.randint(1,6)
	if choice==1:
		return np.concatenate([img[:img.shape[0]//2:-1, :,:,:],
						img[math.ceil(img.shape[0]/2)-1:, :,:,:]], axis=0)
	elif choice==2:
		return np.concatenate([img[:, :img.shape[1]//2:-1 ,:,:],
						img[:, math.ceil(img.shape[1]/2)-1: ,:,:]], axis=1)
	elif choice==3:
		return np.flip(np.concatenate([img[:, :, :img.shape[2]//2:-1,:],
						img[:, : , math.ceil(img.shape[2]/2)-1:,:]], axis=2), axis=1)
	elif choice==4:
		return np.concatenate([img[:math.ceil(img.shape[0]/2)-1, :,:,:],
						img[img.shape[0]//2::-1, :,:,:]], axis=0)
	elif choice==5:
		return np.concatenate([img[:, :math.ceil(img.shape[1]/2)-1,:,:],
						img[:, img.shape[1]//2::-1, :,:]], axis=1)
	else:
		return np.flip(np.concatenate([img[:,:, :math.ceil(img.shape[2]/2)-1, :],
						img[:,:, img.shape[2]//2::-1,:]], axis=2), axis=1)

def offset_phases(img, max_offset=2, max_z_offset=1):
	"""Return an img by offsetting the second and third channels of img
	by a random amount up to max_offset, uniformly distributed."""

	xy = max_offset+1
	z = max_z_offset+1

	img = np.pad(img, [(xy, xy), (xy, xy), (z, z), (0,0)], 'edge')

	offset_ch2 = [random.randint(-max_offset, max_offset),
					random.randint(-max_offset, max_offset),
					random.randint(-max_z_offset, max_z_offset)]
	offset_ch3 = [random.randint(-max_offset, max_offset),
					random.randint(-max_offset, max_offset),
					random.randint(-max_z_offset, max_z_offset)]

	return np.stack([img[xy:-xy, xy:-xy, z:-z, 0],
					img[xy+offset_ch2[0]:-xy+offset_ch2[0],
						xy+offset_ch2[1]:-xy+offset_ch2[1],
						z+offset_ch2[2]:-z+offset_ch2[2], 1],
					img[xy+offset_ch3[0]:-xy+offset_ch3[0],
						xy+offset_ch3[1]:-xy+offset_ch3[1],
						z+offset_ch3[2]:-z+offset_ch3[2], 2]], axis=3)