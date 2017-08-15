from __future__ import absolute_import
import numpy as np
import cv2
import random
import copy
from . import data_augment
import threading
import itertools
from skimage.transform import resize

def get_data(input_path):
	found_bg = False
	all_imgs = {}
	classes_count = {}
	class_mapping = {}
	visualise = True
	
	with open(input_path,'r') as f:

		print('Parsing annotation files')

		for line in f:
			line_split = line.strip().split(',')
			(filename,x1,y1,z1,x2,y2,z2,class_name) = line_split

			if class_name not in classes_count:
				classes_count[class_name] = 1
			else:
				classes_count[class_name] += 1

			if class_name not in class_mapping:
				if class_name == 'bg' and found_bg == False:
					print('Found class name with special name bg. Will be treated as a background region (this is usually for hard negative mining).')
					found_bg = True
				class_mapping[class_name] = len(class_mapping)

			if filename not in all_imgs:
				all_imgs[filename] = {}
				
				img = np.load(filename)
				(h,w,d) = img.shape[:3]
				all_imgs[filename]['filepath'] = filename
				all_imgs[filename]['w'] = w
				all_imgs[filename]['h'] = h
				all_imgs[filename]['depth'] = d
				all_imgs[filename]['bboxes'] = []
				if np.random.randint(0,6) > 0:
					all_imgs[filename]['imageset'] = 'trainval'
				else:
					all_imgs[filename]['imageset'] = 'test'

			all_imgs[filename]['bboxes'].append({'class': class_name, 'x1': int(x1), 'x2': int(x2),
				'y1': int(y1), 'y2': int(y2), 'z1': int(z1), 'z2': int(z2)})


		all_data = []
		for key in all_imgs:
			all_data.append(all_imgs[key])
		
		# make sure the bg class is last in the list
		if found_bg:
			if class_mapping['bg'] != len(class_mapping) - 1:
				key_to_switch = [key for key in class_mapping.keys() if class_mapping[key] == len(class_mapping)-1][0]
				val_to_switch = class_mapping['bg']
				class_mapping['bg'] = len(class_mapping) - 1
				class_mapping[key_to_switch] = val_to_switch
		
		return all_data, classes_count, class_mapping


def union(au, bu, vol_intersection):
	# au and bu should be (x1,y1,z1,x2,y2,z2)
	vol_a = (au[3] - au[0]) * (au[4] - au[1]) * (au[5] - au[2])
	vol_b = (bu[3] - bu[0]) * (bu[4] - bu[1]) * (au[5] - au[2])
	vol_union = vol_a + vol_b - vol_intersection
	return vol_union


def intersection(ai, bi):
	w = min(ai[3], bi[3]) - max(ai[0], bi[0])
	h = min(ai[4], bi[4]) - max(ai[1], bi[1])
	d = min(ai[5], bi[5]) - max(ai[2], bi[2])
	if w < 0 or h < 0 or d < 0:
		return 0
	return w*h*d


def iou(a, b):
	# a and b should be (x1,y1,z1,x2,y2,z2)

	if a[0] >= a[3] or a[1] >= a[4] or a[2] >= a[5] or b[0] >= b[2] or b[1] >= b[3] or b[2] >= b[5]:
		return 0.0

	vol_i = intersection(a, b)
	vol_u = union(a, b, vol_i)

	return float(vol_i) / float(vol_u + 1e-6)


def get_new_img_size(w, h, d, img_min_side=600):
	if w <= h and w <= d:
		f = float(img_min_side) / w
		resized_w = img_min_side
		resized_h = int(f * h)
		resized_d = int(f * d)
	elif d <= w and d <= h:
		f = float(img_min_side) / d
		resized_w = int(f * w)
		resized_h = int(f * h)
		resized_d = img_min_side
	else:
		f = float(img_min_side) / h
		resized_w = int(f * w)
		resized_h = img_min_side
		resized_d = int(f * d)

	return resized_w, resized_h, resized_d


class SampleSelector:
	def __init__(self, class_count):
		# ignore classes that have zero samples
		self.classes = [b for b in class_count.keys() if class_count[b] > 0]
		self.class_cycle = itertools.cycle(self.classes)
		self.curr_class = next(self.class_cycle)

	def skip_sample_for_balanced_class(self, img_data):

		class_in_img = False

		for bbox in img_data['bboxes']:

			cls_name = bbox['class']

			if cls_name == self.curr_class:
				class_in_img = True
				self.curr_class = next(self.class_cycle)
				break

		if class_in_img:
			return False
		else:
			return True


def calc_rpn(C, img_data, w, h, d, resized_w, resized_h, resized_d, img_length_calc_function):
	dims = 3
	downscale = float(C.rpn_stride)
	anchor_sizes = C.anchor_box_scales
	anchor_ratios = C.anchor_box_ratios
	num_anchors = len(anchor_sizes) * len(anchor_ratios)	

	# calculate the output map size based on the network architecture

	(output_w, output_h, output_d) = img_length_calc_function(resized_w, resized_h, resized_d)

	n_anchratios = len(anchor_ratios)
	
	# initialise empty output objectives
	y_rpn_overlap = np.zeros((output_h, output_w, output_d, num_anchors))
	y_is_box_valid = np.zeros((output_h, output_w, output_d, num_anchors))
	y_rpn_regr = np.zeros((output_h, output_w, output_d, num_anchors * dims * 2))

	num_bboxes = len(img_data['bboxes'])

	num_anchors_for_bbox = np.zeros(num_bboxes).astype(int)
	best_anchor_for_bbox = -1*np.ones((num_bboxes, dims+2)).astype(int)
	best_iou_for_bbox = np.zeros(num_bboxes).astype(np.float32)
	best_x_for_bbox = np.zeros((num_bboxes, dims*2)).astype(int)
	best_dx_for_bbox = np.zeros((num_bboxes, dims*2)).astype(np.float32)

	# get the GT box coordinates, and resize to account for image resizing
	gta = np.zeros((num_bboxes, 6))
	for bbox_num, bbox in enumerate(img_data['bboxes']):
		# get the GT box coordinates, and resize to account for image resizing
		gta[bbox_num, 0] = bbox['x1'] * (resized_w / float(w))
		gta[bbox_num, 1] = bbox['x2'] * (resized_w / float(w))
		gta[bbox_num, 2] = bbox['y1'] * (resized_h / float(h))
		gta[bbox_num, 3] = bbox['y2'] * (resized_h / float(h))
		gta[bbox_num, 4] = bbox['z1'] * (resized_d / float(d))
		gta[bbox_num, 5] = bbox['z2'] * (resized_d / float(d))
	
	# rpn ground truth

	for anchor_size_idx in range(len(anchor_sizes)):
		for anchor_ratio_idx in range(n_anchratios):
			anchor_x = anchor_sizes[anchor_size_idx] * anchor_ratios[anchor_ratio_idx][0]
			anchor_y = anchor_sizes[anchor_size_idx] * anchor_ratios[anchor_ratio_idx][1]
			anchor_z = anchor_sizes[anchor_size_idx] * anchor_ratios[anchor_ratio_idx][2]
			
			for ix in range(output_w):					
				# x-coordinates of the current anchor box	
				x1_anc = downscale * (ix + 0.5) - anchor_x / 2
				x2_anc = downscale * (ix + 0.5) + anchor_x / 2
				
				# ignore boxes that go across image boundaries					
				if x1_anc < 0 or x2_anc > resized_w:
					continue
					
				for jy in range(output_h):

					# y-coordinates of the current anchor box
					y1_anc = downscale * (jy + 0.5) - anchor_y / 2
					y2_anc = downscale * (jy + 0.5) + anchor_y / 2

					# ignore boxes that go across image boundaries
					if y1_anc < 0 or y2_anc > resized_h:
						continue
					
					for iz in range(output_h):

						# y-coordinates of the current anchor box
						z1_anc = downscale * (iz + 0.5) - anchor_z / 2
						z2_anc = downscale * (iz + 0.5) + anchor_z / 2

						# ignore boxes that go across image boundaries
						if z1_anc < 0 or z2_anc > resized_d:
							continue

						# bbox_type indicates whether an anchor should be a target 
						bbox_type = 'neg'

						# this is the best IOU for the (x,y) coord and the current anchor
						# note that this is different from the best IOU for a GT bbox
						best_iou_for_loc = 0.0

						for bbox_num in range(num_bboxes):
							
							# get IOU of the current GT box and the current anchor box
							curr_iou = iou([gta[bbox_num, 0], gta[bbox_num, 3],
								gta[bbox_num, 1], gta[bbox_num, 4]],
								gta[bbox_num, 2], gta[bbox_num, 5]],
								[x1_anc, y1_anc, z1_anc, x2_anc, y2_anc, z2_anc])
							# calculate the regression targets if they will be needed
							if curr_iou > best_iou_for_bbox[bbox_num] or curr_iou > C.rpn_max_overlap:
								cx = (gta[bbox_num, 0] + gta[bbox_num, 1]) / 2.0
								cy = (gta[bbox_num, 2] + gta[bbox_num, 3]) / 2.0
								cz = (gta[bbox_num, 4] + gta[bbox_num, 5]) / 2.0
								cxa = (x1_anc + x2_anc)/2.0
								cya = (y1_anc + y2_anc)/2.0
								cza = (z1_anc + z2_anc)/2.0

								tx = (cx - cxa) / (x2_anc - x1_anc)
								ty = (cy - cya) / (y2_anc - y1_anc)
								tz = (cz - cza) / (z2_anc - z1_anc)
								tw = np.log((gta[bbox_num, 1] - gta[bbox_num, 0]) / (x2_anc - x1_anc))
								th = np.log((gta[bbox_num, 3] - gta[bbox_num, 2]) / (y2_anc - y1_anc))
								td = np.log((gta[bbox_num, 5] - gta[bbox_num, 4]) / (z2_anc - z1_anc))
							
							if img_data['bboxes'][bbox_num]['class'] != 'bg':

								# all GT boxes should be mapped to an anchor box, so we keep track of which anchor box was best
								if curr_iou > best_iou_for_bbox[bbox_num]:
									best_anchor_for_bbox[bbox_num] = [jy, ix, iz, anchor_ratio_idx, anchor_size_idx]
									best_iou_for_bbox[bbox_num] = curr_iou
									best_x_for_bbox[bbox_num,:] = [x1_anc, x2_anc, y1_anc, y2_anc, z1_anc, z2_anc]
									best_dx_for_bbox[bbox_num,:] = [tx, ty, tz, tw, th, td]

								# we set the anchor to positive if the IOU is >0.7 (it does not matter if there was another better box, it just indicates overlap)
								if curr_iou > C.rpn_max_overlap:
									bbox_type = 'pos'
									num_anchors_for_bbox[bbox_num] += 1
									# we update the regression layer target if this IOU is the best for the current (x,y) and anchor position
									if curr_iou > best_iou_for_loc:
										best_iou_for_loc = curr_iou
										best_regr = (tx, ty, tz, tw, th, td)

								# if the IOU is >0.3 and <0.7, it is ambiguous and no included in the objective
								if C.rpn_min_overlap < curr_iou < C.rpn_max_overlap:
									# gray zone between neg and pos
									if bbox_type != 'pos':
										bbox_type = 'neutral'

						# turn on or off outputs depending on IOUs
						if bbox_type == 'neg':
							y_is_box_valid[jy, ix, iz, anchor_ratio_idx + n_anchratios * anchor_size_idx] = 1
							y_rpn_overlap[jy, ix, iz, anchor_ratio_idx + n_anchratios * anchor_size_idx] = 0
						elif bbox_type == 'neutral':
							y_is_box_valid[jy, ix, iz, anchor_ratio_idx + n_anchratios * anchor_size_idx] = 0
							y_rpn_overlap[jy, ix, iz, anchor_ratio_idx + n_anchratios * anchor_size_idx] = 0
						elif bbox_type == 'pos':
							y_is_box_valid[jy, ix, iz, anchor_ratio_idx + n_anchratios * anchor_size_idx] = 1
							y_rpn_overlap[jy, ix, iz, anchor_ratio_idx + n_anchratios * anchor_size_idx] = 1
							start = dims*2 * (anchor_ratio_idx + n_anchratios * anchor_size_idx)
							y_rpn_regr[jy, ix, iz, start:start+dims*2] = best_regr

	# we ensure that every bbox has at least one positive RPN region

	for idx in range(num_anchors_for_bbox.shape[0]):
		if num_anchors_for_bbox[idx] == 0:
			# no box with an IOU greater than zero ...
			if best_anchor_for_bbox[idx, 0] == -1:
				continue
			y_is_box_valid[
				best_anchor_for_bbox[idx,0], best_anchor_for_bbox[idx,1], best_anchor_for_bbox[idx,2],
				best_anchor_for_bbox[idx,3] + n_anchratios * best_anchor_for_bbox[idx,4]] = 1
			y_rpn_overlap[
				best_anchor_for_bbox[idx,0], best_anchor_for_bbox[idx,1], best_anchor_for_bbox[idx,2],
				best_anchor_for_bbox[idx,3] + n_anchratios * best_anchor_for_bbox[idx,4]] = 1
			start = dims*2 * (best_anchor_for_bbox[idx,3] + n_anchratios * best_anchor_for_bbox[idx,4])
			y_rpn_regr[
				best_anchor_for_bbox[idx,0], best_anchor_for_bbox[idx,1], best_anchor_for_bbox[idx,2],
				start:start+dims*2] = best_dx_for_bbox[idx, :]

	y_rpn_overlap = np.transpose(y_rpn_overlap, (3, 0, 1, 2))
	y_rpn_overlap = np.expand_dims(y_rpn_overlap, axis=0)

	y_is_box_valid = np.transpose(y_is_box_valid, (3, 0, 1, 2))
	y_is_box_valid = np.expand_dims(y_is_box_valid, axis=0)

	y_rpn_regr = np.transpose(y_rpn_regr, (3, 0, 1, 2))
	y_rpn_regr = np.expand_dims(y_rpn_regr, axis=0)

	pos_locs = np.where(np.logical_and(y_rpn_overlap[0, :, :, :, :] == 1, y_is_box_valid[0, :, :, :, :] == 1))
	neg_locs = np.where(np.logical_and(y_rpn_overlap[0, :, :, :, :] == 0, y_is_box_valid[0, :, :, :, :] == 1))

	num_pos = len(pos_locs[0])

	# one issue is that the RPN has many more negative than positive regions, so we turn off some of the negative
	# regions. We also limit it to 256 regions.
	num_regions = 256

	if len(pos_locs[0]) > num_regions/2:
		val_locs = random.sample(range(len(pos_locs[0])), len(pos_locs[0]) - num_regions/2)
		y_is_box_valid[0, pos_locs[0][val_locs], pos_locs[1][val_locs], pos_locs[2][val_locs], pos_locs[3][val_locs]] = 0
		num_pos = num_regions/2

	if len(neg_locs[0]) + num_pos > num_regions:
		val_locs = random.sample(range(len(neg_locs[0])), len(neg_locs[0]) - num_pos)
		y_is_box_valid[0, neg_locs[0][val_locs], neg_locs[1][val_locs], neg_locs[2][val_locs], pos_locs[3][val_locs]] = 0

	y_rpn_cls = np.concatenate([y_is_box_valid, y_rpn_overlap], axis=1)
	y_rpn_regr = np.concatenate([np.repeat(y_rpn_overlap, dims+2, axis=1), y_rpn_regr], axis=1)

	return np.copy(y_rpn_cls), np.copy(y_rpn_regr)


class threadsafe_iter:
	"""Takes an iterator/generator and makes it thread-safe by
	serializing call to the `next` method of given iterator/generator.
	"""
	def __init__(self, it):
		self.it = it
		self.lock = threading.Lock()

	def __iter__(self):
		return self

	def next(self):
		with self.lock:
			return next(self.it)		

def threadsafe_generator(f):
	"""A decorator that takes a generator function and makes it thread-safe.
	"""
	def g(*a, **kw):
		return threadsafe_iter(f(*a, **kw))
	return g

def get_anchor_gt(all_img_data, class_count, C, img_length_calc_function, backend, mode='train'):

	# The following line is not useful with Python 3.5, it is kept for the legacy
	# all_img_data = sorted(all_img_data)

	sample_selector = SampleSelector(class_count)

	while True:
		if mode == 'train':
			random.shuffle(all_img_data)

		for img_data in all_img_data:
			try:

				if C.balanced_classes and sample_selector.skip_sample_for_balanced_class(img_data):
					continue

				# read in image, and optionally add augmentation

				if mode == 'train':
					img_data_aug, x_img = data_augment.augment(img_data, C, augment=True)
				else:
					img_data_aug, x_img = data_augment.augment(img_data, C, augment=False)

				(w, h, d) = (img_data_aug['w'], img_data_aug['h'], img_data_aug['d'])
				(rows, cols, slices, _) = x_img.shape

				assert cols == w
				assert rows == h
				assert slices == d

				# get image dimensions for resizing
				(resized_w, resized_h, resized_d) = get_new_img_size(w, h, d, C.im_size)

				# resize the image so that smalles side is length = 600px
				x_img = resize(x_img, (resized_w, resized_h, resized_d))
				#x_img = cv2.resize(x_img, (resized_w, resized_h, resized_d), interpolation=cv2.INTER_CUBIC) # does not work in 3D

				try:
					y_rpn_cls, y_rpn_regr = calc_rpn(C, img_data_aug, w, h, d,
						resized_w, resized_h, resized_d, img_length_calc_function)
				except:
					continue

				# Zero-center by mean pixel, and preprocess image

				x_img = x_img[:,:,:, (2, 1, 0)]  # BGR -> RGB
				x_img = x_img.astype(np.float32)
				x_img[:, :, :, 0] -= C.img_channel_mean[0]
				x_img[:, :, :, 1] -= C.img_channel_mean[1]
				x_img[:, :, :, 2] -= C.img_channel_mean[2]
				x_img /= C.img_scaling_factor

				x_img = np.transpose(x_img, (3, 0, 1, 2))
				x_img = np.expand_dims(x_img, axis=0)

				y_rpn_regr[:, y_rpn_regr.shape[1]//2:, :, :, :] *= C.std_scaling

				x_img = np.transpose(x_img, (0, 2, 3, 4, 1))
				y_rpn_cls = np.transpose(y_rpn_cls, (0, 2, 3, 4, 1))
				y_rpn_regr = np.transpose(y_rpn_regr, (0, 2, 3, 4, 1))

				yield np.copy(x_img), [np.copy(y_rpn_cls), np.copy(y_rpn_regr)], img_data_aug

			except Exception as e:
				print(e)
				continue
