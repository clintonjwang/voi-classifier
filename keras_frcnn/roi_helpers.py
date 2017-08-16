import numpy as np
import math
from . import data_generators
import copy


def calc_iou(R, img_data, C, class_mapping):
	bboxes = img_data['bboxes']
	(w, h, d) = (img_data['width'], img_data['height'], img_data['depth'])
	# get image dimensions for resizing
	(resized_w, resized_h, resized_d) = data_generators.get_new_img_size(w, h, d, C.im_size)

	gta = np.zeros((len(bboxes), 6))

	for bbox_num, bbox in enumerate(bboxes):
		# get the GT box coordinates, and resize to account for image resizing
		gta[bbox_num, 0] = int(round(bbox['x1'] * (resized_w / float(w))/C.rpn_stride))
		gta[bbox_num, 1] = int(round(bbox['x2'] * (resized_w / float(w))/C.rpn_stride))
		gta[bbox_num, 2] = int(round(bbox['y1'] * (resized_h / float(h))/C.rpn_stride))
		gta[bbox_num, 3] = int(round(bbox['y2'] * (resized_h / float(h))/C.rpn_stride))
		gta[bbox_num, 4] = int(round(bbox['z1'] * (resized_d / float(d))/C.rpn_stride))
		gta[bbox_num, 5] = int(round(bbox['z2'] * (resized_d / float(d))/C.rpn_stride))

	x_roi = []
	y_class_num = []
	y_class_regr_coords = []
	y_class_regr_label = []
	IoUs = [] # for debugging only

	for ix in range(R.shape[0]):
		(x1, y1, z1, x2, y2, z2) = R[ix, :]
		x1 = int(round(x1))
		y1 = int(round(y1))
		z1 = int(round(z1))
		x2 = int(round(x2))
		y2 = int(round(y2))
		z2 = int(round(z2))

		best_iou = 0.0
		best_bbox = -1
		for bbox_num in range(len(bboxes)):
			curr_iou = data_generators.iou([gta[bbox_num, 0], gta[bbox_num, 3],
				gta[bbox_num, 1], gta[bbox_num, 4], gta[bbox_num, 2],
				gta[bbox_num, 5]], [x1, y1, z1, x2, y2, z2])
			if curr_iou > best_iou:
				best_iou = curr_iou
				best_bbox = bbox_num

		if best_iou < C.classifier_min_overlap:
				continue
		else:
			w = x2 - x1
			h = y2 - y1
			d = z2 - z1
			x_roi.append([x1, y1, z1, w, h, d])
			IoUs.append(best_iou)

			if C.classifier_min_overlap <= best_iou < C.classifier_max_overlap:
				# hard negative example
				cls_name = 'bg'
			elif C.classifier_max_overlap <= best_iou:
				cls_name = bboxes[best_bbox]['class']
				cxg = (gta[best_bbox, 0] + gta[best_bbox, 1]) / 2.0
				cyg = (gta[best_bbox, 2] + gta[best_bbox, 3]) / 2.0
				czg = (gta[best_bbox, 4] + gta[best_bbox, 5]) / 2.0

				cx = x1 + w / 2.0
				cy = y1 + h / 2.0
				cz = z1 + d / 2.0

				tx = (cxg - cx) / float(w)
				ty = (cyg - cy) / float(h)
				tz = (czg - cz) / float(d)
				tw = np.log((gta[best_bbox, 1] - gta[best_bbox, 0]) / float(w))
				th = np.log((gta[best_bbox, 3] - gta[best_bbox, 2]) / float(h))
				td = np.log((gta[best_bbox, 5] - gta[best_bbox, 4]) / float(d))
			else:
				print('roi = {}'.format(best_iou))
				raise RuntimeError

		nb_params = 6
		class_num = class_mapping[cls_name]
		class_label = len(class_mapping) * [0]
		class_label[class_num] = 1
		y_class_num.append(copy.deepcopy(class_label))
		coords = [0] * nb_params * (len(class_mapping) - 1)
		labels = [0] * nb_params * (len(class_mapping) - 1)
		if cls_name != 'bg':
			label_pos = nb_params * class_num
			sx, sy, sz, sw, sh, sd = C.classifier_regr_std
			coords[label_pos:nb_params+label_pos] = [sx*tx, sy*ty, sz*tz, sw*tw, sh*th, sd*td]
			labels[label_pos:nb_params+label_pos] = [1, 1, 1, 1, 1, 1]
			y_class_regr_coords.append(copy.deepcopy(coords))
			y_class_regr_label.append(copy.deepcopy(labels))
		else:
			y_class_regr_coords.append(copy.deepcopy(coords))
			y_class_regr_label.append(copy.deepcopy(labels))

	if len(x_roi) == 0:
		return None, None, None, None, None, None

	X = np.array(x_roi)
	Y1 = np.array(y_class_num)
	Y2 = np.concatenate([np.array(y_class_regr_label),np.array(y_class_regr_coords)],axis=1)

	return np.expand_dims(X, axis=0), np.expand_dims(Y1, axis=0), np.expand_dims(Y2, axis=0), IoUs

def apply_regr(x, y, z, w, h, d, tx, ty, tz, tw, th, td):
	raise ValueError("Should not be reachable")

	try:
		cx = x + w/2.
		cy = y + h/2.
		cz = z + d/2.
		cx1 = tx * w + cx
		cy1 = ty * h + cy
		cz1 = tz * d + cz
		w1 = round(math.exp(tw) * w)
		h1 = round(math.exp(th) * h)
		d1 = round(math.exp(td) * d)
		x1 = round(cx1 - w1/2.)
		y1 = round(cy1 - h1/2.)
		z1 = round(cz1 - d1/2.)

		return x1, y1, z1, w1, h1, d1

	except ValueError:
		return x, y, z, w, h, d
	except OverflowError:
		return x, y, z, w, h, d
	except Exception as e:
		print(e)
		return x, y, z, w, h, d

def apply_regr_np(X, T):
	try:
		x = X[0, :, :]
		y = X[1, :, :]
		z = X[2, :, :]
		w = X[3, :, :]
		h = X[4, :, :]
		d = X[5, :, :]

		tx = T[0, :, :]
		ty = T[1, :, :]
		tz = T[2, :, :]
		tw = T[3, :, :]
		th = T[4, :, :]
		td = T[5, :, :]

		cx = x + w/2.
		cy = y + h/2.
		cz = z + d/2.
		cx1 = tx * w + cx
		cy1 = ty * h + cy
		cz1 = tz * d + cz

		w1 = np.round(np.exp(tw.astype(np.float64)) * w)
		h1 = np.round(np.exp(th.astype(np.float64)) * h)
		d1 = np.round(np.exp(td.astype(np.float64)) * d)
		x1 = np.round(cx1 - w1/2.)
		y1 = np.round(cy1 - h1/2.)
		z1 = np.round(cz1 - d1/2.)

		return np.stack([x1, y1, z1, w1, h1, d1])
	except Exception as e:
		print(e)
		return X

def non_max_suppression_fast(boxes, probs, overlap_thresh=0.9, max_boxes=300):
	# code used from here: http://www.pyimagesearch.com/2015/02/16/faster-non-maximum-suppression-python/
	# if there are no boxes, return an empty list
	if len(boxes) == 0:
		return []

	# grab the coordinates of the bounding boxes
	x1 = boxes[:, 0]
	y1 = boxes[:, 1]
	z1 = boxes[:, 2]
	x2 = boxes[:, 3]
	y2 = boxes[:, 4]
	z2 = boxes[:, 5]

	np.testing.assert_array_less(x1, x2)
	np.testing.assert_array_less(y1, y2)
	np.testing.assert_array_less(z1, z2)

	# if the bounding boxes integers, convert them to floats --
	# this is important since we'll be doing a bunch of divisions
	if boxes.dtype.kind == "i":
		boxes = boxes.astype("float")

	# initialize the list of picked indexes	
	pick = []

	# calculate the areas
	vol = (x2 - x1) * (y2 - y1) * (z2 - z1)

	# sort the bounding boxes 
	idxs = np.argsort(probs)

	# keep looping while some indexes still remain in the indexes
	# list
	while len(idxs) > 0:
		# grab the last index in the indexes list and add the
		# index value to the list of picked indexes
		last = len(idxs) - 1
		i = idxs[last]
		pick.append(i)

		# find the intersection

		xx1_int = np.maximum(x1[i], x1[idxs[:last]])
		yy1_int = np.maximum(y1[i], y1[idxs[:last]])
		zz1_int = np.maximum(z1[i], z1[idxs[:last]])
		xx2_int = np.minimum(x2[i], x2[idxs[:last]])
		yy2_int = np.minimum(y2[i], y2[idxs[:last]])
		zz2_int = np.minimum(z2[i], z2[idxs[:last]])

		ww_int = np.maximum(0, xx2_int - xx1_int)
		hh_int = np.maximum(0, yy2_int - yy1_int)
		dd_int = np.maximum(0, zz2_int - zz1_int)

		vol_int = ww_int * hh_int * dd_int

		# find the union
		vol_union = vol[i] + vol[idxs[:last]] - vol_int

		# compute the ratio of overlap
		overlap = vol_int/(vol_union + 1e-6)

		# delete all indexes from the index list that have
		idxs = np.delete(idxs, np.concatenate(([last],
			np.where(overlap > overlap_thresh)[0])))

		if len(pick) >= max_boxes:
			break

	# return only the bounding boxes that were picked using the integer data type
	boxes = boxes[pick].astype("int")
	probs = probs[pick]
	return boxes, probs

def rpn_to_roi(rpn_layer, regr_layer, C, use_regr=True, max_boxes=300,overlap_thresh=0.9):

	regr_layer = regr_layer / C.std_scaling

	anchor_sizes = C.anchor_box_scales
	anchor_ratios = C.anchor_box_ratios

	assert rpn_layer.shape[0] == 1

	(rows, cols, slices) = rpn_layer.shape[1:4]

	curr_layer = 0
	A = np.zeros((4, rpn_layer.shape[1], rpn_layer.shape[2], rpn_layer.shape[3], rpn_layer.shape[4]))

	for anchor_size in anchor_sizes:
		for anchor_ratio in anchor_ratios:

			anchor_x = (anchor_size * anchor_ratio[0])/C.rpn_stride
			anchor_y = (anchor_size * anchor_ratio[1])/C.rpn_stride
			anchor_z = (anchor_size * anchor_ratio[2])/C.rpn_stride
			regr = regr_layer[0, :, :, :, 4 * curr_layer:4 * curr_layer + 4]
			regr = np.transpose(regr, (3, 0, 1, 2))

			X, Y, Z = np.meshgrid(np.arange(cols), np.arange(rows), np.arange(slices))

			A[0, :, :, :, curr_layer] = X - anchor_x/2
			A[1, :, :, :, curr_layer] = Y - anchor_y/2
			A[2, :, :, :, curr_layer] = Z - anchor_z/2
			A[3, :, :, :, curr_layer] = anchor_x
			A[4, :, :, :, curr_layer] = anchor_y
			A[5, :, :, :, curr_layer] = anchor_z

			if use_regr:
				A[:, :, :, :, curr_layer] = apply_regr_np(A[:, :, :, :, curr_layer], regr)

			A[3, :, :, :, curr_layer] = np.maximum(1, A[3, :, :, :, curr_layer]) + A[0, :, :, :, curr_layer]
			A[4, :, :, :, curr_layer] = np.maximum(1, A[4, :, :, :, curr_layer]) + A[1, :, :, :, curr_layer]
			A[5, :, :, :, curr_layer] = np.maximum(1, A[5, :, :, :, curr_layer]) + A[2, :, :, :, curr_layer]

			A[0, :, :, :, curr_layer] = np.maximum(0, A[0, :, :, :, curr_layer])
			A[1, :, :, :, curr_layer] = np.maximum(0, A[1, :, :, :, curr_layer])
			A[2, :, :, :, curr_layer] = np.maximum(0, A[2, :, :, :, curr_layer])
			A[3, :, :, :, curr_layer] = np.minimum(cols-1, A[3, :, :, :, curr_layer])
			A[4, :, :, :, curr_layer] = np.minimum(rows-1, A[4, :, :, :, curr_layer])
			A[5, :, :, :, curr_layer] = np.minimum(slices-1, A[5, :, :, :, curr_layer])

			curr_layer += 1

	all_boxes = np.reshape(A.transpose((0,4,1,2,3)), (6,-1)).transpose((1, 0))
	all_probs = rpn_layer.transpose((0,4,1,2,3)).reshape((-1))

	x1 = all_boxes[:, 0]
	y1 = all_boxes[:, 1]
	z1 = all_boxes[:, 2]
	x2 = all_boxes[:, 3]
	y2 = all_boxes[:, 4]
	z2 = all_boxes[:, 5]

	idxs = np.where((x1 - x2 >= 0) | (y1 - y2 >= 0) | (z1 - z2 >= 0))

	all_boxes = np.delete(all_boxes, idxs, 0)
	all_probs = np.delete(all_probs, idxs, 0)

	result = non_max_suppression_fast(all_boxes, all_probs, overlap_thresh=overlap_thresh, max_boxes=max_boxes)[0]

	return result
