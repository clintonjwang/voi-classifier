
@njit
def get_spatial_overlap(w, f_c3_ch, ch_weights, num_rel_f):
	relevant_features = ch_weights.keys()
	
	ch_weights2 = np.zeros(num_rel_f,128)
	for ch in range(128):
		f_c3_ch[:,:,:,:,ch] = (f_c3_ch[:,:,:,:,ch] - np.mean(f_c3_ch[:,:,:,:,ch]))# / np.std(f_c3_ch[:,:,:,:,ch])
		
		if ch_weights[ch] > 0:
			w = np.zeros(f_c3_ch.shape[:4])
			for i in range(128):
				if ch_weights[i] > 0:
					w += ch_weights[i] * f_c3_ch[:,:,:,:,i]
			w = w / np.sum(ch_weights)
			ch_weights2[ch] = np.sum(f_c3_ch[:,:,:,:,ch] * w)
			
	return ch_weights2

def calculate_W(f_conv_ch, all_features, relevant_features, num_rel_f, num_channels):
	gauss = get_gaussian_mask(1)
	feature_avgs = np.zeros((num_rel_f, num_channels*4))
	for i, f_ix in enumerate(relevant_features):
		f = all_features[f_ix]
		for ch_ix in range(num_channels):
			f_conv_ch[f][:,:,:,ch_ix] *= gauss / np.mean(gauss)
		feature_avgs[i] = get_shells(f_conv_ch[f], f_conv_ch[f].shape[:3])
		
	channel_separations = np.empty((num_rel_f, num_channels*4)) # separation between channel mean activations for the relevant features
	for i in range(num_rel_f):
		channel_separations[i] = (np.amax(feature_avgs, 0) - feature_avgs[i]) / np.mean(feature_avgs, 0)

	channel_separations *= 10

	W = np.zeros((num_rel_f, num_channels*4))
	for ch_ix in range(num_channels*4):
		#f_ix = list(channel_separations[:,ch_ix]).index(0)
		#W[f_ix,ch_ix] = channel_separations[:,ch_ix].mean()
		W[:,ch_ix] = np.median(channel_separations[:,ch_ix]) - channel_separations[:,ch_ix]
		
	for f_ix in range(num_rel_f):
		for i in range(4):
			W[f_ix, num_channels*i:num_channels*(i+1)] += np.mean(W[f_ix, num_channels*i:num_channels*(i+1)])

	W[W < 0] = 0

	WW = np.zeros((num_rel_f, num_channels))
	for ch_ix in range(num_channels):
		WW[:,ch_ix] = np.mean(W[:,[ch_ix, ch_ix+num_channels, ch_ix+num_channels*2, ch_ix+num_channels*3]], 1)

	return W

def get_saliency_map(W, test_neurons, num_rel_f):
	D = np.empty(test_neurons.shape[:3])
	for x in range(D.shape[0]):
		for y in range(D.shape[1]):
			for z in range(D.shape[2]):
				D[x,y,z] = -((D.shape[0]//2-.5-x)**2 + (D.shape[1]//2-.5-y)**2 + 4*(D.shape[2]//2-.5-z)**2)

	num_ch = test_neurons.shape[-1]
	sal_map = np.zeros((num_rel_f, *test_neurons.shape[:3]))
	for f_num in range(num_rel_f):
		for ch_ix in range(num_ch):
			sal_map[f_num, D <= np.percentile(D, 25)] += W[f_num, ch_ix] * test_neurons[D <= np.percentile(D, 25),ch_ix]
			sal_map[f_num, (D <= np.percentile(D, 50)) & (D > np.percentile(D, 25))] += \
					W[f_num, ch_ix+num_ch] * test_neurons[(D <= np.percentile(D, 50)) & (D > np.percentile(D, 25)),ch_ix]
			sal_map[f_num, (D <= np.percentile(D, 75)) & (D > np.percentile(D, 50))] += \
					W[f_num, ch_ix+num_ch*2] * test_neurons[(D <= np.percentile(D, 75)) & (D > np.percentile(D, 50)),ch_ix]
			sal_map[f_num, D > np.percentile(D, 75)] += W[f_num, ch_ix+num_ch*3] * test_neurons[D > np.percentile(D, 75),ch_ix]
		sal_map[f_num] /= np.sum(W[f_num])
		
	#for f_num in range(num_rel_f):
	#    for spatial_ix in np.ndindex(test_neurons.shape[:-1]):
	#        sal_map[f_num, spatial_ix] = sal_map[f_num, spatial_ix]**2 / sal_map[:, spatial_ix].sum()
			
	return sal_map

def tsne(filter_results):
	X = []
	z = [0]
	for i,cls in enumerate(C.cls_names):
		X.append(filter_results[cls])
		z.append(len(filter_results[cls]) + z[-1])
	z.append(len(X))
	X = np.concatenate(X, axis=0)

	X_emb = TSNE(n_components=2, init='pca').fit_transform(X)

	fig = plt.figure()
	ax = fig.add_subplot(111)
	for i, cls in enumerate(C.cls_names):
		ax.scatter(X_emb[z[i]:z[i+1], 0], X_emb[z[i]:z[i+1], 1], color=plt.cm.Set1(i/6.), marker='.', alpha=.8)

	ax.legend(C.short_cls_names, framealpha=0.5)
	ax.set_title("t-SNE")
	ax.xaxis.set_major_formatter(NullFormatter())
	ax.yaxis.set_major_formatter(NullFormatter())
	ax.axis('tight')

	return fig

###########################
### Matrix selection/filtering
###########################

def get_shells(X, dims=(8,8,4)):
	D = np.empty(dims)
	for x in range(D.shape[0]):
		for y in range(D.shape[1]):
			for z in range(D.shape[2]):
				D[x,y,z] = -((D.shape[0]//2-.5-x)**2 + (D.shape[1]//2-.5-y)**2 + 4*(D.shape[2]//2-.5-z)**2)

	shell4 = X[D > np.percentile(D, 75), :].mean(axis=0)
	shell3 = X[(D <= np.percentile(D, 75)) & (D > np.percentile(D, 50)), :].mean(axis=0)
	shell2 = X[(D <= np.percentile(D, 50)) & (D > np.percentile(D, 25)), :].mean(axis=0)
	shell1 = X[D <= np.percentile(D, 25), :].mean(axis=0)

	return np.expand_dims(np.concatenate([shell1, shell2, shell3, shell4]), 0)

def average_shells(X, dims=(8,8,4)):
	D = np.empty(dims)
	for x in range(D.shape[0]):
		for y in range(D.shape[1]):
			for z in range(D.shape[2]):
				D[x,y,z] = -((D.shape[0]//2-.5-x)**2 + (D.shape[1]//2-.5-y)**2 + 4*(D.shape[2]//2-.5-z)**2)

	shell4 = X[D > np.percentile(D, 75), :].mean(axis=0)
	shell3 = X[(D <= np.percentile(D, 75)) & (D > np.percentile(D, 50)), :].mean(axis=0)
	shell2 = X[(D <= np.percentile(D, 50)) & (D > np.percentile(D, 25)), :].mean(axis=0)
	shell1 = X[D <= np.percentile(D, 25), :].mean(axis=0)
	
	num_ch = X.shape[-1]
	for ch_ix in range(num_ch):
		X[D > np.percentile(D, 75), ch_ix] = shell4[ch_ix]
		X[(D <= np.percentile(D, 75)) & (D > np.percentile(D, 50)), ch_ix] = shell3[ch_ix]
		X[(D <= np.percentile(D, 50)) & (D > np.percentile(D, 25)), ch_ix] = shell2[ch_ix]
		X[D <= np.percentile(D, 25), ch_ix] = shell1[ch_ix]

	return X

def get_gaussian_mask(divisor=3):
	gauss = np.zeros((12,12))

	for i in range(gauss.shape[0]):
		for j in range(gauss.shape[1]):
			dx = abs(i - gauss.shape[0]/2+.5)
			dy = abs(j - gauss.shape[1]/2+.5)
			gauss[i,j] = scipy.stats.norm.pdf((dx**2 + dy**2)**.5, 0, gauss.shape[0]//divisor)
	gauss = np.transpose(np.tile(gauss, (6,1,1)), (1,2,0))

	return gauss

def get_rotations(x, front_model, rcnn=False):
	h_ic = [front_model.predict(np.expand_dims(np.rot90(x,r),0))[0] for r in range(4)]
	h_ic += [front_model.predict(np.expand_dims(np.flipud(np.rot90(x,r)),0))[0] for r in range(4)]
	h_ic += [front_model.predict(np.expand_dims(np.fliplr(np.rot90(x,r)),0))[0] for r in range(4)]

	if rcnn:
		for r in range(12):
			h_ic[r] = np.concatenate(h_ic[r], -1)

	h_ic_rot = [np.rot90(h_ic[r], 4-r) for r in range(4)] #rotated back into original frame
	h_ic_rot += [np.rot90(np.flipud(h_ic[r]), 4-r) for r in range(4,8)]
	h_ic_rot += [np.rot90(np.fliplr(h_ic[r]), 4-r) for r in range(8,12)]

	return np.array(h_ic_rot)


###########################
### Feature visualization
###########################

def visualize_activations(model, save_path, target_values, init_img=None, rotate=True, stepsize=.01, num_steps=25):
	"""Visualize the model inputs that would match an activation pattern.
	channel_ixs is the set of channels to optimize over; keep as None to use the whole layer
	Original code by the Keras Team at
	https://github.com/keras-team/keras/blob/master/examples/conv_filter_visualization.py"""
	

	layer_dict = dict([(layer.name, layer) for layer in model.layers[1:]])

	input_img = model.input
	init_img0 = K.constant(copy.deepcopy(np.expand_dims(init_img,0)), 'float32')
	init_img1 = tr.rotate(copy.deepcopy(init_img), 10*pi/180)
	init_img1 = K.constant(np.expand_dims(init_img1,0), 'float32')
	init_img2 = tr.rotate(copy.deepcopy(init_img), -10*pi/180)
	init_img2 = K.constant(np.expand_dims(init_img2,0), 'float32')

	gauss = get_gaussian_mask(2)

	# build a loss function that maximizes the activation
	# of the nth filter of the layer considered
	#layer_output = layer_dict[layer_name].output
	loss = K.sum(K.square(model.output - target_values)) + \
			K.sum(K.abs(input_img - init_img0))/2 + \
			K.sum(K.abs(input_img - init_img1))/5 + \
			K.sum(K.abs(input_img - init_img2))/5
			#10*K.sum(K.std(input_img,(3,4)) - K.std(init_img0,(3,4)))

	# compute the gradient of the input picture wrt this loss
	grads = K.gradients(loss, input_img)[0]
	# normalization trick: we normalize the gradient
	grads /= (K.sqrt(K.mean(K.square(grads))) + 1e-5)
	# this function returns the loss and grads given the input picture
	iterate = K.function([input_img, K.learning_phase()], [loss, grads])

	#loss2 = K.sum(K.square(grads))
	#grads2 = K.gradients(loss, target_values)[0]
	#grads2 /= (K.sqrt(K.mean(K.square(grads2))) + 1e-5)
	#iterate2 = K.function([target_values, K.learning_phase()], [loss, grads2])


	if init_img is None:
		input_img_data = np.random.random((1, C.dims[0], C.dims[1], C.dims[2], 3))
	else:
		input_img_data = np.expand_dims(copy.deepcopy(init_img), 0)

	# run gradient ascent for 20 steps
	if True:
		step = stepsize
		for i in range(num_steps):
			loss_value, grads_value = iterate([input_img_data, 0])
			input_img_data += grads_value * step
			if i % 2 == 0:
				step *= .98
			if rotate and i % 2 == 0:
				#random rotations for transformation robustness, see https://distill.pub/2017/feature-visualization/#enemy-of-feature-vis
				input_img_data = np.pad(input_img_data[0], ((5,5),(5,5),(0,0),(0,0)), 'constant')
				input_img_data = tr.rotate(input_img_data, random.uniform(-5,5)*pi/180)
				input_img_data = np.expand_dims(input_img_data[5:-5, 5:-5, :, :], 0)

	img = input_img_data[0]
	#img = deprocess_image(img)
	hf.draw_slices(img, save_path=save_path)

	return img

def visualize_layer_weighted(model, layer_name, save_path, channel_weights=None, init_img=None):
	"""Visualize the model inputs that would maximally activate a layer.
	channel_ixs is the set of channels to optimize over; keep as None to use the whole layer
	Original code by the Keras Team at
	https://github.com/keras-team/keras/blob/master/examples/conv_filter_visualization.py"""
	

	layer_dict = dict([(layer.name, layer) for layer in model.layers[1:]])

	input_img = model.input

	# build a loss function that maximizes the activation
	# of the nth filter of the layer considered
	layer_output = layer_dict[layer_name].output
	layer_output = K.mean(layer_output, (0,1,2,3))
	loss = K.dot(K.expand_dims(layer_output,0), K.expand_dims(channel_weights))

	# compute the gradient of the input picture wrt this loss
	grads = K.gradients(loss, input_img)[0]

	# normalization trick: we normalize the gradient
	grads /= (K.sqrt(K.mean(K.square(grads))) + 1e-5)

	# this function returns the loss and grads given the input picture
	iterate = K.function([input_img], [loss, grads])

	if init_img is None:
		input_img_data = np.random.random((1, C.dims[0], C.dims[1], C.dims[2], 3))
	else:
		input_img_data = np.expand_dims(init_img, 0)

	# run gradient ascent for 20 steps
	step = 5.
	for i in range(250):
		loss_value, grads_value = iterate([input_img_data])
		input_img_data += grads_value * step
		if i % 2 == 0:
			step *= .98
		if i % 5 == 0:
			#random rotations for transformation robustness, see https://distill.pub/2017/feature-visualization/#enemy-of-feature-vis
			input_img_data = np.pad(input_img_data[0], ((5,5),(5,5),(0,0),(0,0)), 'constant')
			input_img_data = tr.rotate(input_img_data, random.uniform(-5,5)*pi/180)
			input_img_data = np.expand_dims(input_img_data[5:-5, 5:-5, :, :], 0)

	img = input_img_data[0]
	img = deprocess_image(img)
	hf.draw_slices(img, save_path=join(save_path, "%s_filter.png" % layer_name))

def visualize_layer(model, layer_name, save_path, channel_ixs=None, init_img=None):
	"""Visualize the model inputs that would maximally activate a layer.
	channel_ixs is the set of channels to optimize over; keep as None to use the whole layer
	Original code by the Keras Team at
	https://github.com/keras-team/keras/blob/master/examples/conv_filter_visualization.py"""
	from keras import backend as K
	K.set_learning_phase(0)

	

	layer_dict = dict([(layer.name, layer) for layer in model.layers[1:]])

	input_img = model.input

	if channel_ixs is None:
		channel_ixs = list(range(layer_dict[layer_name].output.shape[-1]))

	# build a loss function that maximizes the activation
	# of the nth filter of the layer considered
	layer_output = layer_dict[layer_name].output
	layer_output = K.permute_dimensions(layer_output, (4,0,1,2,3))
	layer_output = K.gather(layer_output, channel_ixs)
	loss = K.mean(layer_output)#[:, :, :, :, channel_ixs])

	# compute the gradient of the input picture wrt this loss
	grads = K.gradients(loss, input_img)[0]

	# normalization trick: we normalize the gradient
	grads /= (K.sqrt(K.mean(K.square(grads))) + 1e-5)

	# this function returns the loss and grads given the input picture
	iterate = K.function([input_img], [loss, grads])

	if init_img is None:
		input_img_data = np.random.random((1, C.dims[0], C.dims[1], C.dims[2], 3))
	else:
		input_img_data = np.expand_dims(init_img, 0)

	# run gradient ascent for 20 steps
	step = 1.
	for i in range(250):
		loss_value, grads_value = iterate([input_img_data])
		input_img_data += grads_value * step
		if i % 2 == 0:
			step *= .99
		if i % 5 == 0:
			#random rotations for transformation robustness, see https://distill.pub/2017/feature-visualization/#enemy-of-feature-vis
			input_img_data = np.pad(input_img_data[0], ((5,5),(5,5),(0,0),(0,0)), 'constant')
			input_img_data = tr.rotate(input_img_data, random.uniform(-5,5)*pi/180)
			input_img_data = np.expand_dims(input_img_data[5:-5, 5:-5, :, :], 0)

	img = input_img_data[0]
	img = deprocess_image(img)
	hf.draw_slices(img, save_path=join(save_path, "%s_filter.png" % layer_name))

def visualize_channel(model, layer_name, save_path, num_ch=None):
	"""Visualize the model inputs that would maximally activate a layer.
	num_ch is the number of channels to optimize over; keep as None to use the whole layer
	Original code by the Keras Team at
	https://github.com/keras-team/keras/blob/master/examples/conv_filter_visualization.py"""
	

	layer_dict = dict([(layer.name, layer) for layer in model.layers[1:]])

	input_img = model.input

	if num_ch is None:
		num_ch = layer_dict[layer_name].output.shape[-1]

	for filter_index in range(num_ch):
		# build a loss function that maximizes the activation
		# of the nth filter of the layer considered
		layer_output = layer_dict[layer_name].output
		loss = K.mean(layer_output[:, :, :, :, filter_index])

		# compute the gradient of the input picture wrt this loss
		grads = K.gradients(loss, input_img)[0]

		# normalization trick: we normalize the gradient
		grads /= (K.sqrt(K.mean(K.square(grads))) + 1e-5)

		# this function returns the loss and grads given the input picture
		iterate = K.function([input_img], [loss, grads])

		input_img_data = np.random.random((1, C.dims[0], C.dims[1], C.dims[2], 3)) * 2.

		# run gradient ascent for 20 steps
		step = 1.
		for i in range(20):
			loss_value, grads_value = iterate([input_img_data])
			input_img_data += grads_value * step
			input_img_data = np.pad(input_img_data[0], ((5,5),(5,5),(0,0),(0,0)), 'constant')
			input_img_data = tr.rotate(input_img_data, random.uniform(-5,5)*pi/180)
			input_img_data = np.expand_dims(input_img_data[5:-5, 5:-5, :, :], 0)
			#random rotations for transformation robustness, see https://distill.pub/2017/feature-visualization/#enemy-of-feature-vis

		img = input_img_data[0]
		img = deprocess_image(img)
		hf.save_slices(img, save_path=join(save_path, "%s_filter_%d.png" % (layer_name, filter_index)))



def deprocess_image(x):
	# set mean to center and std to 10% of the full range
	x -= x.mean()
	x /= (x.std() + 1e-5)
	x = (np.clip(x*.1+.5, 0, 1) * 255).astype('uint8')
	return x