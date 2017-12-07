import copy
import math
import numpy as np
import random
import os
import transforms as tr
from scipy.ndimage.filters import gaussian_filter


####################################
### MAIN GENERATION/PROCESSING METHODS
####################################

def gen_imgs(cls, C, n):
	if cls == "cyst":
		imgs = gen_cysts(C, n)
	elif cls == "hcc":
		imgs = gen_hccs(C, n)
	elif cls == "hemangioma":
		imgs = gen_hemangiomas(C, n)
	elif cls == "cholangio":
		imgs = gen_cholangios(C, n)
	elif cls == "colorectal":
		imgs = gen_colorectals(C, n)
	elif cls == "fnh":
		imgs = gen_fnhs(C, n)
	
	if not os.path.exists(C.artif_dir + cls):
		os.makedirs(C.artif_dir + cls)
	for cnt, img in enumerate(imgs):
		np.save(C.artif_dir + cls + "\\artificial_" + str(cnt), post_process_img(img, C))


def post_process_img(img, C, blur_range = [.8, 1.9]):
	"""Post processing applied to all artificial images.
	Currently, rotate, add noise, add edge, blur and offset phases."""
		
	img = tr.rotate(img, random.randint(0, 359))
	img += np.random.normal(scale = C.noise_std, size = img.shape)
	img = add_edge(img)
	img = tr.offset_phases(img)
	img = blur_2d(img, random.uniform(blur_range[0], blur_range[1]))
	
	return img

####################################
### POST-PROCESSING METHODS
####################################

def add_edge(img, edge_frac=0.2):
	"""Add an artificial edge along one of the x/y sides (spans the entire z axis"""

	dims = img.shape
	edge_slope = random.uniform(-.2, .2)
	
	edge_choice = random.randint(1, 4)
	if edge_choice == 1: #vertical
		edge_start = random.uniform(0, dims[0]*edge_frac)
		for y in range(dims[1]):
			img[:max(math.floor(edge_start + edge_slope*y),0), y, :, :] = -1

	elif edge_choice == 2:
		edge_start = random.uniform(dims[0]*(1-edge_frac), dims[0])
		for y in range(dims[1]):
			img[math.ceil(edge_start + edge_slope*y):, y, :, :] = -1

	elif edge_choice == 3: #horizontal
		edge_start = random.uniform(0, dims[1]*edge_frac)
		for x in range(dims[0]):
			img[x, :max(math.floor(edge_start + edge_slope*x),0), :, :] = -1

	else:
		edge_start = random.uniform(dims[1]*(1-edge_frac), dims[1])
		for x in range(dims[0]):
			img[x, math.ceil(edge_start + edge_slope*x):, :, :] = -1
	
	return img


def blur_2d(orig_img, sigma):
	"""Apply gaussian blur to images in the x/y plane only."""

	img = copy.deepcopy(orig_img)
	for ch in range(img.shape[3]):
		for sl in range(img.shape[2]):
			img[:,:,sl,ch] = gaussian_filter(img[:,:,sl,ch], sigma)
		
	return img


def get_sizes(C, n):
	side_rat = np.linspace(1/C.max_side_ratio, 1, 6)
	side_rat = list(sum(zip(reversed(side_rat), side_rat), ())[:len(side_rat)]) * math.ceil(n/6)
	sizes = np.linspace(C.long_size_frac[0], C.long_size_frac[1], num=n)

	return side_rat, sizes


####################################
### SPECIFIC LESION TYPES
####################################

def gen_cysts(C, n):
	"""Generate n images of cysts with dimensions of C.dims plus channels defined by the config file.
	Should be round and hypointense in all phases."""
	
	shades = [-0.8, -0.8, -0.8]
	return gen_round_lesions(n, shades, C)

def gen_hccs(C, n):
	"""Generate n images of hccs with dimensions of C.dims plus channels defined by the config file.
	Should be round, enhancing in arterial, with washout in venous and delayed."""
	
	shades = [0.2, -0.2, -0.25]
	shade_offset = 0.15
	return gen_round_lesions(n, shades, C, shade_offset=shade_offset)

def gen_hemangiomas(C, n):
	"""Generate n images of hemangiomas with dimensions of C.dims plus channels defined by the config file.
	Should be round, enhancing in arterial, with washout in venous and delayed."""
	
	shades = [-0.8, -0.8, -0.8]
	return gen_rimmed_lesions(n, shades, C, rim_shades=[0.1, 0.15, 0.15], shrink_factor=[0.35, 0.8], rim_ratio=0.97, prob_discont=0.05)

def gen_cholangios(C, n):
	"""Generate n images of cholangiocarcinomas with dimensions of C.dims plus channels defined by the config file.
	Mass-forming should have irregular, ragged rim enhancement with gradual centripetal enhancement."""
	
	shades = [-0.6, -0.5, -0.4]
	return gen_heterogen_lesions(n, shades, C, shrink_factor=[.6, .9], rim_ratio=0.92)

def gen_colorectals(C, n):
	"""Generate n images of colorectal mets with dimensions of C.dims plus channels defined by the config file.
	Should be hypointense in all phases (necrotic core) with an enhancing rim. Sometimes enhances or shrinks over time(?)"""
	
	shades = [-0.5, -0.5, -0.45]
	return gen_rimmed_lesions(n, shades, C, rim_shades=[.5,.35,.35], shrink_factor=[.8, .95], rim_ratio=0.86)

def gen_fnhs(C, n, scar_fraction = .5):
	"""Generate n images of FNHs with dimensions of C.dims plus channels defined by the config file.
	Should be hypointense in all phases (necrotic core) with an enhancing rim."""
	
	shades = [0.35, 0, 0]
	return gen_scarring_lesions(math.ceil(n*scar_fraction), shades, C) + gen_round_lesions(math.floor(n*(1-scar_fraction)), shades, C)




####################################
### GENERAL METHODS FOR LESION TYPES
####################################

def gen_round_lesions(n, shades, C, shade_offset=0):
	"""Round lesions that remain the same size in all phases. Includes HCCs and cysts.
	Shade_offset randomly offsets the lesion shades in all phases by the same amount."""
	
	imgs = []
	side_rat, sizes = get_sizes(C, n)
	
	midx = C.dims[0]//2
	midy = C.dims[1]//2
	midz = C.dims[2]//2
	z_ratio = midy/midz
	
	for i in range(n):
		r = midx * sizes[i]
		shade_offset_i = random.uniform(-shade_offset, shade_offset)
		
		shades_i = [shade+random.gauss(0, C.shade_std)+shade_offset_i for shade in shades]
		
		img = np.zeros(C.dims + [C.nb_channels])
		for x in range(-math.floor(r), math.floor(r)):
			for y in range(-math.floor(r), math.floor(r)):
				z = (r**2 - x**2 - (y/side_rat[i])**2)
				if z <= 0:
					continue
					
				z = int(round(z**(.5)/z_ratio))
				
				if z > midz:
					z = midz
					
				img[x+midx, y+midy, midz-z:midy+z, :] = shades_i
		
		imgs.append(img)
		
	return imgs

def gen_scarring_lesions(n, shades, C, scar_shades=[-.5, -.5, .25]):
	"""Lesions that have a central scar. Includes FNHs."""
	
	imgs = []
	side_rat, sizes = get_sizes(C, n)
	
	midx = C.dims[0]//2
	midy = C.dims[1]//2
	midz = C.dims[2]//2
	z_ratio = midy/midz
	
	for i in range(n):
		r = midx * sizes[i]
		if sizes[i] < .65 or side_rat[i] < 0.75:
			r_scar = 0
		else:
			r_scar = r * random.uniform(.3,.8) * sizes[i] #more pronounced scarring in large lesions
		
		shades_i = [shade+random.gauss(0, C.shade_std) for shade in shades]
		scar_shades_i = [scar_shade+random.gauss(0, C.shade_std) for scar_shade in scar_shades]
		
		img = np.zeros(C.dims + [C.nb_channels])
		for x in range(-math.floor(r), math.floor(r)):
			for y in range(-math.floor(r), math.floor(r)):
				z = (r**2 - x**2 - (y/side_rat[i])**2)
				if z <= 0:
					continue
					
				z = int(round(z**(.5)/z_ratio))
				
				if z > midz:
					z = midz
					
				img[x+midx, y+midy, midz-z:midy+z, :] = shades_i
		

		for x in range(-math.floor(r_scar), math.floor(r_scar)):
			for y in range(-int(round(random.uniform(.1,.5)*r_scar)), math.ceil(random.uniform(.1,.5)*r_scar)):
				z = (r_scar**2 - x**2 - (y/side_rat[i])**2)
				if z <= 0:
					continue
				z = math.ceil(z**(.5)/z_ratio)
				img[x+midx, y+midy, midz-z:midy+z, :] = [scar_shades_ix/z for scar_shades_ix in scar_shades_i]
		
		imgs.append(img)
		
	return imgs


def gen_shrinking_lesions(n, shades, C, shrink_factor = [0.5, 0.8]):
	"""Lesions that shrink over phases. Unused."""
	
	imgs = []
	side_rat, sizes = get_sizes(C, n)
	midx = C.dims[0]//2
	midy = C.dims[1]//2
	midz = C.dims[2]//2
	z_ratio = midy/midz
	
	for i in range(n):
		r = midx * sizes[i]
		rven = r * random.uniform(shrink_factor[0], shrink_factor[1])
		req = rven * random.uniform(shrink_factor[0], shrink_factor[1])
		shades_i = [shade+random.gauss(0, C.shade_std) for shade in shades]
		
		img = np.zeros(C.dims + [C.nb_channels])
		for x in range(-math.floor(r), math.floor(r)):
			for y in range(-math.floor(r), math.floor(r)):
				for ch, rad in enumerate([r, rven, req]):
					z = (rad**2 - x**2 - (y/side_rat[i])**2)
					if z <= 0:
						continue

					z = int(round(z**(.5)/z_ratio))

					if z > midz:
						z = midz

					img[x+midx, y+midy, midz-z:midy+z, ch] = shades_i[ch]
				
		imgs.append(img)
		
	return imgs


def gen_rimmed_lesions(n, shades, C, rim_shades=[.3, .3, .3], rim_ratio = 0.9, prob_discont = 0, shrink_factor = [1,1]):
	"""Lesions that have an enhancing rim. Includes hemangiomas and colorectal mets."""
	
	imgs = []
	side_rat, sizes = get_sizes(C, n)
	midx = C.dims[0]//2
	midy = C.dims[1]//2
	midz = C.dims[2]//2
	z_ratio = midy/midz

	spread = 2
	
	for i in range(n):
		r = midx * sizes[i]
		r_core = r * rim_ratio * random.uniform(.9,1.05)
		rven = r_core * random.uniform(shrink_factor[0], shrink_factor[1])
		req = rven * random.uniform(shrink_factor[0], shrink_factor[1])
		
		shades_i = [shade+random.gauss(0, C.shade_std) for shade in shades]
		
		img = np.zeros(C.dims + [C.nb_channels])
		for x in range(-math.floor(r), math.floor(r)):
			for y in range(-math.floor(r), math.floor(r)):
				z_sq = (r**2 - x**2 - (y/side_rat[i])**2)
				if z_sq <= 0:
					continue
					
				z = int(round(z_sq**(.5)/z_ratio))
				
				if z > midz:
					z = midz
				
				img[x+midx, y+midy, midz-z:midy+z, :] = rim_shades
				
				for ch, rad in enumerate([r_core, rven, req]):
					z_sq = (rad**2 - x**2 - (y/side_rat[i])**2)
					if z_sq <= 0:
						continue

					z = int(round(z_sq**(.5)/z_ratio))

					if z > midz:
						z = midz

					if random.random() < prob_discont and z_sq > rad**2 * .75:
						for ch in range(3):
							img[x+midx-spread:x+midx+round(spread*random.random()),
								y+midy-spread:y+midy+round(spread*random.random()),
								midz-z:midy+z, ch] = shades_i[ch]*.5*(1+z_sq/rad**2)
					else:
						img[x+midx, y+midy, midz-z:midy+z, ch] = shades_i[ch]*.5*(1+z_sq/rad**2)
				
		imgs.append(img)
		
	return imgs

def gen_heterogen_lesions(n, shades, C, rim_shades=[.3, .3, .3], rim_ratio = 0.9, shrink_factor = [1,1]):
	"""Lesions that have a heterogeneous texture. Includes cholangiocarcinomas."""
	
	imgs = []
	side_rat, sizes = get_sizes(C, n)
	midx = C.dims[0]//2
	midy = C.dims[1]//2
	midz = C.dims[2]//2
	z_ratio = midy/midz

	spread = 2
	
	for i in range(n):
		r = midx * sizes[i]
		r_core = r * rim_ratio
		rven = r_core * random.uniform(shrink_factor[0], shrink_factor[1])
		req = rven * random.uniform(shrink_factor[0], shrink_factor[1])
		
		shades_i = [shade+random.gauss(0, C.shade_std) for shade in shades]
		rim_shades_i = [rim_shade+random.gauss(0, C.shade_std) for rim_shade in rim_shades]
		
		img = np.zeros(C.dims + [C.nb_channels])
		for x in range(-math.floor(r), math.floor(r)):
			for y in range(-math.floor(r), math.floor(r)):
				z = (r**2 - x**2 - (y/side_rat[i])**2)
				if z <= 0:
					continue
					
				z = int(round(z**(.5)/z_ratio))
				
				if z > midz:
					z = midz
					
				img[x+midx, y+midy, midz-z:midy+z, :] = rim_shades
				
				if random.random() < 0.8:
					z = (r_core**2 - x**2 - (y/side_rat[i])**2)
					if z <= 0:
						continue
					z = int(round(z**(.5)/z_ratio))
					if z > midz:
						z = midz
						
					img[x+midx, y+midy, midz-z:midy+z, :] = rim_shades_i
					
				elif random.random() < 0.5:
					for ch, rad in enumerate([r_core, rven, req]):
						z_sq = (rad**2 - x**2 - (y/side_rat[i])**2)
						if z_sq <= 0:
							continue
						z = int(round(z_sq**(.5)/z_ratio))
						if z > midz:
							z = midz
							
						img[x+midx-spread:x+midx+spread, y+midy-spread:y+midy+spread, midz-z:midy+z, ch] = shades_i[ch]*random.uniform(.8,1.2)
						
				else:
					for ch, rad in enumerate([r_core, rven, req]):
						z_sq = (rad**2 - x**2 - (y/side_rat[i])**2)
						if z_sq <= 0:
							continue
						z = int(round(z_sq**(.5)/z_ratio))
						if z > midz:
							z = midz
							
						img[x+midx, y+midy, midz-z:midy+z, ch] = shades_i[ch]/max(z_sq**.3,.9)
				
		imgs.append(img)
		
	return imgs