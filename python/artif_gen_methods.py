"""
Converts a nifti file to a numpy array.
Accepts either a single nifti file or a folder of niftis as the input argument.

Usage:
	python artif_gen_methods.py

Author: Clinton Wang, E-mail: `clintonjwang@gmail.com`, Github: `https://github.com/clintonjwang/voi-classifier`
"""

import config
import copy
import math
import numpy as np
import random
import dr_methods as drm
import voi_methods as vm
import os
import niftiutils.helper_fxns as hf
import niftiutils.transforms as tr
from scipy.ndimage.filters import gaussian_filter

####################################
### MAIN GENERATION/PROCESSING METHODS
####################################

@drm.autofill_cls_arg
def gen_imgs(cls=None, n=None):
	C = config.Config()
	if n is None:
		n = C.n_aug

	if cls == "cyst":
		imgs = gen_cysts(n)
	elif cls == "hcc":
		imgs = gen_hccs(n)
	elif cls == "hemangioma":
		imgs = gen_hemangiomas(n)
	elif cls == "cholangio":
		imgs = gen_cholangios(n)
	elif cls == "colorectal":
		imgs = gen_colorectals(n)
	elif cls == "fnh":
		imgs = gen_fnhs(n)
	
	if not os.path.exists(C.artif_dir + cls):
		os.makedirs(C.artif_dir + cls)
	for cnt, img in enumerate(imgs):
		np.save(C.artif_dir + cls + "\\artificial_" + str(cnt), post_process_img(img))

def visualize_gen_img(cls):
	img = gen_hemangiomas(30)[random.randint(1,29)]
	img = post_process_img(img)
	hf.plot_section_auto(img, normalize=[-.8,.5])
	return img

def post_process_img(img, blur_range = [.7, 2.1]):
	"""Post processing applied to all artificial images.
	Currently, rotate, add noise, add edge, blur and offset phases."""
	C = config.Config()

	img = tr.rotate(img, random.randint(0, 359))
	img = add_edge(img, min_val=random.gauss(-.7,.2), fill=random.random()<.7)
	img = tr.offset_phases(img)
	img += np.random.normal(scale = C.noise_std, size = img.shape)
	img = blur_2d(img, random.uniform(blur_range[0], blur_range[1])) 
	img = tr.normalize_intensity(img, max_intensity=1, min_intensity=-1)
	img = img * random.gauss(1, C.intensity_scaling[0]) + random.gauss(0,C.intensity_scaling[1])
	
	return img

####################################
### POST-PROCESSING METHODS
####################################

def init_img():
	C = config.Config()
	img = np.zeros(C.dims + [C.nb_channels])
	enh_parenchyma_int = .15 * random.random() + .1
	img[:,:,:,1] = enh_parenchyma_int
	img[:,:,:,2] = enh_parenchyma_int - .03

	return img

def add_edge(img, edge_frac=0.2, min_val = -.7, fill=True):
	"""Add an artificial edge along one of the x/y sides (spans the entire z axis"""

	dims = img.shape
	edge_slope = random.uniform(-.2, .2)
	edge_choice = random.randint(1, 4)
	
	if fill:
		if edge_choice == 1: #vertical
			edge_start = random.uniform(0, dims[0]*edge_frac)
			for y in range(dims[1]):
				img[:max(math.floor(edge_start + edge_slope*y),0), y, :, :] = min_val

		elif edge_choice == 2:
			edge_start = random.uniform(dims[0]*(1-edge_frac), dims[0])
			for y in range(dims[1]):
				img[math.ceil(edge_start + edge_slope*y):, y, :, :] = min_val

		elif edge_choice == 3: #horizontal
			edge_start = random.uniform(0, dims[1]*edge_frac)
			for x in range(dims[0]):
				img[x, :max(math.floor(edge_start + edge_slope*x),0), :, :] = min_val

		else:
			edge_start = random.uniform(dims[1]*(1-edge_frac), dims[1])
			for x in range(dims[0]):
				img[x, math.ceil(edge_start + edge_slope*x):, :, :] = min_val
	
	else:
		if edge_choice == 1: #vertical
			edge_start = random.uniform(0, dims[0]*edge_frac)
			for y in range(dims[1]):
				x = max(math.floor(edge_start + edge_slope*y),1)
				img[x-1:x, y, :, :] = min_val

		elif edge_choice == 2:
			edge_start = random.uniform(dims[0]*(1-edge_frac), dims[0])
			for y in range(dims[1]):
				x=math.ceil(edge_start + edge_slope*y)-1
				img[x:x+1, y, :, :] = min_val

		elif edge_choice == 3: #horizontal
			edge_start = random.uniform(0, dims[1]*edge_frac)
			for x in range(dims[0]):
				y = max(math.floor(edge_start + edge_slope*x),1)
				img[x, y-1:y, :, :] = min_val

		else:
			edge_start = random.uniform(dims[1]*(1-edge_frac), dims[1])
			for x in range(dims[0]):
				y=math.ceil(edge_start + edge_slope*x)-1
				img[x, y:y+1, :, :] = min_val

	return img

def blur_2d(orig_img, sigma):
	"""Apply gaussian blur to images in the x/y plane only."""

	img = copy.deepcopy(orig_img)
	for ch in range(img.shape[3]):
		for sl in range(img.shape[2]):
			img[:,:,sl,ch] = gaussian_filter(img[:,:,sl,ch], sigma)
		
	return img

def get_sizes(n):
	C = config.Config()
	side_rat = np.linspace(1/C.max_side_ratio, 1, 6)
	side_rat = list(sum(zip(reversed(side_rat), side_rat), ())[:len(side_rat)]) * math.ceil(n/6)
	sizes = np.linspace(C.long_size_frac[0], C.long_size_frac[1], num=n)

	return side_rat, sizes

####################################
### SPECIFIC LESION TYPES
####################################

def gen_cysts(n):
	"""Generate n images of cysts with dimensions of C.dims plus channels defined by the config file.
	Should be round and hypointense in all phases."""
	
	shades = [-0.65, -0.65, -0.65]
	return gen_round_lesions(n, shades)

def gen_hccs(n):
	"""Generate n images of hccs with dimensions of C.dims plus channels defined by the config file.
	Should be round, enhancing in arterial, with washout in delayed and usually venous."""
	C = config.Config()

	shades = [0.2, -0.1, -0.3]
	shade_offset = 0.2
	shade_std = .1
	prob_rim = .7
	rim_continuity = .8
	prob_hetero = .5

	imgs = []
	side_rat, sizes = get_sizes(n)
	
	midx = C.dims[0]//2
	midy = C.dims[1]//2
	midz = C.dims[2]//2
	z_ratio = midy/midz

	spread = 2.4
	
	for i in range(n):
		has_rim = random.random()<prob_rim
		patchwork = random.random()<prob_hetero
		r = midx * sizes[i]
		shade_offset_i = random.uniform(-shade_offset, shade_offset)
		
		shades_i = [shade+random.gauss(0, shade_std)+shade_offset_i for shade in shades]
		rim_shade = shades_i[0]*random.uniform(1.05,1.6)
		
		img = init_img()
		for x in range(-math.floor(r), math.floor(r)):
			for y in range(-math.floor(r), math.floor(r)):
				z_sq = (r**2 - x**2 - (y/side_rat[i])**2)
				if z_sq <= 0:
					continue
					
				z = int(round(z_sq**(.5)/z_ratio))
				
				if z > midz:
					z = midz

				if has_rim and z_sq < r**2 * .4 and random.random()<rim_continuity:
					img[x+midx, y+midy, midz-z:midz+z, :] = rim_shade
				elif patchwork:
					if random.random() < 0.7:
						img[x+midx, y+midy, midz-z:midz+z, :] = rim_shade*random.uniform(.3,1)
					else:
						img[x+midx-round(spread*random.uniform(0,1)):x+midx+1+round(spread*random.uniform(0,1)),
							y+midy-round(spread*random.uniform(0,1)):y+midy+1+round(spread*random.uniform(0,1)),
							midz-z:midz+z, :] = [shade*random.uniform(.8,1.1) for shade in shades_i]
				else:
					img[x+midx, y+midy, midz-z:midz+z, :] = [shade*random.uniform(.8,1.1) for shade in shades_i]
					
		
		imgs.append(img)
		
	return imgs

def gen_hemangiomas(n):
	"""Generate n images of hemangiomas with dimensions of C.dims plus channels defined by the config file.
	Should be round with a nodularly enhancing rim that fills in over time."""
	C = config.Config()
	
	shades = [-0.65, -0.65, -0.65]
	rim_shades = [.4,.4,.4]
	shrink_factor = [0.8, 1]
	rim_ratio = [0.95,1.05]

	imgs = []
	side_rat, sizes = get_sizes(n)
	midx = C.dims[0]//2
	midy = C.dims[1]//2
	midz = C.dims[2]//2
	z_ratio = midy/midz

	spread = [1,2,5]
	prob_nodule = 0.2
	
	for i in range(n):
		r = midx * sizes[i]
		r_core = r * random.uniform(rim_ratio[0], rim_ratio[1])
		rven = r_core * random.uniform(shrink_factor[0], shrink_factor[1])
		req = rven * random.uniform(shrink_factor[0], shrink_factor[1])
		
		shades_i = [shade+random.gauss(0, C.shade_std) for shade in shades]
		
		img = init_img()
		for x in range(-math.floor(r), math.floor(r)):
			for y in range(-math.floor(r), math.floor(r)):
				z_sq = (r**2 - x**2 - (y/side_rat[i])**2)
				if z_sq <= 0:
					continue
					
				z = int(round(z_sq**(.5)/z_ratio))
				
				if z > midz:
					z = midz
				
				img[x+midx, y+midy, midz-z:midz+z, :] = rim_shades
				
				for ch, rad in enumerate([r_core, rven, req]):
					z_sq = (rad**2 - x**2 - (y/side_rat[i])**2)
					if z_sq <= 0:
						continue

					z = int(round(z_sq**(.5)/z_ratio))

					if z > midz:
						z = midz

					if random.random() < prob_nodule and z_sq < rad**2 * .3:
						for ch in range(3):
							x1 = x+midx - int(min((rad**2-y**2)**.5, round(spread[ch]*random.uniform(0,(ch+1)*.5))))
							x2 = x+midx+1 + int(min((rad**2-y**2)**.5, round(spread[ch]*random.uniform(0,ch))))
							y1 = y+midy - int(min((rad**2-x**2)**.5, round(spread[ch]*random.uniform(0,ch*.5))))
							y2 = y+midy+1 + int(min((rad**2-x**2)**.5, round(spread[ch]*random.uniform(0,ch*.5))))
							img[x1:x2, y1:y2,
								midz-z:midz+z, ch] = rim_shades[ch]*random.uniform(.8,1)
					else:
						img[x+midx, y+midy, midz-z:midz+z, ch] = shades_i[ch]
						#img[x+midx, y+midy, midz-z:midz+z, ch] = shades_i[ch]*.5*(1+z_sq/rad**2)
				
		imgs.append(img)
		
	return imgs

def gen_cholangios(n):
	"""Generate n images of cholangiocarcinomas with dimensions of C.dims plus channels defined by the config file.
	Mass-forming should have irregular, ragged rim enhancement with heterogeneous texture and gradual centripetal enhancement."""
	C = config.Config()
	
	base_shade = -0.5
	enhancement_rate = 1.15
	rim_shades=[.3, .3, .3]
	shrink_factor=[.7, .9]
	rim_ratio=[.9, 1]
	
	
	imgs = []
	side_rat, sizes = get_sizes(n)
	midx = C.dims[0]//2
	midy = C.dims[1]//2
	midz = C.dims[2]//2
	z_ratio = midy/midz

	spread = 2
	
	for i in range(n):
		r = midx * sizes[i]
		r_core = r * random.uniform(rim_ratio[0], rim_ratio[1])
		rven = r_core * random.uniform(shrink_factor[0], shrink_factor[1])
		req = rven * random.uniform(shrink_factor[0], shrink_factor[1])
		
		enhancement_rate_i = enhancement_rate ** random.uniform(.5,3)
		ven_shade = base_shade / enhancement_rate
		eq_shade = ven_shade / enhancement_rate**(1.5 * random.uniform(.8,1.4))
		shades_i = [base_shade, ven_shade, eq_shade]
		rim_shades_i = [rim_shade+random.gauss(0, C.shade_std) for rim_shade in rim_shades]
		
		img = init_img()
		for x in range(-math.floor(r), math.floor(r)):
			for y in range(-math.floor(r), math.floor(r)):
				z = (r**2 - x**2 - (y/side_rat[i])**2)
				if z <= 0:
					continue
					
				z = int(round(z**(.5)/z_ratio))
				
				if z > midz:
					z = midz
					
				img[x+midx, y+midy, midz-z:midz+z, :] = [rim_shade*random.uniform(.6,1) for rim_shade in rim_shades]
				
				if random.random() < 0.8:
					z = (r_core**2 - x**2 - (y/side_rat[i])**2)
					if z <= 0:
						continue
					z = int(round(z**(.5)/z_ratio))
					if z > midz:
						z = midz
						
					img[x+midx, y+midy, midz-z:midz+z, :] = [rim_shade*random.uniform(.6,1) for rim_shade in rim_shades_i]
					
				elif random.random() < 0.5:
					for ch, rad in enumerate([r_core, rven, req]):
						z_sq = (rad**2 - x**2 - (y/side_rat[i])**2)
						if z_sq <= 0:
							continue
						z = int(round(z_sq**(.5)/z_ratio))
						if z > midz:
							z = midz
							
						img[x+midx-spread:x+midx+spread, y+midy-spread:y+midy+spread, midz-z:midz+zh] = shades_i[ch]*random.uniform(.8,1.2)
						
				else:
					for ch, rad in enumerate([r_core, rven, req]):
						z_sq = (rad**2 - x**2 - (y/side_rat[i])**2)
						if z_sq <= 0:
							continue
						z = int(round(z_sq**(.5)/z_ratio))
						if z > midz:
							z = midz
							
						img[x+midx, y+midy, midz-z:midz+zh] = shades_i[ch]/max(z_sq**.3,.9)*random.uniform(.8,1.2)
				
		imgs.append(img)
		
	return imgs

def gen_colorectals(n):
	"""Generate n images of colorectal mets with dimensions of C.dims plus channels defined by the config file.
	Should be hypointense in all phases (necrotic core) with a continuous enhancing rim. Sometimes enhances or shrinks over time."""
	C = config.Config()
	
	shades = [-0.5, -0.45, -0.4]
	rim_shades=[.5,.4,.4]
	shrink_factor=[.85, 1]
	rim_ratio=[.83,0.98]
	internal_structure = 0.2
	
	imgs = []
	side_rat, sizes = get_sizes(n)
	midx = C.dims[0]//2
	midy = C.dims[1]//2
	midz = C.dims[2]//2
	z_ratio = midy/midz

	spread = 2
	
	for i in range(n):
		r = midx * sizes[i]
		r_core = r * random.uniform(rim_ratio[0], rim_ratio[1])
		rven = r_core * random.uniform(shrink_factor[0], shrink_factor[1])
		req = rven * random.uniform(shrink_factor[0], shrink_factor[1])
		
		shades_i = [shade+random.gauss(0, C.shade_std) for shade in shades]
		
		img = init_img()
		for x in range(-math.floor(r), math.floor(r)):
			for y in range(-math.floor(r), math.floor(r)):
				z_sq = (r**2 - x**2 - (y/side_rat[i])**2)
				if z_sq <= 0:
					continue
					
				z = int(round(z_sq**(.5)/z_ratio))
				
				if z > midz:
					z = midz
				
				img[x+midx, y+midy, midz-z:midz+z, :] = rim_shades
				
				for ch, rad in enumerate([r_core, rven, req]):
					z_sq = (rad**2 - x**2 - (y/side_rat[i])**2)
					if z_sq <= 0:
						continue

					z = int(round(z_sq**(.5)/z_ratio))

					if z > midz:
						z = midz

					if random.random() < internal_structure and z_sq > rad**2 * .7:
						for ch in range(3):
							img[x+midx+round(spread*random.random()),
								y+midy+round(spread*random.random()),
								midz-round(z*random.random()):midz+max(1,round(z*random.random())),
								ch] = rim_shades[ch]*random.uniform(.2,.8)
					else:
						img[x+midx, y+midy, midz-z:midz+zh] = shades_i[ch]*random.uniform(.6,1)
				
		imgs.append(img)
		
	return imgs

def gen_fnhs(n, scar_fraction = .3):
	"""Generate n images of FNHs with dimensions of C.dims plus channels defined by the config file.
	Should be hypointense in all phases (necrotic core) with an enhancing rim. Large ones have a central scar."""
	
	shades = [0.35, 0.05, 0.05]
	return gen_scarring_lesions(math.ceil(n*scar_fraction), shades) + gen_round_lesions(math.floor(n*(1-scar_fraction)), shades)

####################################
### GENERAL METHODS FOR LESION TYPES
####################################

def gen_round_lesions(n, shades, shade_offset=0.02):
	"""Round, homogeneous lesions that remain the same size in all phases. Includes cysts and FNH.
	Shade_offset randomly offsets the lesion shades in all phases by the same amount."""
	C = config.Config()
	
	imgs = []
	side_rat, sizes = get_sizes(n)
	
	midx = C.dims[0]//2
	midy = C.dims[1]//2
	midz = C.dims[2]//2
	z_ratio = midy/midz
	
	for i in range(n):
		r = midx * sizes[i]
		shade_offset_i = random.uniform(-shade_offset, shade_offset)
		
		shades_i = [shade+random.gauss(0, C.shade_std)+shade_offset_i for shade in shades]
		
		img = init_img()
		for x in range(-math.floor(r), math.floor(r)):
			for y in range(-math.floor(r), math.floor(r)):
				z = (r**2 - x**2 - (y/side_rat[i])**2)
				if z <= 0:
					continue
					
				z = int(round(z**(.5)/z_ratio))
				
				if z > midz:
					z = midz
					
				img[x+midx, y+midy, midz-z:midz+z, :] = [shade*random.uniform(.6,1.1) for shade in shades_i]
		
		imgs.append(img)
		
	return imgs

def gen_scarring_lesions(n, shades, scar_shades=[-.5, -.5, .3]):
	"""Lesions that have a central scar. Includes FNHs."""
	C = config.Config()
	
	imgs = []
	side_rat, sizes = get_sizes(n)
	
	midx = C.dims[0]//2
	midy = C.dims[1]//2
	midz = C.dims[2]//2
	z_ratio = midy/midz
	
	for i in range(n):
		r = midx * sizes[i]
		if sizes[i] < .65 or side_rat[i] < 0.75:
			r_scar = 0
		else:
			r_scar = r * random.uniform(.3,.6) * sizes[i] #more pronounced scarring in large lesions
		
		shades_i = [shade+random.gauss(0, C.shade_std) for shade in shades]
		scar_shades_i = [scar_shade+random.gauss(0, C.shade_std) for scar_shade in scar_shades]
		
		img = init_img()
		for x in range(-math.floor(r), math.floor(r)):
			for y in range(-math.floor(r), math.floor(r)):
				z = (r**2 - x**2 - (y/side_rat[i])**2)
				if z <= 0:
					continue
					
				z = int(round(z**(.5)/z_ratio))
				
				if z > midz:
					z = midz
					
				img[x+midx, y+midy, midz-z:midz+z, :] = shades_i
		

		for x in range(-math.floor(r_scar), math.floor(r_scar)):
			y = int(random.uniform(-.5,.5)*r_scar)
			z = (r_scar**2 - x**2 - (y/side_rat[i])**2)
			if z <= 0:
				continue
			z = math.ceil(z**(.5)/z_ratio)
			img[x+midx, y+midy, midz-z:midz+z, :] = [scar_shades_ix for scar_shades_ix in scar_shades_i]
		
		imgs.append(img)
		
	return imgs

if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='Generate artificial lesion images.')
	parser.add_argument('-n', '--num_lesions', type=int, help='number of lesions of each class to generate')
	#parser.add_argument('-o', '--overwrite', action='store_true', help='overwrite')
	args = parser.parse_args()

	gen_imgs(n=args.num_lesions)