import helper_fxns as hf
import numpy as np
import os
import pandas as pd
import time

def add_voi(voi_df, acc_num, x, y, z, vox_dims=None, cls=None, flipz=None, return_id=False):
	"""Append voi info to the dataframe voi_df. Overwrite any previous entries."""
	
	if return_id:
		try:
			lesion_num = max(voi_df[voi_df["Filename"] == str(acc_num) + ".npy"]["lesion_num"]) + 1
		except ValueError:
			lesion_num = 0
			
		row_id = str(acc_num)+'_'+str(lesion_num)
	else:
		row_id = acc_num
	
	voi_df = voi_df[voi_df["id"] != row_id]
	
	if len(voi_df) == 0:
		i = 0
	else:
		i = voi_df.index[-1]+1
		
	if return_id:
		dx = (x[1] - x[0])*vox_dims[0]
		dy = (y[1] - y[0])*vox_dims[1]
		dz = (z[1] - z[0])*vox_dims[2]
		
		voi_df.loc[i] = [str(acc_num) + ".npy", x[0], x[1], y[0], y[1], z[0], z[1], cls, flipz, dx, dy, dz, row_id, lesion_num]
		return voi_df, row_id
		
	else:
		voi_df.loc[i] = [row_id, x[0], x[1], y[0], y[1], z[0], z[1]]
		return voi_df


def add_intensity_df(intensity_df, img, acc_num):
	"""Append scale info to the dataframe dims_df. Overwrite any previous entries."""
	
	intensity_df = intensity_df[intensity_df["AccNum"] != acc_num]
	
	if len(intensity_df) == 0:
		i = 0
	else:
		i = intensity_df.index[-1] + 1
		
	intensity_df.loc[i] = [acc_num, get_scaling_intensity(img[:,:,:,0]),
						   get_scaling_intensity(img[:,:,:,1]),
						   get_scaling_intensity(img[:,:,:,2])]
	
	return intensity_df

def get_scaling_intensity(img):
	"""Return intensity value to normalize img and all its transforms to. img should be 3D with no channels."""

	"""temp_img = img[img.shape[0]//5:img.shape[0]*3//5,
				   img.shape[1]//5:img.shape[1]*3//5,
				   img.shape[2]//5:img.shape[2]*4//5]
	temp_img = temp_img[temp_img > np.mean(temp_img)*2/3]
	hist = np.histogram(temp_img, bins=15)
	a = list(hist[0])
	max_value = max(a)
	max_index = a.index(max_value)
	ret = (hist[1][max_index] + hist[1][max_index+1]) / 2"""
	temp_img = img[img.shape[0]//5:img.shape[0]*3//5,
				   img.shape[1]//5:img.shape[1]//2,:]
	ret = np.amax(temp_img)
	
	return ret


def load_vois(cls, xls_name, sheetname, voi_dfs, dims_df, C, verbose=False, target_dims=None):
	"""Load all vois belonging to a class based on the contents of the spreadsheet.
	If target_dims is None, do not rescale images."""
	
	s = time.time()
	print("\nLoading VOIs for class", cls)
	
	voi_df_art, voi_df_ven, voi_df_eq = voi_dfs
	df = pd.read_excel(xls_name, sheetname=sheetname)
	df = preprocess_df(df, C)
	
	acc_nums = list(set(df['Patient E Number'].dropna().astype(str).tolist()))

	for cnt, acc_num in enumerate(acc_nums):
		df_subset = df.loc[df['Patient E Number'].astype(str) == acc_num]
		img = np.load(C.full_img_dir + "\\" + cls + "\\" + str(acc_num) + ".npy")
		
		for _, row in df_subset.iterrows():
			x = (int(row['x1']), int(row['x2']))
			y = (int(row['y1']), int(row['y2']))
			z = (int(row['z1']), int(row['z2']))
			
			try:
				cur_dims = dims_df[dims_df["AccNum"] == acc_num].iloc[0].values[1:]
			except:
				print("dims_df not yet loaded for", acc_num)
				
			if target_dims is not None:
				vox_scale = [float(cur_dims[i]/target_dims[i]) for i in range(3)]
				x,y,z = hf.scale_vois(x, y, z, vox_scale)
			
			y = (img.shape[1]-y[1], img.shape[1]-y[0]) # flip y
			if row['Flipped'] != "Yes":
				z = (img.shape[2]-z[1], img.shape[2]-z[0]) # flip z
			
			voi_df_art, art_id = add_voi(voi_df_art, acc_num, x,y,z, vox_dims=cur_dims,
										 cls=cls, flipz=(row['Flipped'] == "Yes"), return_id = True)

			if "Image type2" in row.keys() and row['Image type2'] == 'VP-T1':
				x = (int(row['x3']), int(row['x4']))
				y = (int(row['y3']), int(row['y4']))
				z = (int(row['z3']), int(row['z4']))
				
				if target_dims is not None:
					x,y,z = hf.scale_vois(x, y, z, vox_scale)
				
				y = (img.shape[1]-y[1], img.shape[1]-y[0]) # flip y
				if row['Flipped'] != "Yes":
					z = (img.shape[2]-z[1], img.shape[2]-z[0]) # flip z
					
				voi_df_ven = add_voi(voi_df_ven, art_id, x,y,z)
				
			if "Image type3" in row.keys() and row['Image type3'] == 'EQ-T1':
				x = (int(row['x5']), int(row['x6']))
				y = (int(row['y5']), int(row['y6']))
				z = (int(row['z5']), int(row['z6']))
				
				if target_dims is not None:
					x,y,z = hf.scale_vois(x, y, z, vox_scale)
				
				y = (img.shape[1]-y[1], img.shape[1]-y[0]) # flip y
				if row['Flipped'] != "Yes":
					z = (img.shape[2]-z[1], img.shape[2]-z[0]) # flip z
					
				voi_df_eq = add_voi(voi_df_eq, art_id, x,y,z)

		if verbose:
			print(acc_num, "%d out of %d acc_nums loaded" % (cnt+1, len(acc_nums)))
		else:
			print(".", end="")
			
	print("Overall time: %s" % str(time.time() - s))
	return voi_df_art, voi_df_ven, voi_df_eq


def load_ints(C):
	"""Return a dataframe with the normalizing intensities of each image's channels"""

	intensity_df = pd.DataFrame(columns = ["AccNum", "art_int", "ven_int", "eq_int"])

	for cls in C.classes_to_include:
		for fn in os.listdir(C.full_img_dir + "\\" + cls):
			img = np.load(C.full_img_dir + "\\" + cls + "\\" + fn)
			intensity_df = add_intensity_df(intensity_df, img, fn[:-4])
		
	return intensity_df

def add_to_dims_df(dims_df, acc_num, cur_dims):
	"""Append scale info to the dataframe dims_df. Overwrite any previous entries."""
	
	dims_df = dims_df[dims_df["AccNum"] != acc_num]
	
	if len(dims_df) == 0:
		i = 0
	else:
		i = dims_df.index[-1] + 1
		
	dims_df.loc[i] = [acc_num] + list(cur_dims)
	
	return dims_df

def preprocess_df(df, C):
	"""Select only rows for this run. Collect acc_nums and voi coordinates."""
	
	df = df[df['Run'] <= C.run_num].dropna(subset=["x1"])
	
	return df.drop(set(df.columns).difference(['Patient E Number', 
		  'x1', 'x2', 'y1', 'y2', 'z1', 'z2', 'Image type', 'Flipped',
		  'x3', 'x4', 'y3', 'y4', 'z3', 'z4', 'Image type2',
		  'x5', 'x6', 'y5', 'y6', 'z5', 'z6', 'Image type3']), axis=1)

def load_imgs(img_dir, cls, xls_name, sheetname, dims_df, C, verbose=False, target_dims=None, num_ch=3):
	"""Load images stored in folder cls and excel spreadsheet xls_name with name sheetname.
	Saves images to C.full_img_dir and saves vois to the global vois variable.
	Scales images and VOIs so that each voxel is 1.5 x 1.5 x 4 cm
	"""
	
	s = time.time()
	print("\nLoading DCMs of type", sheetname)
	df = pd.read_excel(xls_name, sheetname=sheetname)
	df = preprocess_df(df, C)
	acc_nums = list(set(df['Patient E Number'].dropna().astype(str).tolist()))

	for cnt, acc_num in enumerate(acc_nums):
		if os.path.exists(C.full_img_dir + "\\" + cls + "\\" + str(acc_num) + ".npy"):
			print(acc_num, "has already been saved. Skipping.")
			continue
			
		df_subset = df.loc[df['Patient E Number'].astype(str) == acc_num]
		subdir = img_dir+"\\"+acc_num
		#try:
		art, cur_dims = hf.dcm_load(subdir+r"\T1_AP")
		#except:
		#	print(subdir+"\\T1_AP error")
		#	continue
		try:
			ven, _ = hf.dcm_load(subdir+r"\T1_VP")
		except:
			print(subdir+"\\T1_VP missing")
			continue

		# register phases if venous was not specified separately
		if "Image type2" not in df_subset.columns or df_subset.iloc[0]["Image type2"] != "VP-T1":
			ven, _ = hf.reg_imgs(moving=ven, fixed=art, params=C.params, rescale_only=False)
			
		dims_df = add_to_dims_df(dims_df, acc_num, cur_dims)

		if num_ch == 3:
			try:
				eq, _ = hf.dcm_load(subdir+"\\T1_EQ")
			except:
				print(subdir+"\\T1_EQ missing")
				continue
			if "Image type3" not in df_subset.columns or df_subset.iloc[0]["Image type3"] != "EQ-T1":
				eq, _ = hf.reg_imgs(moving=eq, fixed=art, params=C.params, rescale_only=False)
			img = np.transpose(np.stack((art, ven, eq)), (1,2,3,0))
		else:
			img = np.transpose(np.stack((art, ven)), (1,2,3,0))
			
		if target_dims is not None:
			img, vox_scale = hf.rescale(img, target_dims, cur_dims)
			
		np.save(C.full_img_dir + "\\" + cls + "\\" + str(acc_num), img)

		if verbose:
			print(acc_num, "%d out of %d acc_nums loaded" % (cnt+1, len(acc_nums)))
		else:
			print(".", end="")
			
	print("Overall time: %s" % str(time.time() - s))
	return dims_df

def remove_voi(voi_df_art, voi_df_ven, acc_num, voi_num):
	try:
		voi_row = voi_df_art[voi_df_art["Filename"] == acc_num + ".npy"].iloc[voi_num]
		if len(voi_df_ven[voi_df_ven["id"] == voi_row["id"]]) > 0:
			voi_df_ven = voi_df_ven[voi_df_ven["id"] != voi_row["id"]]
		voi_df_art = voi_df_art[voi_df_art["id"] != voi_row["id"]]
		
	except:
		print(acc_num, "with lesion number", voi_num, "not found.")
		
	return voi_df_art, voi_df_ven


def delete_imgs(acc_nums, cls, C, xls_name=None, sheetname=None):
	if xls_name is not None:
		df = pd.read_excel(xls_name, sheetname=sheetname)
		df = preprocess_df(df, C)

		acc_nums = list(set(df['Patient E Number'].dropna().astype(str).tolist()))
	
	for acc_num in acc_nums:
		#try:
		os.remove(C.full_img_dir + "\\" + cls + "\\" + str(acc_num) + ".npy")
		#except:
		#	continue

def check_folders(img_dir, xls_name, sheetname, C):
	df = pd.read_excel(xls_name, sheetname=sheetname)
	df = preprocess_df(df, C)
	acc_nums = list(set(df['Patient E Number'].dropna().astype(str).tolist()))

	for cnt, acc_num in enumerate(acc_nums):
		df_subset = df.loc[df['Patient E Number'].astype(str) == acc_num]

		if not os.path.exists(img_dir + "\\" +acc_num + "\\T1_AP"):
			print(img_dir + "\\" +acc_num + "\\T1_AP is missing")
		if not os.path.exists(img_dir + "\\" +acc_num + "\\T1_VP"):
			print(img_dir + "\\" +acc_num + "\\T1_VP is missing")
		if not os.path.exists(img_dir + "\\" +acc_num + "\\T1_EQ"):
			print(img_dir + "\\" +acc_num + "\\T1_EQ is missing")