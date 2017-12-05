import numpy as np
import os

###########################
### FOR TRAINING
###########################

def rescale_int(img, intensity_row):
	"""Rescale intensities in img by the """
	img[:,:,:,0] = img[:,:,:,0] / float(intensity_row["art_int"])
	img[:,:,:,1] = img[:,:,:,1] / float(intensity_row["ven_int"])
	img[:,:,:,2] = img[:,:,:,2] / float(intensity_row["eq_int"])

	return img

def collect_unaug_data(classes_to_include, C, voi_df, intensity_df):
    """Return dictionary pointing to X (img data) and Z (filenames) and dictionary storing number of samples of each class."""
    orig_data_dict = {}
    num_samples = {}

    for class_name in classes_to_include:
        x = np.empty((10000, C.dims[0], C.dims[1], C.dims[2], C.nb_channels))
        x2 = np.empty((10000, 2))
        z = []

        for index, img_fn in enumerate(os.listdir(C.orig_dir+class_name)):
            try:
                x[index] = np.load(C.orig_dir+class_name+"\\"+img_fn)
            except:
                raise ValueError(C.orig_dir+class_name+"\\"+img_fn + " not found")
            z.append(img_fn)
            
            row = voi_df[(voi_df["Filename"] == img_fn[:img_fn.find('_')] + ".npy") &
                         (voi_df["lesion_num"] == int(img_fn[img_fn.find('_')+1:-4]))]
            
            try:
                x2[index] = [(float(row["real_dx"]) * float(row["real_dy"]) * float(row["real_dz"])) ** (1/3) / 50,
                            max(float(row["real_dx"]), float(row["real_dy"])) / float(row["real_dz"])]
            except TypeError:
                raise ValueError(img_fn + " is probably missing a voi_df entry.")
            
            x[index] = rescale_int(x[index], intensity_df[intensity_df["AccNum"] == img_fn[:img_fn.find('_')]])

        x.resize((index, C.dims[0], C.dims[1], C.dims[2], C.nb_channels)) #shrink first dimension to fit
        x2.resize((index, 2)) #shrink first dimension to fit
        orig_data_dict[class_name] = [x,x2,np.array(z)]
        num_samples[class_name] = index
        
    return orig_data_dict, num_samples


###########################
### FOR OUTPUTTING IMAGES AFTER TRAINING
###########################


def save_output(Z, y_pred, cls_mapping, C):
    """Parent method; saves all imgs in """
    save_dir = C.output_img_dir
    for cls in cls_mapping:
        if not os.path.exists(save_dir + "\\correct\\" + cls):
            os.makedirs(save_dir + "\\correct\\" + cls)
        if not os.path.exists(save_dir + "\\incorrect\\" + cls):
            os.makedirs(save_dir + "\\incorrect\\" + cls)

    for i in range(len(Z)):
        if y_pred[i] != y_true[i]:
            plot_multich_with_bbox(Z[i], cls_mapping[y_pred[i]], save_dir=save_dir + "\\incorrect\\" + cls_mapping[y_true[i]], C=C)
        else:
            plot_multich_with_bbox(Z[i], cls_mapping[y_pred[i]], save_dir=save_dir + "\\correct\\" + cls_mapping[y_true[i]], C=C)  

def plot_multich_with_bbox(fn, pred_class, num_ch=3, save_dir=None, C=None):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        
    img_fn = fn[:fn.find('_')] + ".npy"
    voi = voi_df_art[(voi_df_art["Filename"] == img_fn) &
                     (voi_df_art["lesion_num"] == int(fn[fn.find('_')+1:fn.rfind('.')]))].iloc[0]
    
    img = np.load(C.crops_dir + voi["cls"] + "\\" + fn)
    img_slice = img[:,:, img.shape[2]//2, :].astype(float)
    for ch in range(img_slice.shape[-1]):
        img_slice[:, :, ch] *= 255/np.amax(img_slice[:, :, ch])
    img_slice = np.stack([img_slice, img_slice, img_slice], axis=2)
    
    img_slice = draw_bbox(img_slice, C.dims, small_vois[fn[:-4]])
        
    ch1 = np.transpose(img_slice[:,::-1,:,0], (1,0,2))
    ch2 = np.transpose(img_slice[:,::-1,:,1], (1,0,2))
    
    if num_ch == 2:
        ret = np.empty([ch1.shape[0]*2, ch1.shape[1], 3])
        ret[:ch1.shape[0],:,:] = ch1
        ret[ch1.shape[0]:,:,:] = ch2
        
    elif num_ch == 3:
        ch3 = np.transpose(img_slice[:,::-1,:,2], (1,0,2))

        ret = np.empty([ch1.shape[0]*3, ch1.shape[1], 3])
        ret[:ch1.shape[0],:,:] = ch1
        ret[ch1.shape[0]:ch1.shape[0]*2,:,:] = ch2
        ret[ch1.shape[0]*2:,:,:] = ch3
        
    else:
        raise ValueError("Invalid num channels")
        
    imsave("%s\\%s (pred %s).png" % (save_dir, fn[:-4], pred_class), ret)

def condense_cm(y_true, y_pred, cls_mapping):
    simplify_map = {'hcc': 0, 'cyst': 1, 'hemangioma': 1, 'fnh': 1, 'cholangio': 2, 'colorectal': 2}
    y_true_simp = np.array([simplify_map[cls_mapping[y]] for y in y_true])
    y_pred_simp = np.array([simplify_map[cls_mapping[y]] for y in y_pred])
    
    return y_true_simp, y_pred_simp, ['hcc', 'benign', 'malignant non-hcc']


def draw_bbox(img_slice, final_dims, voi):
    x1 = voi[0]
    x2 = voi[1]
    y1 = voi[2]
    y2 = voi[3]
    z1 = voi[4]
    z2 = voi[5]
    dx = x2 - x1
    dy = y2 - y1
    dz = z2 - z1
    
    buffer = 0.85
    scale_ratios = [final_dims[0]/dx * buffer, final_dims[1]/dy * buffer, final_dims[2]/dz * buffer]
    
    crop = [img_slice.shape[i] - round(final_dims[i]/scale_ratios[i]) for i in range(2)]
    
    for i in range(2):
        assert crop[i]>=0
        
    x1 = crop[0]//2
    x2 = -crop[0]//2
    y1 = crop[1]//2
    y2 = -crop[1]//2

    img_slice[x1:x2, y2, 2, :] = 255
    img_slice[x1:x2, y2, :2, :] = 0

    img_slice[x1:x2, y1, 2, :] = 255
    img_slice[x1:x2, y1, :2, :] = 0

    img_slice[x1, y1:y2, 2, :] = 255
    img_slice[x1, y1:y2, :2, :] = 0

    img_slice[x2, y1:y2, 2, :] = 255
    img_slice[x2, y1:y2, :2, :] = 0
    
    return img_slice