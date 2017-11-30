from convert_dicom import dicom_series_to_nifti
from convert_siemens import dicom_to_nifti
import copy
import math
import matplotlib
import matplotlib.pyplot as plt
import nibabel
import numpy as np
import os
import pyelastix
import requests
import random
import transforms as tr

###########################
### IMAGE LOADING
###########################

def dcm_load(path2series):
    """
    Load a dcm series as a 3D array along with its dimensions.
    
    returns as a tuple:
    - the normalized (0-255) image
    - the spacing between pixels in cm
    """

    try:
        tmp_fn = "tmp.nii"
        dicom_series_to_nifti(path2series, tmp_fn)
        #dicom_to_nifti(path2series, tmp_fn)

        ret = ni_load(tmp_fn)

    except Exception as e:
        print(path2series, e)
        return None

    try:
        os.remove(tmp_fn)
    except:
        print("Cannot delete %s" % tmp_fn)

    return ret

def ni_load(filename):
    """
    Load a nifti image as a 3D array along with its dimensions.
    
    returns as a tuple:
    - the normalized (0-255) image
    - the spacing between pixels in cm
    """
    
    img = nibabel.load(filename)
    img = nibabel.as_closest_canonical(img) # make sure it is in the correct orientation

    dims = img.header['pixdim'][1:4]
    dim_units = img.header['xyzt_units']
    
    img = np.asarray(img.dataobj).astype(dtype='float64')
    img = img[::-1,:,:]
    img =  (255 * (img / np.amax(img))).astype(dtype='uint8')
    
    if dim_units == 2: #or np.sum(img) * dims[0] * dims[1] * dims[2] > 10000:
        dims = [d/10 for d in dims]
    
    return img, dims


###########################
### IMAGE PREPROCESSING
###########################

def apply_mask(img, mask_file):
    """Apply the mask in mask_file to img and return the masked image."""

    with open(mask_file, 'rb') as f:
        mask = f.read()
        mask = np.fromstring(mask, dtype='uint8')
        mask = np.array(mask).reshape((img.shape[2], img.shape[1], img.shape[0]))
        #mask = np.reshape(mask, img.shape, order='F')
        mask = np.transpose(mask, (2,1,0))
        #mask = mask[:,::-1,:]
        
    img[:,:,:,0][mask == 0] = 0
    #img[mask <= 0] = 0

    return img

def create_diff(art_img, pre_img):
    diff = art_img - pre_img # Calculate the subtracted image.
    diff[diff < 0] = 0  # The pre-contrast should never be more than the arterial. Clamp negative values to zero.

    draw_fig(diff, 'whole')
    
    return diff

def rescale(img, target_dims, cur_dims):
    vox_scale = [float(cur_dims[i]/target_dims[i]) for i in range(3)]
    img = tr.scale3d(img, vox_scale)
    
    return img, vox_scale

def normalize(img):
    #t2[mrn] = t2[mrn] * 255/np.amax(t2[mrn])
    i_min = np.amin(img)
    return (img - i_min) / (np.amax(img) - i_min) * 255

def flipz(img_fn, voi_df):
    def func(row):
        if row["Filename"] == img_fn:
            z1 = row['z1']
            row['z1'] = img.shape[2]-row['z2']
            row['z2'] = img.shape[2]-z1
        return row
    
    img = np.load("full_imgs\\"+img_fn)
    
    return voi_df.apply(lambda row: func(row), axis=1)

def reg_imgs(moving, fixed, params, rescale_only=False):    
    fshape = fixed.shape
    mshape = moving.shape
    scale = [fshape[i]/mshape[i] for i in range(3)]
    moving = tr.scale3d(moving, scale)
    
    assert moving.shape == fixed.shape, ("Shapes not aligned in reg_imgs", moving.shape, fixed.shape)
    
    if not rescale_only:
        moving = np.ascontiguousarray(moving).astype('float32')
        fixed = np.ascontiguousarray(fixed).astype('float32')

        moving, _ = pyelastix.register(moving, fixed, params, verbose=0)
        
    return moving, scale


###########################
### IMAGE AUGMENTATION
##########################

"""def augment(img, final_size, num_samples = 100, exceed_ratio=1, translate=None):
    aug_imgs = []
    
    for _ in range(num_samples):
        temp_img = img
        angle = random.randint(0, 359)
        temp_img = tr.rotate(temp_img, angle)
        
        if translate is not None:
            trans = [random.randint(-translate[0], translate[0]),
                     random.randint(-translate[1], translate[1]),
                     random.randint(-translate[2], translate[2])]
        else:
            trans = [0,0,0]
        
        flip = [random.choice([-1, 1]), random.choice([-1, 1]), random.choice([-1, 1])]

        if exceed_ratio < 1:
            scales = [(1 + 1/exceed_ratio) / 2, (1 + 5/exceed_ratio) / 6]
            scale = [random.uniform(scales[0],scales[1]), random.uniform(scales[0],scales[1]), random.uniform(scales[0],scales[1])]
        else:
            scale = 1

        crops = [(temp_img.shape[i] - final_size[i])*scale[i] for i in range(3)]

        #temp_img = add_noise(temp_img)

        temp_img = temp_img[math.floor(crops[0]/2)*flip[0] + trans[0] : -math.ceil(crops[0]/2)*flip[0] + trans[0] : flip[0],
                                 math.floor(crops[1]/2)*flip[1] + trans[1] : -math.ceil(crops[1]/2)*flip[1] + trans[1] : flip[1],
                                 math.floor(crops[2]/2)*flip[2] + trans[2] : -math.ceil(crops[2]/2)*flip[2] + trans[2] : flip[2], :]

        if scale != 1:
            temp_img = tr.scale3d(temp_img, scale)
        
        aug_imgs.append(temp_img)
    
    return aug_imgs"""

###########################
### VOIs
###########################

def align(img, voi, ven_voi, ch):
    temp_ven = copy.deepcopy(img[:,:,:,ch])
    dx = ((ven_voi["x1"] + ven_voi["x2"]) - (voi["x1"] + voi["x2"])) // 2
    dy = ((ven_voi["y1"] + ven_voi["y2"]) - (voi["y1"] + voi["y2"])) // 2
    dz = ((ven_voi["z1"] + ven_voi["z2"]) - (voi["z1"] + voi["z2"])) // 2
    
    pad = int(max(abs(dx), abs(dy), abs(dz)))+1
    temp_ven = np.pad(temp_ven, pad, 'constant')[pad+dx:-pad+dx, pad+dy:-pad+dy, pad+dz:-pad+dz]
    
    if ch == 1:
        return np.stack([img[:,:,:,0], temp_ven, img[:,:,:,2]], axis=3)
    elif ch == 2:
        return np.stack([img[:,:,:,0], img[:,:,:,1], temp_ven], axis=3)

def scale_vois(x, y, z, pre_reg_scale, field=None, post_reg_scale=None):
    scale = pre_reg_scale
    x = (round(x[0]*scale[0]), round(x[1]*scale[0]))
    y = (round(y[0]*scale[1]), round(y[1]*scale[1]))
    z = (round(z[0]*scale[2]), round(z[1]*scale[2]))
    
    if field is not None:
        xvoi_distortions = field[0][x[0]:x[1]+1, y[0]:y[1]+1, z[0]:z[1]+1]
        yvoi_distortions = field[1][x[0]:x[1]+1, y[0]:y[1]+1, z[0]:z[1]+1]
        zvoi_distortions = field[2][x[0]:x[1]+1, y[0]:y[1]+1, z[0]:z[1]+1]

        x = (x[0] + int(np.amin(xvoi_distortions[0,:,:])), x[1] + int(np.amax(xvoi_distortions[-1,:,:])))
        y = (y[0] + int(np.amin(yvoi_distortions[:,0,:])), y[1] + int(np.amax(yvoi_distortions[:,-1,:])))
        z = (z[0] + int(np.amin(zvoi_distortions[:,:,0])), z[1] + int(np.amax(zvoi_distortions[:,:,-1])))
    
        scale = post_reg_scale
        x = (round(x[0]*scale[0]), round(x[1]*scale[0]))
        y = (round(y[0]*scale[1]), round(y[1]*scale[1]))
        z = (round(z[0]*scale[2]), round(z[1]*scale[2]))
    
    return x, y, z

def add_deltas(voi_df):
    """No longer in use"""
    voi_df = voi_df.astype({"x1": int, "x2": int, "y1": int, "y2": int, "z1": int, "z2": int})
    voi_df['dx'] = voi_df.apply(lambda row: row['x2'] - row['x1'], axis=1)
    voi_df['dy'] = voi_df.apply(lambda row: row['y2'] - row['y1'], axis=1)
    voi_df['dz'] = voi_df.apply(lambda row: row['z2'] - row['z1'], axis=1)
    
    return voi_df

def align_phases(img, voi, ven_voi):
    """Translates venous phase to align with arterial phase"""
    temp_ven = copy.deepcopy(img[:,:,:,1])
    dx = ((voi["x1"] + voi["x2"]) - (ven_voi["x1"] + ven_voi["x2"])) // 2
    dy = ((voi["y1"] + voi["y2"]) - (ven_voi["y1"] + ven_voi["y2"])) // 2
    dz = ((voi["z1"] + voi["z2"]) - (ven_voi["z1"] + ven_voi["z2"])) // 2
    
    pad = int(max(abs(dx), abs(dy), abs(dz)))+1
    temp_ven = np.pad(temp_ven, pad, 'constant')[pad+dx:-pad+dx, pad+dy:-pad+dy, pad+dz:-pad+dz]
    
    return np.stack([img[:,:,:,0], temp_ven], axis=3)


###########################
### IMAGE FEATURES
###########################

def get_vol(img, dims, dim_units):
    return np.sum(img>0) * np.prod(dims)

def get_hist(img):
    """Returns histogram in array and graphical forms."""
    h = plt.hist(flatten(img, times=2))
    plt.title("Histogram")
    plt.xlabel("Value")
    plt.ylabel("Frequency")
    
    #mean_intensity = np.mean(diff[img > 0])
    #std_intensity = np.std(diff[img > 0])

    return h, plt.gcf()
    
#########################
### UTILITY
#########################

def get_voi_id(acc_num, x, y, z):
    return ''.join(map(str, [acc_num, x[0], y[0], z[0]]))

def plot_section_auto(img):
    plt.subplot(131)
    plt.imshow(np.transpose(img[:, ::-1, img.shape[2]//2, 0], (1,0)), cmap='gray')
    plt.subplot(132)
    plt.imshow(np.transpose(img[:, ::-1, img.shape[2]//2, 1], (1,0)), cmap='gray')
    plt.subplot(133)
    plt.imshow(np.transpose(img[:, ::-1, img.shape[2]//2, 2], (1,0)), cmap='gray')

def plot_section_auto_scan(img, frac):
    plt.subplot(231)
    plt.imshow(np.transpose(img[:, ::-1, img.shape[2]//2, 0], (1,0)), cmap='gray')
    plt.subplot(232)
    plt.imshow(np.transpose(img[:, ::-1, img.shape[2]//2, 1], (1,0)), cmap='gray')
    plt.subplot(233)
    plt.imshow(np.transpose(img[:, ::-1, img.shape[2]//2, 2], (1,0)), cmap='gray')

    plt.subplot(234)
    plt.imshow(np.transpose(img[:, ::-1, int(img.shape[2]*frac), 0], (1,0)), cmap='gray')
    plt.subplot(235)
    plt.imshow(np.transpose(img[:, ::-1, int(img.shape[2]*frac), 1], (1,0)), cmap='gray')
    plt.subplot(236)
    plt.imshow(np.transpose(img[:, ::-1, int(img.shape[2]*frac), 2], (1,0)), cmap='gray')

def plot_section_scan(img, frac=None):
    plt.subplot(231)
    plt.imshow(np.transpose(img[:, ::-1, 0, 0], (1,0)), cmap='gray')
    plt.subplot(232)
    plt.imshow(np.transpose(img[:, ::-1, 0, 1], (1,0)), cmap='gray')
    plt.subplot(233)
    plt.imshow(np.transpose(img[:, ::-1, 0, 2], (1,0)), cmap='gray')

    if frac is None:
        plt.subplot(234)
        plt.imshow(np.transpose(img[:, ::-1, -1, 0], (1,0)), cmap='gray')
        plt.subplot(235)
        plt.imshow(np.transpose(img[:, ::-1, -1, 1], (1,0)), cmap='gray')
        plt.subplot(236)
        plt.imshow(np.transpose(img[:, ::-1, -1, 2], (1,0)), cmap='gray')

    else:
        plt.subplot(234)
        plt.imshow(np.transpose(img[:, ::-1, int(img.shape[2]*frac), 0], (1,0)), cmap='gray')
        plt.subplot(235)
        plt.imshow(np.transpose(img[:, ::-1, int(img.shape[2]*frac), 1], (1,0)), cmap='gray')
        plt.subplot(236)
        plt.imshow(np.transpose(img[:, ::-1, int(img.shape[2]*frac), 2], (1,0)), cmap='gray')

def plot_section(img, df, pad=30, flipz="both"):
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
    
def plot_section_mrn(mrn,x,y,z, pad=30):
    plt.subplot(211)
    plt.imshow(np.transpose(art[mrn][x[0]-pad:x[1]+pad, y[1]+pad:y[0]-pad:-1, (z[0]+z[1])//2], (1,0)), cmap='gray')

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