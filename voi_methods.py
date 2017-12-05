import copy
import helper_fxns as hf
import math
import numpy as np
import time

def extract_voi(img, voi, min_dims, ven_voi=[], eq_voi=[]):
    """Input: image, a voi to center on, and the min dims of the unaugmented img.
    Outputs voi-centered image and classes.
    Todo: new_voi should never be negative!
    """
    
    voi_imgs = []
    classes = []
    temp_img = copy.deepcopy(img)
    
    x1 = voi['x1']
    x2 = voi['x2']
    y1 = voi['y1']
    y2 = voi['y2']
    z1 = voi['z1']
    z2 = voi['z2']
    dx = x2 - x1
    dy = y2 - y1
    dz = z2 - z1
    assert dx > 0
    assert dy > 0
    assert dz > 0
    
    # align all phases
    if len(ven_voi) > 0:
        ven_voi = ven_voi.iloc[0]
        temp_img = hf.align(temp_img, voi, ven_voi, 1)
        
    if len(eq_voi) > 0:
        eq_voi = eq_voi.iloc[0]
        temp_img = hf.align(temp_img, voi, eq_voi, 2)

    #padding around lesion
    xpad = max(min_dims[0], dx) * 2*math.sqrt(2) - dx
    ypad = max(min_dims[1], dy) * 2*math.sqrt(2) - dy
    zpad = max(min_dims[2], dz) * 2*math.sqrt(2) - dz
    
    #padding in case voi is too close to edge
    side_padding = math.ceil(max(xpad, ypad, zpad) / 2)
    pad_img = []
    for ch in range(temp_img.shape[-1]):
        pad_img.append(np.pad(temp_img[:,:,:,ch], side_padding, 'constant'))
    pad_img = np.stack(pad_img, axis=3)
    
    assert xpad > 0
    assert ypad > 0
    assert zpad > 0
    
    #choice of ceil/floor needed to make total padding amount correct
    x1 += side_padding - math.floor(xpad/2)
    x2 += side_padding + math.ceil(xpad/2)
    y1 += side_padding - math.floor(ypad/2)
    y2 += side_padding + math.ceil(ypad/2)
    z1 += side_padding - math.floor(zpad/2)
    z2 += side_padding + math.ceil(zpad/2)
    
    #x1 = voi['x1'] + side_padding - math.ceil(xpad/2)
    
    new_voi = [xpad//2, dx + xpad//2,
               ypad//2, dy + ypad//2,
               zpad//2, dz + zpad//2]

    pad_img = pad_img[x1:x2, y1:y2, z1:z2, :]
    
    for i in new_voi:
        assert i>=0
        
    return pad_img, voi['cls'], [int(x) for x in new_voi]

def augment_img(img, final_dims, voi, num_samples, translate=None, add_reflections=False):
    """For rescaling an img to final_dims while scaling to make sure the image contains the voi."""
    
    x1 = voi[0]
    x2 = voi[1]
    y1 = voi[2]
    y2 = voi[3]
    z1 = voi[4]
    z2 = voi[5]
    dx = x2 - x1
    dy = y2 - y1
    dz = z2 - z1
    
    buffer1 = 0.7
    buffer2 = 0.9
    scale_ratios = [final_dims[0]/dx, final_dims[1]/dy, final_dims[2]/dz]

    aug_imgs = []
    
    for _ in range(num_samples):
        scales = [random.uniform(scale_ratios[0]*buffer1, scale_ratios[0]*buffer2),
                 random.uniform(scale_ratios[1]*buffer1, scale_ratios[1]*buffer2),
                 random.uniform(scale_ratios[2]*buffer1, scale_ratios[2]*buffer2)]
        
        angle = random.randint(0, 359)

        temp_img = tr.scale3d(img, scales)
        temp_img = tr.rotate(temp_img, angle)
        
        if translate is not None:
            trans = [random.randint(-translate[0], translate[0]),
                     random.randint(-translate[1], translate[1]),
                     random.randint(-translate[2], translate[2])]
        else:
            trans = [0,0,0]
        
        flip = [random.choice([-1, 1]), random.choice([-1, 1]), random.choice([-1, 1])]

        crops = [temp_img.shape[i] - final_dims[i] for i in range(3)]
    
        for i in range(3):
            assert crops[i]>=0

        #temp_img = add_noise(temp_img)

        temp_img = temp_img[crops[0]//2 *flip[0] + trans[0] : -crops[0]//2 *flip[0] + trans[0] : flip[0],
                            crops[1]//2 *flip[1] + trans[1] : -crops[1]//2 *flip[1] + trans[1] : flip[1],
                            crops[2]//2 *flip[2] + trans[2] : -crops[2]//2 *flip[2] + trans[2] : flip[2], :]
        
        aug_imgs.append(temp_img)
        
        if add_reflections:
            aug_imgs.append(tr.generate_reflected_img(temp_img))
    
    return aug_imgs

def extract_vois(acc_nums, small_vois, classes_to_include, C, voi_df_art, voi_df_ven, voi_df_eq):
    """Call extract_voi for a list of acc_nums"""
    
    t = time.time()

    # iterate over image series
    for img_fn in os.listdir(C.full_img_dir + cls):
        img = np.load(C.full_img_dir+"\\"+img_fn)
        art_vois = voi_df_art[(voi_df_art["Filename"] == img_fn) & (voi_df_art["cls"].isin(classes_to_include))]

        # iterate over each voi in that image
        for voi_num, voi in enumerate(art_vois.iterrows()):
            ven_voi = voi_df_ven[voi_df_ven["id"] == voi[1]["id"]]
            eq_voi = voi_df_eq[voi_df_eq["id"] == voi[1]["id"]]

            cropped_img, cls, small_voi = extract_voi(img, copy.deepcopy(voi[1]), final_size, ven_voi=ven_voi, eq_voi=eq_voi)
            fn = img_fn[:-4] + "_" + str(voi[1]["lesion_num"])
            np.save(C.crops_dir + cls + "\\" + fn, cropped_img)
            small_vois[fn] = small_voi

            if voi_num % 20 == 0:
                print(".", end="")
    print("")
    print(time.time()-t)
    
    return small_vois

def resize_img(img, final_dims, voi):
    """For rescaling an img to final_dims while scaling to make sure the image contains the voi.
    """
    
    x1 = voi[0]
    x2 = voi[1]
    y1 = voi[2]
    y2 = voi[3]
    z1 = voi[4]
    z2 = voi[5]
    dx = x2 - x1
    dy = y2 - y1
    dz = z2 - z1
    
    padding = 0.85
    #consider making the padding bigger for small lesions
    scale_ratios = [final_dims[0]/dx * padding, final_dims[1]/dy * padding, final_dims[2]/dz * padding]
    
    img = tr.scale3d(img, scale_ratios)
    #if scale_ratio < 0.9: #need to shrink original image to fit
    #    img = tr.scale3d(img, [scale_ratio]*3)
    #elif scale_ratio > 1.4: #need to enlarge original image
    #    img = tr.scale3d(img, [scale_ratio]*3)
    
    crop = [img.shape[i] - final_dims[i] for i in range(3)]
    
    for i in range(3):
        assert crop[i]>=0
    
    return img[crop[0]//2:-crop[0]//2, crop[1]//2:-crop[1]//2, crop[2]//2:-crop[2]//2, :]