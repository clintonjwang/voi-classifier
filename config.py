"""
Config file

Author: Clinton Wang, E-mail: `clintonjwang@gmail.com`, Github: `https://github.com/clintonjwang/voi-classifier`
"""

from os.path import *
from keras.callbacks import EarlyStopping
from keras.optimizers import Adam
import numpy as np
import torch

import data_retrieval, preprocessing, model, classify_feats
import multiprocessing

def config_A(A):
    # paths and folders
    A["CPU cores to use"] = multiprocessing.cpu_count() - 2
    A["base folder"] = "/data/vision/polina/users/clintonw/code/voi-classifier"

    A["raw df path"] = join(A["base folder"], 'excel/Prototype1e.xlsx')
    A["accnum df path"] = join(A["base folder"], "excel", "accnum_data.csv")
    A["lesion df path"] = join(A["base folder"], "excel", "lesion_data.csv")
    A["run stats path"] = join(A["base folder"], "excel", "overnight_run.csv")
    A["label df path"] = join(A["base folder"], "excel", "lesion_labels.csv")

    # A["dcm dirs"] = [join("DICOMs", cls) for cls in A["lesion classes"]]
    A["full img folder"] = join(A["base folder"], "full_imgs")
    A["cropped img folder"] = join(A["base folder"], "imgs", "rough_crops")
    A["non-augmented img folder"] = join(A["base folder"], "imgs", "unaug_imgs")
    A["augmented img folder"] = join(A["base folder"], "imgs", "aug_imgs")
    A["model folder"] = join(A["base folder"], "models")
    A["phase dirs"] = ["T1_20s", "T1_70s", "T1_3min"]

    # dataframe headers
    A["voxel size columns"] = ["voxdim_x", "voxdim_y", "voxdim_z"]
    A["accnum columns"] = ["MRN", "Sex", "AgeAtImaging", "Ethnicity"] + A["voxel size columns"] + ["downsample"]
    A["VOI columns"] = [list(np.array([[ph + ch + '1', ph + ch + '2'] for ch in ['x', 'y', 'z']]).flatten()) for ph in
                     ['a_', 'v_', 'e_']]
    A["art cols"], A["ven cols"], A["equ cols"] = A["VOI columns"]
    A["pad cols"] = ['pad_x', 'pad_y', 'pad_z']
    A["VOI columns"] = list(np.array(A["VOI columns"]).flatten()) + A["pad cols"]

    # data retrieval
    A["stage"] = 2
    A["number of channels"] = 3
    A["lesion classes"] = ['hcc', 'cholangio', 'colorectal', 'cyst', 'hemangioma', 'fnh']
    A["number of classes"] = len(A["lesion classes"])
    A["sheetnames"] = ['HCC', 'Cholangio', 'Colorectal', 'Cyst', 'Hemangioma', 'FNH']
    A["short class names"] = ['HCC', 'ICC', 'CRC Met.', 'Cyst', 'Hemang.', 'FNH']
    A["simplify map"] = {'hcc': 0, 'cyst': 1, 'hemangioma': 1, 'fnh': 1, 'cholangio': 2, 'colorectal': 2}

    # data preprocessing and augmentation
    A["image dimensions"] = [32,32,16]
    A["augmentation factor"] = 100

    A["lesion ratio"] = .7 # ratio of the lesion side length to the length of the cropped image
    A["pre scale"] = .8 # normalizes images while saving augmented/unaugmented images
    A["post scale"] = 0. # normalizes images at train/test time
    A["intensity scaling"] = [.1,.01]

    # CNN training
    A["variable to predict"] = "lesion class"
    A["input shape"] = (A["number of channels"], *A["image dimensions"])
    A["max number of runs"] = "infinite"
    A["fix test lesions"] = True
    A["fixed test lesion ids"] = ['E103312835_1', '12823036_0', '12569915_0', 'E102093118_0', 'E102782525_0', '12799652_0',
         'E100894274_0', '12874178_3', 'E100314676_0', '12842070_0', '13092836_2', '12239783_0',
         '12783467_0', '13092966_0', 'E100962970_0', 'E100183257_1', 'E102634440_0', 'E106182827_0',
         '12582632_0', 'E100121654_0', 'E100407633_0', 'E105310461_0', '12788616_0', 'E101225606_0',
         '12678910_1', 'E101083458_1', '12324408_0', '13031955_0', 'E101415263_0', 'E103192914_0',
         '12888679_2', 'E106096969_0', 'E100192709_1', '13112385_1', 'E100718398_0', '12207268_0',
         'E105244287_0', 'E102095465_0', 'E102613189_0', '12961059_0', '11907521_0', 'E105311123_0',
         '12552705_0', 'E100610622_0', '12975280_0', 'E105918926_0', 'E103020139_1', 'E101069048_1',
         'E105427046_0', '13028374_0', 'E100262351_0', '12302576_0', '12451831_0', 'E102929168_0',
         'E100383453_0', 'E105344747_0', '12569826_0', 'E100168661_0', '12530153_0', 'E104697262_0']

    # training hyperparameters
    P = {}
    P["batch size"] = 24
    P["steps per epoch"] = 64 #256
    P["epochs"] = 2 #350
    P["optimizer"] = "adam"
    P["loss"] = "categorical cross-entropy"
    #P["early stopping"] = EarlyStopping(monitor='loss', min_delta=0.0001, patience=25)
    A("set training settings")(**P)

    for mod in [data_retrieval, preprocessing, model, classify_feats]:
        for keywords, action in mod.get_actions(A):
            for k in keywords:
                if k == "TKTK":
                    print(mod)
                A("add action")(k, action)

    obj_cls = []
    for mod in [model]:
        obj_cls += mod.get_obj_cls(A)

    props = []
    for mod in [model]:
        props += mod.get_props(A)
