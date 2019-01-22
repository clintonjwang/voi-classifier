import datetime
import glob
import os
import random
import time
from os.path import *
import subprocess

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from am.dispositio.action import Action

from joblib import Parallel, delayed

def get_actions(A):
    acts = [
        # i/o
        (["convert dcm to nii"], Action(dcm2nii(A))),
        (["convert dcm to npy"], Action(dcm2npy(A))),
        (["convert nii to npy"], Action(nii2npy(A))),

        (["for all lesion classes"], Action(for_all_classes(A))),
        (["get raw input df"], Action(get_raw_input_df(A))),
        (["get lesion df"], Action(get_lesion_df(A))),
        (["get accnum df"], Action(get_accnum_df(A))),
    ]

    return acts

def for_all_classes(A):
    def fxn(act):
        return [A(act)(cls=cls) for cls in A["lesion classes"]]
    return fxn

def nii2npy(A):
    def fxn():
        raise NotImplementedError("nii2npy")
    return fxn

###########################
### METHODS FOR EXTRACTING VOIS FROM THE SPREADSHEET
###########################

def build_coords_df(accnum_xls_path):
    """Builds all coords from scratch, without the ability to add or change individual coords"""
    input_df = pd.read_excel(accnum_xls_path, "Prelim Analysis Patients", index_col=0, parse_cols="A,J")

    accnum_dict = {category: list(input_df[input_df["Category"] == category].index.astype(str)) for category in A["sheetnames"]}

    writer = pd.ExcelWriter(A["coord xls path"])

    for category in A["sheetnames"]:
        print(category)
        #if exists(A["coord xls path"]):
        #   coords_df = pd.read_excel(A["coord xls path"], sheet_name=category, index_col=0)
        #   if not overwrite:
        #       accnum_dict[category] = list(set(accnum_dict[category]).difference(coords_df['acc #'].values.astype(str)))
        #else:
        coords_df = pd.DataFrame(columns=['acc #', 'Run', 'Flipped',
              'x1', 'y1', 'z1', 'x2', 'y2', 'z2'])

        for ix,accnum in enumerate(accnum_dict[category]):
            """load_dir = join(A["dcm dirs"][0], accnum)
            if not exists(join(load_dir, 'Segs', 'tumor_20s_0.ids')):
                masks.off2ids(join(load_dir, 'Segs', 'tumor_20s.off'))

            try:
                art,D = io.nii_load(join(load_dir, "nii_dir", "20s.nii.gz"))
            except:
                raise ValueError(load_dir)
            #ven,_ = io.dcm_load(join(load_dir, A["phases"][1]))
            #equ,_ = io.dcm_load(join(load_dir, A["phases"][2]))

            for fn in glob.glob(join(load_dir, 'Segs', 'tumor_20s_*.ids')):
                try:
                    _,coords = masks.crop_img_to_mask_vicinity([art,D], fn[:-4], return_crops=True)
                except:
                    raise ValueError(fn)
                lesion_id = accnum + fn[fn.rfind('_'):-4]
                coords_df.loc[lesion_id] = [accnum, "1", ""] + coords[0] + coords[1]
                #   M = masks.get_mask(fn, D, img.shape)
                #   M = io.crop_nonzero(M, C)[0]"""
            coords_df.loc[accnum+"_0"] = [accnum, "1", ""] + [0]*6

            print('.', end='')
            if ix % 5 == 2:
                coords_df.to_excel(writer, sheet_name=category)
                writer.save()

        coords_df.to_excel(writer, sheet_name=category)
        writer.save()

def dcm2nii(A):
    def fxn(cls=None, accnums=None, overwrite=False, exec_reg=False):
        """Converts dcms to full-size npy, update accnum_df. Requires coords_df."""

        src_data_df = get_coords_df(cls)
        if accnums is None:
            accnums = list(set(src_data_df['acc #'].values))
        else:
            accnums = set(accnums).intersection(src_data_df['acc #'].values)

        if A['dcm_dirs'] is not None:
            root = A["dcm dirs"][A["lesion classes"].index(cls)]
        else:
            root = A["dcm dir"]

        for cnt, accnum in enumerate(accnums):
            load_dir = join(root, accnum)

            if not exists(join(load_dir, A["phase dirs"][0])):
                load_dir = join("Z:\\LIRADS\\DICOMs\\hcc", accnum)
                if not exists(join(load_dir, A["phase dirs"][0])):
                    continue
            if not overwrite and exists(join(load_dir, "nii_dir", "20s.nii.gz")):
                continue

            try:
                art,D = io.dcm_load(join(load_dir, A["phase dirs"][0]), flip_x=False, flip_y=False)
                ven,_ = io.dcm_load(join(load_dir, A["phase dirs"][1]), flip_x=False, flip_y=False)
                eq,_ = io.dcm_load(join(load_dir, A["phase dirs"][2]), flip_x=False, flip_y=False)

                if exec_reg:
                    ven,_ = reg.reg_elastix(moving=ven, fixed=art)
                    eq,_ = reg.reg_elastix(moving=eq, fixed=art)

                nii_dir = join(load_dir, "nii_dir")
                if not exists(nii_dir):
                    os.makedirs(nii_dir)
                io.save_nii(art, join(nii_dir, "20s.nii.gz"), D)
                io.save_nii(ven, join(nii_dir, "70s.nii.gz"), D)
                io.save_nii(eq, join(nii_dir, "3min.nii.gz"), D)
            except:
                raise ValueError(accnum)
    return fxn

def dcm2npy(A):
    def fxn(cls=None, accnums=None, overwrite=False, exec_reg=False, save_seg=False, downsample=1):
        """Converts dcms to full-size npy, update accnum_df. Requires coords_df."""

        src_data_df = get_coords_df(cls)
        accnum_df = get_accnum_df()
        if accnums is None:
            accnums = list(set(src_data_df['acc #'].values))
        else:
            accnums = set(accnums).intersection(src_data_df['acc #'].values)

        if A['dcm_dirs'] is not None:
            root = A["dcm dirs"][A["lesion classes"].index(cls)]
        else:
            root = A["dcm dir"]

        for cnt, accnum in enumerate(accnums):
            load_dir = join(root, accnum)
            save_path = join(A["full img dir"], accnum + ".npy")

            if not exists(join(load_dir, A["phase dirs"][0])):
                load_dir = join("Z:\\LIRADS\\DICOMs\\hcc", accnum)
                if not exists(join(load_dir, A["phase dirs"][0])):
                    continue
            if not overwrite and exists(save_path) and accnum in accnum_df.index and \
                    not np.isnan(accnum_df.loc[accnum, "voxdim_x"]):
                continue

            flip = src_data_df.loc[src_data_df['acc #'] == accnum, "Flipped"].values[0]
            if type(flip) != str:
                flip_z = [False]*3
            elif flip == 'Yes':
                flip_z = [True]*3
            else:
                flip_z = [char in flip for char in ['A','V','E']]
            try:
                if exists(join(load_dir, "nii_dir", "20s.nii.gz")):
                    art,D = io.nii_load(join(load_dir, "nii_dir", "20s.nii.gz"), flip_x=True, flip_y=True, flip_z=flip_z[0])
                    ven,_ = io.nii_load(join(load_dir, "nii_dir", "70s.nii.gz"), flip_x=True, flip_y=True, flip_z=flip_z[1])
                    eq,_ = io.nii_load(join(load_dir, "nii_dir", "3min.nii.gz"), flip_x=True, flip_y=True, flip_z=flip_z[2])
                else:
                    art,D = io.dcm_load(join(load_dir, A["phase dirs"][0]), flip_z=flip_z[0])
                    ven,_ = io.dcm_load(join(load_dir, A["phase dirs"][1]), flip_z=flip_z[1])
                    eq,_ = io.dcm_load(join(load_dir, A["phase dirs"][2]), flip_z=flip_z[2])

                if exec_reg:
                    art, ven, eq, slice_shift = reg.crop_reg(art, ven, eq)#, "bspline", num_iter=30)
                else:
                    slice_shift = 0

                img = np.stack((art, ven, eq), -1)

                if np.product(art.shape) > A["max size"]:
                    downsample = min((np.product(art.shape) / A["max size"])**(1/3), 1.5)

                if downsample != 1:
                    img = tr.scale3d(img, [1/downsample, 1/downsample, 1])
                    D = [D[0]*downsample, D[1]*downsample, D[2]]
            except:
                raise ValueError(accnum)

            np.save(join(A["full img dir"], accnum+".npy"), img)

            if save_seg:
                import seg_methods as sm
                sm.save_segs([accnum], downsample, slice_shift, art.shape[-1])

            if cnt % 3 == 2:
                print(".", end="")
            accnum_df.loc[accnum] = get_patient_row(load_dir) + list(D) + [downsample]
            accnum_df.to_csv(A["accnum df path"])
    return fxn

def load_vois(A):
    def fxn(cls=None, accnums=None, overwrite=False, save_seg=False):
        """Updates the voi_dfs based on the raw spreadsheet.
        dcm2npy() must be run first to produce full size npy images."""

        if accnums is None:
            accnums = set(src_data_df['acc #'].values)
        else:
            accnums = set(accnums).intersection(src_data_df['acc #'].values)

        if overwrite:
            lesion_df = lesion_df[~((lesion_df["accnum"].isin(accnums)) & (lesion_df["cls"] == cls))]
        else:
            accnums = set(accnums).difference(lesion_df[lesion_df["cls"] == cls]["accnum"].values)

        for cnt, accnum in enumerate(accnums):
            df_subset = src_data_df[src_data_df['acc #'] == accnum]

            """if save_seg:
                load_dir = join(A["dcm dir"], accnum)
                I,_ = io.nii_load(join(load_dir, "nii_dir", "20s.nii.gz"))
                
                downsample = 1
                if np.product(I.shape) > A["max size"]:
                    downsample = 2
                    for i in ['x','y']:
                        for j in ['1','2']:
                            df_subset[i+j] = df_subset[i+j] / downsample
                sm.save_segs([accnum], downsample)"""

            for _, row in df_subset.iterrows():
                x,y,z = [[int(row[ch+'1']), int(row[ch+'2'])] for ch in ['x','y','z']]
                if accnum_df.loc[accnum, "downsample"] != 1:
                    x /= accnum_df.loc[accnum, "downsample"]
                    y /= accnum_df.loc[accnum, "downsample"]

                lesion_ids = lesion_df[lesion_df["accnum"] == accnum].index
                if len(lesion_ids) > 0:
                    lesion_nums = [int(lid[lid.find('_')+1:]) for lid in lesion_ids]
                    for num in range(len(lesion_nums)+1):
                        if num not in lesion_nums:
                            new_num = num
                else:
                    new_num = 0

                l_id = accnum + "_" + str(new_num)
                lesion_df.loc[l_id, ["accnum", "cls", "run_num"] + A["art cols"]] = \
                            [accnum, cls, int(row["Run"])]+list([*x,*y,*z])

                if 'x3' in row and not np.isnan(row['x3']):
                    x,y,z = [[int(row[ch+'3']), int(row[ch+'4'])] for ch in ['x','y','z']]
                    if accnum_df.loc[accnum, "downsample"] != 1:
                        x /= accnum_df.loc[accnum, "downsample"]
                        y /= accnum_df.loc[accnum, "downsample"]
                    lesion_df.loc[l_id, A["ven cols"]] = list([*x,*y,*z])

                if 'x5' in row and not np.isnan(row['x5']):
                    x,y,z = [[int(row[ch+'5']), int(row[ch+'6'])] for ch in ['x','y','z']]
                    if accnum_df.loc[accnum, "downsample"] != 1:
                        x /= accnum_df.loc[accnum, "downsample"]
                        y /= accnum_df.loc[accnum, "downsample"]
                    lesion_df.loc[l_id, A["equ cols"]] = list([*x,*y,*z])

            print(".", end="")
            if cnt % 5 == 2:
                lesion_df.to_csv(A["lesion df path"])
        lesion_df.to_csv(A["lesion df path"])
    return fxn

###########################
### Build/retrieve dataframes
###########################

def get_lesion_df(A):
    def fxn():
        if not A["lesion df"]:
            if exists(A["lesion df path"]):
                A("load df")("lesion df path")
                df = A["active df"]
                df["accnum"] = df["accnum"].astype(str)
                df[A["art cols"]] = df[A["art cols"]].astype(int)
                df = df[df["run_num"] <= A["stage"]]
            else:
                df = pd.DataFrame(columns=["accnum", "class", "stage"]+A["voi cols"])

            A["lesion df"] = df

        return A["lesion df"]
    return fxn

def get_accnum_df(A):
    def fxn():
        if not A["accnum df"]:
            if exists(A["accnum df path"]):
                A("load df")("accnum df path")
                df = A["active df"]
                df.index = df.index.map(str)
            else:
                df = pd.DataFrame(columns=A["accnum cols"])

            A["accnum df"] = df

        return A["accnum df"]
    return fxn

def get_raw_input_df(A):
    def fxn(cls=None):
        if A['sheetnames'] is not None:
            if cls is not None:
                df = pd.read_excel(A["coord xls path"], A["sheetnames"][A["lesion classes"].index(cls)])
            else:
                df = pd.concat([pd.read_excel(A["coord xls path"], A["sheetnames"][A["lesion classes"].index(cls)], sort=False) for cls in A["lesion classes"]])
        else:
            df = pd.read_excel(A["coord xls path"], A["sheetname"])
            if cls is not None:
                df = df[df["cls"] == cls]
        df = df[df['Run'] <= A["stage"]].dropna(subset=["x1"])
        df['acc #'] = df['acc #'].astype(str)

        return df.drop(set(df.columns).difference(['acc #', 'Run', 'Flipped',
              'x1', 'x2', 'y1', 'y2', 'z1', 'z2',
              'x3', 'x4', 'y3', 'y4', 'z3', 'z4',
              'x5', 'x6', 'y5', 'y6', 'z5', 'z6']), axis=1)
    return fxn


###########################
### QC methods
###########################

def open_dcm_folder(cls, accnum):
    raise ValueError("No DICOMs")
    if A['dcm_dirs'] is not None:
        root = A["dcm dirs"][A["lesion classes"].index(cls)]
    else:
        root = A["dcm dir"]
    os.startfile(join(root,accnum))

def check_accnum_df(cls=None):
    """Checks to see if accnum_df is missing any accession numbers."""
    df = get_coords_df(cls)
    accnums = set(df['acc #'].tolist())
    accnum_df = get_accnum_df()
    missing = accnums.difference(accnum_df.index)
    if len(missing) > 0:
        print(cls, missing)

def missing_dcms(cls=None):
    """Checks to see if any image phases are missing from the DICOM directories"""
    raise ValueError("No DICOMs")
    df = get_coords_df(cls)
    accnums = list(set(df['acc #'].tolist()))

    for cnt, accnum in enumerate(accnums):
        df_subset = df.loc[df['acc #'].astype(str) == accnum]
        if A['dcm_dirs'] is not None:
            subfolder = join(A["dcm dirs"][i], accnum)
        else:
            subfolder = join(A["dcm dir"], accnum)

        #if not exists(subfolder + "\\T1_multiphase"):
        for ph in A["phase dirs"]:
            if not exists(join(subfolder, ph)) and not exists(join("Z:\\LIRADS\\DICOMs\\hcc", accnum, ph)):
                print(subfolder, "is missing", ph)
                break
