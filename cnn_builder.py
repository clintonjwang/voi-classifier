import keras.backend as K
from keras.layers import Input, Dense, Concatenate, Flatten, Dropout, Conv3D, MaxPooling3D, Conv2D, MaxPooling2D, ZeroPadding3D, Activation
from keras.layers.normalization import BatchNormalization
from keras.models import Model
from keras.regularizers import l1, l2
from keras.optimizers import Adam
from keras.constraints import max_norm
from keras.layers.noise import GaussianNoise
from keras.callbacks import EarlyStopping
from keras.utils import np_utils
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score

import cnn_methods as cfunc
import copy
import config
import helper_fxns as hf
import numpy as np
import operator
import os
import pandas as pd
import random

def build_cnn(C, optimizer='adam', inputs=4, batch_norm=True):
    nb_classes = len(C.classes_to_include)

    if batch_norm:
        art_img = Input(shape=(C.dims[0], C.dims[1], C.dims[2], 1))
        art_x = art_img
        art_x = Conv3D(filters=64, kernel_size=(3,3,2), padding='same')(art_x)
        art_x = BatchNormalization()(art_x)
        art_x = Activation('relu')(art_x)
        art_x = MaxPooling3D((2, 2, 2))(art_x)

        ven_img = Input(shape=(C.dims[0], C.dims[1], C.dims[2], 1))
        ven_x = ven_img
        ven_x = Conv3D(filters=64, kernel_size=(3,3,2), padding='same')(ven_x)
        ven_x = BatchNormalization()(ven_x)
        ven_x = Activation('relu')(ven_x)
        ven_x = MaxPooling3D((2, 2, 2))(ven_x)

        eq_img = Input(shape=(C.dims[0], C.dims[1], C.dims[2], 1))
        eq_x = eq_img
        eq_x = Conv3D(filters=64, kernel_size=(3,3,2), padding='same')(eq_x)
        eq_x = BatchNormalization()(eq_x)
        eq_x = Activation('relu')(eq_x)
        eq_x = MaxPooling3D((2, 2, 2))(eq_x)

        intermed = Concatenate(axis=4)([art_x, ven_x, eq_x])
        x = Conv3D(filters=128, kernel_size=(3,3,2), padding='same')(intermed)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Conv3D(filters=128, kernel_size=(3,3,2), padding='same')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Conv3D(filters=100, kernel_size=(3,3,2))(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = MaxPooling3D((2, 2, 1))(x)
        x = Flatten()(x)

        img_traits = Input(shape=(2,)) #bounding volume and aspect ratio of lesion

        intermed = Concatenate(axis=1)([x, img_traits])
        x = Dense(100)(intermed)#, kernel_initializer='normal', kernel_regularizer=l1(.01), kernel_constraint=max_norm(3.))(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Dropout(0.2)(x)
        x = Dense(nb_classes)(x)
        x = BatchNormalization()(x)
        pred_class = Activation('softmax')(x)

        model = Model([art_img, ven_img, eq_img, img_traits], pred_class)

    elif inputs == 2:
        voi_img = Input(shape=(C.dims[0], C.dims[1], C.dims[2], C.nb_channels))
        x = voi_img
        #x = GaussianNoise(1)(x)
        #x = ZeroPadding3D(padding=(3,3,2))(voi_img)
        x = Conv3D(filters=128, kernel_size=(3,3,2), activation='relu')(x)
        x = Dropout(0.5)(x)
        x = Conv3D(filters=128, kernel_size=(3,3,2), activation='relu')(x)
        x = MaxPooling3D((2, 2, 2))(x)
        x = Dropout(0.5)(x)
        #x = Conv3D(filters=64, kernel_size=(3,3,2), strides=(2, 2, 2), activation='relu', kernel_constraint=max_norm(4.))(x)
        #x = Dropout(0.5)(x)
        x = Conv3D(filters=64, kernel_size=(3,3,2), activation='relu')(x)
        x = MaxPooling3D((2, 2, 1))(x)
        x = Dropout(0.5)(x)
        x = Flatten()(x)

        img_traits = Input(shape=(2,)) #bounding volume and aspect ratio of lesion

        intermed = Concatenate(axis=1)([x, img_traits])
        x = Dense(64, activation='relu')(intermed)#, kernel_initializer='normal', kernel_regularizer=l1(.01), kernel_constraint=max_norm(3.))(x)
        x = Dropout(0.5)(x)
        pred_class = Dense(nb_classes, activation='softmax')(x)#Dense(nb_classes, activation='softmax')(x)

        model = Model([voi_img, img_traits], pred_class)

    elif inputs == 4:
        art_img = Input(shape=(C.dims[0], C.dims[1], C.dims[2], 1))
        art_x = art_img
        art_x = Conv3D(filters=64, kernel_size=(3,3,2), activation='relu')(art_x)
        art_x = MaxPooling3D((2, 2, 2))(art_x)

        ven_img = Input(shape=(C.dims[0], C.dims[1], C.dims[2], 1))
        ven_x = ven_img
        ven_x = Conv3D(filters=64, kernel_size=(3,3,2), activation='relu')(ven_x)
        ven_x = MaxPooling3D((2, 2, 2))(ven_x)

        eq_img = Input(shape=(C.dims[0], C.dims[1], C.dims[2], 1))
        eq_x = eq_img
        eq_x = Conv3D(filters=64, kernel_size=(3,3,2), activation='relu')(eq_x)
        eq_x = MaxPooling3D((2, 2, 2))(eq_x)

        intermed = Concatenate(axis=4)([art_x, ven_x, eq_x])
        x = Conv3D(filters=128, kernel_size=(3,3,2), activation='relu')(intermed)
        x = Conv3D(filters=100, kernel_size=(3,3,2), activation='relu')(intermed)
        x = MaxPooling3D((2, 2, 1))(x)
        x = Dropout(0.5)(x)
        x = Flatten()(x)

        img_traits = Input(shape=(2,)) #bounding volume and aspect ratio of lesion

        intermed = Concatenate(axis=1)([x, img_traits])
        x = Dense(100, activation='relu')(intermed)#, kernel_initializer='normal', kernel_regularizer=l1(.01), kernel_constraint=max_norm(3.))(x)
        x = Dropout(0.5)(x)
        pred_class = Dense(nb_classes, activation='softmax')(x)#Dense(nb_classes, activation='softmax')(x)

        model = Model([art_img, ven_img, eq_img, img_traits], pred_class)
    
    #optim = Adam(lr=0.01)#5, decay=0.001)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

    return model

def build_2d_cnn(C, optimizer='adam', inputs=4, batch_norm=True):
    nb_classes = len(C.classes_to_include)

    if batch_norm:
        art_img = Input(shape=(C.dims[0], C.dims[1], 1))
        art_x = art_img
        art_x = Conv2D(filters=64, kernel_size=(3,3))(art_x)
        art_x = BatchNormalization()(art_x)
        art_x = Activation('relu')(art_x)
        art_x = MaxPooling2D((2, 2))(art_x)

        ven_img = Input(shape=(C.dims[0], C.dims[1], 1))
        ven_x = ven_img
        ven_x = Conv2D(filters=64, kernel_size=(3,3))(ven_x)
        ven_x = BatchNormalization()(ven_x)
        ven_x = Activation('relu')(ven_x)
        ven_x = MaxPooling2D((2, 2))(ven_x)

        eq_img = Input(shape=(C.dims[0], C.dims[1],  1))
        eq_x = eq_img
        eq_x = Conv2D(filters=64, kernel_size=(3,3))(eq_x)
        eq_x = BatchNormalization()(eq_x)
        eq_x = Activation('relu')(eq_x)
        eq_x = MaxPooling2D((2, 2))(eq_x)

        intermed = Concatenate(axis=3)([art_x, ven_x, eq_x])
        x = Conv3D(filters=128, kernel_size=(3,3), padding='same')(intermed)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Conv2D(filters=128, kernel_size=(3,3))(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = MaxPooling2D((2, 2))(x)
        x = Flatten()(x)

        img_traits = Input(shape=(2,)) #bounding volume and aspect ratio of lesion

        intermed = Concatenate(axis=1)([x, img_traits])
        x = Dense(128)(intermed)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Dropout(0.2)(x)
        x = Dense(nb_classes)(x)
        x = BatchNormalization()(x)
        pred_class = Activation('softmax')(x)

        model = Model([art_img, ven_img, eq_img, img_traits], pred_class)

    elif inputs == 2:
        voi_img = Input(shape=(C.dims[0], C.dims[1], C.nb_channels))
        x = voi_img
        x = Conv2D(filters=128, kernel_size=(3,3), activation='relu')(x)
        x = Dropout(0.5)(x)
        x = Conv2D(filters=128, kernel_size=(3,3), activation='relu')(x)
        x = MaxPooling2D((2, 2))(x)
        x = Dropout(0.5)(x)
        x = Conv2D(filters=64, kernel_size=(3,3), activation='relu')(x)
        x = MaxPooling2D((2, 2))(x)
        x = Dropout(0.5)(x)
        x = Flatten()(x)

        img_traits = Input(shape=(2,)) #bounding volume and aspect ratio of lesion

        intermed = Concatenate(axis=1)([x, img_traits])
        x = Dense(64, activation='relu')(intermed)#, kernel_initializer='normal', kernel_regularizer=l1(.01), kernel_constraint=max_norm(3.))(x)
        x = Dropout(0.5)(x)
        pred_class = Dense(nb_classes, activation='softmax')(x)#Dense(nb_classes, activation='softmax')(x)

        model = Model([voi_img, img_traits], pred_class)

    elif inputs == 4:
        art_img = Input(shape=(C.dims[0], C.dims[1], 1))
        art_x = art_img
        art_x = Conv2D(filters=64, kernel_size=(3,3), activation='relu')(art_x)
        art_x = MaxPooling2D((2, 2))(art_x)

        ven_img = Input(shape=(C.dims[0], C.dims[1], 1))
        ven_x = ven_img
        ven_x = Conv2D(filters=64, kernel_size=(3,3), activation='relu')(ven_x)
        ven_x = MaxPooling2D((2, 2))(ven_x)

        eq_img = Input(shape=(C.dims[0], C.dims[1],  1))
        eq_x = eq_img
        eq_x = Conv2D(filters=64, kernel_size=(3,3), activation='relu')(eq_x)
        eq_x = MaxPooling2D((2, 2))(eq_x)

        intermed = Concatenate(axis=3)([art_x, ven_x, eq_x])
        x = Conv2D(filters=128, kernel_size=(3,3), activation='relu')(intermed)
        x = MaxPooling2D((2, 2))(x)
        x = Flatten()(x)

        img_traits = Input(shape=(2,)) #bounding volume and aspect ratio of lesion

        intermed = Concatenate(axis=1)([x, img_traits])
        x = Dense(128, activation='relu')(intermed)#, kernel_initializer='normal', kernel_regularizer=l1(.01), kernel_constraint=max_norm(3.))(x)
        x = Dropout(0.5)(x)
        pred_class = Dense(nb_classes, activation='softmax')(x)#Dense(nb_classes, activation='softmax')(x)

        model = Model([art_img, ven_img, eq_img, img_traits], pred_class)
    
    #optim = Adam(lr=0.01)#5, decay=0.001)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

    return model

def build_pretrain_model(trained_model, C):
    """CNN with pretrained weights"""

    nb_classes = len(C.classes_to_include)

    voi_img = Input(shape=(C.dims[0], C.dims[1], C.dims[2], C.nb_channels))
    x = voi_img
    x = Conv3D(filters=128, kernel_size=(3,3,2), activation='relu', trainable=False)(x)
    x = Dropout(0.5)(x)
    x = Conv3D(filters=128, kernel_size=(3,3,2), activation='relu', trainable=False)(x)
    x = MaxPooling3D((2, 2, 2))(x)
    x = Dropout(0.5)(x)
    x = Conv3D(filters=64, kernel_size=(3,3,2), activation='relu', trainable=False)(x)
    x = MaxPooling3D((2, 2, 1))(x)
    x = Dropout(0.5)(x)
    x = Flatten()(x)

    img_traits = Input(shape=(2,)) #bounding volume and aspect ratio of lesion

    intermed = Concatenate(axis=1)([x, img_traits])
    x = Dense(64, activation='relu')(intermed)#, kernel_initializer='normal', kernel_regularizer=l1(.01), kernel_constraint=max_norm(3.))(x)
    x = Dropout(0.5)(x)
    pred_class = Dense(nb_classes, activation='softmax')(x)


    model_pretrain = Model([voi_img, img_traits], pred_class)
    model_pretrain.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    for l in range(1,5):
        if type(model_pretrain.layers[l]) == Conv3D:
            model_pretrain.layers[l].set_weights(trained_model.layers[l].get_weights())

    return model_pretrain

def run_cnn(model, C, n=4, n_art=4, steps_per_epoch=25, epochs=50, run_2d=True, verbose=False):
    """Run CNN"""

    nb_classes = len(C.classes_to_include)
    voi_df = pd.read_csv(C.art_voi_path)
    intensity_df = pd.read_csv(C.int_df_path)
    #intensity_df.loc[intensity_df["art_int"] == 0, "art_int"] = np.mean(intensity_df[intensity_df["art_int"] > 0]["art_int"])
    #intensity_df.loc[intensity_df["ven_int"] == 0, "ven_int"] = np.mean(intensity_df[intensity_df["ven_int"] > 0]["ven_int"])
    #intensity_df.loc[intensity_df["eq_int"] == 0, "eq_int"] = np.mean(intensity_df[intensity_df["eq_int"] > 0]["eq_int"])

    orig_data_dict, num_samples = cfunc.collect_unaug_data(C, voi_df, intensity_df)

    avg_X2 = {}
    train_ids = {} #filenames of training set originals
    test_ids = {} #filenames of test set
    X_test = []
    X2_test = []
    Y_test = []
    Z_test = []
    X_train_orig = []
    X2_train_orig = []
    Y_train_orig = []
    Z_train_orig = []

    train_samples = {}

    for cls in orig_data_dict:
        cls_num = C.classes_to_include.index(cls)
        avg_X2[cls] = np.mean(orig_data_dict[cls][1], axis=0)
        
        train_samples[cls] = round(num_samples[cls]*C.train_frac)
        
        order = np.random.permutation(list(range(num_samples[cls])))
        train_ids[cls] = list(orig_data_dict[cls][2][order[:train_samples[cls]]])
        test_ids[cls] = list(orig_data_dict[cls][2][order[train_samples[cls]:]])
        
        X_test = X_test + list(orig_data_dict[cls][0][order[train_samples[cls]:]])
        X2_test = X2_test + list(orig_data_dict[cls][1][order[train_samples[cls]:]])
        Y_test = Y_test + [[0] * cls_num + [1] + [0] * (nb_classes - cls_num - 1)] * \
                            (num_samples[cls] - train_samples[cls])
        Z_test = Z_test + test_ids[cls]
        
        X_train_orig = X_train_orig + list(orig_data_dict[cls][0][order[:train_samples[cls]]])
        X2_train_orig = X2_train_orig + list(orig_data_dict[cls][1][order[:train_samples[cls]]])
        Y_train_orig = Y_train_orig + [[0] * cls_num + [1] + [0] * (nb_classes - cls_num - 1)] * \
                            (train_samples[cls])
        Z_train_orig = Z_train_orig + train_ids[cls]
        
        if verbose:
            print("%s has %d samples for training (%d after augmentation) and %d for testing" %
                  (cls, train_samples[cls], train_samples[cls] * C.aug_factor, num_samples[cls] - train_samples[cls]))
        
    #Y_test = np_utils.to_categorical(Y_test, nb_classes)
    #Y_train_orig = np_utils.to_categorical(Y_train_orig, nb_classes)
    X_test = [np.array(X_test), np.array(X2_test)]
    X_train_orig = [np.array(X_train_orig), np.array(X2_train_orig)]

    Y_test = np.array(Y_test)
    Y_train_orig = np.array(Y_train_orig)

    Z_test = np.array(Z_test)
    Z_train_orig = np.array(Z_train_orig)

    if run_2d:
        X_test = cfunc.separate_phases_2d(X_test)
        X_train_orig = cfunc.separate_phases_2d(X_train_orig)

        train_generator = train_generator_func_2d(C, train_ids, intensity_df, voi_df, avg_X2, n=n, n_art=n_art)
    else:
        X_test = cfunc.separate_phases(X_test)
        X_train_orig = cfunc.separate_phases(X_train_orig)

        train_generator = train_generator_func(C, train_ids, intensity_df, voi_df, avg_X2, n=n, n_art=n_art)

    #early_stopping = EarlyStopping(monitor='acc', min_delta=0.01, patience=4)
    model.fit_generator(train_generator, steps_per_epoch=steps_per_epoch, epochs=epochs, verbose=verbose)#, callbacks=[early_stopping])

    return model, X_test, Y_test

def overnight_run(C):
    running_stats = pd.DataFrame(columns = ["n", "n_art", "steps_per_epoch", "epochs", "2d", "batchnorm", "num_cnn_inputs", "Acc6cls", "Acc3cls", "time_elapsed"])
    running_acc_6 = []
    running_acc_3 = []
    index = 0
    n=4
    n_art=4
    steps_per_epoch=25
    epochs=50
    run_2d=False

    while True:
        model = build_cnn(C, 'adam')
        model, X_test, Y_test = run_cnn(model, C, n=4, n_art=4, steps_per_epoch=25, epochs=50, run_2d=run_2d)
        Y_pred = model.predict(X_test)
        y_true = np.array([max(enumerate(x), key=operator.itemgetter(1))[0] for x in Y_test])
        y_pred = np.array([max(enumerate(x), key=operator.itemgetter(1))[0] for x in Y_pred])
        running_acc_6.append(accuracy_score(y_true, y_pred))
        print("6cls accuracy:", running_acc_6[-1], " - average:", np.mean(running_acc_6))
        y_true_simp, y_pred_simp, _ = cfunc.condense_cm(y_true, y_pred, C.classes_to_include)
        running_acc_3.append(accuracy_score(y_true_simp, y_pred_simp))
        print("3cls accuracy:", running_acc_3[-1], " - average:", np.mean(running_acc_3))

        running_stats.iloc[index] = [n, n_art, steps_per_epoch, epochs, run_2d, running_acc_6[-1], running_acc_3[-1]]
        running_stats.to_csv("overnight_run.csv", index=False)
        index += 1


def train_generator_func(C, train_ids, intensity_df, voi_df, avg_X2, n=12, n_art=0):
    """n is the number of samples from each class, n_art is the number of artificial samples"""
    classes_to_include = C.classes_to_include
    
    num_classes = len(classes_to_include)
    while True:
        x1 = np.empty(((n+n_art)*num_classes, C.dims[0], C.dims[1], C.dims[2], C.nb_channels))
        x2 = np.empty(((n+n_art)*num_classes, 2))
        y = np.zeros(((n+n_art)*num_classes, num_classes))

        train_cnt = 0
        for cls in classes_to_include:
            if n_art>0:
                img_fns = os.listdir(C.artif_dir+cls)
                for _ in range(n_art):
                    img_fn = random.choice(img_fns)
                    x1[train_cnt] = np.load(C.artif_dir + cls + "\\" + img_fn)
                    x2[train_cnt] = avg_X2[cls]
                    y[train_cnt][C.classes_to_include.index(cls)] = 1

                    train_cnt += 1

            img_fns = os.listdir(C.aug_dir+cls)
            while n>0:
                img_fn = random.choice(img_fns)
                if img_fn[:img_fn.rfind('_')] + ".npy" in train_ids[cls]:
                    x1[train_cnt] = np.load(C.aug_dir+cls+"\\"+img_fn)
                    #x1[train_cnt] = cfunc.rescale_int(x1[train_cnt],
                    #                      intensity_df[intensity_df["AccNum"] == img_fn[:img_fn.find('_')]])

                    row = voi_df[(voi_df["Filename"] == img_fn[:img_fn.find('_')] + ".npy") &
                                 (voi_df["lesion_num"] == int(img_fn[img_fn.find('_')+1:img_fn.rfind('_')]))]
                    x2[train_cnt] = [(float(row["real_dx"]) * float(row["real_dy"]) * float(row["real_dz"])) ** (1/3) / 50,
                                        max(float(row["real_dx"]), float(row["real_dy"])) / float(row["real_dz"])]
                    
                    y[train_cnt][C.classes_to_include.index(cls)] = 1
                    
                    train_cnt += 1
                    if train_cnt % (n+n_art) == 0:
                        break
            
        yield cfunc.separate_phases([np.array(x1), np.array(x2)]), np.array(y) #[np.array(x1), np.array(x2)], np.array(y) #


def train_generator_func_2d(C, train_ids, intensity_df, voi_df, avg_X2, n=12, n_art=0):
    """n is the number of samples from each class, n_art is the number of artificial samples"""

    classes_to_include = C.classes_to_include
    
    num_classes = len(classes_to_include)
    while True:
        x1 = np.empty(((n+n_art)*num_classes, C.dims[0], C.dims[1], C.nb_channels))
        x2 = np.empty(((n+n_art)*num_classes, 2))
        y = np.zeros(((n+n_art)*num_classes, num_classes))

        train_cnt = 0
        for cls in classes_to_include:
            if n_art>0:
                img_fns = os.listdir(C.artif_dir+cls)
                for _ in range(n_art):
                    img_fn = random.choice(img_fns)
                    temp = np.load(C.artif_dir + cls + "\\" + img_fn)
                    x1[train_cnt] = temp[:,:,temp.shape[2]//2,:]
                    x2[train_cnt] = avg_X2[cls]
                    y[train_cnt][C.classes_to_include.index(cls)] = 1

                    train_cnt += 1

            img_fns = os.listdir(C.aug_dir+cls)
            while n>0:
                img_fn = random.choice(img_fns)
                if img_fn[:img_fn.rfind('_')] + ".npy" in train_ids[cls]:
                    temp = np.load(C.aug_dir+cls+"\\"+img_fn)
                    x1[train_cnt] = temp[:,:,temp.shape[2]//2,:]
                    #x1[train_cnt] = rescale_int_2d(x1[train_cnt],
                    #                      intensity_df[intensity_df["AccNum"] == img_fn[:img_fn.find('_')]])

                    row = voi_df[(voi_df["Filename"] == img_fn[:img_fn.find('_')] + ".npy") &
                                 (voi_df["lesion_num"] == int(img_fn[img_fn.find('_')+1:img_fn.rfind('_')]))]
                    x2[train_cnt] = [(float(row["real_dx"]) * float(row["real_dy"]) * float(row["real_dz"])) ** (1/3) / 50,
                                        max(float(row["real_dx"]), float(row["real_dy"])) / float(row["real_dz"])]
                    
                    y[train_cnt][C.classes_to_include.index(cls)] = 1
                    
                    train_cnt += 1
                    if train_cnt % (n+n_art) == 0:
                        break
            
        
        yield cfunc.separate_phases_2d([np.array(x1), np.array(x2)]), np.array(y) #[np.array(x1), np.array(x2)], np.array(y) #