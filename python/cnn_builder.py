import keras.backend as K
from keras.layers import Input, Dense, Concatenate, Flatten, Dropout, Conv3D, MaxPooling3D, Conv2D, MaxPooling2D, ZeroPadding3D, Activation, ELU
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
import time

####################################
### OVERNIGHT PROCESSES
####################################

def run_all():
    """Reruns everything except dimensions. Meant for overnight runs."""
    
    import dr_methods as drm
    import voi_methods as vm
    import artif_gen_methods as agm
    import config
    import cnn_builder as cbuild
    
    C = config.Config()
    drm.load_all_vois(C)
    
    intensity_df = drm.load_ints(C)
    intensity_df.to_csv(C.int_df_path, index=False)
    
    n = 1500
    for cls in C.classes_to_include:
        agm.gen_imgs(cls, C, n)
        if not os.path.exists(C.orig_dir + cls):
            os.makedirs(C.orig_dir + cls)
        if not os.path.exists(C.aug_dir + cls):
            os.makedirs(C.aug_dir + cls)
        if not os.path.exists(C.crops_dir + cls):
            os.makedirs(C.crops_dir + cls)
            
    final_size = C.dims

    voi_df_art = pd.read_csv(C.art_voi_path)
    voi_df_ven = pd.read_csv(C.ven_voi_path)
    voi_df_eq = pd.read_csv(C.eq_voi_path)
    intensity_df = pd.read_csv(C.int_df_path)
    
    small_vois = {}
    small_vois = vm.extract_vois(small_vois, C, voi_df_art, voi_df_ven, voi_df_eq, intensity_df)

    with open(C.small_voi_path, 'w', newline='') as csv_file:
        writer = csv.writer(csv_file)
        for key, value in small_vois.items():
            writer.writerow([key, value])
            
    # scaled imgs
    t = time.time()
    for cls in C.classes_to_include:
        for fn in os.listdir(C.crops_dir + cls):
            img = np.load(C.crops_dir + cls + "\\" + fn)
            unaug_img = vm.resize_img(img, C.dims, small_vois[fn[:-4]])
            np.save(C.orig_dir + cls + "\\" + fn, unaug_img)
    print(time.time()-t)
    
    # augmented imgs
    t = time.time()
    for cls in C.classes_to_include:
        vm.parallel_augment(cls, small_vois, C)
        print(cls, time.time()-t)
        
    for cls in C.classes_to_include:
        vm.save_vois_as_imgs(cls, C)
        
    overnight_run(C)

def overnight_run(C_list, overwrite=False, max_runs=999):
    """Runs the CNN indefinitely, saving performance metrics."""
    if overwrite:
        running_stats = pd.DataFrame(columns = ["n", "n_art", "steps_per_epoch", "epochs",
            "num_phases", "input_res", "training_fraction", "augment_factor", "non_imaging_inputs",
            "kernel_size", "batchnorm", "conv_filters", "conv_padding",
            "dropout", "activation_type", "dilation", "dense_units",
            "acc6cls", "acc3cls", "time_elapsed(s)", "loss_hist",
            'hcc', 'cholangio', 'colorectal', 'cyst', 'hemangioma', 'fnh',
            'confusion_matrix', 'f1', 'timestamp', 'comments', 'run_num'])
        index = 0
    else:
        running_stats = pd.read_csv(C_list[0].run_stats_path)
        index = len(running_stats)


    running_acc_6 = []
    running_acc_3 = []
    n = [4]
    n_art = [0]
    steps_per_epoch = [500]
    epochs = [30]
    run_2d = False
    batch_norm = True
    non_imaging_inputs = False
    f = [[64,128,128]]
    padding = [['same','valid']]
    dropout = [[0.1,0.1]]
    dense_units = [100]
    dilation_rate = [(1, 1, 1)]
    kernel_size = [(3,3,2)]
    activation_type = ['relu']
    merge_layer = [1]
    cycle_len = 1
    early_stopping = EarlyStopping(monitor='loss', min_delta=0.002, patience=3)

    C_index = 0
    while index < max_runs:
        C = C_list[C_index % len(C_list)]

        X_test, Y_test, train_generator, num_samples, _ = get_cnn_data(C, n=n[C_index % len(n)], n_art=n_art[C_index % len(n_art)], run_2d=run_2d)

        for _ in range(cycle_len):
            model = build_cnn(C, 'adam', activation_type=activation_type[index % len(activation_type)],
                    dilation_rate=dilation_rate[index % len(dilation_rate)], f=f[index % len(f)],
                    padding=padding[index % len(padding)], dropout=dropout[index % len(dropout)],
                    dense_units=dense_units[index % len(dense_units)], kernel_size=kernel_size[index % len(kernel_size)],
                    merge_layer=merge_layer[index % len(merge_layer)], non_imaging_inputs=non_imaging_inputs)

            t = time.time()
            hist = model.fit_generator(train_generator, steps_per_epoch=steps_per_epoch[index % len(steps_per_epoch)], epochs=epochs[index % len(epochs)], callbacks=[early_stopping], verbose=False)
            loss_hist = hist.history['loss']

            Y_pred = model.predict(X_test)
            y_true = np.array([max(enumerate(x), key=operator.itemgetter(1))[0] for x in Y_test])
            y_pred = np.array([max(enumerate(x), key=operator.itemgetter(1))[0] for x in Y_pred])

            running_acc_6.append(accuracy_score(y_true, y_pred))
            print("6cls accuracy:", running_acc_6[-1], " - average:", np.mean(running_acc_6))

            y_true_simp, y_pred_simp, _ = cfunc.condense_cm(y_true, y_pred, C.classes_to_include)
            running_acc_3.append(accuracy_score(y_true_simp, y_pred_simp))
            #print("3cls accuracy:", running_acc_3[-1], " - average:", np.mean(running_acc_3))

            running_stats.loc[index] = [n[C_index % len(n)], n_art[C_index % len(n_art)], steps_per_epoch[index % len(steps_per_epoch)], epochs[index % len(epochs)],
                                C.nb_channels, C.dims, C.train_frac, C.aug_factor, non_imaging_inputs,
                                kernel_size[index % len(kernel_size)], batch_norm, f[index % len(f)], padding[index % len(padding)],
                                dropout[index % len(dropout)], activation_type[index % len(activation_type)], dilation_rate[index % len(dilation_rate)], dense_units[index % len(dense_units)],
                                running_acc_6[-1], running_acc_3[-1], time.time()-t, loss_hist,
                                num_samples['hcc'], num_samples['cholangio'], num_samples['colorectal'], num_samples['cyst'], num_samples['hemangioma'], num_samples['fnh'],
                                confusion_matrix(y_true, y_pred), f1_score(y_true, y_pred, average="weighted"), time.time(), str(C.hard_scale), C.run_num]
            running_stats.to_csv(C.run_stats_path, index=False)
            index += 1

        C_index += 1


####################################
### BUILD CNNS
####################################

def build_cnn(C, optimizer='adam', batch_norm=True, dilation_rate=(1, 1, 1), padding=['same', 'valid'],
    dropout=[0.1,0.1], activation_type='relu', f=[64,128,128], dense_units=100, kernel_size=(3,3,2), merge_layer=1,
    non_imaging_inputs=False):
    """Main class for setting up a CNN. Returns the compiled model."""

    if activation_type == 'elu':
        ActivationLayer = ELU
        activation_args = 1
    elif activation_type == 'relu':
        ActivationLayer = Activation
        activation_args = 'relu'

    nb_classes = len(C.classes_to_include)

    if merge_layer == 1:
        art_img = Input(shape=(C.dims[0], C.dims[1], C.dims[2], 1))
        art_x = art_img
        art_x = Conv3D(filters=f[0], kernel_size=kernel_size, dilation_rate=dilation_rate, padding=padding[0])(art_x)
        art_x = BatchNormalization()(art_x)
        art_x = ActivationLayer(activation_args)(art_x)
        art_x = MaxPooling3D((2, 2, 1))(art_x)

        ven_img = Input(shape=(C.dims[0], C.dims[1], C.dims[2], 1))
        ven_x = ven_img
        ven_x = Conv3D(filters=f[0], kernel_size=kernel_size, dilation_rate=dilation_rate, padding=padding[0])(ven_x)
        ven_x = BatchNormalization()(ven_x)
        ven_x = ActivationLayer(activation_args)(ven_x)
        ven_x = MaxPooling3D((2, 2, 1))(ven_x)

        eq_img = Input(shape=(C.dims[0], C.dims[1], C.dims[2], 1))
        eq_x = eq_img
        eq_x = Conv3D(filters=f[0], kernel_size=kernel_size, dilation_rate=dilation_rate, padding=padding[0])(eq_x)
        eq_x = BatchNormalization()(eq_x)
        eq_x = ActivationLayer(activation_args)(eq_x)
        eq_x = MaxPooling3D((2, 2, 1))(eq_x)

        x = Concatenate(axis=4)([art_x, ven_x, eq_x])

        for layer_num in range(1,len(f)):
            x = Conv3D(filters=f[layer_num], kernel_size=kernel_size, padding=padding[1])(x)
            x = BatchNormalization()(x)
            x = ActivationLayer(activation_args)(x)
            x = Dropout(dropout[0])(x)

        x = MaxPooling3D((2, 2, 2))(x)
        x = Flatten()(x)

        img_traits = Input(shape=(2,)) #bounding volume and aspect ratio of lesion

        if non_imaging_inputs:
            x = Concatenate(axis=1)([x, img_traits])
        x = Dense(dense_units)(x)#, kernel_initializer='normal', kernel_regularizer=l2(.01), kernel_constraint=max_norm(3.))(x)
        x = BatchNormalization()(x)
        x = Dropout(dropout[1])(x)
        x = ActivationLayer(activation_args)(x)
        x = Dense(nb_classes)(x)
        x = BatchNormalization()(x)
        pred_class = Activation('softmax')(x)

        model = Model([art_img, ven_img, eq_img, img_traits], pred_class)

    elif merge_layer == 0:
        art_img = Input(shape=(C.dims[0], C.dims[1], C.dims[2], 1))
        art_x = art_img
        ven_img = Input(shape=(C.dims[0], C.dims[1], C.dims[2], 1))
        ven_x = ven_img
        eq_img = Input(shape=(C.dims[0], C.dims[1], C.dims[2], 1))
        eq_x = eq_img
        #voi_img = Input(shape=(C.dims[0], C.dims[1], C.dims[2], C.nb_channels))
        #x = voi_img

        x = Concatenate(axis=4)([art_x, ven_x, eq_x])
        #x = GaussianNoise(1)(x)
        #x = ZeroPadding3D(padding=(3,3,2))(x)
        for layer_num in range(1,len(f)):
            x = Conv3D(filters=f[layer_num], kernel_size=kernel_size, padding=padding[0])(x)
            x = BatchNormalization()(x)
            x = ActivationLayer(activation_args)(x)
            x = Dropout(dropout[0])(x)
        x = MaxPooling3D((2, 2, 1))(x)
        x = Flatten()(x)

        img_traits = Input(shape=(2,)) #bounding volume and aspect ratio of lesion

        intermed = Concatenate(axis=1)([x, img_traits])
        x = Dense(dense_units)(intermed)#, kernel_initializer='normal', kernel_regularizer=l2(.01), kernel_constraint=max_norm(3.))(x)
        x = BatchNormalization()(x)
        x = Dropout(dropout[1])(x)
        x = ActivationLayer(activation_args)(x)
        x = Dense(nb_classes)(x)
        x = BatchNormalization()(x)
        pred_class = Activation('softmax')(x)

        model = Model([art_img, ven_img, eq_img, img_traits], pred_class)
    
    #optim = Adam(lr=0.01)#5, decay=0.001)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

    return model

def build_2d_cnn(C, optimizer='adam', inputs=4, batch_norm=True, dilation_rate=(2, 2)):
    """Sets up a 2D CNN. Returns the compiled model."""

    nb_classes = len(C.classes_to_include)

    if batch_norm:
        art_img = Input(shape=(C.dims[0], C.dims[1], 1))
        art_x = art_img
        art_x = Conv2D(filters=64, kernel_size=(3,3), dilation_rate=dilation_rate)(art_x)
        art_x = BatchNormalization()(art_x)
        art_x = Activation('relu')(art_x)
        art_x = MaxPooling2D((2, 2))(art_x)

        ven_img = Input(shape=(C.dims[0], C.dims[1], 1))
        ven_x = ven_img
        ven_x = Conv2D(filters=64, kernel_size=(3,3), dilation_rate=dilation_rate)(ven_x)
        ven_x = BatchNormalization()(ven_x)
        ven_x = Activation('relu')(ven_x)
        ven_x = MaxPooling2D((2, 2))(ven_x)

        eq_img = Input(shape=(C.dims[0], C.dims[1],  1))
        eq_x = eq_img
        eq_x = Conv2D(filters=64, kernel_size=(3,3), dilation_rate=dilation_rate)(eq_x)
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
    """Sets up CNN with pretrained weights"""

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


####################################
### TRAIN CNNS
####################################

def get_cnn_data(C, n=4, n_art=4, run_2d=False, verbose=False):
    """Subroutine to run CNN"""

    nb_classes = len(C.classes_to_include)
    voi_df = pd.read_csv(C.art_voi_path)
    intensity_df = pd.read_csv(C.int_df_path)
    orig_data_dict, num_samples = cfunc.collect_unaug_data(C, voi_df)

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

        train_generator = train_generator_func_2d(C, train_ids, voi_df, avg_X2, n=n, n_art=n_art)
    else:
        X_test = cfunc.separate_phases(X_test)
        X_train_orig = cfunc.separate_phases(X_train_orig)

        train_generator = train_generator_func(C, train_ids, voi_df, avg_X2, n=n, n_art=n_art)

    return X_test, Y_test, train_generator, num_samples, [X_train_orig, Y_train_orig, Z_train_orig]

def train_generator_func(C, train_ids, voi_df, avg_X2, n=12, n_art=0):
    """n is the number of samples from each class, n_art is the number of artificial samples"""

    import voi_methods as vm
    classes_to_include = C.classes_to_include
    
    num_classes = len(classes_to_include)
    while True:
        x1 = np.empty(((n+n_art)*num_classes, C.dims[0], C.dims[1], C.dims[2], C.nb_channels))
        x2 = np.empty(((n+n_art)*num_classes, 2))
        y = np.zeros(((n+n_art)*num_classes, num_classes))

        train_cnt = 0
        for cls in classes_to_include:
            if n_art > 0:
                img_fns = os.listdir(C.artif_dir+cls)
                for _ in range(n_art):
                    img_fn = random.choice(img_fns)
                    x1[train_cnt] = np.load(C.artif_dir + cls + "\\" + img_fn)
                    x2[train_cnt] = avg_X2[cls]
                    y[train_cnt][C.classes_to_include.index(cls)] = 1

                    train_cnt += 1

            img_fns = os.listdir(C.aug_dir+cls)
            while n > 0:
                img_fn = random.choice(img_fns)
                if img_fn[:img_fn.rfind('_')] + ".npy" in train_ids[cls]:
                    x1[train_cnt] = np.load(C.aug_dir+cls+"\\"+img_fn)
                    if C.hard_scale:
                        x1[train_cnt] = vm.scale_intensity(x1[train_cnt], 1, max_int=2)#, keep_min=True)

                    row = voi_df[(voi_df["Filename"] == img_fn[:img_fn.find('_')] + ".npy") &
                                 (voi_df["lesion_num"] == int(img_fn[img_fn.find('_')+1:img_fn.rfind('_')]))]
                    x2[train_cnt] = [(float(row["real_dx"]) * float(row["real_dy"]) * float(row["real_dz"])) ** (1/3) / 50,
                                        max(float(row["real_dx"]), float(row["real_dy"])) / float(row["real_dz"])]
                    
                    y[train_cnt][C.classes_to_include.index(cls)] = 1
                    
                    train_cnt += 1
                    if train_cnt % (n+n_art) == 0:
                        break
            
        yield cfunc.separate_phases([np.array(x1), np.array(x2)]), np.array(y) #[np.array(x1), np.array(x2)], np.array(y) #

def train_generator_func_2d(C, train_ids, voi_df, avg_X2, n=12, n_art=0):
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