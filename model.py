import copy
import glob
import os
from os.path import *
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

from am.dispositio.action import Action

def get_actions(A):
    acts = [
        (["collect all images"], Action(collect_unaug_data(A))),
        (["split training and test data", "split data"], Action(split_training_test(A))),
        (["set training data generator"], Action(set_train_data_gen(A))),
        (["build CNN"], Action(build_cnn(A))),
        (["run CNN"], Action(run_fixed_hyperparams(A)))
    ]

    return acts

####################################
### Data Setup
####################################

def collect_unaug_data(A):
    def fxn():
        """Return dictionary pointing to X (img data) and Z (filenames) and dictionary storing number of samples of each class."""
        A["images by class"] = {}
        A["datasize by class"] = {}
        lesion_df = A("get lesion df")()

        for cls in A["lesion classes"]:
            x = np.empty((10000, *A["input shape"])) #A("instantiate collection")(cls="image")
            z = []

            lesion_ids = lesion_df[lesion_df["cls"] == cls].index

            A["datasize by class"][cls] = len(lesion_ids)

            for ix, lesion_id in enumerate(lesion_ids):
                img_path = join(A["unaug_dir"], lesion_id+".npy")
                x[ix] = np.load(img_path)
                if A["post_scale"] > 0:
                    x[ix] = A("normalize intensity")(x[ix], 1., -1., A["post_scale"])
                z.append(lesion_id)

            x.resize((A["datasize by class"][cls], *A["input shape"])) #shrink first dimension to fit
            A["images by class"][cls] = [x, np.array(z)]
    return fxn

def split_training_test(A):
    def fxn():
        A("remove state keys")("test images", "test labels", "test accnums",
            "training images", "training labels", "training accnums")

        if A["fix test accnums"]:
            orders = {cls: np.where(np.isin(A["images by class"][cls][1], A["fixed test accnums"])) for cls in A["lesion classes"]}
            for cls in A["lesion classes"]:
                orders[cls] = list(set(range(A["datasize by class"][cls])).difference(list(orders[cls][0]))) + list(orders[cls][0])

        for cls_num, cls in enumerate(A["lesion classes"]):
            n_train = A["datasize by class"][cls] - A["test num"]
            n_test = A["test num"]

            if test_accnums is None:
                order = np.random.permutation(list(range(A["datasize by class"][cls])))
                test_accnums = list(A["images by class"][cls][-1][order[n_train:]])
            else:
                order = orders[cls]
            train_accnums = list(A["images by class"][cls][-1][order[:n_train]])

            A("append to state")("training images", A["images by class"][cls][order[:n_train]])
            A("append to state")("training labels", [cls_num] * n_train)
            A("append to state")("training accnums", train_accnums)
            A("append to state")("test images", A["images by class"][cls][order[n_train:]])
            A("append to state")("test labels", [cls_num] * n_test)
            A("append to state")("test accnums", test_accnums)

            if verbose:
                print("%s has %d samples for training (%d after augmentation) and %d for testing" %
                      (cls, n_train, n_train * A["aug_factor"], A["datasize by class"][cls] - n_train))


        A["test labels"] = np_utils.to_categorical(A["test labels"], A["number of classes"])
        A["training labels"] = np_utils.to_categorical(A["training labels"], A["number of classes"])
        A["test images"] = np.array(A["test images"])
        A["training images"] = np.array(A["training images"])

        A["test labels"] = np.array(A["test labels"])
        A["training labels"] = np.array(A["training labels"])

        A["test accnums"] = np.array(A["test accnums"])
        A["training accnums"] = np.array(A["training accnums"])

    return fxn

def set_train_data_gen(A):
    def fxn():
        A["training data generator"] = _train_gen_classifier(A)
    return fxn

def _train_gen_classifier(A):
    """n is the number of samples from each class"""
    if accnums is None:
        lesion_df = drm.get_lesion_df()
        accnums = {}
        for cls in A["lesion classes"]:
            accnums[cls] = lesion_df.loc[(lesion_df["cls"]==cls) & \
                (lesion_df["run_num"] <= A["run_num"]), "accnum"].values

    if A["clinical_inputs"] > 0:
        train_path = join(A["base_dir"], "excel", "clinical_data_test.xlsx")
        clinical_df = pd.read_excel(train_path, index_col=0)
        clinical_df.index = clinical_df.index.astype(str)

    if type(accnums) == dict:
        img_fns = {cls:[fn for fn in os.listdir(A["aug_dir"]) if fn[:fn.find('_')] in accnums[cls]] for cls in A["lesion classes"]}

    x = np.empty((n*A["number of classes"], *A["input shape"]))
    y = np.zeros((n*A["number of classes"], A["number of classes"]))

    while True:
        train_cnt = 0
        for cls in A["lesion classes"]:
            while n > 0:
                img_fn = random.choice(img_fns[cls])
                lesion_id = img_fn[:img_fn.rfind('_')]
                if lesion_id in test_ids[cls]:
                    continue

                try:
                    x[train_cnt] = np.load(join(A["aug_dir"], img_fn))
                except:
                    print(join(A["aug_dir"], img_fn))
                if A["post_scale"] > 0:
                    try:
                        x[train_cnt] = tr.normalize_intensity(x[train_cnt], 1., -1., A["post_scale"])
                    except:
                        raise ValueError(lesion_id)
                        #vm.reset_accnum(lesion_id[:lesion_id.find('_')])

                y[train_cnt][A["lesion classes"].index(cls)] = 1

                train_cnt += 1
                if train_cnt % n == 0:
                    break

        yield np.array(x), np.array(y)

def _separate_phases(A):
    def fxn(X, clinical_inputs=False):
        """Assumes X[0] contains imaging and X[1] contains dimension data.
        Reformats such that X[0:2] has 3 phases and X[3] contains dimension data.
        Image data still is 5D (nb_samples, 3D, 1 channel).
        Handles both 2D and 3D images"""

        if clinical_inputs:
            dim_data = copy.deepcopy(X[1])
            img_data = X[0]

            axis = len(X[0].shape) - 1
            X[1] = np.expand_dims(X[0][...,1], axis=axis)
            X += [np.expand_dims(X[0][...,2], axis=axis)]
            X += [dim_data]
            X[0] = np.expand_dims(X[0][...,0], axis=axis)

        else:
            X = np.array(X)
            axis = len(X[0].shape) - 1
            X = [np.expand_dims(X[...,ix], axis=axis) for ix in range(3)]

        return X
    return fxn

####################################
### CNN
####################################

def build_cnn(A):
    def fxn():
        A("set input shape")(A["nb channels"], *A["dims"])

        A("make block from layers")("192conv_1", ["conv3d", "batch norm", "relu", "max pool"],
                                    **{"out channels": 192, "kernel size": (3, 3, 2)})
        A("make block from layers")("128conv_1", ["conv3d", "batch norm", "relu"],
                                    **{"out channels": 128, "kernel size": (3, 3, 2)})
        A("make block from layers")("pool_1", ["max pool"], **{"stride": (2, 2, 1)})
        A("make block from layers")("fc_1", ["fc", "relu"], **{"number of units": 128})
        A("make block from layers")("cls_1", ["fc", "softmax"], **{"number of units": A["number of classes"]})

        A("build model")(model_name="model1", blocks=["192conv_1", "3x 128conv_1", "pool_1", "fc_1", "cls_1"])

        A("set training settings")()
    return fxn

def run_fixed_hyperparams(A):
    def fxn(overwrite=False, max_runs=999, Z_test=None, model_name='models_', verbose=True):
        """Runs the CNN for max_runs times, saving performance metrics."""
        if overwrite and exists(A["run stats path"]):
            os.remove(A["run stats path"])

        running_acc_6 = []

        for _ in range(max_runs):

            X_test, Y_test, train_gen, num_samples, train_orig, Z = cbuild.get_cnn_data(n=A["T"].n,
                    Z_test_fixed=Z_test)
            A["Z_test"], A["Z_train_orig"] = Z
            X_train_orig, Y_train_orig = train_orig


            if A["aleatoric"]:
                A["pred_model"], A["train_model"] = cbuild.build_cnn_hyperparams(A["T"])
                val_data = [X_test, Y_test], None
            else:
                A["pred_model"] = cbuild.build_cnn_hyperparams(A["T"])
                A["train_model"] = A["pred_model"]
                val_data = [X_test, Y_test]

            t = time.time()

            if A["T"].steps_per_epoch > 32:
                hist = A["train_model"].fit_generator(train_gen, A["T"].steps_per_epoch,
                        A["T"].epochs, verbose=verbose, callbacks=[A["T"].early_stopping], validation_data=val_data)
                loss_hist = hist.history['val_loss']
            else:
                A["pred_model"] = A["train_model"].layers[-2]
                A["pred_model"].compile('adam', loss='categorical_crossentropy', metrics=['accuracy'])
                for _ in range(5):
                    hist = A["train_model"].fit_generator(train_gen, A["T"].steps_per_epoch,
                            A["T"].epochs, verbose=verbose, callbacks=[A["T"].early_stopping])#, validation_data=[X_test, Y_test])
                    print(A["pred_model"].evaluate(X_test, Y_test, verbose=False))
                loss_hist = hist.history['loss']

            Y_pred = A["pred_model"].predict(X_train_orig)
            y_true = np.array([max(enumerate(x), key=operator.itemgetter(1))[0] for x in Y_train_orig])
            y_pred = np.array([max(enumerate(x), key=operator.itemgetter(1))[0] for x in Y_pred])
            miscls_train = list(A["Z_train_orig"][~np.equal(y_pred, y_true)])

            if A["aug pred"]:
                x = np.empty((A["aug factor"], *A["dims"], A["nb channels"]))
                Y_pred = []
                for z in A["Z_test"]:
                    x = np.stack([np.load(fn) for fn in glob.glob(join(A["aug dir"],"*")) if basename(fn).startswith(z)], 0)
                    y = A["pred_model"].predict(x)
                    Y_pred.append(np.median(y, 0))
                Y_pred = np.array(Y_pred)
            elif A["T"].mc_sampling:
                Y_pred = []
                for ix in range(len(A["Z_test"])):
                    x = np.tile(X_test[ix], (256, 1,1,1,1))
                    y = A["pred_model"].predict(x)
                    Y_pred.append(np.median(y, 0))
                Y_pred = np.array(Y_pred)
            else:
                Y_pred = A["pred_model"].predict(X_test)
            y_true = np.array([max(enumerate(x), key=operator.itemgetter(1))[0] for x in Y_test])
            y_pred = np.array([max(enumerate(x), key=operator.itemgetter(1))[0] for x in Y_pred])
            miscls_test = list(A["Z_test"][~np.equal(y_pred, y_true)])

            cm = confusion_matrix(y_true, y_pred)
            #f1 = f1_score(y_true, y_pred, average="weighted")
            running_acc_6.append(accuracy_score(y_true, y_pred))
            print("Accuracy: %d%% (avg: %d%%), time: %ds" % (running_acc_6[-1]*100, np.mean(running_acc_6)*100, time.time()-t))

            """if hasattr(A["C"],'simplify_map'):
                y_true_simp, y_pred_simp = merge_classes(y_true, y_pred, A["C"])
                acc_3 = accuracy_score(y_true_simp, y_pred_simp)
                row = _get_hyperparams_as_list(A["C"], A["T"]) + [num_samples[k] for k in A["cls names"]] + [running_acc_6[-1], acc_3]
            else:"""
            row = _get_hyperparams_as_list(A["C"], A["T"]) + [num_samples[k] for k in A["cls names"]] + [running_acc_6[-1]]

            model_names = glob.glob(join(A["model dir"], model_name+"*"))
            if len(model_names) > 0:
                model_num = max([int(x[x.rfind('_')+1:x.find('.')]) for x in model_names]) + 1
            else:
                model_num = 0
            running_stats = get_run_stats_csv(A["C"])
            running_stats.loc[len(running_stats)] = row + [loss_hist, cm, time.time()-t, time.time(),
                            miscls_test, miscls_train, model_name+str(model_num), y_true, str(Y_pred), list(A["Z_test"])]

            running_stats.to_csv(A["run stats path"], index=False)
            A["pred_model"].save(join(A["model dir"], model_name+'%d.hdf5' % model_num))
            model_num += 1
    return fxn

def _get_hyperparams_as_list(C, T):
    return [A["hyperparameters"]["n"], A["hyperparameters"]["steps_per_epoch"], A["hyperparameters"]["epochs"],
            A["test num"], A["aug factor"], A["clinical inputs"],
            A["hyperparameters"]["kernel_size"], A["hyperparameters"]["f"], A["hyperparameters"]["padding"], A["hyperparameters"]["dropout"], A["hyperparameters"]["dense_units"],
            A["hyperparameters"]["pool_sizes"],
            A["hyperparameters"]["cnn_type"]+A["aleatoric"]*'-al'+A["aug pred"]*'-aug'+A["hyperparameters"]["mc_sampling"]*'-mc'+'-foc%.1f'%A["focal loss"],
            A["ensemble num"],
            A["hyperparameters"]["optimizer"].get_config()['lr']]

def get_run_stats_csv(A):
    def fxn():
        try:
            running_stats = pd.read_csv(A["run stats path"])
            index = len(running_stats)
        except FileNotFoundError:
            running_stats = pd.DataFrame(columns = ["n", "steps_per_epoch", "epochs",
                "test_num", "augment_factor", "clinical_inputs",
                "kernel_size", "conv_filters", "conv_padding",
                "dropout", "dense_units", "pooling", "cnn_type", "ensemble_num", "learning_rate"] + \
                A["cls names"] + ["acc6cls"] + ["acc3cls"]*(hasattr(C,'simplify_map')) + \
                ["loss_hist", 'confusion_matrix', "time_elapsed(s)", 'timestamp',
                'miscls_test', 'miscls_train', 'model_num',
                'y_true', 'y_pred_raw', 'z_test'])

        return running_stats
    return fxn

def merge_classes(A):
    def fxn(y_true, y_pred):
        """From lists y_true and y_pred with class numbers, """
        y_true_simp = np.array([A["simplify map"][A["cls names"][y]] for y in y_true])
        y_pred_simp = np.array([A["simplify map"][A["cls names"][y]] for y in y_pred])

        return y_true_simp, y_pred_simp
    return fxn
