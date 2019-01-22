import functools
import os
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data

from am.world.obj_class import create_obj_cls_f
from am.world.obj_inst import create_obj_inst_f
from am.world.property import create_prop_f
from am.dispositio.action import Action

# files and folders
def get_obj_cls(A):
    create_obj_cls = functools.partial(create_obj_cls_f, A)
    create_noun_inst = functools.partial(create_obj_inst_f, A, cls="noun")

    objs = [
        create_obj_cls(name='LIRADS datapoint',
                       attributes=["image path", "lesion id", "accession number", "image", "one hot label", "true class", "predicted class"],
                       parent="datapoint"),
    ]

    return objs

def get_props(A):
    create_prop = functools.partial(create_prop_f, A)
    def create_props(*args):
        return [create_prop(name=x) for x in args]
    return [
        # dummy properties that are actually objects or collections of objects
        *create_props("image path", "lesion id", "accession number", "image", "one hot label", "true class", "predicted class") #parent='performance measure'
    ]

def get_actions(A):
    acts = [
        (["construct datasets"], Action(construct_datasets(A))),
        (["set balanced training data generator"], Action(set_bal_train_data_gen(A))), # class balancing by separate sampling from each class
        (["build lesion classifier"], Action(build_cnn(A))),
        (["assess lesion classifier"], Action(assess_model(A))),

        # alternative to the balanced training data generator, which implements class balancing via weighted sampling
        (["get dataset weights for class balancing"], Action(get_weights_for_balancing(A))),
        (["dataset to dataloader"], Action(construct_dataloaders(A))),

        (["get augmented image paths"], Action(get_augmented_paths(A))),
        #(["get image for lesion id"], Action(lesion_id_to_datapoint(A))),
        (["get class of lesion id"], Action(get_class_of_lesion_id(A))),
        (["new datapoint"], Action(new_datapoint(A))),
        (["datapoint from lesion id"], Action(new_datapoint_from_id(A))),
        (["datapoint from augmented path"], Action(new_datapoint_from_aug_path(A))),

        (["get_hyperparams_as_list"], Action(get_hyperparams_as_list(A))),
        #(["get image from path"], Action(get_image_from_path(A))),
    ]

    return acts

####################################
### Data Setup
####################################

def get_class_of_lesion_id(A):
    def fxn(lesion_id):
        if A["lesion df"] is None:
            A("get lesion df")()
        return A["lesion df"].loc[lesion_id, "cls"]
    return fxn

def lesion_id_to_datapoint(A):
    def fxn(lesion_id):
        return A("permute image")(A("load image")(os.path.join(A["non-augmented img folder"], lesion_id + ".npy")), (3, 0, 1, 2))
    return fxn

def new_datapoint(A):
    def fxn(lesion_id=None, path=None, load_img=True):
        if path is None:
            path = os.path.join(A["non-augmented img folder"], lesion_id + ".npy")
        if lesion_id is None:
            fn = os.path.basename(path)
            lesion_id = fn[:fn.rfind("_")]

        kwargs = {"lesion id": lesion_id, "image path": path}
        kwargs["accession number"] = lesion_id[:lesion_id.find("_")]
        kwargs["true class"] = A("get class of lesion id")(lesion_id)
        kwargs["one hot label"] = A("1-hot encode")(A["lesion classes"].index(kwargs["true class"]))
        if load_img:
            img = np.load(path)
            kwargs["image"] = A("permute image")(img, (3, 0, 1, 2))

        return A("instantiate")(cls="LIRADS datapoint", **kwargs)
    return fxn

def new_datapoint_from_id(A):
    def fxn(lesion_id):
        return A("new datapoint")(lesion_id=lesion_id, load_img=True)
    return fxn

def new_datapoint_from_aug_path(A):
    def fxn(path):
        return A("new datapoint")(path=path, load_img=False)
    return fxn

def get_augmented_paths(A):
    def fxn():
        paths = []
        for path in A("get folder contents")(A["augmented img folder"]):
            fn = os.path.basename(path)
            lesion_id = fn[:fn.rfind("_")]
            if lesion_id in A["training lesion ids"]:
                paths.append(path)
        return paths
    return fxn

def construct_datasets(A):
    def fxn():
        if A["lesion df"] is None:
            A("get lesion df")()

        A("remove state keys")("datapoints", "test lesion ids", "training lesion ids")

        for cls_num, cls in enumerate(A["lesion classes"]):
            df = A["lesion df"]
            lesion_ids = df[df["cls"] == cls].index

            if A["fix test lesions"]:
                test_ids, train_ids = A("split list conditionally")(lambda x: x in A["fixed test lesion ids"], lesion_ids)
            else:
                test_ids, train_ids = A("random split with N in the first partition")(lesion_ids, A["test num"])

            A("add to state var")("test lesion ids", test_ids)
            A("add to state var")("training lesion ids", train_ids)

        A["test dataset"] = list(map(A("datapoint from lesion id"), A["test lesion ids"]))
        #A("parallelize")("datapoint from lesion id", A["test lesion ids"])
        A["training dataset"] = list(map(A("datapoint from augmented path"), A("get augmented image paths")()))
        #A("parallelize")("datapoint from augmented path", A("get augmented image paths")())
    return fxn

#def partition_dataset_by_cls(A):
#    def fxn(dataset):
#        A["training data by class"] = {cls:[dp for dp in A["training dataset"] if dp["true class"] == cls] for cls in A["lesion classes"]}
#    return fxn

def set_bal_train_data_gen(A):
    def bal_data_gen():
        A["training data by class"] = {cls:[dp for dp in A["training dataset"] if \
                dp["true class"] == cls] for cls in A["lesion classes"]}

        assert A["number of classes"] % A["batch size"] == 0
        n = A["batch size"]//A["number of classes"]

        imgs = np.empty((A["batch size"], *A["input shape"]))
        labels = np.empty((A["batch size"], A["number of classes"]))

        while True:
            for ix,cls in enumerate(A["lesion classes"]):
                dps = np.random.permutation(A["training data by class"][cls])[:n]
                imgs[ix*n : (ix+1)*n] = np.array([A("load image")(dp["image path"]) for dp in dps])
                labels[ix*n : (ix+1)*n] = np.array([dp["true class"] for dp in dps])

            yield imgs, labels

    def fxn():
        A["training data generator"] = bal_data_gen()
    return fxn

####################################
### CNN
####################################

def build_cnn(A):
    def fxn(model_name):
        A("set input shape")(A["number of channels"], *A["dims"])

        A("make block from layers")("192conv_1", ["conv3d", "batch norm", "relu", "max pool"],
                                    **{"out channels": 192, "kernel size": (3, 3, 2)})
        A("make block from layers")("128conv_1", ["conv3d", "batch norm", "relu"],
                                    **{"out channels": 128, "kernel size": (3, 3, 2)})
        A("make block from layers")("pool_1", ["max pool"], **{"stride": (2, 2, 1)})
        A("make block from layers")("fc_1", ["fc", "relu"], **{"number of units": 128})
        A("make block from layers")("cls_1", ["fc", "softmax"], **{"number of units": A["number of classes"]})

        A("build model")(model_name=model_name, blocks=["192conv_1", "3x 128conv_1", "pool_1", "fc_1", "cls_1"])

        A["active model"] = nn.DataParallel(A["active model"])
    return fxn

def assess_model(A):
    def fxn():
        """Runs the CNN for max_runs times, saving performance metrics."""
        A["run status"] = A("instantiate")(cls="LIRADS run status")

        for run in range(1,A["max runs"]+1):
            A["run status"]["current run"] = run
            A("construct datasets")()
            gen = A("set balanced training data generator")()
            CNN = A("build lesion classifier")()
            A("set training settings")()

            CNN, hist = A("train model")(CNN, data=gen)
            results = A("test model")(CNN, data=)
            optimizer = A("compile optimizer")(CNN)

        return hist, args, epoch
    return fxn
    y_true = np.array([max(enumerate(x), key=operator.itemgetter(1))[0] for x in Y_test])
    y_pred = np.array([max(enumerate(x), key=operator.itemgetter(1))[0] for x in Y_pred])
    miscls_test = list(A["Z_test"][~np.equal(y_pred, y_true)])

    cm = confusion_matrix(y_true, y_pred)
    #f1 = f1_score(y_true, y_pred, average="weighted")
    running_acc_6.append(accuracy_score(y_true, y_pred))
    print("Accuracy: %d%% (avg: %d%%), time: %ds" % (running_acc_6[-1]*100, np.mean(running_acc_6)*100, time.time()-t))

    row = A("get_hyperparams_as_list")() + [num_samples[k] for k in A["cls names"]] + [running_acc_6[-1]]

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

def get_hyperparams_as_list(A):
    def fxn():
        return [A["hyperparameters"]["n"], A["hyperparameters"]["steps_per_epoch"], A["hyperparameters"]["epochs"],
                A["test num"], A["aug factor"], A["clinical inputs"],
                A["hyperparameters"]["kernel_size"], A["hyperparameters"]["f"], A["hyperparameters"]["padding"], A["hyperparameters"]["dropout"], A["hyperparameters"]["dense_units"],
                A["hyperparameters"]["pool_sizes"],
                A["hyperparameters"]["cnn_type"]+A["aleatoric"]*'-al'+A["aug pred"]*'-aug'+A["hyperparameters"]["mc_sampling"]*'-mc'+'-foc%.1f'%A["focal loss"],
                A["ensemble num"],
                A["hyperparameters"]["optimizer"].get_config()['lr']]
    return fxn

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


### delete?

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

def get_weights_for_balancing(A):
    def fxn():
        classes = dp["true class"] for dp in A["training dataset"]
        ordering, counts = np.unique(classes, return_counts=True)
        A["class weights"] =
    return fxn

def construct_dataloaders(A):
    def fxn():
        sampler = data.WeightedRandomSampler(weights=[dp["true class"] for dp in A["training dataset"]], num_samples=A["batch size"])
        A["training data loader"] = data.DataLoader(DS(A["training dataset"]),
            batch_size=A["batch size"], sampler=sampler, num_workers=8, pin_memory=True)
        class DS(data.Dataset):
            def __init__(self, ds_inst):
                self.ds_inst = ds_inst
            def __getitem__(self, index):
                path = self.ds_inst[index]["image path"]
                return np.load(self.ds_inst[index]["image path"]), self.ds_inst[index]["image path"]
            def __len__(self):
                return len(self.ds


            A["training data loader"] = data.DataLoader(self, sampler=train_sampler, batch_size=batch_size, **kwargs)
