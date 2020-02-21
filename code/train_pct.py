import glob
import json
import os
import keras
import nibabel as nib
import numpy as np
import scipy
from keras.callbacks import ModelCheckpoint, CSVLogger, LearningRateScheduler, ReduceLROnPlateau, EarlyStopping
from keras.optimizers import Adam
from skimage import measure
from metrics import weighted_dice_coefficient_loss, get_label_dice_coefficient_function
from model import Unet3D_model

os.environ['CUDA_VISIBLE_DEVICES'] = '0'


def step_decay(epoch, initial_lrate, drop, epochs_drop):
    return initial_lrate * math.pow(drop, math.floor((1 + epoch) / float(epochs_drop)))


def get_callbacks(model_file, initial_learning_rate=0.0001, learning_rate_drop=0.5, learning_rate_epochs=None,
                  learning_rate_patience=10, logging_file="training.log", verbosity=1,
                  early_stopping_patience=50):
    callbacks = list()
    callbacks.append(ModelCheckpoint(model_file, save_best_only=True))
    callbacks.append(CSVLogger(logging_file, append=True))
    if learning_rate_epochs:
        callbacks.append(LearningRateScheduler(partial(step_decay, initial_lrate=initial_learning_rate,
                                                       drop=learning_rate_drop, epochs_drop=learning_rate_epochs)))
    else:
        callbacks.append(ReduceLROnPlateau(factor=learning_rate_drop, patience=learning_rate_patience,
                                           verbose=verbosity))
    if early_stopping_patience:
        callbacks.append(EarlyStopping(verbose=verbosity, patience=early_stopping_patience))
    return callbacks


def get_random_index(image_shape, patch_shape):
    gap = np.array(image_shape) - np.array(patch_shape)
    index = [np.random.choice(g + 1) for g in gap]
    return [index]


def get_indices(image_shape, patch_shape, overlap):
    gap = np.array(image_shape) - np.array(patch_shape)
    step = np.array(patch_shape) - np.array(overlap)
    num = gap // step + 1
    D_indices = list(set([i * step[0] for i in range(num[0])] + [gap[0]]))
    H_indices = list(set([i * step[1] for i in range(num[1])] + [gap[1]]))
    W_indices = list(set([i * step[2] for i in range(num[2])] + [gap[2]]))
    return np.asarray(np.meshgrid(D_indices, H_indices, W_indices), dtype=np.int).reshape(3, -1).T


def pad_gap(gap):
    pad = []
    for g in gap:
        if g < 0:
            g_b = np.abs(g) // 2
            g_a = np.abs(g) - g_b
        else:
            g_b = 0
            g_a = 0
        pad.append((g_b, g_a))
    return tuple(pad)


def data_generate(X, Y, patch_shape, overlap, batch_size=1, n_classes=6, random=False):
    while True:
        data_x = list()
        data_y = list()
        for i in range(len(X)):
            x = X[i]
            y = Y[i]
            gap = np.array(x.shape) - np.array(patch_shape)
            if np.any(gap < 0):
                pad = pad_gap(gap)
                x = np.pad(x, pad, mode="constant")
                y = np.pad(y, pad, mode="constant")
            if random:
                indices = get_random_index(x.shape, patch_shape)
            else:
                indices = get_indices(x.shape, patch_shape, overlap)
            for j, index in enumerate(indices):
                x_patch = x[index[0]:index[0] + patch_shape[0], index[1]:index[1] + patch_shape[1],
                          index[2]:index[2] + patch_shape[2]]
                y_patch = y[index[0]:index[0] + patch_shape[0], index[1]:index[1] + patch_shape[1],
                          index[2]:index[2] + patch_shape[2]]
                data_x.append(x_patch)
                data_y.append(y_patch)
                if len(data_x) == batch_size or (i == len(X) - 1 and j == len(indices) - 1):
                    batch_x = np.array(data_x)[..., np.newaxis]
                    batch_y = keras.utils.to_categorical(data_y, num_classes=n_classes)
                    data_x = list()
                    data_y = list()
                    yield batch_x, batch_y


def get_steps(X, patch_shape, overlap, batch_size=8, random=False):
    n_patches = 0
    for x in X:
        gap = np.array(x.shape) - np.array(patch_shape)
        if np.any(gap < 0):
            pad = pad_gap(gap)
            x = np.pad(x, pad, mode="constant")
        if random:
            indices = get_random_index(x.shape, patch_shape)
        else:
            indices = get_indices(x.shape, patch_shape, overlap)
        n_patches += len(indices)
    steps = int(np.ceil(n_patches / batch_size))
    return steps


def normalization(image3D, vmin=-200, vmax=300):
    image3D = np.clip(image3D, vmin, vmax)
    image3D = (image3D - image3D.min()) / (image3D.max() - image3D.min())
    return image3D


def max_area_rigion(labels_image):
    # bbox = [ymin, xmin, ymax, xmax]
    rigions = measure.regionprops(labels_image)
    max_area = -1
    bbox = []
    for i, r in enumerate(rigions):
        if r.area > max_area:
            max_area = r.area
            bbox = r.bbox
    return bbox


def get_body_bbox(image3D, threshold=0):
    bboxes = []
    for i in range(image3D.shape[2]):
        image = image3D[:, :, i]
        binary_image = image > threshold
        labels_image = measure.label(binary_image, connectivity=2)
        bbox = max_area_rigion(labels_image)
        bboxes.append(bbox)
    bbox_choice = np.median(np.vstack(bboxes), axis=0).astype(np.int)
    return bbox_choice


def get_data(patients_list, resample_shape, HU_window):
    X = list()
    Y = list()
    for patient in patients_list:
        print("import patient:", patient)
        x = nib.load(os.path.join(patient, "image.nii.gz"))
        spacing = x.header["pixdim"][1:4]
        x = x.get_data()
        y = nib.load(os.path.join(patient, "label.nii.gz")).get_data()

        # remove abnormal slices (value=-1024)
        ids = np.where(np.max(x, axis=(0, 1)) > -1000)[0]
        x = x[:, :, ids.min():ids.max() + 1]
        y = y[:, :, ids.min():ids.max() + 1]

        # cropping foreground
        # bbox = get_body_bbox(x, -200)
        # x = x[bbox[0]:bbox[2], bbox[1]:bbox[3], :]
        # y = y[bbox[0]:bbox[2], bbox[1]:bbox[3], :]

        # resampling
        if resample_shape:
            factor = np.array(resample_shape) / x.shape
            x = scipy.ndimage.zoom(x, factor, order=3)
            y = scipy.ndimage.zoom(y, factor, order=0)

        x = normalization(x, HU_window[0], HU_window[1])
        # import matplotlib.pyplot as plt
        # plt.imshow(x[:, :, x.shape[2]//2])
        # plt.show()
        X.append(x)
        Y.append(y)
    return X, Y


def train(model_dir, data_path, patch_shape, overlap, HU_window, n_classes=6, train_batch_size=8, valid_batch_size=8,
          learning_rate=0.001, resample_shape=None, num_folds=None, id_fold_test=None):
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    model_file = os.path.join(model_dir, "model.h5")
    logging_file = os.path.join(model_dir, "training.log")

    id_folds = np.arange(num_folds)
    id_folds = np.r_[id_folds[id_fold_test:], id_folds[:id_fold_test]]
    id_fold_train = id_folds[1:4]
    id_fold_valid = id_folds[4]
    patients_train = []
    for idx in id_fold_train:
        patients_train += glob.glob(os.path.join(data_path, "fold" + str(idx), "*"))
    patients_valid = glob.glob(os.path.join(data_path, "fold" + str(id_fold_valid), "*"))

    X_train, Y_train = get_data(patients_train, resample_shape, HU_window)
    X_valid, Y_valid = get_data(patients_valid, resample_shape, HU_window)
    print("train patients:", len(X_train))
    print("valid patients:", len(X_valid))
    gen_train = data_generate(X_train, Y_train, patch_shape, overlap=overlap, batch_size=train_batch_size,
                              n_classes=n_classes, random=True)
    gen_valid = data_generate(X_valid, Y_valid, patch_shape, overlap=overlap, batch_size=valid_batch_size,
                              n_classes=n_classes, random=True)
    steps_train = get_steps(X_train, patch_shape, overlap=overlap, batch_size=train_batch_size, random=True)
    steps_valid = get_steps(X_valid, patch_shape, overlap=overlap, batch_size=valid_batch_size, random=True)

    input_shape = tuple(list(patch_shape) + [1])
    model = Unet3D_model(input_shape=input_shape, n_base_filters=16, n_classes=n_classes)
    model.summary()
    json_string = model.to_json()
    with open(os.path.join(model_dir, "model.json"), "w") as f:
        json.dump(json_string, f)
    print("training steps:", steps_train)
    print("validation steps:", steps_valid)
    metrics = [get_label_dice_coefficient_function(index) for index in range(n_classes)]
    model.compile(optimizer=Adam(lr=learning_rate), loss=weighted_dice_coefficient_loss, metrics=metrics)
    callbacks = get_callbacks(model_file, initial_learning_rate=learning_rate, learning_rate_drop=0.5,
                              learning_rate_epochs=None,
                              learning_rate_patience=30, logging_file=logging_file, verbosity=1,
                              early_stopping_patience=50)
    model.fit_generator(gen_train, validation_data=gen_valid, epochs=3000, steps_per_epoch=steps_train,
                        validation_steps=steps_valid, callbacks=callbacks)


config = dict()
config["data_path"] = os.path.join("..", "data", "PCT_crop")
config["result_path"] = os.path.join("..", "result", "PCT_crop")
config["model_name"] = "Unet3D_PCT_Patch-128-128-128_HU--200-300_Reshape-144-144-144_fold"
config["resample_shape"] = (144, 144, 144)
config["HU_window"] = (-200, 300)
config["patch_shape"] = (128, 128, 128)
config["overlap"] = (120, 120, 120)
config["n_classes"] = 9

if __name__ == "__main__":
    for i in range(5):
        config["model_dir"] = os.path.join(config["result_path"], config["model_name"]+str(i))
        train(model_dir=config["model_dir"],
              data_path=config["data_path"],
              patch_shape=config["patch_shape"],
              HU_window=config["HU_window"],
              n_classes=config["n_classes"],
              overlap=config["overlap"],
              train_batch_size=1,
              valid_batch_size=4,
              learning_rate=5e-4,
              resample_shape=config["resample_shape"],
              num_folds=5,
              id_fold_test=i)