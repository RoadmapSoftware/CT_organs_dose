import json
import os
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import pandas as pd
from keras.models import model_from_json
from scipy.ndimage import zoom
from skimage import measure, morphology
from train import config, get_indices, normalization, get_body_bbox, pad_gap
from keras_contrib.layers.normalization import InstanceNormalization

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

def load_old_model(model_dir, json_file, weights_file):
    print("Load:", os.path.join(model_dir, json_file))
    with open(os.path.join(model_dir, json_file), "r") as f:
        json_string = json.load(f)
    model = model_from_json(json_string)
    model.load_weights(os.path.join(model_dir, weights_file))
    return model

def loss_plot(model_dir):
    training_file = os.path.join(model_dir, "training.log")
    training_df = pd.read_csv(training_file).set_index('epoch')
    plt.figure()
    plt.plot(training_df['loss'].values, label='training loss')
    plt.plot(training_df['val_loss'].values, label='validation loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.xlim((0, len(training_df.index)))
    plt.legend(loc='upper right')
    plt.savefig(os.path.join(model_dir, "loss_graph.png"))
    plt.close()

def post_process(label, long_organ_id=None):
    new_label = np.zeros_like(label)
    for i in range(1, np.max(label)+1):
        mask = label == i
        label_image = measure.label(mask, connectivity=3)
        props = measure.regionprops(label_image)
        areas = [region.area for region in props]
        if i in long_organ_id:
            mask = morphology.remove_small_objects(mask, min_size=np.max(areas) * 0.1, connectivity=3)
        else:
            mask = morphology.remove_small_objects(mask, min_size=np.max(areas), connectivity=3)
        new_label[mask>0] = i
    return new_label

def predict_lctsc(model, image, spacing, standard_spacing, HU_window, patch_shape, overlap, n_classes):
    # resampling
    old_shape = image.shape
    scale = np.array(spacing) / np.array(standard_spacing)
    new_shape = (old_shape * scale).astype(np.int)
    factor = new_shape / old_shape
    x = zoom(image, factor, order=3)

    # cropping foreground
    bbox = get_body_bbox(x, -200)
    ymin, xmin, ymax, xmax = bbox
    ypad = (ymin, x.shape[0] - ymax)
    xpad = (xmin, x.shape[1] - xmax)
    x = x[ymin:ymax, xmin:xmax, :]

    x = normalization(x, HU_window[0], HU_window[1])

    gap = np.array(x.shape) - np.array(patch_shape)
    if np.any(gap < 0):
        pad = pad_gap(gap)
        x = np.pad(x, pad, mode="constant")

    indices = get_indices(x.shape, patch_shape, overlap)
    y_pred = np.zeros(x.shape + (n_classes,))
    count = np.zeros(x.shape + (n_classes,))
    print("patch number:", len(indices))
    for index in indices:
        x_patch = x[index[0]:index[0] + patch_shape[0], index[1]:index[1] + patch_shape[1],
                  index[2]:index[2] + patch_shape[2]]
        x_patch = x_patch[np.newaxis, ..., np.newaxis]
        y_patch_pred = model.predict(x_patch, batch_size=1)
        y_patch_pred = np.squeeze(y_patch_pred)
        y_pred[index[0]:index[0] + patch_shape[0], index[1]:index[1] + patch_shape[1],
        index[2]:index[2] + patch_shape[2], :] += y_patch_pred
        count[index[0]:index[0] + patch_shape[0], index[1]:index[1] + patch_shape[1],
        index[2]:index[2] + patch_shape[2], :] += 1
    y_pred = y_pred / count
    mask = np.argmax(y_pred, axis=-1)
    if np.any(gap < 0):
        mask = mask[pad[0][0]:x.shape[0] - pad[0][1], pad[1][0]:x.shape[1] - pad[1][1], pad[2][0]:x.shape[2] - pad[2][1]]
    mask = np.pad(mask, (ypad, xpad, (0, 0)), mode="constant")
    mask = zoom(mask, 1 / factor, order=0)
    mask = mask.astype(np.int)
    mask = post_process(mask, long_organ_id=[1, 5])
    return mask


def predict(model, data_path, save_path, standard_spacing, HU_window, patch_shape, overlap, n_classes):
    patients_list = os.listdir(data_path)
    for patient in patients_list:
        print("import patient:", patient)
        x = nib.load(os.path.join(data_path, patient, "image.nii.gz"))
        header = x.header
        affine = x.affine
        spacing = x.header["pixdim"][1:4]
        x = x.get_data()
        mask = predict_lctsc(model, x, spacing, standard_spacing=standard_spacing, HU_window=HU_window,
                             patch_shape=patch_shape, overlap=overlap, n_classes=n_classes)
        patient_path = os.path.join(save_path, patient)
        if not os.path.exists(patient_path):
            os.makedirs(patient_path)
        nib.save(nib.Nifti1Image(mask, affine=affine, header=header), os.path.join(patient_path, "label.nii.gz"))

if __name__ == "__main__":
    for i in range(5):
        config["model_dir"] = os.path.join(config["result_path"], config["model_name"] + str(i))
        test_data_path = os.path.join(config["data_path"], "fold"+str(i))
        prediction_path = os.path.join(config["model_dir"], "prediction")
        model = load_old_model(config["model_dir"], "model.json", "model.h5")
        loss_plot(config["model_dir"])
        predict(model=model, data_path=test_data_path, save_path=prediction_path,
                standard_spacing=config["standard_spacing"], HU_window=config["HU_window"],
                patch_shape=config["patch_shape"], overlap=config["overlap"], n_classes=config["n_classes"])