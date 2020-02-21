import json
import os
import nibabel as nib
import numpy as np
from keras.models import model_from_json
from scipy.ndimage import zoom
from skimage import measure, morphology
from train_pct import config, get_indices, normalization, pad_gap
from keras_contrib.layers.normalization import InstanceNormalization

os.environ['CUDA_VISIBLE_DEVICES'] = '0'


def load_old_model(model_dir, json_file, weights_file):
    print("Loading pre-trained model")
    with open(os.path.join(model_dir, json_file), "r") as f:
        json_string = json.load(f)
    model = model_from_json(json_string)
    model.load_weights(os.path.join(model_dir, weights_file))
    return model

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

def predict(model, data_path, save_path, HU_window, patch_shape, overlap, n_classes, resample_shape=None):
    patients_list = os.listdir(data_path)
    for patient in patients_list:
        print("import patient:", patient)
        x = nib.load(os.path.join(data_path, patient, "image.nii.gz"))
        header = x.header
        affine = x.affine
        spacing = x.header["pixdim"][1:4]
        old_shape = x.shape
        x = x.get_data()

        # remove abnormal slices (value=-1024)
        ids = np.where(np.max(x, axis=(0, 1)) > -1000)[0]
        zmin = ids.min()
        zmax = ids.max() + 1
        zpad = (zmin, x.shape[2] - zmax)
        x = x[:, :, zmin:zmax]
        
        # resampling
        if resample_shape:
            factor = np.array(resample_shape) / x.shape
            x = zoom(x, factor, order=3)

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
            mask = mask[pad[0][0]:x.shape[0] - pad[0][1], pad[1][0]:x.shape[1] - pad[1][1],
                   pad[2][0]:x.shape[2] - pad[2][1]]

        if resample_shape:
            mask = zoom(mask, 1 / factor, order=0)

        mask = np.pad(mask, ((0, 0), (0, 0), zpad), mode="constant")
        mask = mask.astype(np.int)
        mask = post_process(mask, long_organ_id=[2, 5, 8])

        patient_path = os.path.join(save_path, patient)
        if not os.path.exists(patient_path):
            os.makedirs(patient_path)
        nib.save(nib.Nifti1Image(mask, affine=affine, header=header), os.path.join(patient_path, "label.nii.gz"))


if __name__ == "__main__":
    for i in range(5):
        config["model_dir"] = os.path.join(config["result_path"], config["model_name"] + str(i))
        test_data_path = os.path.join(config["data_path"], "fold" + str(i))
        prediction_path = os.path.join(config["model_dir"], "prediction")
        model = load_old_model(config["model_dir"], "model.json", "model.h5")
        predict(model=model,
                data_path=test_data_path,
                save_path=prediction_path,
                HU_window=config["HU_window"],
                patch_shape=config["patch_shape"],
                overlap=config["overlap"],
                n_classes=config["n_classes"],
                resample_shape=config["resample_shape"])