import glob
import os
import nibabel as nib
import numpy as np
import pandas as pd
from ct_to_density import convert


def organs_dose(image_path, label_path, dose_path, organs, organs_id, number_of_turns):
    factor = 1.94e6
    image = nib.load(os.path.join(image_path, "image.nii.gz")).get_data()
    label = nib.load(os.path.join(label_path, "label.nii.gz")).get_data()
    dose = nib.load(os.path.join(dose_path, "dose.nii.gz")).get_data()
    rsd = nib.load(os.path.join(dose_path, "rsd.nii.gz")).get_data()
    density = np.reshape(list(map(convert, image.flatten())), image.shape)
    Dose = list()
    Rsd = list()
    for i in range(len(organs)):
        if isinstance(organs_id[i], list):
            organ_mask = np.zeros_like(label)
            for j in organs_id[i]:
                organ_mask += label == j
            organ_mask = organ_mask > 0
        else:
            organ_mask = label == organs_id[i]
        organ_dose = np.sum(dose * density * organ_mask) / np.sum(density* organ_mask)
        organ_rsd = np.sqrt(np.sum((dose * density * rsd * organ_mask) ** 2)) / (np.sum(density * organ_mask) * organ_dose)
        Dose.append(organ_dose * factor * number_of_turns)
        Rsd.append(organ_rsd)
    result = np.r_[Dose, Rsd]
    return result

def organs_dose_for_dataset(patients_image_path, patients_pred_path, dirname_pred, patients_dose_path, patients_info_file,
                            result_path, organs, organs_id):
    df = pd.read_csv(patients_info_file, index_col=0)
    patients_list = os.listdir(patients_dose_path)
    true_results = list()
    pred_results = list()
    for patient in patients_list:
        print(patient)
        image_path = glob.glob(os.path.join(patients_image_path, "*", patient))[0]
        true_label_path = image_path
        pred_label_path = glob.glob(os.path.join(patients_pred_path, "*", dirname_pred, patient))[0]
        dose_path = os.path.join(patients_dose_path, patient)
        number_of_turns = df["number of turns"][patient]
        result1 = organs_dose(image_path, true_label_path, dose_path, organs, organs_id, number_of_turns)
        result2 = organs_dose(image_path, pred_label_path, dose_path, organs, organs_id, number_of_turns)
        true_results.append(result1)
        pred_results.append(result2)
    true_results = np.array(true_results)
    pred_results = np.array(pred_results)
    organs_rsd_name = [o + "_rsd" for o in organs]
    columns = np.r_[organs, organs_rsd_name]
    df_true = pd.DataFrame(true_results, columns=columns, index=patients_list)
    df_pred = pd.DataFrame(pred_results, columns=columns, index=patients_list)
    df_true.to_csv(os.path.join(result_path, "true_organs_dose.csv"))
    df_pred.to_csv(os.path.join(result_path, "pred_organs_dose.csv"))
    

if __name__ == "__main__":
    print("Calculating organs dose for LCTSC ...")
    organs = ["lung", "heart", "esophagus"]
    organs_id = [[2, 3], 4, 5]
    # the path is the same for image and true label
    patients_image_path = os.path.join("..", "data", "LCTSC")
    patients_pred_path = os.path.join("..", "result", "LCTSC")
    dirname_pred = "prediction"
    patients_dose_path = os.path.join("..", "dose", "LCTSC_dose")
    patients_info_file = os.path.join("..", "data", "LCTSC_info.csv")
    result_path = os.path.join("..", "result", "LCTSC")
    organs_dose_for_dataset(patients_image_path, patients_pred_path, dirname_pred, patients_dose_path,
                            patients_info_file, result_path, organs, organs_id)

    print("Calculating organs dose for PCT ...")
    organs = ["spleen", "pancreas", "left kidney", "gallbladder", "liver", "stomach"]
    organs_id = [1, 2, 3, 4, 6, 7]
    patients_image_path = os.path.join("..", "data", "PCT_crop")
    patients_pred_path = os.path.join("..", "result", "PCT_crop")
    dirname_pred = "prediction"
    patients_dose_path = os.path.join("..", "dose", "PCT_crop_dose")
    patients_info_file = os.path.join("..", "data", "PCT_crop_info.csv")
    result_path = os.path.join("..", "result", "PCT_crop")
    organs_dose_for_dataset(patients_image_path, patients_pred_path, dirname_pred, patients_dose_path,
                            patients_info_file, result_path, organs, organs_id)

