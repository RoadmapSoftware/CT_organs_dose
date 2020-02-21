import os
from functools import partial
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import pandas as pd
from train import config


def get_organs_mask(data, index=1):
    return data == index

def dice_coefficient(truth, prediction):
    if np.sum(truth) == 0:
        return np.nan
    else:
        return 2 * np.sum(truth * prediction) / (np.sum(truth) + np.sum(prediction))
    
def dice_plot(df, file_path):
    scores = dict()
    for index, score in enumerate(df.columns):
        values = df.values.T[index]
        scores[score] = values[np.isnan(values) == False]
    plt.boxplot(list(scores.values()), labels=list(scores.keys()))
    plt.ylabel("Dice Coefficient")
    plt.grid()
    plt.gca().set_xticklabels(list(scores.keys()), rotation=30, fontsize=8)
    plt.savefig(file_path)
    plt.close()

def evaluate(data_dir, prediction_dir, header):
    masking_functions = [partial(get_organs_mask, index=index+1) for index in range(len(header))]
    dice_rows = list()
    subject_ids = list()
    patients_list = os.listdir(prediction_dir)
    for patient in patients_list:
        dice = list()
        subject_ids.append(patient)
        truth_file = os.path.join(data_dir, patient, "label.nii.gz")
        truth = nib.load(truth_file).get_data()
        
        prediction_file = os.path.join(prediction_dir, patient, "label.nii.gz")
        prediction = nib.load(prediction_file).get_data()

        # cord
        ids_true = np.where(truth == 1)[2]
        min_z = np.min(ids_true)
        max_z = np.max(ids_true)
        cord_truth = truth[:, :, min_z:max_z+1]
        cord_prediction = prediction[:, :, min_z:max_z+1]
        func = masking_functions[0]
        dice.append(dice_coefficient(func(cord_truth), func(cord_prediction)))
        # lung and heart
        dice.extend([dice_coefficient(func(truth), func(prediction))for func in masking_functions[1:4]])
        # esophagus
        ids_true = np.where(truth == 5)[2]
        min_z = np.min(ids_true)
        max_z = np.max(ids_true)
        eso_truth = truth[:, :, min_z:max_z + 1]
        eso_prediction = prediction[:, :, min_z:max_z + 1]
        func = masking_functions[4]
        dice.append(dice_coefficient(func(eso_truth), func(eso_prediction)))
        
        dice_rows.append(dice)
        
    return dice_rows, subject_ids


if __name__ == "__main__":
    header = ('SpinalCord', 'Lung_R', 'Lung_L', 'Heart', 'Esophagus')
    Dice = list()
    Patients = list()
    for i in range(5):
        print("Evaluating fold" + str(i) + "...")
        config["model_dir"] = os.path.join(config["result_path"], config["model_name"] + str(i))
        test_data_path = os.path.join(config["data_path"], "fold" + str(i))
        prediction_path = os.path.join(config["model_dir"], "prediction")
        dice_rows, subject_ids = evaluate(test_data_path, prediction_path, header)
        Dice.extend(dice_rows)
        Patients.extend(subject_ids)
    df = pd.DataFrame.from_records(Dice, columns=header, index=Patients)
    df.to_csv(os.path.join(config["result_path"], "test_dice_scores.csv"))
    df.describe().to_csv(os.path.join(config["result_path"], "test_dice_statistics.csv"))
    dice_plot(df, os.path.join(config["result_path"], "test_dice_scores_boxplot.png"))






