import os
import nibabel as nib
import numpy as np
import pandas as pd


def organs_dose(data_path, dose_path, organs, organs_id, number_of_turns):
    factor = 1.94e6
    image = nib.load(os.path.join(data_path, "image.nii.gz")).get_data()
    dose = nib.load(os.path.join(dose_path, "dose.nii.gz")).get_data()
    rsd = nib.load(os.path.join(dose_path, "rsd.nii.gz")).get_data()
    Dose = list()
    Rsd = list()
    for i in range(len(organs)):
        if isinstance(organs_id[i], list):
            organ_mask = np.zeros_like(image)
            for j in organs_id[i]:
                organ_mask += image == j
            organ_mask = organ_mask > 0
        else:
            organ_mask = image == organs_id[i]
        organ_dose = np.sum(dose * organ_mask) / np.sum(organ_mask)
        organ_rsd = np.sqrt(np.sum((dose * rsd * organ_mask) ** 2)) / (np.sum(organ_mask) * organ_dose)
        Dose.append(organ_dose * factor * number_of_turns)
        Rsd.append(organ_rsd)
    result = np.r_[Dose, Rsd]
    return result

def phantom_for_LCTSC():
    print("Calculating thorax organs dose ...")
    result_path = os.path.join("..", "result", "LCTSC")
    organs = ["lung", "heart", "esophagus"]
    organs_id = [[97, 99], [87, 88], 110]

    data_path = os.path.join("..", "data", "Phantom", "rpi_am_73")
    dose_path = os.path.join("..", "dose", "phantom_dose", "rpi_am_73_thorax_118-158")
    number_of_turns =20
    am_result = organs_dose(data_path, dose_path, organs, organs_id, number_of_turns)

    data_path = os.path.join("..", "data", "Phantom", "rpi_af_63")
    dose_path = os.path.join("..", "dose", "phantom_dose", "rpi_af_63_thorax_106-144")
    number_of_turns = 19
    af_result = organs_dose(data_path, dose_path, organs, organs_id, number_of_turns)

    result = np.array([am_result, af_result])
    result = np.reshape(result, (2, len(organs)*2))
    organs_rsd_name = [o + "_rsd" for o in organs]
    columns = np.r_[organs, organs_rsd_name]
    df = pd.DataFrame(result, columns=columns, index=["Male", "Female"])
    df.to_csv(os.path.join(result_path, "phantom_organs_dose.csv"))

def phantom_for_PCT():
    print("Calculating abdomen organs dose ...")
    result_path = os.path.join("..", "result", "PCT_crop")
    data_path = os.path.join("..", "data", "Phantom", "rpi_am_73")
    dose_path = os.path.join("..", "dose", "phantom_dose", "rpi_am_73_abdomen_106-126")
    organs = ["spleen", "pancreas", "left kidney", "gallbladder", "liver", "stomach"]
    organs_id = [127, 113, [89, 90, 91], [70, 71], 95, [72, 73]]
    number_of_turns = 10
    result = organs_dose(data_path, dose_path, organs, organs_id, number_of_turns)
    result = np.reshape(result, (1, len(organs)*2))
    organs_rsd_name = [o + "_rsd" for o in organs]
    columns = np.r_[organs, organs_rsd_name]
    df = pd.DataFrame(result, columns=columns, index=["Male"])
    df.to_csv(os.path.join(result_path, "phantom_organs_dose.csv"))

if __name__ == "__main__":
    phantom_for_LCTSC()
    phantom_for_PCT()


