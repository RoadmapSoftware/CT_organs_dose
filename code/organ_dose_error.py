import os
import numpy as np
import pandas as pd

def error_prediction(true_path, pred_path):
    truth = pd.read_csv(true_path, index_col=0)
    prediction = pd.read_csv(pred_path, index_col=0)
    num_organs = len(truth.columns) // 2
    error = (prediction.values[:, :num_organs] - truth.values[:, :num_organs]) / truth.values[:, :num_organs]
    df_error = pd.DataFrame(error, columns=truth.columns[:num_organs], index=truth.index)
    df_error.to_csv(os.path.join(os.path.dirname(true_path), "error_pred.csv"))
    df_error_statistic = df_error.describe()
    columns_abs = [o + "_abs" for o in truth.columns[:num_organs]]
    df_abserror = pd.DataFrame(np.abs(error), columns=columns_abs, index=truth.index)
    df_abserror_statistic = df_abserror.describe()
    df_statistic = pd.concat([df_error_statistic, df_abserror_statistic], axis=1)
    df_statistic.to_csv(os.path.join(os.path.dirname(true_path), "error_pred_statistic.csv"))

def error_phantom(true_path, phantom_path, info_file):
    truth = pd.read_csv(true_path, index_col=0)
    phantom = pd.read_csv(phantom_path, index_col=0)
    num_organs = len(truth.columns) // 2
    error = list()
    if info_file:
        df_info = pd.read_csv(info_file, index_col=0)
        for i, patient in enumerate(truth.index):
            sex = 0 if df_info["sex"][patient] == "M" else 1
            err = (phantom.values[sex, :num_organs] - truth.values[i, :num_organs]) / truth.values[i, :num_organs]
            error.append(err)
    else:
        for i, patient in enumerate(truth.index):
            err = (phantom.values[0, :num_organs] - truth.values[i, :num_organs]) / truth.values[i, :num_organs]
            error.append(err)
    error = np.array(error)
    df_error = pd.DataFrame(error, columns=truth.columns[:num_organs], index=truth.index)
    df_error.to_csv(os.path.join(os.path.dirname(true_path), "error_phantom.csv"))
    df_error_statistic = df_error.describe()
    columns_abs = [o + "_abs" for o in truth.columns[:num_organs]]
    df_abserror = pd.DataFrame(np.abs(error), columns=columns_abs, index=truth.index)
    df_abserror_statistic = df_abserror.describe()
    df_statistic = pd.concat([df_error_statistic, df_abserror_statistic], axis=1)
    df_statistic.to_csv(os.path.join(os.path.dirname(true_path), "error_phantom_statistic.csv"))

if __name__ == "__main__":
    print("Organs dose error statistics for LCTSC")
    true_path = os.path.join("..", "result", "LCTSC", "true_organs_dose.csv")
    pred_path = os.path.join("..", "result", "LCTSC", "pred_organs_dose.csv")
    phantom_path = os.path.join("..", "result", "LCTSC", "phantom_organs_dose.csv")
    info_file = os.path.join("..", "data", "LCTSC_info.csv")
    error_prediction(true_path, pred_path)
    error_phantom(true_path, phantom_path, info_file)

    print("Organs dose error statistics for PCT")
    true_path = os.path.join("..", "result", "PCT_crop", "true_organs_dose.csv")
    pred_path = os.path.join("..", "result", "PCT_crop", "pred_organs_dose.csv")
    phantom_path = os.path.join("..", "result", "PCT_crop", "phantom_organs_dose.csv")
    info_file = None
    error_prediction(true_path, pred_path)
    error_phantom(true_path, phantom_path, info_file)



