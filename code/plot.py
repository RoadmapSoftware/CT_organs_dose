import os
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def plot_dice(data_path, figsize):
    data = pd.read_csv(data_path)
    sns.set(font_scale=1.1)
    sns.set_style("whitegrid")
    plt.figure(figsize=figsize)
    sns.boxplot(data=data)
    plt.ylabel("Dice similarity coefficient")
    plt.savefig(os.path.join(os.path.dirname(data_path), "dice_boxplot.png"), dpi=1200, bbox_inches="tight", pad_inches=0)

def read_data(error_path, method):
    df_pred = pd.read_csv(error_path, index_col=0)
    dict_error = dict()
    dict_error["Relative Dose Error(%)"] = df_pred.values.flatten(order="F") * 100
    dict_error["Organs"] = []
    for organ in df_pred.columns:
        dict_error["Organs"] += [organ] * len(df_pred.index)
    dict_error["Method"] = [method] * len(dict_error["Organs"])
    df = pd.DataFrame(dict_error)
    return df

def plot_dose_error(pred_error_path, phantom_error_path, figsize=(8, 8), yrange=(-50, 50)):
    df_pred = read_data(pred_error_path, "Proposed method")
    df_phantom = read_data(phantom_error_path, "Phantom-based method")
    df = pd.concat([df_pred, df_phantom], axis=0)
    sns.set(font_scale=1.5)
    sns.set_style("whitegrid")
    plt.figure(figsize=figsize)
    sns.boxplot(x="Organs", y="Relative Dose Error(%)", hue="Method", data=df, palette=sns.color_palette())
    plt.ylim(yrange[0], yrange[1])
    plt.legend(loc="upper left")
    plt.savefig(os.path.join(os.path.dirname(pred_error_path),"RDE_boxplot.png"), dpi=1000, bbox_inches="tight", pad_inches=0)

if __name__ == "__main__":
    data_path = os.path.join("..", "result", "LCTSC", "test_dice_scores.csv")
    plot_dice(data_path, figsize=(6.25, 6))
    data_path = os.path.join("..", "result", "PCT_crop", "test_dice_scores.csv")
    plot_dice(data_path, figsize=(10, 6))

    print("LCTSC: plot organ dose error boxplot")
    pred_error_path = os.path.join("..", "result", "LCTSC", "error_pred.csv")
    phantom_error_path = os.path.join("..", "result", "LCTSC", "error_phantom.csv")
    plot_dose_error(pred_error_path, phantom_error_path, figsize=(7, 7), yrange=(-40, 130))

    print("PCT: plot organ dose error boxplot")
    pred_error_path = os.path.join("..", "result", "PCT_crop", "error_pred.csv")
    phantom_error_path = os.path.join("..", "result", "PCT_crop", "error_phantom.csv")
    plot_dose_error(pred_error_path, phantom_error_path, figsize=(14, 7), yrange=(-60, 90))