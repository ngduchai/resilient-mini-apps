import sewar.full_ref as metrics_cal
from skimage.exposure import adjust_gamma, rescale_intensity
import h5py as h5
import numpy as np
import sys

import pickle

import matplotlib.pyplot as plt

import os

def plot_data(data, colors, name, unit, figpath, numiter, data_gd=1):
  plt.figure()
  for skip_ratio in data:
    pdata = np.array(data[skip_ratio])
    if pdata.ndim > 1:
      # Temporarily collect only one dimension
      pdata = pdata.transpose()[0]
    if data_gd == 1:
      pdata = np.ones(len(pdata))-pdata
    plt.plot(pdata, color=colors[skip_ratio], label=skip_labels[skip_ratio])
  plt.xlabel(unit)
  plt.xlim(0, numiter)
  if data_gd == 1:
    name = "$1-$" + name
  plt.ylabel(name)
  plt.yscale("log")
  
  plt.legend(loc="best")
  plt.tight_layout()
  plt.savefig(figpath)

def export_figure(data, colors, name, unit, figpath, numiter, labels, linestyles, ylim=[0, 1], data_gd=1):
  plt.figure()
  for skip_ratio in data:
    pdata = np.array(data[skip_ratio])
    if pdata.ndim > 1:
      # Temporarily collect only one dimension
      pdata = pdata.transpose()[0]
    if data_gd == 1:
      pdata = np.ones(len(pdata))-pdata
    plt.plot(pdata, color=colors[skip_ratio], label=labels[skip_ratio], linewidth=2, linestyle=linestyles[skip_ratio])
  plt.xlabel(unit)
  # plt.ylim(ylim)
  plt.xlim(0, numiter)
  if data_gd == 1:
    name = "$1-$" + name
  plt.ylabel(name)
  # plt.yscale("log")
  
  plt.legend(loc="best")
  plt.tight_layout()
  plt.savefig(figpath + ".png")
  plt.savefig(figpath + ".pdf")

reconpath = "recons/"
skip_configures = ["0", "10", "50", "90"]
# skip_configures = ["0", "10", "10-add-every-10", "10-drop-at-50", "50"]
# skip_configures = ["0", "10", "10-add-every-10", "10-add-every-20", "20-add-every-10", "20-add-every-20", "10-drop-at-50", "50"]
# skip_configures = ["0", "10", "10-add-every-10", "10-add-every-20", "20-add-every-10", "20-add-every-20", "50"]
# skip_configures = ["0", "10", "25", "40", "50", "10-drop-at-50", "10-add-every-10"]
# skip_configures = ["0",
#   "10", "10-drop-at-50", "10-drop-at-100",
#   "50", "50-drop-at-50", "50-drop-at-100",
#   "90", "90-drop-at-50", "90-drop-at-100"
#   ]
skip_data_paths = {
  "0" : "indicators-00",
  "10" : "indicators-10",
  "10-add-every-10" : "indicators-10-add-every-10",
  "10-add-every-20" : "indicators-10-add-every-20",
  "10-drop-at-100" : "indicators-10-drop-at-100",
  "10-drop-at-50" : "indicators-10-drop-at-50",
  "20-add-every-10" : "indicators-20-add-every-10",
  "20-add-every-20" : "indicators-20-add-every-20",
  "25" : "indicators-25",
  "25-drop-at-50" : "indicators-25-drop-at-50",
  "40" : "indicators-40",
  "50" : "indicators-50",
  "50-drop-at-50" : "indicators-50-drop-at-50",
  "50-drop-at-100" : "indicators-50-drop-at-100",
  "90" : "indicators-90",
  "90-drop-at-50" : "indicators-90-drop-at-50",
  "90-drop-at-100" : "indicators-90-drop-at-100",
}
skip_labels = {
  "0" : "Ideal",
  "10" : "$10\%$ data missed",
  "10-drop-at-50": "$10\%$ data missed after iter #50",
  "10-drop-at-100": "$10\%$ data missed after iter #100",
  "10-add-every-10" : "$10\%$ data available every 10 iter.",
  "25" : "$25\%$ data missed",
  "25-drop-at-50" : "$25\%$ data missed after iter #50",
  "40" : "$40\%$ data missed",
  "50" : "$50\%$ data missed",
  "50-drop-at-50": "$50\%$ data missed after iter #50",
  "50-drop-at-100": "$50\%$ data missed after iter #100",
  "90" : "$90\%$ data missed",
  "90-drop-at-50": "$90\%$ data missed after iter #50",
  "90-drop-at-100": "$90\%$ data missed after iter #100",
}
skip_colors = {
  "0" : "blue",
  "10" : "orange",
  "10-drop-at-50" : "brown",
  "10-drop-at-100" : "indigo",
  "10-add-every-10" : "violet",
  "10-add-every-20" : "olivedrab",
  "20-add-every-10" : "chocolate",
  "20-add-every-20" : "salmon",
  "25" : "cornflowerblue",
  "25-drop-at-50" : "pink",
  "40" : "lightseagreen",
  "50" : "green",
  "50-drop-at-50" : "cyan",
  "50-drop-at-100" : "olivedrab",
  "90" : "purple",
  "90-drop-at-50" : "greenyellow",
  "90-drop-at-100" : "yellowgreen",
}

quality_labels = {
  "0" : "$\\theta$ = 181",
  "10" : "$\\theta$ = 162",
  "50" : "$\\theta$ = 90",
  "90" : "$\\theta$ = 18"
}
skip_linestyles = {
  "0" : "-",
  "10" : "--",
  "50" : "-.",
  "90" : ":"
}

metrics = ["MS-SSIM", "SSIM", "UQI", "MSE", "PSNR"]
metric_paths = {
  "MS-SSIM": 'msssim',
  "SSIM": 'ssim',
  "UQI": 'uqi',
  "MSE": 'mse',
  "PSNR": 'psnr'
}

metric_gd = {
  "MS-SSIM": 1,
  "SSIM": 1,
  "UQI": 1,
  "MSE": -1,
  "PSNR": 2
}

if __name__ == "__main__":
  
  if len(sys.argv) < 4:
    print("Usage: python plt-quality-indicators.py <data folder> <fig folder> <num iter>")
    sys.exit(1)


  datapath = sys.argv[1]
  figpath = sys.argv[2]
  numiter = int(sys.argv[3])
  if not os.path.exists(figpath + "/indicators"):
    os.makedirs(figpath + "/indicators")
  if not os.path.exists(figpath + "/quality"):
    os.makedirs(figpath + "/quality")

  indicator_data = {}
  quality_data = {}
  for metric in metrics:
    indicator_data[metric] = {}
    quality_data[metric] = {}

  for skip_ratio in skip_configures:
    with open(datapath + "/" + skip_data_paths[skip_ratio], "rb") as f:
      for metric in metrics:
        indicator_data[metric][skip_ratio] = pickle.load(f)
      for metric in metrics:
        quality_data[metric][skip_ratio] = pickle.load(f)
      

  for metric in metrics:
    plot_data(indicator_data[metric], skip_colors, metric, "# iterations", figpath + "/indicators/" + metric_paths[metric], numiter, data_gd=metric_gd[metric])
    plot_data(quality_data[metric], skip_colors, metric, "# iterations", figpath + "/quality/" + metric_paths[metric], numiter, data_gd=metric_gd[metric])

plt.rcParams['axes.labelsize'] = 16     # X and Y labels font size
plt.rcParams['xtick.labelsize'] = 16    # X-axis tick labels font size
plt.rcParams['ytick.labelsize'] = 16    # Y-axis tick labels font size
plt.rcParams['legend.fontsize'] = 16

export_figure(quality_data["MS-SSIM"], skip_colors, "MS-SSIM", "Number of Iterations", figpath + "/quality/quality-indicators-" + metric_paths["MS-SSIM"], numiter, skip_labels, skip_linestyles, data_gd=-1)
export_figure(quality_data["MSE"], skip_colors, "MSE", "Number of Iterations", figpath + "/quality/quality-indicators-" + metric_paths["MSE"], numiter, skip_labels, skip_linestyles, data_gd=-1)
export_figure(quality_data["UQI"], skip_colors, "UQI", "Number of Iterations", figpath + "/quality/quality-indicators-" + metric_paths["UQI"], numiter, skip_labels, skip_linestyles, data_gd=-1)
export_figure(quality_data["SSIM"], skip_colors, "SSIM", "Number of Iterations", figpath + "/quality/quality-indicators-" + metric_paths["SSIM"], numiter, skip_labels, skip_linestyles, data_gd=-1)

export_figure(indicator_data["MS-SSIM"], skip_colors, "MS-SSIM", "Number of Iterations", figpath + "/indicator/indicators-" + metric_paths["MS-SSIM"], numiter, skip_labels, skip_linestyles, data_gd=-1)
export_figure(indicator_data["MSE"], skip_colors, "MSE", "Number of Iterations", figpath + "/indicator/indicators-" + metric_paths["MSE"], numiter, skip_labels, skip_linestyles, data_gd=-1)
export_figure(indicator_data["UQI"], skip_colors, "UQI", "Number of Iterations", figpath + "/indicator/indicators-" + metric_paths["UQI"], numiter, skip_labels, skip_linestyles, data_gd=-1)
export_figure(indicator_data["SSIM"], skip_colors, "SSIM", "Number of Iterations", figpath + "/indicator/indicators-" + metric_paths["SSIM"], numiter, skip_labels, skip_linestyles, data_gd=-1)



 
