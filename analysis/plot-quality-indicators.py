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
  # ßßß
  
  plt.legend(loc="best")
  plt.tight_layout()
  plt.savefig(figpath)

reconpath = "recons/"
# skip_configures = ["0", "10", "50", "90"]
# skip_configures = ["0", "10", "10-add-every-10", "10-skip-at-50", "50"]
# skip_configures = ["0", "10", "10-add-every-10", "10-add-every-20", "20-add-every-10", "20-add-every-20", "10-skip-at-50", "50"]
# skip_configures = ["0", "10", "10-add-every-10", "10-add-every-20", "20-add-every-10", "20-add-every-20", "50"]
skip_configures = ["0", "10", "10-skip-at-50", "10-add-every-10"]
skip_data_paths = {
  "0" : "indicators-00",
  "10" : "indicators-10",
  "10-add-every-10" : "indicators-10-add-every-10",
  "10-add-every-20" : "indicators-10-add-every-20",
  "20-add-every-10" : "indicators-20-add-every-10",
  "20-add-every-20" : "indicators-20-add-every-20",
  "10-skip-at-50" : "indicators-10-drop-at-50",
  "10-skip-at-100" : "indicators-10-drop-at-100",
  "50" : "indicators-50",
  "50-skip-at-100" : "indicators-50-drop-at-100",
  "90" : "indicators-90"
}
skip_labels = {
  "0" : "Ideal",
  "10" : "$10\%$ data missed",
  "10-skip-at-50": "$10\%$ data missed after iter #50",
  "10-add-every-10" : "$10\%$ data available every 10 iter.",
}
skip_colors = {
  "0" : "orange",
  "10" : "blue",
  "10-add-every-10" : "violet",
  "10-add-every-20" : "olivedrab",
  "20-add-every-10" : "chocolate",
  "20-add-every-20" : "salmon",
  "10-skip-at-50" : "brown",
  "10-skip-at-100" : "indigo",
  "50" : "green",
  "50-skip-at-100" : "yellowgreen",
  "90" : "purple"
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




 
