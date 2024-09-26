import sewar.full_ref as metrics_cal
from skimage.exposure import adjust_gamma, rescale_intensity
import h5py as h5
import numpy as np
import sys

import pickle

import matplotlib.pyplot as plt

import os

def plot_quality(steps, data, colors, name, unit, figpath, numiter, data_gd=1):
  plt.figure()
  for inner in data:
    pdata = np.array(data[inner])
    if pdata.ndim > 1:
      # Temporarily collect only one dimension
      pdata = pdata.transpose()[0]
    if data_gd == 1:
      pdata = np.ones(len(pdata))-pdata
    plt.plot(steps[inner], pdata, color=colors[inner], label=inner)
  plt.xlabel(unit)
  plt.xlim(0, numiter)
  if data_gd == 1:
    name = "$1-$" + name
  plt.ylabel(name)
  plt.yscale("log")
  
  plt.legend(loc="best")
  plt.tight_layout()
  plt.savefig(figpath)

reconpath = "recons/"
inner_configures = [1, 2, 4, 8]
inner_data_paths = {
  1 : "inner-1",
  2 : "inner-2",
  4 : "inner-4",
  8 : "inner-8"
}
inners_colors = {
  1 : "orange",
  2 : "blue",
  4 : "green",
  8 : "purple"
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
    print("Usage: python plt-quality-time.py <data folder> <fig folder> <num iter>")
    sys.exit(1)


  datapath = sys.argv[1]
  figpath = sys.argv[2]
  numiter = int(sys.argv[3])

  metric_data = {}
  for metric in metrics:
    metric_data[metric] = {}

  for inner in inner_configures:
    with open(datapath + "/" + inner_data_paths[inner], "rb") as f:
      for metric in metrics:
        metric_data[metric][inner] = pickle.load(f)

  for metric in metrics:
    metric_steps = {}
    metric_steps_adj = {}
    for inner in inner_configures:
      metric_steps[inner] = np.array(range(len(metric_data[metric][inner])))
      metric_steps_adj[inner] = np.array(range(len(metric_data[metric][inner]))) * inner
    plot_quality(metric_steps, metric_data[metric], inners_colors, metric, "outer iter.", figpath + "/" + metric_paths[metric], numiter, data_gd=metric_gd[metric])
    plot_quality(metric_steps_adj, metric_data[metric], inners_colors, metric, "inner iter.", figpath + "/" + "adj-" + metric_paths[metric], numiter*max(inner_configures), data_gd=metric_gd[metric])




 
