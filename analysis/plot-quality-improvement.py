import sewar.full_ref as metrics_cal
from skimage.exposure import adjust_gamma, rescale_intensity
import h5py as h5
import numpy as np
import sys

import pickle

import matplotlib.pyplot as plt

import os

def plot_compute(quality, compute, colors, name, unit, figpath):
  plt.figure()
  width = 0.15
  linner = None
  for inner in quality:
    if linner == None or len(quality[inner]) > len(quality[linner]):
      linner = inner
  m = 0
  for inner in compute:
    x = np.arange(len(quality[inner]))
    print(inner)
    print(quality[inner])
    print(compute[inner])
    plt.bar(x + width*m, np.array(compute[inner])*inner, width, color=colors[inner], label=inner)
    m += 1
  plt.xlabel(name)
  plt.xticks(np.arange(len(quality[linner])), quality[linner])
  plt.ylabel(unit)
  
  plt.legend(loc="best")
  plt.tight_layout()
  plt.savefig(figpath)


def cal_improvement(data, data_gd=1):
  pdata = np.array(data)
  if pdata.ndim > 1:
    # Temporarily collect only one dimension
    pdata = pdata[0]
  if data_gd == 1:
    pdata = np.ones(len(pdata)) - pdata
  quality = 0.1
  m = 0.1
  qualities = []
  computes = []

  print(pdata)
  
  for i in range(len(pdata)):
    if pdata[i] < quality:
      qualities.append(quality)
      computes.append(i+1)
      quality *= m
      if quality < 0.0000001:
        break
  
  if data_gd == 1:
    qualities = np.ones(len(qualities)) - np.array(qualities)
  
  return qualities, computes


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
  "PSNR": 1
}

if __name__ == "__main__":
  
  if len(sys.argv) < 3:
    print("Usage: python plt-quality-improvement.py <data folder> <fig folder>")
    sys.exit(1)


  datapath = sys.argv[1]
  figpath = sys.argv[2]

  metric_data = {}
  quality_data = {}
  compute_data = {}
  for metric in metrics:
    metric_data[metric] = {}
    quality_data[metric] = {}
    compute_data[metric] = {}

  for inner in inner_configures:
    with open(datapath + "/" + inner_data_paths[inner], "rb") as f:
      for metric in metrics:
        metric_data[metric][inner] = pickle.load(f)
        quality_data[metric][inner], compute_data[metric][inner] = cal_improvement(metric_data[metric][inner], metric_gd[metric])

  for metric in metrics:
    plot_compute(quality_data[metric], compute_data[metric], inners_colors, metric, "inner iter.", figpath + "/" + metric_paths[metric])




 
