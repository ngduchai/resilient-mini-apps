import sewar.full_ref as metrics_cal
from skimage.exposure import adjust_gamma, rescale_intensity
import h5py as h5
import numpy as np
import sys

import pickle

import matplotlib.pyplot as plt

import os

def plot_quality(data, colors, name, figpath):
  plt.figure()
  for inner in data:
    plt.plot(data[inner], color=colors[inner], label=inner)
  plt.xlabel("# outer iterations")
  plt.ylabel(name)
  
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

if __name__ == "__main__":
  
  if len(sys.argv) < 3:
    print("Usage: python plt-quality-time.py <data folder> <fig folder>")
    sys.exit(1)


  datapath = sys.argv[1]
  figpath = sys.argv[2]

  msssims = {}
  ssims = {}
  uqis = {}
  mses = {}
  psnrs = {}

  for inner in inner_configures:
    with open(datapath + "/" + inner_data_paths[inner], "rb") as f:
      msssims[inner] = pickle.load(f)
      ssims[inner] = pickle.load(f)
      uqis[inner] = pickle.load(f)
      mses[inner] = pickle.load(f)
      psnrs[inner] = pickle.load(f)


  plot_quality(msssims, inners_colors, "MS-SSIM", figpath + "/msssim")
  plot_quality(ssims, inners_colors, "SSIM", figpath + "/ssim")
  plot_quality(uqis, inners_colors, "UQI", figpath + "/uqi")
  plot_quality(mses, inners_colors, "MSE", figpath + "/mse")
  plot_quality(psnrs, inners_colors, "PSNR", figpath + "/psnr")




 
