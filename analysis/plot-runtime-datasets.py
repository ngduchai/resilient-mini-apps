import sewar.full_ref as metrics_cal
from skimage.exposure import adjust_gamma, rescale_intensity
import h5py as h5
import numpy as np
import sys
import json

import pickle

import matplotlib.pyplot as plt

import os

import scipy.stats as stats

def plot_totaltime(data, probs, figpath):
  width = 0.15
  # plt.figure(figsize=(10, 6))
  plt.figure()
  m = -1
  tomos = ["shale", "chip"]

  normalized_value = {}
  for tomo in tomos:
    found_normalized_value = False
    for info in data[tomo]["elapsed-time"]:
      if info["prob"] == 0:
        normalized_value[tomo] = info["total"]
        found_normalized_value = True
        break
    if not found_normalized_value:
      normalized_value[tomo] = 1

  # for approach in data:
  for tomo in tomos:
    appconf = data[tomo]
    appdata = data[tomo]["elapsed-time"]
    total = []
    for prob in probs:
      found_prob = False
      for info in appdata:
        if prob == info["prob"]:
          total.append(info["total"] / normalized_value[tomo])
          found_prob = True
          break
      if not found_prob:
        total.append(0)
    x = np.arange(len(probs))
    total = np.array(total)
    plt.bar(x + width*(m+0.5), total, width, facecolor="none", edgecolor=appconf["color"], hatch="//", label=appconf["label"])
    print(total)
    m += 1
  plt.xlabel("Mean Time to Failure (sec)")
  # plt.xticks(np.arange(len(probs)), 1/np.array(probs))
  plt.xticks(np.arange(len(probs)), ["$\infty$" if prob == 0 else int(round(1/prob)) for prob in probs])
  
  # plt.ylabel("Reconstrucution Time (sec)")
  plt.ylabel("Normalized Reconstruction Time")
  # plt.ylim(1, 31536000) # A year
  # plt.ylim(1, 200000) # A year
  # plt.ylim(1, 25000) # A year
  plt.ylim(0.1, 100) # A year
  plt.yscale("log")
  plt.grid(True)
  
  
  plt.legend(loc="best")
  plt.tight_layout()
  plt.savefig(figpath + ".png")
  plt.savefig(figpath + ".pdf")

if __name__ == "__main__":

  # python plot-runtime-datasets.py data/execinfo-runtimes-datasets.json figures/runtime
  


  if len(sys.argv) < 3:
    print("Usage: python plot-time.py <data file> <fig folder>")
    sys.exit(1)

  datapath = sys.argv[1]
  figpath = sys.argv[2]
  with open(datapath, 'r') as file:
    plotdata = json.load(file)

  plt.rcParams['axes.labelsize'] = 16     # X and Y labels font size
  plt.rcParams['xtick.labelsize'] = 16    # X-axis tick labels font size
  plt.rcParams['ytick.labelsize'] = 16    # Y-axis tick labels font size
  plt.rcParams['legend.fontsize'] = 16

  probs = [0, 0.00001, 0.0001, 0.001, 0.01]
  # probs = [0, 0.00001, 0.0001, 0.001]
  

  plot_totaltime(plotdata["datasets"], probs, figpath + "/elapsed-time-resilient-datasets")

 



 
