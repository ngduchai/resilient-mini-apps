import sewar.full_ref as metrics_cal
from skimage.exposure import adjust_gamma, rescale_intensity
import h5py as h5
import numpy as np
import sys
import json

import pickle

import matplotlib.pyplot as plt

import os


# Making a plot showing the checkpoint overhead varying input data size


if __name__ == "__main__":
  
  if len(sys.argv) < 3:
    print("Usage: python plot-checkpoint-overhead.py <data file> <fig folder>")
    sys.exit(1)


  datapath = sys.argv[1]
  figpath = sys.argv[2]
  with open(datapath, 'r') as file:
    plotdata = json.load(file)
  
  width = 0.15
  plt.figure()
  lapp = None
  for approach in plotdata:
    if lapp == None or len(plotdata[approach]["overhead"]) > len(plotdata[lapp]["overhead"]):
      lapp = approach
  m = 0
  for approach in plotdata:
    appdata = plotdata[approach]
    x = np.arange(len(appdata["overhead"]))
    plt.bar(x + width*m, np.array(list(appdata["overhead"].values())), width, color=appdata["color"], label=appdata["label"])
    m += 1
  plt.xlabel("# slices")
  plt.xticks(np.arange(len(plotdata[lapp]["overhead"])), list(plotdata[lapp]["overhead"].keys()))
  plt.ylabel("Elapsed time (s)")
  
  plt.legend(loc="best")
  plt.tight_layout()
  plt.savefig(figpath + "elapsed-time.png")



 
