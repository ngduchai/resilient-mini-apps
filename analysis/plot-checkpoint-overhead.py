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
def plot_time(data, expname, xlab, ylab, figpath):
  width = 0.15
  plt.figure()
  lapp = None
  for approach in data:
    if lapp == None or len(data[approach][expname]["total"]) > len(data[lapp][expname]["total"]):
      lapp = approach
  m = 0
  for approach in data:
    appconf = data[approach]
    appdata = data[approach][expname]
    x = np.arange(len(appdata["total"]))
    overhead = np.array(list(appdata["overhead"].values()))
    exectime = np.array(list(appdata["total"].values())) - overhead
    plt.bar(x + width*m, exectime, width, facecolor="none", edgecolor=appconf["color"], hatch="//")
    exectime = np.zeros(len(x))
    plt.bar(x + width*m, overhead, width, bottom=exectime, color=appconf["color"], label=appconf["label"])
    m += 1
  plt.xlabel(xlab)
  plt.xticks(np.arange(len(data[lapp][expname]["total"])), list(data[lapp][expname]["overhead"].keys()))
  plt.ylabel(ylab)
  plt.ylabel("Overhead (s)")
  # plt.yscale("log")
  
  plt.legend(loc="best")
  plt.tight_layout()
  plt.savefig(figpath)

if __name__ == "__main__":
  
  if len(sys.argv) < 3:
    print("Usage: python plot-checkpoint-overhead.py <data file> <fig folder>")
    sys.exit(1)


  datapath = sys.argv[1]
  figpath = sys.argv[2]
  with open(datapath, 'r') as file:
    plotdata = json.load(file)

  plot_time(plotdata, "varying-tasks", "# processes", "Time (s)", figpath + "varying-tasks.png")
  plot_time(plotdata, "varying-data-per-task", "# slices", "Time (s)", figpath + "varying-slices")
  plot_time(plotdata, "drop-slices-64", "Fraction of dropped data", "Time (s)", figpath + "varying-drop-64")
  plot_time(plotdata, "drop-slices-32", "Fraction of dropped data", "Time (s)", figpath + "varying-drop-32")
  


 
