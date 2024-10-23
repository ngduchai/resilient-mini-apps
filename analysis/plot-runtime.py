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
def plot_fig(data, xlab, ylab, figpath):
  width = 0.15
  plt.figure()
  lapp = None
  for approach in data:
    if lapp == None or len(data[approach]["elapsed-time"]) > len(data[lapp]["elapsed-time"]):
      lapp = approach
  probs = None
  m = 0
  for approach in data:
    appconf = data[approach]
    appdata = data[approach]["elapsed-time"]
    prob = []
    ckpt = []
    comm = []
    recovery = []
    total = []
    for info in appdata:
      prob.append(info["prob"])
      ckpt.append(info["ckpt"])
      recovery.append(info["recover"])
      comm.append(info["comm"])
      total.append(info["total"])
    if probs == None:
      probs = prob
    x = np.arange(len(prob))
    ckpt = np.array(ckpt)
    recovery = np.array(recovery)
    comm = np.array(comm) - recovery
    total = np.array(total)
    exectime = total - recovery - ckpt - comm
    # plt.bar(x + width*m, exectime, width, facecolor="none", edgecolor="appconf["color"]", hatch="//")
    # plt.bar(x + width*m, ckpt, width, bottom=exectime, facecolor="none", edgecolor=appconf["color"], hatch="*")
    # plt.bar(x + width*m, comm, width, bottom=exectime+ckpt, facecolor="none", edgecolor=appconf["color"], hatch="\\")
    # plt.bar(x + width*m, recovery, width, bottom=exectime+ckpt+comm, facecolor="none", edgecolor=appconf["color"], label=appconf["label"], hatch="||")
    plt.bar(x + width*m, exectime, width, facecolor="none", edgecolor="green", hatch="//", label="Reconstruction")
    plt.bar(x + width*m, ckpt, width, bottom=exectime, facecolor="none", edgecolor="orange", hatch="*", label="Checkpointing")
    plt.bar(x + width*m, comm, width, bottom=exectime+ckpt, facecolor="none", edgecolor="blue", hatch="\\", label="Sync")
    plt.bar(x + width*m, recovery, width, bottom=exectime+ckpt+comm, facecolor="none", edgecolor="purple", label=appconf["label"], hatch="||", "Recovery")
    m += 1
  plt.xlabel(xlab)
  plt.xticks(np.arange(len(probs)), probs)
  plt.ylabel(ylab)
  # plt.yscale("log")
  
  plt.legend(loc="best")
  plt.tight_layout()
  plt.savefig(figpath)

if __name__ == "__main__":
  
  if len(sys.argv) < 3:
    print("Usage: python plot-time.py <data file> <fig folder>")
    sys.exit(1)


  datapath = sys.argv[1]
  figpath = sys.argv[2]
  with open(datapath, 'r') as file:
    plotdata = json.load(file)

  plot_fig(plotdata["exp_failure"], "Failure Frequency (per sec)", "Elapsed time (s)", figpath + "elapsed-time.png")
  
  


 



 
