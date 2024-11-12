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

def plot_profiling(data, xlabl, figname):
  plt.figure()
  width = 0.15
  x = np.arange(len(nslices_per_proc))
  plt.bar(x, data, width)
  plt.xlabel(xlabl)
  plt.xticks(x, nslices_per_proc)
  plt.ylabel("Elapsed time (s)")
  # plt.yscale("log")
  
  plt.legend(loc="best")
  plt.tight_layout()
  plt.savefig(figpath + "/" + figname + ".png")
  plt.savefig(figpath + "/" + figname + ".pdf")

if __name__ == "__main__":

  if len(sys.argv) < 3:
    print("Usage: python plot-time.py <data file> <fig folder>")
    sys.exit(1)


  datapath = sys.argv[1]
  figpath = sys.argv[2]
  with open(datapath, 'r') as file:
    plotdata = json.load(file)

  # Plot data
  plt.rcParams['axes.labelsize'] = 16     # X and Y labels font size
  plt.rcParams['xtick.labelsize'] = 16    # X-axis tick labels font size
  plt.rcParams['ytick.labelsize'] = 16    # Y-axis tick labels font size
  plt.rcParams['legend.fontsize'] = 16

  # Collect data for varying slices
  nslices_per_proc = set()
  total_times = {}
  for info in plotdata["vary-slices"]:
    ns = int(info["nslices"] / info["nprocs"])
    nslices_per_proc.add(ns)
    total_times[ns] = info["exec"]
  
  nslices_per_proc = sorted(list(nslices_per_proc))
  plot_total_times = []
  for ns in nslices_per_proc:
    if ns not in total_times:
      plot_total_times.append(0)
    else:
      plot_total_times.append(total_times[ns])
  print(plot_total_times)

  plot_profiling(plot_total_times, "# slices per process", "slices-per-proc")


  # Collect data for varying iter
  num_iters = set()
  total_times = {}
  for info in plotdata["vary-iter"]:
    ni = info["num_iter"]
    num_iters.add(ni)
    total_times[ni] = info["exec"]
  
  num_iters = sorted(list(num_iters))
  plot_total_times = []
  for ni in num_iters:
    if ni not in total_times:
      plot_total_times.append(0)
    else:
      plot_total_times.append(total_times[ni])
  print(plot_total_times)
  plot_profiling(plot_total_times, "# iterations", "iterations")


 



 
