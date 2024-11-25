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
import scipy.integrate as integrate
import math

if __name__ == "__main__":

  plt.rcParams['axes.labelsize'] = 16     # X and Y labels font size
  plt.rcParams['xtick.labelsize'] = 16    # X-axis tick labels font size
  plt.rcParams['ytick.labelsize'] = 16    # Y-axis tick labels font size
  plt.rcParams['legend.fontsize'] = 16
  
  approach_colors = {
    "naive" : "orange",
    "balance-aware" : "blue"
  }
  appraoch_hatches = {
    "naive" : "//",
    "balance-aware" : "o"
  }
  approach_labels = {
    "naive" : "Naive",
    "balance-aware" : "Balance-Aware"
  }

  if len(sys.argv) < 3:
    print("Usage: python plot-sync-async.py <data file> <fig folder>")
    sys.exit(1)

  datapath = sys.argv[1]
  figpath = sys.argv[2]
  with open(datapath, 'r') as file:
    plotdata = json.load(file)
 
  failure_ratios = set()
  approaches = ["naive", "balance-aware"]

  total_times = {}
  for approach in approaches:
    total_times[approach] = {}
    for info in plotdata["approaches"][approach]:
      fr = info["failure_ratio"]
      failure_ratios.add(fr)
      total_times[approach][fr] = info["total"]
  
  plot_times = {}
  failure_ratios = sorted(list(failure_ratios))
  for approach in total_times:
    plot_times[approach] = []
    for fr in failure_ratios:
      if fr in total_times[approach]:
        plot_times[approach].append(total_times[approach][fr])
        print(approach, fr, total_times[approach][fr])
      else:
        plot_times[approach].append(0)
  
  # Plot
  plt.figure()
  m = -0.5
  width = 0.25
  for approach in approaches:
    x = np.arange(len(failure_ratios))
    plt.bar(x + width*m, plot_times[approach], width=width, facecolor="none", edgecolor=approach_colors[approach], hatch=appraoch_hatches[approach], label=approach_labels[approach])
    m += 1
  
  ideal_runtime = plotdata["baseline"]["total"]
  print(failure_ratios)
  # plt.xlabel("Time Between Failures / Ideal Runtime")
  # plt.xticks(np.arange(len(time_bw_failues)), np.round(np.array(time_bw_failues) / ideal_runtime, 1))
  plt.xlabel("Failure Ratio")
  plt.xticks(np.arange(len(failure_ratios)), failure_ratios)
  plt.ylabel("Elapsed time (s)")
  # plt.yscale("log")
  
  plt.legend(loc="best")
  plt.tight_layout()
  figname = "/runtime-vs-failure-ratios"
  plt.savefig(figpath + figname + ".png")
  plt.savefig(figpath + figname + ".pdf")



 
