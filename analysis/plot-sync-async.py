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
    "sync" : "orange",
    "async" : "blue"
  }
  approach_failure_colors = {
    "sync" : "brown",
    "async" : "violet"
  }
  approach_failure_markers = {
    "sync" : "s",
    "async" : "o"
  }
  appraoch_hatches = {
    "sync" : "//",
    "async" : "o"
  }
  approach_labels = {
    "sync" : "Sync",
    "async" : "Async"
  }

  if len(sys.argv) < 3:
    print("Usage: python plot-imbalance-impact.py <data file> <fig folder>")
    sys.exit(1)

  datapath = sys.argv[1]
  figpath = sys.argv[2]
  with open(datapath, 'r') as file:
    plotdata = json.load(file)
 
  time_bw_failues = set()
  approaches = ["sync", "async"]

  total_times = {}
  total_failures = {}
  for approach in approaches:
    total_times[approach] = {}
    total_failures[approach] = {}
    for info in plotdata["approaches"][approach]:
      gap = info["failure_gaps"]
      time_bw_failues.add(gap)
      total_times[approach][gap] = info["total"]
      total_failures[approach][gap] = info["task_failures"]
  
  plot_times = {}
  plot_failures = {}
  time_bw_failues = sorted(list(time_bw_failues), reverse=True)
  for approach in total_times:
    plot_times[approach] = []
    plot_failures[approach] = []
    for t in time_bw_failues:
      if t in total_times[approach]:
        plot_times[approach].append(total_times[approach][t])
        plot_failures[approach].append(total_failures[approach][t])
        print(approach, t, total_times[approach][t], total_failures[approach][t])
      else:
        plot_times[approach].append(0)
        plot_failures[approach].append(0)
  
  # Plot
  plt.figure()
  fig, axtime = plt.subplots()
  m = -0.5
  width = 0.25

  # plt.xlabel("Time Between Failures (sec)")
  # plt.xticks(np.arange(len(time_bw_failues)), time_bw_failues)
  # ideal_runtime = plotdata["baseline"]["total"]
  # plt.xlabel("# Failures")
  # plt.xticks(np.arange(len(time_bw_failues)), np.round(ideal_runtime / np.array(time_bw_failues), 0))
  # plt.yscale("log")
  axtime.set_xlabel("Time Between Failures (sec)")
  
  axfailure = axtime.twinx()

  axtime.set_ylabel("Elapsed Time (sec)")
  axfailure.set_ylabel("# Failures")

  for approach in approaches:
    x = np.arange(len(time_bw_failues))
    axtime.bar(x + width*m, plot_times[approach], width=width, facecolor="none", edgecolor=approach_colors[approach], hatch=appraoch_hatches[approach], label=approach_labels[approach] + " (Time)")
    axfailure.plot(x, plot_failures[approach], color=approach_failure_colors[approach], label=approach_labels[approach] + " (# Failures)", marker=approach_failure_markers[approach])
    m += 1
  
  plt.xticks(x, time_bw_failues)
  
  # plt.ylabel("Elapsed time (s)")
  
  fig.legend(bbox_to_anchor=[0.15, 0.95], loc="upper left")
  plt.tight_layout()
  figname = "/sync-vs-async"
  plt.savefig(figpath + figname + ".png")
  plt.savefig(figpath + figname + ".pdf")



 
