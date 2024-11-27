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


if __name__ == "__main__":

# python plot-load-balancing.py data/load-balancing.json figures/load-balancing

  if len(sys.argv) < 3:
    print("Usage: python plot-time.py <data file> <fig folder>")
    sys.exit(1)


  datapath = sys.argv[1]
  figpath = sys.argv[2]
  with open(datapath, 'r') as file:
    plotdata = json.load(file)

  # Collect data
  nprocs = set()
  total_times = {}
  for approach in plotdata:
    total_times[approach] = {}
    traces = plotdata[approach]["traces"]
    for info in traces:
      nps = info["nprocs"]
      nprocs.add(nps)
      total_times[approach][nps] = info["total"]
  
  nprocs = sorted(list(nprocs), reverse=True)
  nprocs = nprocs[1:] # Collect data for failures only
  plot_total_times = {}
  for approach in plotdata:
    plot_times = []
    for nps in nprocs:
      if nps not in total_times[approach]:
        plot_times.append(0)
      else:
        plot_times.append(total_times[approach][nps])
    plot_total_times[approach] = plot_times
  
  # Plot data
  plt.rcParams['axes.labelsize'] = 16     # X and Y labels font size
  plt.rcParams['xtick.labelsize'] = 16    # X-axis tick labels font size
  plt.rcParams['ytick.labelsize'] = 16    # Y-axis tick labels font size
  plt.rcParams['legend.fontsize'] = 16

  plt.figure()
  width = 0.15
  x = np.arange(len(nprocs))
  m=0
  approaches = ["static"]
  for approach in approaches:
    print(plot_total_times[approach])
    plt.bar(x + width*m, plot_total_times[approach], width, facecolor="none", edgecolor=plotdata[approach]["color"], hatch="//", label=plotdata[approach]["label"])
    m += 1
  plt.xlabel("# Remaining Processes")
  plt.xticks(x, nprocs)
  plt.ylabel("Elapsed time (s)")
  # plt.yscale("log")
  
  # plt.legend(loc="best")
  plt.tight_layout()
  plt.savefig(figpath + "/load-balancing-scaling.png")
  plt.savefig(figpath + "/load-balancing-scaling.pdf")



 



 
