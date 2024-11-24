import sewar.full_ref as metrics_cal
from skimage.exposure import adjust_gamma, rescale_intensity
import h5py as h5
import numpy as np
import sys
import json

import seaborn as sns

import pickle

import matplotlib.pyplot as plt

import os

def make_index(data):
  index = {}
  for i in range(len(data)):
    index[data[i]] = i
  return index

def draw_heatmap(data, xticks, yticks, figpath):
  plt.figure()
  # plt.imshow(data, cmap="autumn", vmin=0, vmax=100, extent=[0, 8, 0, 8])
  # for i in range(8): 
  #   for j in range(8): 
  #       plt.annotate(str(data[i][j]), xy=(j+0.5, i+0.5), ha='center', va='center', color='white')
  # plt.colorbar()

  # plt.xticks(range(len(xticks)), xticks)
  # plt.yticks(range(len(yticks)), yticks)

  # plt.xlabel("Time of Failure")
  # plt.ylabel("Fraction of Failure Processes")

  hm = sns.heatmap(
    data=data,
    annot=True,
    fmt=".2f",
    cmap="cool",
    vmin=0, vmax=2.5,
    xticklabels=xticks,
    yticklabels=yticks)

  plt.xlabel("Progress when failures happen")
  plt.ylabel("Fraction of failure tasks")

  plt.tight_layout()
  plt.savefig(figpath + ".png")
  plt.savefig(figpath + ".pdf")


# Making a plot showing the checkpoint overhead varying input data size
def plot_fig(data, baseline, figpath):
  # failure_iters = [9, 8, 7, 6, 5, 4, 3, 2, 1, 0]
  # failure_iter_ticklabels = [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0]
  # failure_ratios = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
  failure_iters = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
  failure_iter_ticklabels = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
  failure_ratios = [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]
  iter_indexes = make_index(failure_iters)
  ratio_indxes = make_index(failure_ratios)

  ckpt_times = np.zeros([len(failure_ratios), len(failure_iters)])
  sync_times = np.zeros([len(failure_ratios), len(failure_iters)])
  compute_times = np.zeros([len(failure_ratios), len(failure_iters)])
  total_times = np.zeros([len(failure_ratios), len(failure_iters)])

  for exp in data:
    fiter = iter_indexes[exp["failure_iter"]]
    fratio = ratio_indxes[exp["failure_ratio"]]
    ckpt_times[fratio][fiter] = exp["ckpt"] / baseline["exec"]
    sync_times[fratio][fiter] = exp["comm"] / baseline["exec"]
    # compute_times[fratio][fiter] = (exp["exec"] - baseline["exec"]) / baseline["exec"]
    compute_times[fratio][fiter] = (exp["total"] - exp["ckpt"] - exp["comm"] - baseline["exec"]) / baseline["exec"]
    total_times[fratio][fiter] = (exp["total"] - baseline["exec"]) / baseline["exec"]

  
  draw_heatmap(ckpt_times, failure_iter_ticklabels, failure_ratios, figpath + "-ckpt_overhead")
  draw_heatmap(sync_times, failure_iter_ticklabels, failure_ratios, figpath + "-sync_overhead")
  draw_heatmap(compute_times, failure_iter_ticklabels, failure_ratios, figpath + "-exec_overhead")
  draw_heatmap(total_times, failure_iter_ticklabels, failure_ratios, figpath + "-total_overhead")



if __name__ == "__main__":
  
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

  baseline = plotdata["baseline"]

  plot_fig(plotdata["ckpt"], baseline, figpath + "/ckpt")
  plot_fig(plotdata["no-resilient"], baseline, figpath + "/no-resilient")
  
  


 



 
