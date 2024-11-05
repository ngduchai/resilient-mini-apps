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

def exp_no_resilient_runtime(lamb, num_processes, runtime):
  if lamb == 0:
    return runtime
  # Probability that a single worker/process complete before failure
  #   = Prb[failure interval > runtime] = 1 - Prob[failure interval <= runtime]
  #   = 1 - cdf(runtime) 
  process_success_prob = 1 - stats.expon.cdf(x=runtime, scale=1/lamb)
  # probability of success reconstruction = Prob[all processes complete before failure] 
  reconstruction_success_prob = stats.binom.pmf(num_processes, num_processes, process_success_prob)
  # exp_runtime = runtime * 1/reconstruction_success_prob
  # For simplicity, we assume the execution of failure reconstruction = MTTF
  fail_runtime = min(runtime, 1/lamb)
  exp_runtime = runtime + (1 - reconstruction_success_prob)/reconstruction_success_prob*fail_runtime
  return exp_runtime
  

if __name__ == "__main__":

  figpath = "figures/runtime/est-runtime-no-resilient"

  plt.rcParams['axes.labelsize'] = 16     # X and Y labels font size
  plt.rcParams['xtick.labelsize'] = 16    # X-axis tick labels font size
  plt.rcParams['ytick.labelsize'] = 16    # Y-axis tick labels font size
  plt.rcParams['legend.fontsize'] = 16

  # mttf = 5.55555556e-8 # 1 per 5k hours
  mttf = 1/(30*24*3600) # Assume a process can run for 1 month without failure
  resolutions = ["640x640", "2000x2000"]
  per_slice_runtime = {
    "640x640" : 6,
    "2000x2000" : 58.59375
  }
  num_iter = 200
  num_slices = 2048

  # num_processes = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048]
  num_processes = [1, 4, 16, 64, 256, 1024]

  serial_runtimes = {}
  exp_runtimes = {}
  ideal_runtimes = {}
  for resolution in resolutions:
    serial_runtimes[resolution] = per_slice_runtime[resolution] * num_slices * num_iter
    exp_runtimes[resolution] = []
    ideal_runtimes[resolution] = []

  for num_process in num_processes:
    slice_per_process = num_slices / num_process
    for resolution in resolutions:
      ideal_runtime = per_slice_runtime[resolution] * slice_per_process * num_iter
      ideal_runtimes[resolution].append(ideal_runtime / serial_runtimes[resolution])
      exp_runtime = exp_no_resilient_runtime(mttf, num_process, ideal_runtime)
      exp_runtimes[resolution].append(exp_runtime / serial_runtimes[resolution])
  
  colors = {
    "640x640" : "orange",
    "2000x2000" : "blue"
  }

  width = 0.15
  plt.figure()
  m = -0.5
  x = np.array(range(len(num_processes)))
  for resolution in resolutions:
    # plt.bar(x + width*m, exp_runtimes[resolution], width, facecolor="none", edgecolor=colors[resolution], hatch="//", label=resolution + " (expected)")
    # plt.plot(x, ideal_runtimes[resolution], color=colors[resolution], label=resolution + " (ideal)")
    plt.bar(x + width*m, exp_runtimes[resolution], width, facecolor="none", edgecolor=colors[resolution], hatch="//", label=resolution)
    m += 1
  plt.plot(x, ideal_runtimes[resolution], color="green", marker="o", label="Ideal")
  plt.xlabel("Number of processes")
  plt.xticks(x, num_processes)
  plt.ylabel("Normalized Reconstruction Time")
  plt.yscale("log")
  plt.grid(which='major', color='black', linestyle='-', zorder=-1)   # Major grid
  plt.grid(which='minor', color='gray', linestyle='--', linewidth=0.5, zorder=-1)   # Minor grid


  plt.legend(loc="lower left")
  plt.tight_layout()
  plt.savefig(figpath + ".png")
  plt.savefig(figpath + ".pdf")


 



 
