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
  
def simulate_no_resilient_resume_runtime(lamb, num_processes, runtime):
  # if num_processes == 1:
  #   return 10000*runtime
  ntries = 1000
  total_tries = 0
  if lamb == 0:
    lamb = 0.00000000001
  for i in range(ntries):
    num_exec = num_processes
    count = 0
    while num_exec > 0:
      count += 1
      process_state = np.random.exponential(scale=1/lamb, size=num_exec)
      finished = np.sum(np.where(process_state > runtime, 1, 0))
      num_exec -= finished
      if count > 10000:
        break
    total_tries += count
  act_runtime = total_tries / ntries * runtime
  print(num_processes, act_runtime)
  return act_runtime

if __name__ == "__main__":

  plt.rcParams['axes.labelsize'] = 16     # X and Y labels font size
  plt.rcParams['xtick.labelsize'] = 16    # X-axis tick labels font size
  plt.rcParams['ytick.labelsize'] = 16    # Y-axis tick labels font size
  plt.rcParams['legend.fontsize'] = 16

  resolutions = ["640x640", "2000x2000"]
  per_slice_runtime = {
    "640x640" : 6,
    "2000x2000" : 58.59375
  }
  colors = {
    "640x640" : "orange",
    "2000x2000" : "blue"
  }

  print("Plot runtime vs. mean time between failures")

  figpath = "figures/runtime/est-runtime-no-resilient-mttf"
  probs = [0, 0.0001, 0.001, 0.01, 0.1]
  num_process = 64
  num_slices = 64
  num_iter = 10
  slice_per_process = num_slices / num_process
  exp_runtimes = {}
  for resolution in resolutions:
    exp_runtimes[resolution] = []
    for prob in probs:
      ideal_runtime = per_slice_runtime[resolution] * slice_per_process * num_iter
      exp_runtime = simulate_no_resilient_resume_runtime(prob, num_process, ideal_runtime)
      exp_runtimes[resolution].append(exp_runtime)
  
  width = 0.15
  plt.figure()
  m = -0.5
  x = np.array(range(len(probs)))
  for resolution in resolutions:
    plt.bar(x + width*m, exp_runtimes[resolution], width, facecolor="none", edgecolor=colors[resolution], hatch="//", label=resolution)
    m += 1
    print(resolution, exp_runtimes[resolution])
  plt.xlabel("Mean time to failures")
  plt.xticks(np.arange(len(probs)), 1/np.array(probs))
  # plt.ylabel("Normalized Reconstruction Time")
  plt.ylabel("Reconstruction Time (sec)")
  plt.yscale("log")
  # plt.grid(which='major', color='black', linestyle='-', zorder=-1)   # Major grid
  # plt.grid(which='minor', color='gray', linestyle='--', linewidth=0.5, zorder=-1)   # Minor grid


  # plt.legend(loc="lower left")
  plt.tight_layout()
  plt.savefig(figpath + ".png")
  plt.savefig(figpath + ".pdf")

  print("Plot runtime vs. num processes")

  figpath = "figures/runtime/est-runtime-no-resilient-np"
  # mttf = 5.55555556e-8 # 1 per 5k hours
  # mttf = 1/(30*24*3600) # Assume a process can run for 1 month without failure
  mttf = 1/(24*3600) # Assume a process can run for 1 day without failure
  # mttf = 1/(365*24*3600) # Assume a process can run for 1 year without failure
  num_iter = 200
  num_slices = 2048
  # num_processes = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048]
  num_processes = [1, 4, 16, 64, 256, 1024]

  serial_runtimes = {}
  exp_runtimes = {}
  ideal_runtimes = {}
  for resolution in resolutions:
    ideal_runtime = per_slice_runtime[resolution] * slice_per_process * num_iter
    serial_runtimes[resolution] = per_slice_runtime[resolution] * num_slices * num_iter
    exp_runtimes[resolution] = []
    ideal_runtimes[resolution] = []

  for num_process in num_processes:
    slice_per_process = num_slices / num_process
    for resolution in resolutions:
      ideal_runtime = per_slice_runtime[resolution] * slice_per_process * num_iter
      # ideal_runtimes[resolution].append(ideal_runtime / serial_runtimes[resolution])
      ideal_runtimes[resolution].append(ideal_runtime)
      # exp_runtime = exp_no_resilient_runtime(mttf, num_process, ideal_runtime)
      exp_runtime = simulate_no_resilient_resume_runtime(mttf, num_process, ideal_runtime)
      # exp_runtimes[resolution].append(exp_runtime / serial_runtimes[resolution])
      exp_runtimes[resolution].append(exp_runtime)

  width = 0.15
  plt.figure()
  m = -0.5
  x = np.array(range(len(num_processes)))
  for resolution in resolutions:
    # plt.bar(x + width*m, exp_runtimes[resolution], width, facecolor="none", edgecolor=colors[resolution], hatch="//", label=resolution + " (expected)")
    # plt.plot(x, ideal_runtimes[resolution], color=colors[resolution], label=resolution + " (ideal)")
    plt.bar(x + width*m, exp_runtimes[resolution], width, facecolor="none", edgecolor=colors[resolution], hatch="//", label=resolution)
    m += 1
    print(resolution, exp_runtimes[resolution])
  plt.plot(x, ideal_runtimes[resolution], color="green", marker="o", label="Ideal")
  plt.xlabel("Number of processes")
  plt.xticks(x, num_processes)
  # plt.ylabel("Normalized Reconstruction Time")
  plt.ylabel("Reconstruction Time (sec)")
  plt.yscale("log")
  # plt.grid(which='major', color='black', linestyle='-', zorder=-1)   # Major grid
  # plt.grid(which='minor', color='gray', linestyle='--', linewidth=0.5, zorder=-1)   # Minor grid


  # plt.legend(loc="lower left")
  plt.tight_layout()
  plt.savefig(figpath + ".png")
  plt.savefig(figpath + ".pdf")


 



 
