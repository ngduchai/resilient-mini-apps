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

def waste_est(lamb, runtime):
  p = stats.expon.cdf(x=runtime, scale=1/lamb)
  W = 1/p * integrate.quad(lambda x: x*stats.expon.pdf(x, scale=1/lamb), 0, runtime)[0]
  return W

def waste_sim(lamb, runtime):
  ntries = 10000000
  fail_times = np.random.exponential(scale=1/lamb, size=ntries)
  total_time = 0
  total_count = 0
  for ftime in fail_times:
    if ftime < runtime:
      total_time += ftime
      total_count += 1
  return total_time / total_count


def exp_no_resilient_runtime(lamb, num_processes, runtime, restart_overhead=0):
  if lamb == 0:
    return runtime
  
  p = stats.expon.cdf(x=runtime, scale=1/lamb)
  W = waste_est(lamb, runtime)
  En = np.zeros(num_processes+1)
  for n in range(1, num_processes+1):
    En[n] = stats.binom.pmf(0, n, p)*runtime + stats.binom.pmf(n, n, p)*(W + restart_overhead)
    for i in range(1, n-1):
      En[n] += stats.binom.pmf(i, n, p)*(W + En[min(i, n-i)])
    if 1-stats.binom.pmf(n, n, p) == 0:
      En[n] /= 0.000000000001
    else:
      En[n] /= 1-stats.binom.pmf(n, n, p)

  return En[num_processes]

  # # Probability that a single worker/process complete before failure
  # #   = Prb[failure interval > runtime] = 1 - Prob[failure interval <= runtime]
  # #   = 1 - cdf(runtime) 
  # process_success_prob = 1 - stats.expon.cdf(x=runtime, scale=1/lamb)
  # # probability of success reconstruction = Prob[all processes complete before failure] 
  # reconstruction_success_prob = stats.binom.pmf(num_processes, num_processes, process_success_prob)
  # # exp_runtime = runtime * 1/reconstruction_success_prob
  # # For simplicity, we assume the execution of failure reconstruction = MTTF
  # fail_runtime = min(runtime, 1/lamb)
  # exp_runtime = runtime + (1 - reconstruction_success_prob)/reconstruction_success_prob*fail_runtime
  # return exp_runtime
  
def simulate_no_resilient_runtime(lamb, num_processes, runtime):
  # if num_processes == 1:
  #   return 10000*runtime
  ntries = 1000
  # total_tries = 0
  # if lamb == 0:
  #   lamb = 0.00000000001
  # for i in range(ntries):
  #   num_exec = num_processes
  #   count = 0
  #   while num_exec > 0:
  #     count += 1
  #     process_state = np.random.exponential(scale=1/lamb, size=num_exec)
  #     finished = np.sum(np.where(process_state > runtime, 1, 0))
  #     num_exec -= finished
  #     if count > 10000:
  #       break
  #   total_tries += count
  # act_runtime = total_tries / ntries * runtime
  # print(num_processes, act_runtime)
  # return act_runtime
  if lamb == 0:
    return runtime
  total_times = 0
  for n in range(ntries):
    num_exec = num_processes
    acc_times = np.zeros(num_exec)
    completed = np.zeros(num_exec)
    while num_exec > 0:
      fail_times = np.random.exponential(scale=1/lamb, size=num_processes)
      for i in range(num_processes):
        if completed[i] == 0:
          if fail_times[i] > runtime:
            completed[i] = 1
            acc_times[i] += runtime
            num_exec -= 1
          else:
            acc_times[i] += fail_times[i]
      if np.max(acc_times) > 10000000:
        break
    total_times += np.max(acc_times)
    # total_times += np.mean(acc_times)
  return total_times / ntries
        



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
  nprocs_colors = {
    1 : "blue",
    4 : "orange",
    64 : "green",
    256 : "purple"
  }
  probs_colors = {
    0 : "brown",
    0.0001 : "blue",
    0.001 : "green",
    0.01 : "orange",
    0.1: "purple"
  }
  probs_hatches = {
    0 : "//",
    0.0001 : "o",
    0.001 : "\\\\",
    0.01 : "+",
    0.1: "||"
  }

  # UNCOMMENT THE LINE BELOW IF CHANGES ARE MADE IN waste_est/waste_sim
  # Test waste estimation -- PASS
  # for niter in [1, 2, 4, 8, 16]:
  #   for lamb in [0.0001, 0.001, 0.01, 0.1]:
  #     runtime = niter * per_slice_runtime["640x640"]
  #     print(niter, lamb, waste_est(lamb, runtime), waste_sim(lamb, runtime))
  # exit(0)

  # # Test simulation and calculation
  # for num_proc in [1, 4, 16, 64]:
  #   for lamb in [0.0001, 0.001, 0.01, 0.1]:
  #     runtime = per_slice_runtime["640x640"] * 10
  #     print(num_proc, lamb, exp_no_resilient_runtime(lamb, num_proc, runtime), simulate_no_resilient_runtime(lamb, num_proc, runtime))
  # exit(0)

  print("Plot runtime vs. mean time between failures varying # procs")

  precalculated_runtimes = {}
  precalculated_runtimes[0] = [15360.0, 3840.0, 960.0, 240.0, 60.0]
  precalculated_runtimes[0.0001] = [36208.84967684865, 6409.166514339338, 1487.30206372819, 361.4484777338541, 89.51053757096128]
  precalculated_runtimes[0.001] = [4637158899.095492, 89681.69285024096, 3778.7936558865404, 590.5789751935984, 125.95471764301672]
  precalculated_runtimes[0.01] = [10000100.711195493, 10000100.62457155, 5058946.069681379, 4061.0682538866236, 277.7976830045413]

  figpath = "figures/runtime/est-runtime-vs-mttf-varying-nprocs"
  probs = [0, 0.0001, 0.001, 0.01, 0.1]
  num_processes = [1, 4, 16, 64, 256]
  num_slices = 256
  num_iter = 10
  resolution = "640x640"
  exp_runtimes = {}
  ideal_runtimes = {}
  for num_process in num_processes:
    slice_per_process = num_slices / num_process
    ideal_runtimes[num_process] = per_slice_runtime[resolution] * slice_per_process * num_iter
  for prob in probs:
    if prob in precalculated_runtimes:
      exp_runtimes[prob] = precalculated_runtimes[prob]
    else:
      exp_runtimes[prob] = []
      for num_process in num_processes:
        ideal_runtime = ideal_runtimes[num_process]
        exp_runtime = simulate_no_resilient_runtime(prob, num_process, ideal_runtime)
        # exp_runtime = exp_no_resilient_runtime(prob, num_process, ideal_runtime)
        exp_runtimes[prob].append(exp_runtime)
    for i in range(len(num_processes)):
      print(prob, num_processes[i], exp_runtimes[prob][i])
  
  width = 0.15
  plt.figure()
  m = -1.5
  x = np.array(range(len(num_processes)))
  for prob in probs:
    plabel = "No Failure"
    if prob != 0:
      plabel = "MTTF = " + str(1/prob) + "s"
    plt.bar(x + width*m, exp_runtimes[prob], width, facecolor="none", edgecolor=probs_colors[prob], hatch=probs_hatches[prob], label=plabel)
    m += 1
    print(prob, exp_runtimes[prob])
  plt.plot(x, [ideal_runtimes[nproc] for nproc in num_processes], color="black", marker="o", label="Ideal Recon. Time")
  plt.xlabel("Number of Reconstruction Tasks")
  plt.xticks(x, num_processes)
  # plt.ylabel("Normalized Reconstruction Time")
  plt.ylabel("Reconstruction Time (sec)")
  plt.ylim(1, 10000000000)
  plt.yscale("log")
  # plt.grid(which='major', color='black', linestyle='-', zorder=-1)   # Major grid
  # plt.grid(which='minor', color='gray', linestyle='--', linewidth=0.5, zorder=-1)   # Minor grid


  plt.legend(loc="upper right")
  plt.tight_layout()
  plt.savefig(figpath + ".png")
  plt.savefig(figpath + ".pdf")

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
      exp_runtime = simulate_no_resilient_runtime(prob, num_process, ideal_runtime)
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
      exp_runtime = simulate_no_resilient_runtime(mttf, num_process, ideal_runtime)
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


 



 
