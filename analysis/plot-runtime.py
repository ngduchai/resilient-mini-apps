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
  fail_runtime_simul = np.random.exponential(scale=1/lamb, size=(10000, num_processes))
  fail_runtime = np.mean(np.min(fail_runtime_simul, 1))
  exp_runtime = runtime + (1 - reconstruction_success_prob)/reconstruction_success_prob*fail_runtime
  return exp_runtime

def exp_no_resilient_resume_runtime(lamb, num_processes, runtime):
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
  fail_runtime_simul = np.random.exponential(scale=1/lamb, size=(10000, num_processes))
  fail_runtime = np.mean(np.min(fail_runtime_simul, 1))
  exp_runtime = runtime + (1 - reconstruction_success_prob)/reconstruction_success_prob*fail_runtime
  return exp_runtime

def exp_resilient_runtime(lamb, num_processes, runtime):
  if lamb == 0:
    return runtime
  # Probability that a single worker/process complete before failure
  #   = Prb[failure interval > runtime] = 1 - Prob[failure interval <= runtime]
  #   = 1 - cdf(runtime) 
  process_success_prob = 1 - stats.expon.cdf(x=runtime, scale=1/lamb)
  # probability of success reconstruction
  #   = Prob[at least one processes complete before failure]
  #   = 1 - Prob[all processes failure before completion]
  reconstruction_success_prob = 1 - stats.binom.pmf(0, num_processes, process_success_prob)
  # exp_runtime = runtime * 1/reconstruction_success_prob
  # For simplicity, we assume the execution of failure reconstruction = MTTF
  fail_runtime = min(runtime, 1/lamb)
  exp_runtime = runtime + (1 - reconstruction_success_prob)/reconstruction_success_prob*fail_runtime
  return exp_runtime

def simulate_no_resilient_runtime(lamb, num_processes, runtime):
  ntries = 1000
  if lamb == 0:
    return runtime
  total_times = 0
  for n in range(ntries):
    num_exec = num_processes
    num_remain = num_processes
    acc_time = 0
    while num_remain > 0:
      fail_times = np.random.exponential(scale=1/lamb, size=num_exec)
      next_num_exec = 0
      complete = 0
      ptime = 0
      for i in range(num_exec):
        num_tasks = int(num_remain / num_exec)
        if i + num_tasks*num_exec < num_remain:
          num_tasks += 1
        actual_runtime = runtime * num_tasks
        if fail_times[i] > actual_runtime:
          complete += num_tasks
          next_num_exec += 1
        ptime = max(ptime, actual_runtime)
      acc_time += ptime
      num_remain -= complete
      if next_num_exec == 0:
        num_exec = num_remain
      
    total_times += acc_time
  return total_times / ntries

def simulate_no_resilient_resume_runtime(lamb, num_processes, runtime):
  ntries = 10
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
    total_tries += count
  act_runtime = total_tries / ntries * runtime
  return act_runtime

def simulate_resilient_runtime(lamb, num_processes, runtime):
  ntries = 10000
  total_tries = 0
  for i in range(ntries):
    count = 0
    while True:
      count += 1
      process_state = np.random.exponential(scale=1/lamb, size=num_processes)
      process_state = np.sum(np.where(process_state > runtime, 1, 0))
      if process_state >= 1:
        break
    total_tries += count
  act_runtime = total_tries / ntries * runtime
  return act_runtime


# Making a plot showing the checkpoint overhead varying input data size
def plot_overhead(data, probs, figpath, normalized_value=1):
  width = 0.15
  # plt.figure(figsize=(10, 6))
  plt.figure()
  m = -1.5
  total_times = []
  compute_times = []
  ckpt_times = []
  comm_times = []
  for prob in probs:
    for info in data["elapsed-time"]:
      if prob == info["prob"]:
        total_times.append(info["total"])
        compute_times.append(info["exec"])
        ckpt_times.append(info["ckpt"])
        comm_times.append(info["comm"])
        break
  
  compute_times = np.array(compute_times) / normalized_value
  ckpt_times = np.array(ckpt_times) / normalized_value
  comm_times = np.array(comm_times) / normalized_value
  
  x = np.arange(len(probs))
  
  plt.bar(x-width, compute_times, width, facecolor="none", edgecolor="green", hatch="//", label="Reconstruction")
  plt.bar(x, ckpt_times, width, facecolor="none", edgecolor="orange", hatch="*", label="Checkpointing")
  plt.bar(x+width, comm_times, width, facecolor="none", edgecolor="blue", hatch="\\", label="Communication")
  
  plt.xlabel("Mean Time to Failure (sec)")
  plt.xticks(np.arange(len(probs)), 1/np.array(probs))
  plt.ylabel("Normalized Elapsed Time")
  plt.yscale("log")
  plt.ylim((0, 100))
  plt.grid(True)
  
  plt.legend(loc="best")
  plt.tight_layout()
  plt.savefig(figpath + ".png")
  plt.savefig(figpath + ".pdf")

def plot_totaltime(data, probs, figpath, normalized_value=1):
  width = 0.15
  # plt.figure(figsize=(10, 6))
  plt.figure()
  m = -1.5
  approaches = ["no-resilient", "ckpt", "balance", "async"]
  # for approach in data:
  for approach in approaches:
    appconf = data[approach]
    appdata = data[approach]["elapsed-time"]
    total = []
    for prob in probs:
      for info in appdata:
        if prob == info["prob"]:
          total.append(info["total"])
          break
    x = np.arange(len(probs))
    total = np.array(total)
    plt.bar(x + width*m, total/normalized_value, width, facecolor="none", edgecolor=appconf["color"], hatch="//", label=appconf["label"])
    print(total/normalized_value)
    m += 1
  plt.xlabel("Mean Time to Failure (sec)")
  plt.xticks(np.arange(len(probs)), 1/np.array(probs))
  if normalized_value == 1:
    plt.ylabel("Reconstrucution Time (sec)")
    plt.ylim(1, 200000) # A year
  else:
    plt.ylabel("Normalized Reconstruction Time")
    # plt.ylim(1, 31536000) # A year
    plt.ylim(1, 200000) # A year
  plt.yscale("log")
  plt.grid(True)
  
  
  plt.legend(loc="best")
  plt.tight_layout()
  plt.savefig(figpath + ".png")
  plt.savefig(figpath + ".pdf")

if __name__ == "__main__":

  # python plot-runtime.py data/runtime-test.json figures/runtime
  
  # lamb = 0.002
  # num_processes = 64
  # runtime = 10
  # exp_runtime = exp_no_resilient_runtime(lamb, num_processes, runtime)
  # act_runtime = simulate_no_resilient_runtime(lamb, num_processes, runtime)
  # print(exp_runtime, act_runtime)
  # lamb = 0.1
  # num_processes = 64
  # runtime = 50
  # exp_runtime = exp_resilient_runtime(lamb, num_processes, runtime)
  # act_runtime = simulate_resilient_runtime(lamb, num_processes, runtime)
  # print(exp_runtime, act_runtime)


  if len(sys.argv) < 3:
    print("Usage: python plot-time.py <data file> <fig folder>")
    sys.exit(1)

  # probs = [0.0001, 0.001, 0.01, 0.1]
  # for prob in probs:
  #   num_process = 64
  #   ideal_runtime = 60
  #   actual_runtime = simulate_no_resilient_runtime(prob, num_process, ideal_runtime)
  #   print(num_process, prob, actual_runtime)
  # exit(0)


  datapath = sys.argv[1]
  figpath = sys.argv[2]
  with open(datapath, 'r') as file:
    plotdata = json.load(file)

  plt.rcParams['axes.labelsize'] = 16     # X and Y labels font size
  plt.rcParams['xtick.labelsize'] = 16    # X-axis tick labels font size
  plt.rcParams['ytick.labelsize'] = 16    # Y-axis tick labels font size
  plt.rcParams['legend.fontsize'] = 16
  
  # Calculate runtime for no resilient implementation
  no_resilience = {}
  no_resilience["label"] = "No Resilience"
  no_resilience["color"] = "orange"
  no_resilience["elapsed-time"] = []

  precalculated_runtimes = {
    0 : 60,
    0.0001 : 81.12,
    0.001 : 131.7,
    0.01 : 387.3,
    0.1 : 116109.54
  }

  ideal_exec_time = plotdata["baseline"]["total"]
  probs = [0, 0.0001, 0.001, 0.01, 0.1]
  for prob in probs:
    info = {}
    info["prob"] = prob
    # info["total"] = exp_no_resilient_runtime(prob, 64, ideal_exec_time)
    # info["total"] = simulate_no_resilient_resume_runtime(prob, 64, ideal_exec_time)
    info["total"] = precalculated_runtimes[prob]
    no_resilience["elapsed-time"].append(info)
  print(no_resilience)
  plotdata["approaches"]["no-resilient"] = no_resilience

  plot_totaltime(plotdata["approaches"], probs, figpath + "/elapsed-time-resilient")
  plot_overhead(plotdata["approaches"]["async"], probs, figpath + "/elapsed-time-breakdown", normalized_value=ideal_exec_time)

 



 
