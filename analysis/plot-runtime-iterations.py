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
  ntries = 10000
  total_tries = 0
  for i in range(ntries):
    count = 0
    while True:
      count += 1
      process_state = np.random.exponential(scale=1/lamb, size=num_processes)
      process_state = np.sum(np.where(process_state > runtime, 1, 0))
      if process_state == num_processes:
        break
    total_tries += count
  act_runtime = total_tries / ntries * runtime
  return act_runtime

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
def plot_fig(data, probs, figpath):
  width = 0.15
  # plt.figure(figsize=(10, 6))
  plt.figure()
  lapp = None
  for approach in data:
    if lapp == None or len(data[approach]["elapsed-time"]) > len(data[lapp]["elapsed-time"]):
      lapp = approach
  m = 0
  for approach in data:
    appconf = data[approach]
    appdata = data[approach]["elapsed-time"]
    ckpt = []
    comm = []
    recovery = []
    exectime = []
    total = []
    for prob in probs:
      for info in appdata:
        if prob == info["prob"]:
          ckpt.append(info["ckpt"])
          recovery.append(info["recover"])
          comm.append(info["comm"])
          exectime.append(info["exec"])
          total.append(info["total"])
    x = np.arange(len(probs))
    ckpt = np.array(ckpt)
    recovery = np.array(recovery)
    # comm = np.array(comm) - recovery
    comm = np.array(comm)
    total = np.array(total)
    # exectime = total - recovery - ckpt - comm
    exectime = np.array(exectime)
    # plt.bar(x + width*m, exectime, width, facecolor="none", edgecolor="appconf["color"]", hatch="//")
    # plt.bar(x + width*m, ckpt, width, bottom=exectime, facecolor="none", edgecolor=appconf["color"], hatch="*")
    # plt.bar(x + width*m, comm, width, bottom=exectime+ckpt, facecolor="none", edgecolor=appconf["color"], hatch="\\")
    # plt.bar(x + width*m, recovery, width, bottom=exectime+ckpt+comm, facecolor="none", edgecolor=appconf["color"], label=appconf["label"], hatch="||")
    plt.bar(x + width*m, exectime, width, facecolor="none", edgecolor="green", hatch="//", label="Data Processing")
    plt.bar(x + width*m, ckpt, width, bottom=exectime, facecolor="none", edgecolor="orange", hatch="*", label="Checkpointing")
    plt.bar(x + width*m, comm, width, bottom=exectime+ckpt, facecolor="none", edgecolor="blue", hatch="\\", label="Sync")
    # plt.bar(x + width*m, recovery, width, bottom=exectime+ckpt+comm, facecolor="none", edgecolor="purple", hatch="||", label="Recovery")
    m += 1
  plt.xlabel("Mean Time to Failure (sec)")
  plt.xticks(np.arange(len(probs)), 1/np.array(probs))
  plt.ylabel("Elapsed time (s)")
  # plt.yscale("log")
  
  plt.legend(loc="best")
  plt.tight_layout()
  plt.savefig(figpath + ".png")
  plt.savefig(figpath + ".pdf")

def plot_resilient(data, probs, figpath, normalized_value=1):
  width = 0.15
  # plt.figure(figsize=(10, 6))
  lapp = None
  for approach in data:
    if lapp == None or len(data[approach]["elapsed-time"]) > len(data[lapp]["elapsed-time"]):
      lapp = approach
  plt.figure()
  m = 0
  approaches = ["no-resilient", "ckpt", "dynamic-redis"]
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
  # plt.ylabel("Reconstrucution Time (sec)")
  plt.ylabel("Normalized Reconstruction Time")
  # plt.yscale("log")
  # plt.ylim(0, 31536000) # A year
  plt.ylim(0, 20) # A year
  
  # plt.legend(loc="best")
  plt.tight_layout()
  plt.savefig(figpath + ".png")
  plt.savefig(figpath + ".pdf")

def plot_totaltime(data, approaches, figpath):

  num_iters, probs, total_times = extract_data(data, approaches)

  for num_iter in num_iters:
    print("Draw for iteration", num_iter)
    width = 0.15
    plt.figure()
    m = 0
    x = np.arange(len(probs))
    for approach in total_times:
      appconf = data[approach]
      plt.bar(x + width*m, total_times[approach][num_iter], width, facecolor="none", edgecolor=appconf["color"], hatch="//", label=appconf["label"])
      m += 1
    plt.xlabel("Mean Time to Failure (sec)")
    plt.xticks(np.arange(len(probs)), 1/np.array(probs))
    plt.ylabel("Reconstrucution Time (sec)")
    # plt.yscale("log")
    # plt.ylim(0, 31536000) # A year
    # plt.ylim(0, 20) # A year
    
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(figpath + "-iters-" + str(num_iter) + ".png")
    plt.savefig(figpath + "-iters-" + str(num_iter) + ".pdf")

def extract_data(data, approaches):
  num_iters = set()
  probs = set()
  total_times = {}
  for approach in approaches:
    total_times[approach] = {}
    traces = data[approach]["elapsed-time"]
    for info in traces:
      niter = info["num_iter"]
      nps = info["prob"]
      num_iters.add(niter)
      probs.add(nps)
      if niter not in total_times[approach]:
        total_times[approach][niter] = {}
      total_times[approach][niter][nps] = info["total"]
  
  num_iters = sorted(list(num_iters))
  probs = sorted(list(probs))

  plot_total_times = {}
  for approach in plotdata:
    plot_total_times[approach] = {}
    for niter in num_iters:
      plot_times = []
      for nps in probs:
        if nps not in total_times[approach][niter]:
          plot_times.append(0)
        else:
          plot_times.append(total_times[approach][niter][nps])
      plot_total_times[approach][niter] = plot_times
      # print(approach, niter, plot_times)

  return num_iters, probs, plot_total_times

if __name__ == "__main__":
  
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
  ideal_exec_time = 0
  elapsed_time_info = plotdata["ckpt"]["elapsed-time"]
  for info in elapsed_time_info:
    if info["prob"] == 0:
      ideal_exec_time = info["total"] - info["ckpt"] - info["comm"]
    # exp_total = exp_resilient_runtime(info["prob"], 64, info["total"])
    # info["total"] = exp_total
  # probs = [0, 0.0001, 0.001, 0.01, 0.1]

  # probs = [0, 0.0001, 0.001, 0.01, 0.1]
  # for prob in probs:
  #   info = {}
  #   info["prob"] = prob
  #   # info["total"] = exp_no_resilient_runtime(prob, 64, ideal_exec_time)
  #   info["total"] = simulate_no_resilient_resume_runtime(prob, 64, ideal_exec_time)
  #   no_resilience["elapsed-time"].append(info)
  # print(no_resilience)
  # plotdata["no-resilient"] = no_resilience

  # approaches = ["no-resilient", "ckpt", "dynamic-redis"]
  approaches = ["ckpt", "dynamic-redis"]
  plot_totaltime(plotdata, approaches, figpath + "/elapsed-time-resilient")
  

 



 
