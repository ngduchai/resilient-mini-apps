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
  process_success_prob = 1 - stats.expon.cdf(x=runtime, scale=1/lamb)
  reconstruction_success_prob = stats.binom.pmf(num_processes, num_processes, process_success_prob)
  exp_runtime = runtime * 1/reconstruction_success_prob
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

# Making a plot showing the checkpoint overhead varying input data size
def plot_fig(data, xlab, ylab, figpath):
  width = 0.15
  plt.figure()
  lapp = None
  for approach in data:
    if lapp == None or len(data[approach]["elapsed-time"]) > len(data[lapp]["elapsed-time"]):
      lapp = approach
  probs = None
  m = 0
  for approach in data:
    appconf = data[approach]
    appdata = data[approach]["elapsed-time"]
    prob = []
    ckpt = []
    comm = []
    recovery = []
    total = []
    for info in appdata:
      prob.append(info["prob"])
      ckpt.append(info["ckpt"])
      recovery.append(info["recover"])
      comm.append(info["comm"])
      total.append(info["total"])
    if probs == None:
      probs = 1/np.array(prob)
    x = np.arange(len(prob))
    ckpt = np.array(ckpt)
    recovery = np.array(recovery)
    # comm = np.array(comm) - recovery
    comm = np.array(comm)
    total = np.array(total)
    exectime = total - recovery - ckpt - comm
    # plt.bar(x + width*m, exectime, width, facecolor="none", edgecolor="appconf["color"]", hatch="//")
    # plt.bar(x + width*m, ckpt, width, bottom=exectime, facecolor="none", edgecolor=appconf["color"], hatch="*")
    # plt.bar(x + width*m, comm, width, bottom=exectime+ckpt, facecolor="none", edgecolor=appconf["color"], hatch="\\")
    # plt.bar(x + width*m, recovery, width, bottom=exectime+ckpt+comm, facecolor="none", edgecolor=appconf["color"], label=appconf["label"], hatch="||")
    plt.bar(x + width*m, exectime, width, facecolor="none", edgecolor="green", hatch="//", label="Data Processing")
    plt.bar(x + width*m, ckpt, width, bottom=exectime, facecolor="none", edgecolor="orange", hatch="*", label="Checkpointing")
    plt.bar(x + width*m, comm, width, bottom=exectime+ckpt, facecolor="none", edgecolor="blue", hatch="\\", label="Sync")
    # plt.bar(x + width*m, recovery, width, bottom=exectime+ckpt+comm, facecolor="none", edgecolor="purple", hatch="||", label="Recovery")
    m += 1
  plt.xlabel(xlab)
  plt.xticks(np.arange(len(probs)), probs)
  plt.ylabel(ylab)
  # plt.yscale("log")
  
  plt.legend(loc="best")
  plt.tight_layout()
  plt.savefig(figpath + ".png")
  plt.savefig(figpath + ".pdf")

def plot_resilient(data, figpath):
  width = 0.15
  lapp = None
  for approach in data:
    if lapp == None or len(data[approach]["elapsed-time"]) > len(data[lapp]["elapsed-time"]):
      lapp = approach
  probs = None
  plt.figure()
  m = 0
  for approach in data:
    print(approach)
    appconf = data[approach]
    appdata = data[approach]["elapsed-time"]
    total = []
    prob = []
    for info in appdata:
      prob.append(info["prob"])
      total.append(info["total"])
    if probs is None:
      probs = 1/np.array(prob)
    x = np.arange(len(prob))
    total = np.array(total)
    plt.bar(x + width*m, total, width, facecolor="none", edgecolor=appconf["color"], hatch="//", label=appconf["label"])
    m += 1
  plt.xlabel("Mean Time to Failure (sec)")
  plt.xticks(np.arange(len(probs)), probs)
  plt.ylabel("End-to-end Reconstrucution Time (sec)")
  plt.yscale("log")
  
  plt.legend(loc="best")
  plt.tight_layout()
  plt.savefig(figpath + ".png")
  plt.savefig(figpath + ".pdf")

if __name__ == "__main__":
  
  # lamb = 0.002
  # num_processes = 64
  # runtime = 10
  # exp_runtime = exp_no_resilient_runtime(lamb, num_processes, runtime)
  # act_runtime = simulate_no_resilient_runtime(lamb, num_processes, runtime)
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

  plot_fig(plotdata["exp_failure"], "Mean Time to Failure (sec)", "Elapsed time (s)", figpath + "/elapsed-time-no-retry")
  plot_fig(plotdata["with-retries"], "Mean Time to Failure (sec)", "Elapsed time (s)", figpath + "/elapsed-time-with-retry")
  
  # Calculate runtime for no resilient implementation
  no_resilience = {}
  no_resilience["label"] = "No Resilience"
  no_resilience["color"] = "orange"
  no_resilience["elapsed-time"] = []
  ideal_exec_time = 0
  elapsed_time_info = plotdata["exp_failure"]["veloc"]["elapsed-time"]
  for info in elapsed_time_info:
    if info["prob"] == 0:
      ideal_exec_time = info["total"] - info["ckpt"] - info["comm"]
  probs = [0, 0.01, 0.05, 0.1]
  for prob in probs:
    info = {}
    info["prob"] = prob
    info["total"] = exp_no_resilient_runtime(prob, 64, ideal_exec_time)
    no_resilience["elapsed-time"].append(info)
  plotdata["exp_failure"]["no-resilient"] = no_resilience
  plot_resilient(plotdata["exp_failure"], figpath + "/elapsed-time-resilient")


 



 
