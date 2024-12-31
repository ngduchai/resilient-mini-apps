import sewar.full_ref as metrics_cal
from skimage.exposure import adjust_gamma, rescale_intensity
import h5py as h5
import numpy as np
import sys
import json
import math

import pickle

import matplotlib.pyplot as plt

import os

plt.rcParams['axes.labelsize'] = 16     # X and Y labels font size
plt.rcParams['xtick.labelsize'] = 16    # X-axis tick labels font size
plt.rcParams['ytick.labelsize'] = 16    # Y-axis tick labels font size
plt.rcParams['legend.fontsize'] = 16

def sim_ckpt_overhead(num_procs, prob, ckpt_period, num_iter, num_sinograms, comp_unit, ckpt_unit, dynamic=False):
  num_tries = 1000
  total_overhead = 0
  total_runtime = 0
  print(prob)
  for i in range(num_tries):
    num_complete = 0
    failure_times = []
    if prob == 0:
      failure_times = np.ones(num_procs) * 100000000000
    else:
      failure_times = np.random.exponential(scale=1/prob, size=num_procs)
    acc_time = 0
    num_fail = 0
    overhead = 0
    surviving_procs = num_procs
    progress = 0
    target_progress = num_iter * num_sinograms
    selected_ckpt_period = ckpt_period
    while progress < target_progress:
      per_proc_load = num_sinograms*selected_ckpt_period / surviving_procs # Assume load distribution is perfectly balance 
      exec_time = comp_unit * per_proc_load
      per_proc_ckpt = num_sinograms / surviving_procs
      ckpt_time = ckpt_unit * per_proc_ckpt
      actual_time = exec_time + ckpt_time
      acc_time += actual_time
      failure_occurences = sum([1 if x < acc_time else 0 for x in failure_times])
      nfails = failure_occurences - num_fail
      num_fail = failure_occurences
      surviving_procs = num_procs - num_fail
      overhead += ckpt_time + nfails * actual_time/2
      progress += per_proc_load * surviving_procs
      if surviving_procs == 0:
        surviving_procs = num_procs
        failure_times = np.random.exponential(scale=1/prob, size=num_procs) + acc_time
      if nfails > 0 and dynamic == True:
        ckpt_overhead = ckpt_unit * num_sinograms / surviving_procs
        iteration_length = num_sinograms / surviving_procs * comp_unit
        prev_ckpt_period = selected_ckpt_period
        selected_ckpt_period = max(1, int(round(math.sqrt(2 * 1 / (prob*surviving_procs)  * ckpt_overhead) / iteration_length, 0)))
        if prev_ckpt_period != selected_ckpt_period:
          print(i, prob, dynamic, target_progress-progress, "-->", selected_ckpt_period, surviving_procs)
    total_overhead += overhead
    total_runtime += acc_time
  return total_overhead/num_tries
    



# Making a plot showing the checkpoint overhead varying input data size
def plot_overhead(data, probs, figpath, normalized_value=1):
  width = 0.15
  # plt.figure(figsize=(10, 6))
  plt.figure()
  m = -1
  approaches = ["low_freq", "high_freq", "dynamic"]
  # approaches = ["high_freq", "static", "dynamic"]
  # for approach in data:
  for approach in approaches:
    appconf = data[approach]
    appdata = data[approach]["elapsed-time"]
    total = []
    for prob in probs:
      for info in appdata:
        if prob == info["prob"]:
          total.append(info["ckpt"])
          break
    x = np.arange(len(probs))
    total = np.array(total)
    plt.bar(x + width*m, total/normalized_value, width, facecolor="none", edgecolor=appconf["color"], hatch="//", label=appconf["label"])
    print(total/normalized_value)
    m += 1
  plt.xlabel("Mean Time to Failure (sec)")
  plt.xticks(np.arange(len(probs)), ["$\infty$" if prob == 0 else int(round(1/prob)) for prob in probs])
  if normalized_value == 1:
    plt.ylabel("Overhead (sec)")
    plt.ylim(0.1, 400000) # A year
  else:
    plt.ylabel("Normalized Overhead")
    # plt.ylim(1, 31536000) # A year
    plt.ylim(0.1, 400000) # A year
  plt.yscale("log")
  plt.grid(True)
  
  
  plt.legend(loc="best")
  plt.tight_layout()
  plt.savefig(figpath + ".png")
  plt.savefig(figpath + ".pdf")


if __name__ == "__main__":
  
  # if len(sys.argv) < 3:
  #   print("Usage: python plot-checkpoint-strategies.py <data file> <fig folder>")
  #   sys.exit(1)

  # datapath = sys.argv[1]
  # figpath = sys.argv[2]
  # with open(datapath, 'r') as file:
  #   plotdata = json.load(file)

  
  # Work with simulation first
  # probs = [0, 0.0001, 0.001, 0.01, 0.1]
  probs = [0, 0.00001, 0.0001, 0.001]
  num_iter = 20
  # num_procs = 64
  # num_sinograms = 64
  num_procs = 64
  num_sinograms = 10*num_procs
  comp_unit = 6
  # ckpt_unit = 6
  ckpt_unit = 0.1

  dynamic_total_overhead = []
  static_total_overhead = []
  high_freq_total_overhead = []
  low_freq_total_overhead = []

  for prob in probs:
    mu = 1000000000000000
    if prob > 0:
      mu = 1 / (prob*num_procs)
    C = ckpt_unit * num_sinograms/num_procs
    iteration_length = num_sinograms / num_procs * comp_unit
    ckpt_period = max(1, int(round(math.sqrt(2 * mu * C) / iteration_length)))
    static_overhead = sim_ckpt_overhead(num_procs, prob, ckpt_period, num_iter, num_sinograms, comp_unit, ckpt_unit, dynamic=False)
    dynamic_overhead = sim_ckpt_overhead(num_procs, prob, ckpt_period, num_iter, num_sinograms, comp_unit, ckpt_unit, dynamic=True)
    high_freq_overhead = sim_ckpt_overhead(num_procs, prob, 1, num_iter, num_sinograms, comp_unit, ckpt_unit, dynamic=False)
    low_freq_overhead = sim_ckpt_overhead(num_procs, prob, num_iter, num_iter, num_sinograms, comp_unit, ckpt_unit, dynamic=False)
    # low_freq_overhead = 0
    dynamic_total_overhead.append({"prob" : prob, "ckpt" : dynamic_overhead})
    static_total_overhead.append({"prob" : prob, "ckpt" : static_overhead})
    high_freq_total_overhead.append({"prob" : prob, "ckpt" : high_freq_overhead})
    low_freq_total_overhead.append({"prob" : prob, "ckpt" : low_freq_overhead})
  
  plotdata = {}
  plotdata["approaches"] = {}
  plotdata["approaches"]["low_freq"] = {
    "color" : "purple",
    "label" : "No Ckpt",
    "elapsed-time" : low_freq_total_overhead
  }
  plotdata["approaches"]["high_freq"] = {
    "color" : "orange",
    "label" : "Once Per Iter.",
    "elapsed-time" : high_freq_total_overhead
  }
  plotdata["approaches"]["static"] = {
    "color" : "blue",
    "label" : "Static",
    "elapsed-time" : static_total_overhead
  }
  plotdata["approaches"]["dynamic"] = {
    "color" : "green",
    "label" : "Dynamic Reconfig.",
    "elapsed-time" : dynamic_total_overhead
  }
  figpath = "figures/checkpoint-overhead"

  plot_overhead(plotdata["approaches"], probs, figpath + "/checkpoint-strategies-overhead")
  


 
