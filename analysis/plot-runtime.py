import sewar.full_ref as metrics_cal
from skimage.exposure import adjust_gamma, rescale_intensity
import h5py as h5
import numpy as np
import sys
import json

import pandas as pd

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
  approaches = ["no-resilient", "veloc", "veloc-dynamic"]
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

def plot_breakdown(data, figpath, approach, normalized_value=1):
  width = 0.15
  # plt.figure(figsize=(10, 6))
  times = ["ckpt", "comm", "exec"]
  colors = {
    "exec" : "orange",
    "ckpt" : "blue",
    "comm" : "green"
  }
  labels = {
    "exec" : "Execution",
    "ckpt" : "Checkpointing",
    "comm" : "Data Redistribution"
  }
  
  probs = []
  mtimes = {}
  for time in times:
    mtimes[time] = []
  for info in data[approach]["elapsed-time"]:
    probs.append(info["prob"])
    for time in times:
      mtimes[time].append(info[time])
    
  plt.figure()
  x = np.arange(len(probs))
  m=0
  for time in times:
    plt.bar(x + width*m, np.array(mtimes[time])/normalized_value, width, facecolor="none", edgecolor=colors[time], hatch="//", label=labels[time])
    print(time, np.array(mtimes[time])/normalized_value)
    m += 1
  plt.xlabel("Mean Time to Failure (sec)")
  plt.xticks(np.arange(len(probs)), 1/np.array(probs))
  if normalized_value == 1:
    plt.ylabel("Reconstrucution Time (sec)")
  else:
    plt.ylabel("Normalized Elapsed Time")
  plt.yscale("log")
  # plt.ylim(0, 31536000) # A year
  # plt.ylim(0, 20) # A year
  
  # plt.legend(loc="best")
  plt.tight_layout()
  plt.savefig(figpath + ".png")
  plt.savefig(figpath + ".pdf")

def plot_clustered_stacked(figpath, varied_unit, dfall, labels=None,  hashes="/", **kwargs):
    """Given a list of dataframes, with identical columns and index, create a clustered stacked bar plot. 
labels is a list of the names of the dataframe, used for the legend
title is a string for the title of the plot
H is the hatch used for identification of the different dataframe"""
    plt.figure()
    n_df = len(dfall)
    n_col = len(dfall[0].columns) 
    n_ind = len(dfall[0].index)
    axe = plt.subplot(111)

    for df in dfall : # for each data frame
        axe = df.plot(kind="bar",
                      linewidth=0,
                      stacked=True,
                      ax=axe,
                      legend=False,
                      grid=False,
                      **kwargs)  # make bar plots

    h,l = axe.get_legend_handles_labels() # get the handles we want to modify
    for i in range(0, n_df * n_col, n_col): # len(h) = n_col * n_df
        for j, pa in enumerate(h[i:i+n_col]):
            for rect in pa.patches: # for each index
                rect.set_x(rect.get_x() + 1 / float(n_df + 1) * i / float(n_col))
                # rect.set_hatch(H * int(i / n_col)) #edited part     
                rect.set_hatch(hashes[int(i / n_col)]) #edited part     
                rect.set_width(1 / float(n_df + 1))

    axe.set_xticks((np.arange(0, 2 * n_ind, 2) + 1 / float(n_df + 1)) / 2.)
    axe.set_xticklabels(df.index, rotation = 0)

    # Add invisible data to add another legend
    n=[]        
    for i in range(n_df):
        n.append(axe.bar(0, 0, color="none", hatch=hashes[i]))

    l1 = axe.legend(h[:n_col], l[:n_col], loc=[0.01, 0.7])
    if labels is not None:
        l2 = plt.legend(n, labels, loc=[0.01, 0.4]) 
    axe.add_artist(l1)
    axe.set_ylim([0, 2000])

    axe.set_ylabel("Reconstruction Time (sec)")
    axe.set_xlabel(varied_unit)

    plt.tight_layout()
    plt.savefig(figpath + ".png")
    plt.savefig(figpath + ".pdf")

    return axe

def plot_resilient_breakdown(data, varied_param, fixed_params, figpath, normalized_value=1):
  width = 0.15
  # plt.figure(figsize=(10, 6))
  plt.figure()
  m = 0
  approaches = ["no-resilient", "veloc", "veloc-dynamic"]
  hatches = {
    "comp" : "//",
    "comm" : "+",
    "comm" : "\\"
  }
  total_times = {}
  for approach in approaches:
    appconf = data[approach]
    appdata = data[approach]["elapsed-time"]
    total = []
    comp = []
    comm = []
    ckpt = []
    for vvalue in varied_param["values"]:
      for info in appdata:
        if vvalue == info[varied_param["key"]]:
          found = True
          for fixed_param in fixed_params:
            if info[fixed_param] != info[fixed_param]:
              found = False
              break
          if found:
            total.append(info["total"])
            comp.append(info["exec"])
            comm.append(info["comm"])
            ckpt.append(info["ckpt"])
            break

    print(varied_param["labels"])      
    print(comp)
    total_times[approach] = pd.DataFrame(np.array([ckpt, comm, comp]).transpose(), index=varied_param["labels"], columns=["Checkpoint", "Communicate", "Compute"])

  plot_clustered_stacked(figpath, varied_param["xlabel"], list(total_times.values()), ["No Resilient", "+Ckpt", "+Ckpt +Dynamic Redist"], hashes=["", "//", "o"])
  


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

  # probs = [0, 0.005, 0.01, 0.02, 0.05, 0.1]
  # plot_fig(plotdata["exp_failure"], probs, figpath + "/elapsed-time-no-retry")
  # # plot_fig(plotdata["with-retries"], probs, figpath + "/elapsed-time-with-retry")
  
  # Calculate runtime for no resilient implementation
  no_resilience = {}
  no_resilience["label"] = "No Resilience"
  no_resilience["color"] = "orange"
  no_resilience["elapsed-time"] = []
  ideal_exec_time = 5.99935 # Processing time of 1 slice in 1 iteration
  print("Generate no resilient runtime")
  probs = [0, 0.0001, 0.001, 0.01, 0.1]
  num_iters = [10, 20, 30, 40, 50]
  nprocs = [1, 2, 4, 8, 16, 32, 64]
  for prob in probs:
    info = {}
    info["prob"] = prob
    info["nprocs"] = 64
    info["num_iter"] = 10
    info["total"] = exp_no_resilient_runtime(prob, 64, 10*ideal_exec_time)
    # info["total"] = simulate_no_resilient_resume_runtime(prob, 64, ideal_exec_time)
    info["exec"] = info["total"]
    info["comm"] = 0
    info["ckpt"] = 0
    info["recover"] = 0
    no_resilience["elapsed-time"].append(info)
  for num_iter in num_iters:
    info = {}
    info["prob"] = 0.01
    info["nprocs"] = 64
    info["num_iter"] = num_iter
    info["total"] = exp_no_resilient_runtime(0.01, 64, num_iter*ideal_exec_time)
    # info["total"] = simulate_no_resilient_resume_runtime(prob, 64, ideal_exec_time)
    info["exec"] = info["total"]
    info["comm"] = 0
    info["ckpt"] = 0
    info["recover"] = 0
    no_resilience["elapsed-time"].append(info)
  for nproc in nprocs:
    info = {}
    info["prob"] = 0.01
    info["nprocs"] = nproc
    info["num_iter"] = 10
    info["total"] = exp_no_resilient_runtime(0.01, 64, 64/nproc*10*ideal_exec_time)
    # info["total"] = simulate_no_resilient_resume_runtime(prob, 64, ideal_exec_time)
    info["exec"] = info["total"]
    info["comm"] = 0
    info["ckpt"] = 0
    info["recover"] = 0
    no_resilience["elapsed-time"].append(info)
  plotdata["with-retries"]["no-resilient"] = no_resilience
  print(no_resilience)
  # normalized_value = ideal_exec_time
  print(ideal_exec_time)
  # normalized_value=59.682124539
  # plot_resilient(plotdata["with-retries"], probs, figpath + "/elapsed-time-resilient", normalized_value)
  # # normalized_value=59.682124539
  # normalized_value=1
  # plot_breakdown(plotdata["with-retries"], figpath + "/elapsed-time-breakdown", "veloc-dynamic", normalized_value)

  normalized_value = 1

  # Varying failure rate
  plot_resilient_breakdown(
    plotdata["with-retries"],
    {
      "key" : "prob",
      "values" : probs,
      "labels" : 1/np.array(probs),
      "xlabel" : "Mean time to Failure (sec)"
    },
    {"nprocs": 64, "num_iter": 10},
    figpath + "/elapsed-time-varying-mttf",
    normalized_value
  )
  # Varying number of iterations
  plot_resilient_breakdown(
    plotdata["with-retries"],
    {
      "key" : "num_iter",
      "values" : num_iters,
      "labels" : num_iters,
      "xlabel" : "Number of iteration"
    },
    {"nprocs": 64, "prob": 0.01},
    figpath + "/elapsed-time-varying-iter",
    normalized_value
  )
  # Varying number of tasks
  plot_resilient_breakdown(
    plotdata["with-retries"],
    {
      "key" : "nprocs",
      "values" : nprocs,
      "labels" : nprocs,
      "xlabel" : "Number of reconstruction tasks"
    },
    {"prob": 0.01, "num_iter": 10},
    figpath + "/elapsed-time-varying-np",
    normalized_value
  )
 



 
