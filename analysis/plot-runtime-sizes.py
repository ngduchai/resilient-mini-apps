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


def plot_totaltime_by_prob(data, probs, figpath, normalized_value=1):
  width = 0.15
  # plt.figure(figsize=(10, 6))
  plt.figure()
  m = -1.5
  sizes = ["640", "1280", "2k"]
  
  # for approach in data:
  for size in sizes:
    appconf = data[size]
    appdata = data[size]["elapsed-time"]
    total = []
    for prob in probs:
      found_data = False
      for info in appdata:
        if prob == info["prob"]:
          total.append(info["total"])
          found_data = True
          break
      if not found_data:
        total.append(0) # No data available, plot zero.
    x = np.arange(len(probs))
    total = np.array(total)
    plt.bar(x + width*(m+0.5), total/normalized_value, width, facecolor="none", edgecolor=appconf["color"], hatch="//", label=appconf["label"])
    print(total/normalized_value)
    m += 1
  plt.xlabel("Mean Time to Failure (sec)")
  # plt.xticks(np.arange(len(probs)), 1/np.array(probs))
  plt.xticks(np.arange(len(probs)), ["$\infty$" if prob == 0 else int(round(1/prob)) for prob in probs])
  if normalized_value == 1:
    plt.ylabel("Reconstrucution Time (sec)")
    # plt.ylim(1, 200000) # A year
  else:
    plt.ylabel("Normalized Reconstruction Time")
    # plt.ylim(1, 31536000) # A year
    # plt.ylim(1, 200000) # A year
    # plt.ylim(1, 25000) # A year
  plt.yscale("log")
  plt.grid(True)
  
  
  plt.legend(loc="best")
  plt.tight_layout()
  plt.savefig(figpath + ".png")
  plt.savefig(figpath + ".pdf")

def plot_totaltime_by_size(data, probs, figpath):
  width = 0.15
  # plt.figure(figsize=(10, 6))
  plt.figure()
  m = -1.5
  sizes = ["640", "1280", "2k"]
  colors = {
    0.0001 : "purple",
    0.001 : "blue",
    0.01: "green"
  }

  normalized_value = {}
  for size in sizes:
    found_normalized_value = False
    for info in data[size]["elapsed-time"]:
      if info["prob"] == 0:
        normalized_value[size] = info["total"]
        found_normalized_value = True
        break
    if not found_normalized_value:
      normalized_value[size] = 1

  # for approach in data:
  for prob in probs:
    total = []
    for size in sizes:
      appconf = data[size]
      appdata = data[size]["elapsed-time"]
      found_data = False
      for info in appdata:
        if prob == info["prob"]:
          total.append(info["total"] / normalized_value[size])
          found_data = True
          break
      if not found_data:
        total.append(0) # No data available, plot zero.
    x = np.arange(len(probs))
    total = np.array(total)
    plt.bar(x + width*(m+0.5), total, width, facecolor="none", edgecolor=colors[prob], hatch="//", label="MTTF = " + ("$\infty$" if prob == 0 else str(int(round(1/prob))) + " sec"))
    print(total)
    m += 1
  plt.xlabel("Sinogram Size")
  # plt.xticks(np.arange(len(probs)), 1/np.array(probs))
  plt.xticks(range(len(sizes)), [data[size]["label"] for size in sizes])
  plt.ylabel("Normalized Reconstruction Time")
  plt.ylim(0.1, 100)
  
  plt.yscale("log")
  plt.grid(True)
  
  
  plt.legend(loc="best")
  plt.tight_layout()
  plt.savefig(figpath + ".png")
  plt.savefig(figpath + ".pdf")

def plot_overhead_with_mttf(data, probs, figpath):
  width = 0.15
  # plt.figure(figsize=(10, 6))
  plt.figure()
  m = -1.5

  sizes = ["640", "1280", "2k"]
  colors = {
    0.0001 : "purple",
    0.001 : "blue",
    0.01: "green"
  }

  normalized_value = {}
  for size in sizes:
    found_normalized_value = False
    for info in data[size]["elapsed-time"]:
      if info["prob"] == 0:
        normalized_value[size] = info["total"]
        found_normalized_value = True
        break
    if not found_normalized_value:
      normalized_value[size] = 0

  overhead = {}
  for prob in probs:
    overhead[prob] = []
    for size in sizes:
      found_prob = False
      for info in data[size]["elapsed-time"]:
        if prob == info["prob"]:
          if normalized_value[size] > 0:
            # overhead[prob].append((info["ckpt"] + info["comm"]) / normalized_value[size])
            overhead[prob].append((info["ckpt"] + info["comm"]) / info["total"])
          else:
            overhead[prob].append(0)
          # print(size, prob, normalized_value, overhead[prob][-1])
          found_prob = True
          break
      if not found_prob:
        overhead[prob].append(0)
    x = np.arange(len(sizes))
    print(overhead[prob])
    overhead[prob] = np.array(overhead[prob])
    plt.bar(x + width*(m+0.5), overhead[prob], width, facecolor="none", edgecolor=colors[prob], hatch="//", label="MTTF = " + ("$\infty$" if prob == 0 else str(int(round(1/prob))) + " sec"))
    m += 1
  
  plt.xlabel("Reconstruction Size")
  # plt.xticks(np.arange(len(probs)), 1/np.array(probs))
  plt.xticks(range(len(sizes)), [data[size]["label"] for size in sizes])
  
  plt.ylabel("Normalized Overhead")
  plt.ylim([0.0001, 0.1])
    
  plt.yscale("log")
  plt.grid(True)
  
  plt.legend(loc="best")
  plt.tight_layout()
  plt.savefig(figpath + ".png")
  plt.savefig(figpath + ".pdf")

# Making a plot showing the checkpoint overhead varying input data size
def plot_overhead_breakdown(data, prob, figpath):
  width = 0.15
  # plt.figure(figsize=(10, 6))
  plt.figure()
  m = -1.5
  total_times = []
  compute_times = []
  ckpt_times = []
  comm_times = []

  sizes = ["640", "1280", "2k"]

  normalized_value = {}
  for size in sizes:
    found_normalized_value = False
    for info in data[size]["elapsed-time"]:
      if info["prob"] == 0:
        normalized_value[size] = info["total"]
        found_normalized_value = True
        break
    if not found_normalized_value:
      normalized_value[size] = 1

  for size in sizes:
    found_prob = False
    for info in data[size]["elapsed-time"]:
      if prob == info["prob"]:
        total_times.append(info["total"] / normalized_value[size])
        compute_times.append(info["exec"] / normalized_value[size])
        ckpt_times.append(info["ckpt"] / normalized_value[size])
        comm_times.append(info["comm"] / normalized_value[size])
        found_prob = True
        break
    if not found_prob:
      total_times.append(0)
      compute_times.append(0)
      ckpt_times.append(0)
      comm_times.append(0)
  
  # compute_times = np.array(compute_times) / normalized_value
  # ckpt_times = np.array(ckpt_times) / normalized_value
  # comm_times = np.array(comm_times) / normalized_value
  compute_times = np.array(compute_times)
  ckpt_times = np.array(ckpt_times)
  comm_times = np.array(comm_times)
  
  x = np.arange(len(sizes))
  
  plt.bar(x-width, compute_times, width, facecolor="none", edgecolor="green", hatch="//", label="Reconstruction")
  plt.bar(x, ckpt_times, width, facecolor="none", edgecolor="orange", hatch="*", label="Checkpointing")
  plt.bar(x+width, comm_times, width, facecolor="none", edgecolor="blue", hatch="\\", label="Communication")

  print(compute_times)
  print(ckpt_times)
  print(comm_times)
  
  plt.xlabel("Sinogram Size")
  # plt.xticks(np.arange(len(probs)), 1/np.array(probs))
  plt.xticks(np.arange(len(sizes)), [data[size]["label"] for size in sizes])
  plt.ylabel("Normalized Elapsed Time")
  plt.yscale("log")
  plt.ylim((0, 100))
  plt.grid(True)
  
  plt.legend(loc="best")
  plt.tight_layout()
  plt.savefig(figpath + ".png")
  plt.savefig(figpath + ".pdf")

if __name__ == "__main__":

  # python plot-runtime-sizes.py data/execinfo-runtimes-sizes.json figures/runtime
  


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

  probs = [0.0001, 0.001, 0.01]
  

  # plot_totaltime_by_prob(plotdata["sizes"], probs, figpath + "/elapsed-time-resilient-sizes")
  plot_totaltime_by_size(plotdata["sizes"], probs, figpath + "/elapsed-time-resilient-sizes")
  # plot_overhead_with_mttf(plotdata["sizes"], probs, figpath + "/elapsed-time-breakdown-sizes")
  plot_overhead_breakdown(plotdata["sizes"], 0.001, figpath + "/elapsed-time-breakdown-sizes")
 



 
