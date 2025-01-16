import numpy as np
import sys
import json
import math

import pickle

import scipy.stats as stats

import matplotlib.pyplot as plt

def scaling_sim(num_procs, comp_unit, num_iter, num_sinograms, deadline, lamb, ckpt_period):
    
    num_tries = 1000
    # num_tries = 10000

    total_runtime = 0
    num_success = 0
    num_complete = 0

    for i in range(num_tries):
        total_computation = comp_unit * num_iter * num_sinograms
        prev_num_survive = num_procs
        progress = 0
        last_ckpt = 0
        failure_times = []
        if lamb == 0:
            failure_times = np.ones(num_procs) * 100000000000
        else:
            failure_times = np.random.exponential(scale=1/lamb, size=num_procs)
        t = 0
        while progress < total_computation:
            surviving_processes = (failure_times > t).astype(int)
            num_survive = np.sum(surviving_processes)
            if num_survive == 0:
                t = -1
                break
            if num_survive < prev_num_survive:
                total_computation += (prev_num_survive - num_survive) * (t - last_ckpt)
                prev_num_survive = num_survive
            progress += num_survive
            if t % ckpt_period:
                last_ckpt = t
            t += 1
        if t > 0:
            total_runtime += t
            num_complete += 1
            if t <= deadline:
                num_success += 1
    
    avg_runtime = 100000000000000000
    if num_complete > 0:
        avg_runtime = total_runtime / num_complete
    success_rate = num_success / num_tries

    return avg_runtime, success_rate

def estimate_scaling_success(num_procs, comp_unit, num_iter, num_sinograms, deadline, lamb, ckpt_period):
    total_computation = comp_unit * num_iter * num_sinograms
    total_success_prob = 0
    for num_failures in range(0, num_procs+1):
        # Process failures are iid with failure rate = lamb (or MTTF = 1/lamb)
        # Since the computation must complete within the deadline, we only care
        # failure happen BEFORE the deadline. Also, assuming each process failure
        # follows an exponential distribution, we can consider the whole pipeline
        # failure also follow an exponential distribution with
        #       failure_rate = num_processes * lamb
        # Thus, the probability of having EXACTLY [num_failures] processes failed
        # within the interval of length [deadline] would follow an Poisson distribution
        # with average incident
        #       mu = num_processes * lamb * deadline 
        failure_prob = stats.poisson.pmf(k=num_failures, mu=num_procs*lamb*deadline)
        # Extra computation = wasted computation left by the failed process
        # here we assume checkpoint is taken every [ckpt_period] sec, so
        # the waste = all computation done before the last checkpoint, which
        # could vary from 0 (best case, failure happens immediately after checkpointing) to
        # [ckpt_period] (worse case, failure happens immediately before checkpointing),
        # again, since failures follow an exponential distribution, it is memoryless,
        # meaning the chance of failure are equal in time, thus, for simplicity,
        # we asume the the waste is averaged = (worst case + best case) / 2 = 0.5 * worst case 
        extra_computation = 0.5* num_failures * ckpt_period
        require_computation = total_computation + extra_computation
        # Require computation now refect the progress made by failed processes by subtracting
        # the progress made by surviving process until the deadline.
        require_computation -= (num_procs - num_failures) * deadline
        success_prob = 1
        if require_computation > 0:
            # if require computation > 0, then we needed computation made by failed processes.
            # Their computation refected by their life time. Thus, their total progress
            # is their total lifetime
            if num_failures > 0:
                # Since we known the number of failed processes, their lifetime is uniformly
                # distributed over the [deadline] period (assuming failures following Exponential
                # distribution, which is memoryless) --> the computation made by failed processes
                # are sum of m iid uniform variables where m = [num_failures]. The sum will
                # follow a Irwin-Hall distribution, according to
                #   - P. Hall, “The distribution of means for samples of size N drawn from a population in which the variate takes values between 0 and 1, all such values being equally probable”, Biometrika, Volume 19, Issue 3-4, December 1927, Pages 240-244, DOI:10.1093/biomet/19.3-4.240.
                #   - J. O. Irwin, “On the frequency distribution of the means of samples from a population having any law of frequency with finite moments, with special reference to Pearson’s Type II, Biometrika, Volume 19, Issue 3-4, December 1927, Pages 225-239, DOI:0.1093/biomet/19.3-4.225.
                success_prob = 1 - stats.irwinhall.cdf(x=require_computation/deadline, n=num_failures)
            else:
                success_prob = 0
        total_success_prob += failure_prob * success_prob
    return total_success_prob

def configure_scaling(target_success, comp_unit, num_iter, num_sinograms, deadline, lamb, ckpt_period):
    for num_procs in range(1, num_sinograms+1):
        success_prob = estimate_scaling_success(num_procs, comp_unit, num_iter, num_sinograms, deadline, lamb, ckpt_period)
        if success_prob >= target_success:
            return num_procs
    return num_sinograms

def examine_scaling(target_success, comp_unit, num_iter, num_sinograms, deadline, lamb, ckpt_period):
    for num_procs in range(int(num_sinograms/4), num_sinograms+1):
    # for num_procs in range(int(num_sinograms/2), num_sinograms+1):
        _, success_prob = scaling_sim(num_procs, comp_unit, num_iter, num_sinograms, deadline, lamb, ckpt_period)
        if success_prob >= target_success:
            return num_procs
    return num_sinograms

# Base configuration, we will varying one of these parameters and keep other fixed
comp_unit = 6
num_iter = 10
num_sinograms = 64
slack = 2
deadline = (comp_unit * num_iter) * slack 
lamb = 0.001
ckpt_period = comp_unit
target_success = 0.9999

# for num_procs in range(int(num_sinograms/2), num_sinograms+1):
#     print("num_procs =", num_procs, " ------------------------------ ")
#     print("Estimation", estimate_scaling_success(num_procs, comp_unit, num_iter, num_sinograms, deadline, lamb, ckpt_period))
#     _, success_rate = scaling_sim(num_procs, comp_unit, num_iter, num_sinograms, deadline, lamb, ckpt_period)
#     print("Simulation", success_rate)


# num_procs = configure_scaling(target_success, comp_unit, num_iter, num_sinograms, deadline, lamb, ckpt_period)
# print("num_procs", num_procs)
# runtime, success_rate = scaling_sim(num_procs, comp_unit, num_iter, num_sinograms, deadline, lamb, ckpt_period)
# print("runtime", runtime, "deadline", deadline)
# print("success_rate", success_rate)

# Static evaluation

# Varying mean time to failure ------------------------------------------------------
comp_unit = 6
num_iter = 10
num_sinograms = 64
slack = 4
deadline = (comp_unit * num_iter) * slack 
ckpt_period = comp_unit
target_success = 0.9999

lambs = [0, 0.0001, 0.001, 0.01]
est_required_procs = []
sim_required_procs = []
for lamb in lambs:
    est_num_procs = configure_scaling(target_success, comp_unit, num_iter, num_sinograms, deadline, lamb, ckpt_period)
    sim_num_procs = examine_scaling(target_success, comp_unit, num_iter, num_sinograms, deadline, lamb, ckpt_period)
    print(lamb, est_num_procs, sim_num_procs)
    est_required_procs.append(est_num_procs)
    sim_required_procs.append(sim_num_procs)

print(lambs)
print(est_required_procs)
print(sim_required_procs)

plt.figure()
width=0.25
x = np.arange(len(lambs))
plt.bar(x-0.5*width, est_required_procs, width, label="Estimation", color="blue")
plt.bar(x+0.5*width, sim_required_procs, width, label="Simulation", color="orange")
plt.xlabel("Mean time to failure")
plt.ylabel("Required processes")
plt.xticks(np.arange(len(lambs)), ["$\infty$" if lamb == 0 else int(round(1/lamb)) for lamb in lambs])
plt.legend()
plt.grid()
plt.savefig("figures/scaling/static-resource-mttf.png")

exit(0)


# Varying slack -----------------------------------------------------------------
comp_unit = 6
num_iter = 10
num_sinograms = 64
slacks = np.arange(1.2, 5.1, 0.2)
lamb = 0.001
ckpt_period = comp_unit
target_success = 0.9999


est_required_procs = [63, 55, 49, 45, 41, 39, 36, 34, 33, 31, 30, 29, 28, 27, 27, 26, 26, 25, 25, 24]
sim_required_procs = [61, 53, 49, 43, 39, 38, 34, 31, 31, 29, 28, 26, 25, 25, 23, 23, 22, 22, 20, 20]
# est_required_procs = []
# sim_required_procs = []
# for slack in slacks:
#     deadline = (comp_unit * num_iter) * slack
#     est_num_procs = configure_scaling(target_success, comp_unit, num_iter, num_sinograms, deadline, lamb, ckpt_period)
#     sim_num_procs = examine_scaling(target_success, comp_unit, num_iter, num_sinograms, deadline, lamb, ckpt_period)
#     est_required_procs.append(est_num_procs)
#     sim_required_procs.append(sim_num_procs)

print(slacks)
print(est_required_procs)
print(sim_required_procs)

plt.figure()
width=0.25
# plt.plot(slacks-1, est_required_procs, label="Estimation", color="blue")
# plt.plot(slacks-1, sim_required_procs, label="Simulation", color="orange")
# plt.xlabel("Slack/Ideal Runtime")
plt.plot(slacks, est_required_procs, label="Estimation", color="blue")
plt.plot(slacks, sim_required_procs, label="Simulation", color="orange")
plt.xlabel("Deadline/Ideal Runtime")
plt.ylabel("Required processes")
plt.legend()
plt.grid()
plt.savefig("figures/scaling/static-resource-slack.png")

exit(0)

# Varying target success rate ----------------------------------------------------
comp_unit = 6
num_iter = 10
num_sinograms = 64
slack = 2
deadline = (comp_unit * num_iter) * slack 
lamb = 0.001
ckpt_period = comp_unit

# target_successes = [0.9, 0.99, 0.999, 0.9999, 0.99999, 0.999999]
target_successes = [0.9, 0.99, 0.999, 0.9999, 0.99999]
est_required_procs = []
sim_required_procs = []
for target_success in target_successes:
    est_num_procs = configure_scaling(target_success, comp_unit, num_iter, num_sinograms, deadline, lamb, ckpt_period)
    sim_num_procs = examine_scaling(target_success, comp_unit, num_iter, num_sinograms, deadline, lamb, ckpt_period)
    print(target_success, est_num_procs, sim_num_procs)
    est_required_procs.append(est_num_procs)
    sim_required_procs.append(sim_num_procs)

print(target_successes)
print(est_required_procs)
print(sim_required_procs)

plt.figure()
width=0.25
x = np.arange(len(target_successes))
plt.bar(x-0.5*width, est_required_procs, width, label="Estimation", color="blue")
plt.bar(x+0.5*width, sim_required_procs, width, label="Simulation", color="orange")
plt.xlabel("Target success rate")
plt.ylabel("Required processes")
plt.xticks(x, target_successes)
plt.legend()
plt.grid()
plt.savefig("figures/scaling/static-resource-processes.png")

# Varying target success rate ----------------------------------------------------
comp_unit = 6
num_iter = 10
num_sinograms = 64
slack = 2
deadline = (comp_unit * num_iter) * slack 
lamb = 0.001
ckpt_period = comp_unit

target_successes = [0.9, 0.99, 0.999, 0.9999, 0.99999, 0.999999]
est_required_procs = []
sim_required_procs = []
for target_success in target_successes:
    est_num_procs = configure_scaling(target_success, comp_unit, num_iter, num_sinograms, deadline, lamb, ckpt_period)
    sim_num_procs = examine_scaling(target_success, comp_unit, num_iter, num_sinograms, deadline, lamb, ckpt_period)
    est_required_procs.append(est_num_procs)
    sim_required_procs.append(sim_num_procs)

print(target_successes)
print(est_required_procs)
print(sim_required_procs)

plt.figure()
width=0.25
x = np.arange(len(target_successes))
plt.bar(x-0.5*width, est_required_procs, width, label="Estimation", color="blue")
plt.bar(x+0.5*width, sim_required_procs, width, label="Simulation", color="orange")
plt.xlabel("Target success rate")
plt.ylabel("Required processes")
plt.xticks(x, target_successes)
plt.legend()
plt.grid()
plt.savefig("figures/scaling/static-resource-processes.png")

# Varying resources -----------------------------------------------------------
comp_unit = 6
num_iter = 10
num_sinograms = 64
slack = 2
deadline = (comp_unit * num_iter) * slack 
lamb = 0.001
ckpt_period = comp_unit
target_success = 0.9999

nprocs = np.arange(1, num_sinograms)
estimated_success_rates = []
simulated_success_rates = []
for num_procs in nprocs:
    estimated_success_rate = estimate_scaling_success(num_procs, comp_unit, num_iter, num_sinograms, deadline, lamb, ckpt_period)
    _, simulated_success_rate = scaling_sim(num_procs, comp_unit, num_iter, num_sinograms, deadline, lamb, ckpt_period)
    estimated_success_rates.append(estimated_success_rate)
    simulated_success_rates.append(simulated_success_rate)


print(nprocs)
print(estimated_success_rates)
print(simulated_success_rates)

plt.figure()
plt.plot(nprocs, estimated_success_rates, label="Estimation", color="blue")
plt.plot(nprocs, simulated_success_rates, label="Simulation", color="orange")
plt.xlabel("# processes")
plt.ylabel("Success rate")
plt.legend(loc="best")
plt.grid()
plt.savefig("figures/scaling/static-processes-miss-rate.png")
    




