import numpy as np
import sys
import json
import math

import pickle

import scipy.stats as stats

import matplotlib.pyplot as plt

def dynamic_scaling_sim(new_procs, mu, num_procs, comp_unit, num_iter, num_sinograms, deadline, lamb, ckpt_period):
    
    num_tries = 1000

    total_runtime = 0
    num_success = 0
    num_complete = 0

    for i in range(num_tries):
        total_computation = comp_unit * num_iter * num_sinograms
        prev_num_failed = 0
        progress = 0
        last_ckpt = 0

        # prepare the arrival time for new processes
        arrival_times = []
        if mu == 0:
            arrival_times = np.ones(num_procs+new_procs) * 100000000000
        else:
            arrival_times = np.random.exponential(scale=1/mu, size=num_procs+new_procs)

        # prepare failure time of existing processes + added processes
        failure_times = []
        if lamb == 0:
            failure_times = np.ones(num_procs+new_procs) * 100000000000
        else:
            failure_times = np.random.exponential(scale=1/lamb, size=num_procs+new_procs)
        
        # For added processes, we need to shift their failure time to make
        # sure they fail after arrived
        for j in range(new_procs):
            failure_times[j] += arrival_times[j]
        for j in range(new_procs, new_procs+num_procs):
            arrival_times[j] = 0

        t = 0
        while progress < total_computation:
            failed_processes = (failure_times <= t).astype(int)
            num_failed = np.sum(failed_processes)
            if num_failed > prev_num_failed:
                total_computation += (num_failed  - prev_num_failed) * (t - last_ckpt)
                prev_num_failed = num_failed

            added_processes = (arrival_times <= t).astype(int)
            num_added = np.sum(added_processes)
            
            num_avail = num_added - num_failed
            if num_avail == 0:
                t = -1
                break

            progress += num_avail
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

"""
    mu:     new processes arrival rate
    lamb:   process failure rate
"""
def estimate_dynamic_scaling_success(new_procs, mu, num_procs, comp_unit, num_iter, num_sinograms, deadline, lamb, ckpt_period):
    total_computation = comp_unit * num_iter * num_sinograms
    total_success_prob = 0
    for num_added in range(0, new_procs+1):
        added_prob = stats.poisson.pmf(k=num_added, mu=new_procs*mu*deadline)
        for num_failures in range(0, num_procs+num_added+1):
        
            failure_prob = stats.poisson.pmf(k=num_failures, mu=(num_procs+num_added)*lamb*deadline)
            
            extra_computation = 0.5 * num_failures * ckpt_period
            require_computation = total_computation + extra_computation
            safe_capacity = (num_procs + num_added) * deadline
            safe_capacity -= require_computation
            
            success_prob = 0
            if safe_capacity >= 0:
                if num_added + num_failures > 0:
                    success_prob = stats.irwinhall.cdf(x=safe_capacity/deadline, n=num_failures+num_added)
                else:
                    success_prob = 1
            total_success_prob += added_prob * failure_prob * success_prob
    return total_success_prob


def configure_dynamic_scaling(target_success, mu, num_procs, comp_unit, num_iter, num_sinograms, deadline, lamb, ckpt_period):
    for new_procs in range(1, num_sinograms-num_procs+1):
        success_prob = estimate_dynamic_scaling_success(new_procs, mu, num_procs, comp_unit, num_iter, num_sinograms, deadline, lamb, ckpt_period)
        if success_prob >= target_success:
            return new_procs
    return num_sinograms-num_procs

comp_unit = 6
num_iter = 10
num_sinograms = 32
deadline = 120
lamb = 0.001
mu = 0.1
# lamb = 0
ckpt_period = comp_unit
target_success = 0.99
num_procs = int(num_sinograms / 4)

for new_procs in range(1, num_sinograms-num_procs+1):
    print("total_procs (num_procs + new_procs) =", num_procs+new_procs, "(", num_procs, "+", new_procs, ") ------------------------------ ")
    print("Estimation", estimate_dynamic_scaling_success(new_procs, mu, num_procs, comp_unit, num_iter, num_sinograms, deadline, lamb, ckpt_period))
    _, success_rate = dynamic_scaling_sim(new_procs, mu, num_procs, comp_unit, num_iter, num_sinograms, deadline, lamb, ckpt_period)
    print("Simulation", success_rate)


new_procs = configure_dynamic_scaling(target_success, mu, num_procs, comp_unit, num_iter, num_sinograms, deadline, lamb, ckpt_period)
print("total_procs (num_procs + new_procs) =", num_procs+new_procs, "(", num_procs, "+", new_procs)
runtime, success_rate = dynamic_scaling_sim(new_procs, mu, num_procs, comp_unit, num_iter, num_sinograms, deadline, lamb, ckpt_period)
print("runtime", runtime, "deadline", deadline)
print("success_rate", success_rate)


    




