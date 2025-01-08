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


# def estimate_dynamic_scaling_success(new_procs, mu, num_procs, comp_unit, num_iter, num_sinograms, deadline, lamb, ckpt_period):
#     """
#     mu:     new processes arrival rate
#     lamb:   process failure rate
#     """
#     total_computation = comp_unit * num_iter * num_sinograms
#     total_success_prob = 0
#     for num_added in range(0, new_procs+1):
#         added_prob = stats.poisson.pmf(k=num_added, mu=new_procs*mu*deadline)
#         for num_failures in range(0, num_procs+num_added+1):
        
#             failure_prob = stats.poisson.pmf(k=num_failures, mu=(num_procs+num_added)*lamb*deadline)
            
#             extra_computation = 0.5 * num_failures * ckpt_period
#             require_computation = total_computation + extra_computation
#             safe_capacity = (num_procs + num_added) * deadline
#             safe_capacity -= require_computation
            
#             success_prob = 0
#             if safe_capacity >= 0:
#                 if num_added + num_failures > 0:
#                     success_prob = stats.irwinhall.cdf(x=safe_capacity/deadline, n=num_failures+num_added)
#                 else:
#                     success_prob = 1
#             total_success_prob += added_prob * failure_prob * success_prob
#     return total_success_prob

def incident_pdf(x, lamb, deadline):
    """
        The pdf of incident given a deadline and occurrence rate [lamb]
        :param x: Array of values where PDF is evaluated.
        :param lambda: Rate parameter , currently assuming incidents following an exponential distribution.
        :param deadline: Deadline of occurrences, beyond this point, no occurences are counted.
        :return: PDF values corresponding to x.
    """
    # The bounded probability is a conditional probability with the given condition is incidents falling into [0, deadline],
    # that is, bounded probabilility(x) = P(x | x <= deadline) = P(x AND x <= deadline) / P(x <= deadline)
    norm_factor = 1 - np.exp(-lamb * deadline)
    return np.where((x >= 0) & (x <= deadline), lamb * np.exp(-lamb * x) / norm_factor, 0)

def gaussian_approx_exponential_params(lamb, deadline):
    norm_factor = 1 - np.exp(-lamb * deadline)
    # mean = (1 - (deadline + 1) * np.exp(-lamb * deadline)) / (lamb * norm_factor)
    # variance = ((1 - (deadline**2 + 2 * deadline + 2) * np.exp(-lamb * deadline)) /
    #             (lamb**2 * norm_factor)) - mean**2
    mean = (1 - (deadline*lamb + 1)*np.exp(-lamb * deadline)) / (norm_factor * lamb)
    variance = (2 - ((deadline*lamb)**2 + 2*deadline*lamb + 2)*np.exp(-lamb * deadline)) / (norm_factor * lamb**2) - mean**2
    return mean, variance

def gaussian_approx_incident_pdf(x, lamb, deadline, n):
    mean, variance = gaussian_approx_exponential_params(lamb, deadline)
    sum_mean = n * mean
    sum_variance = n * variance
    sum_std = np.sqrt(sum_variance)
    return stats.norm.pdf(x, loc=sum_mean, scale=sum_std)


def compute_prob(pdf, x_values, x_target):
    dx = x_values[1] - x_values[0]  # Interval width
    indices = x_values <= x_target  # Find indices where x <= x_target
    probability = np.sum(pdf[indices]) * dx  # Sum up PDF values up to x_target
    return probability

def compute_success_prob_convolve(deadline, mu, lamb, num_added, num_failures, safe_capacity):
    """
        Compute the succesful probability using convolution
        WARNING: Expensive, should consider Gaussian approximation if number of num_added+num_failures is big
    """
    success_prob = 0
    pdf = None
    t = None
    if safe_capacity >= 0:
        if num_added + num_failures > 0:
            # Now we assume of having [num_added] new process added sometime between [0, deadline]
            # and [num_failures] processes are failed also between [0, deadline]
            # Added processes are unavailable until they arrive: unavailable = arrival time
            # Failed processes are unavailable once they fail: unavailable = deadline - failed_time
            # Thus, the time they are available is:
            #       unavailable = sum(arrival time) + sum(deadline - failure time)
            #                   = deadline*num_failures + sum(arrival time) - sum(falure time)
            #   and to succes,
            #       unavailable <= safe_capacity
            # We just need to compute the probability that
            #       sum(arrival time) - sum(falure time) <= safe_capacity - deadline*num_failures
            # This can be done through convolution
            t_values = np.linspace(-deadline, deadline, 1000)
            single_added_pdf = incident_pdf(t_values, mu, deadline)
            single_failure_pdf = incident_pdf(t_values, lamb, deadline)[::-1]
            if num_added > 0:
                conv_pdf = single_added_pdf
                for _ in range(1, num_added):
                    conv_pdf = np.convolve(conv_pdf, single_added_pdf, mode="full") * (t_values[1] - t_values[0])
                fstart = 0
            else:
                conv_pdf = single_failure_pdf
                fstart = 1
            for _ in range(fstart, num_failures):
                conv_pdf = np.convolve(conv_pdf, single_failure_pdf, mode="full") * (t_values[1] - t_values[0])
            t_sum = np.linspace((num_added+num_failures) * t_values[0], (num_added+num_failures) * t_values[-1], len(conv_pdf)) # Adjust range for the sum

            success_prob = compute_prob(conv_pdf, t_sum, safe_capacity-deadline*num_failures)

            pdf = conv_pdf
            t = t_sum
        
            # # Plot the original and summed distributions
            # print(safe_capacity-deadline*num_failures)
            # plt.figure(figsize=(10, 6))
            # plt.plot(t_values, single_added_pdf, label=f"Added")
            # plt.plot(t_values, single_failure_pdf, label=f"Failures")
            # plt.plot(t_sum, conv_pdf, label=f"Total", linestyle="--")
            # plt.axvline(x=safe_capacity-deadline*num_failures, color="black", linestyle=":")
            # plt.xlabel("t")
            # plt.ylabel("PDF")
            # plt.legend()
            # plt.grid()
            # plt.show()
            # plt.savefig("figures/scaling/estimate-compute.png")
        else:
            success_prob = 1

    return success_prob, t, pdf


def compute_success_prob_gaussian_approx(deadline, mu, lamb, num_added, num_failures, safe_capacity):
    """
        Compute the succesful probability using convolution
        WARNING: Inaccurate if num_added + num_failures is small
    """
    success_prob = 0
    pdf = None
    t = None
    if safe_capacity >= 0:
        if num_added + num_failures > 0:
            # t_values = np.linspace(-deadline, deadline, 1000)
            # if num_added > 0:
            #     conv_pdf = gaussian_approx_incident_pdf(t_values, mu, deadline, num_added)
            #     if num_failures > 0:
            #         single_failure_pdf = gaussian_approx_incident_pdf(t_values, lamb, deadline, num_failures)[::-1]
            #         conv_pdf = np.convolve(conv_pdf, single_failure_pdf, mode="full") * (t_values[1] - t_values[0])
            # else:
            #     conv_pdf = incident_pdf(t_values, lamb, deadline, num_failures)[::-1]
            # t_sum = np.linspace((num_added+num_failures) * t_values[0], (num_added+num_failures) * t_values[-1], len(conv_pdf)) # Adjust range for the sum

            value_scale = num_added+num_failures
            t_values = np.linspace(-deadline*value_scale, deadline*value_scale, 1000*value_scale)
            
            mean_add, var_add = gaussian_approx_exponential_params(mu, deadline)
            mean_failure, var_failures = gaussian_approx_exponential_params(lamb, deadline)
            approx_mean = mean_add * num_added - mean_failure * num_failures
            approx_var = var_add * num_added - mean_failure * num_failures
            approx_std = np.sqrt(approx_var)
            approx_pdf = stats.norm.pdf(t_values, loc=approx_mean, scale=approx_std)

            success_prob = compute_prob(approx_pdf, t_values, safe_capacity-deadline*num_failures)

            pdf = approx_pdf
            t = t_values
        
            # # Plot the original and summed distributions
            # print(safe_capacity-deadline*num_failures)
            # plt.figure(figsize=(10, 6))
            # plt.plot(t_values, gaussian_approx_incident_pdf(t_values, mu, deadline, num_added), label=f"Added")
            # plt.plot(t_values, gaussian_approx_incident_pdf(t_values, lamb, deadline, num_failures)[::-1], label=f"Failures")
            # plt.plot(t_values, approx_pdf, label=f"Total", linestyle="--")
            # plt.axvline(x=safe_capacity-deadline*num_failures, color="black", linestyle=":")
            # plt.xlabel("t")
            # plt.ylabel("PDF")
            # plt.legend()
            # plt.grid()
            # plt.show()
            # plt.savefig("figures/scaling/estimate-compute.png")
        else:
            success_prob = 1

    return success_prob, t, pdf

def compute_success_prob(deadline, mu, lamb, num_added, num_failures, safe_capacity):
    conv_prob, conv_t, conv_pdf = compute_success_prob_convolve(deadline, mu, lamb, num_added, num_failures, safe_capacity)
    apprx_prob, apprx_t, apprx_pdf = compute_success_prob_gaussian_approx(deadline, mu, lamb, num_added, num_failures, safe_capacity)

    # # Plot the original and summed distributions
    # print(safe_capacity-deadline*num_failures)
    # plt.figure(figsize=(10, 6))
    # plt.plot(conv_t, conv_pdf, label=f"Conv", linestyle="--")
    # plt.plot(apprx_t, apprx_pdf, label=f"Approx", linestyle="-")
    # plt.axvline(x=safe_capacity-deadline*num_failures, color="black", linestyle=":")
    # plt.xlabel("t")
    # plt.ylabel("PDF")
    # plt.legend()
    # plt.grid()
    # plt.show()
    # plt.savefig("figures/scaling/estimate-compute.png")

    success_prob = conv_prob

    return success_prob


def estimate_dynamic_scaling_success(new_procs, mu, num_procs, comp_unit, num_iter, num_sinograms, deadline, lamb, ckpt_period):
    """
    mu:     new processes arrival rate
    lamb:   process failure rate
    """
    total_computation = comp_unit * num_iter * num_sinograms
    arrival_before_deadline_prob = stats.expon.cdf(deadline, scale=1/mu)
    total_success_prob = 0
    for num_added in range(0, new_procs+1):
        added_prob = stats.binom.pmf(k=num_added, n=new_procs, p=arrival_before_deadline_prob)
        if added_prob < 0.00000000001:
            continue
        for num_failures in range(0, num_procs+num_added+1):
        
            failure_prob = stats.poisson.pmf(k=num_failures, mu=(num_procs+num_added)*lamb*deadline)
            if failure_prob < 0.00000000001:
                continue

            extra_computation = 0.5 * num_failures * ckpt_period
            require_computation = total_computation + extra_computation
            safe_capacity = (num_procs + num_added) * deadline
            safe_capacity -= require_computation

            success_prob = compute_success_prob(deadline, mu, lamb, num_added, num_failures, safe_capacity)
            total_success_prob += added_prob * failure_prob * success_prob
            # print(num_added, num_failures, safe_capacity, added_prob, failure_prob, success_prob, "-->", added_prob * failure_prob * success_prob)
    return total_success_prob


def configure_dynamic_scaling(target_success, mu, num_procs, comp_unit, num_iter, num_sinograms, deadline, lamb, ckpt_period):
    for new_procs in range(1, num_sinograms-num_procs+1):
        success_prob = estimate_dynamic_scaling_success(new_procs, mu, num_procs, comp_unit, num_iter, num_sinograms, deadline, lamb, ckpt_period)
        if success_prob >= target_success:
            return new_procs
    return num_sinograms-num_procs


# deadline = 10
# mu = 0.1
# lamb = 0.001
# num_added = 8
# num_failures = 8
# safe_capacity = 60
# prob = compute_success_prob(deadline, mu, lamb, num_added, num_failures, safe_capacity)
# print(prob)
# exit(0)

comp_unit = 6
num_iter = 10
num_sinograms = 32
deadline = 120
lamb = 0.001
mu = 0.1
# lamb = 0
ckpt_period = comp_unit
target_success = 0.9999
num_procs = int(num_sinograms / 4)

# new_procs = 8
# estimate_dynamic_scaling_success(new_procs, mu, num_procs, comp_unit, num_iter, num_sinograms, deadline, lamb, ckpt_period)
# exit(0)

for new_procs in range(1, num_sinograms-num_procs+1):
    print("total_procs (num_procs + new_procs) =", num_procs+new_procs, "(", num_procs, "+", new_procs, ") ------------------------------ ")
    print("Estimation", estimate_dynamic_scaling_success(new_procs, mu, num_procs, comp_unit, num_iter, num_sinograms, deadline, lamb, ckpt_period))
    _, success_rate = dynamic_scaling_sim(new_procs, mu, num_procs, comp_unit, num_iter, num_sinograms, deadline, lamb, ckpt_period)
    print("Simulation", success_rate)


new_procs = configure_dynamic_scaling(target_success, mu, num_procs, comp_unit, num_iter, num_sinograms, deadline, lamb, ckpt_period)
print("total_procs (num_procs + new_procs) =", num_procs+new_procs, "(", num_procs, "+", new_procs, ")")
runtime, success_rate = dynamic_scaling_sim(new_procs, mu, num_procs, comp_unit, num_iter, num_sinograms, deadline, lamb, ckpt_period)
print("runtime", runtime, "deadline", deadline, "target", target_success)
print("success_rate", success_rate)


    




