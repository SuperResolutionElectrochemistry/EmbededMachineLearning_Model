# Title: Ensemble-machine-learning accelerated high-throughput screening of high-performance PtPd-based high-entropy alloy hydrogen evolution electrocatalysts
# Author: Xiangyi Shan, Yiyang Pan, Furong Cai, Min Zhou*
# Corresponding author. Email: mzhou1982@ciac.ac.cn (M. Z.)

#  This code could generate element combinations of fixed step sizes.

import numpy as np
import pandas as pd

def generate_value_range(start, end, step):
    return np.arange(start, end + step, step)

def find_unique_combinations(start, end, step, total_sum=100, last_two_sum=40):
    value_range = generate_value_range(start, end, step)
    unique_combinations = []
    for a in value_range:
        for b in value_range:
            for c in value_range:
                for d in value_range:
                    for e in value_range:
                        if a + b + c + d + e == total_sum and d + e == last_two_sum:
                            unique_combinations.append((a, b, c, d, e))

    return unique_combinations

