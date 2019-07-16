#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 15 10:02:53 2019

@author: waylonchen
"""
#from scmc import scmc
import matplotlib.pyplot as plt
from constraints import Constraint
from SCMC_module import SCMC






#module testing
#input_path = '../example.txt'
input_path = '../formulation.txt'
n_results = 1000
sampling_method = SCMC(input_path, n_results, track_correctness=True)



print(sampling_method.get_correctness())

print(sampling_method.print_constraints())

n_dim = sampling_method.n_dim
for i in range(n_dim ):
    for j in range(i+1,n_dim):
        sampling_method.plot_results(i,j)
        plt.xlabel('x_{}'.format(i));
        plt.ylabel('x_{}'.format(j));

input_path = '../mixture.txt'
n_results = 1000
sampling_method = SCMC(input_path, n_results, track_correctness=True)



#test with
input_path = '../example.txt'
n_results = 1000
sampling_method = SCMC(input_path, n_results)
print(sampling_method.get_correctness())

print(sampling_method.print_constraints())

n_dim = sampling_method.n_dim
for i in range(n_dim ):
    for j in range(i+1,n_dim):
        sampling_method.plot_results(i,j)
        plt.xlabel('x_{}'.format(i));
        plt.ylabel('x_{}'.format(j));
        
        
        
        
        
input_path = '../alloy.txt'
n_results = 1000
sampling_method = SCMC(input_path, n_results,show_correctness=True)
print(sampling_method.get_correctness())



n_dim = sampling_method.n_dim
for i in range(n_dim ):
    for j in range(i+1,n_dim):
        sampling_method.plot_results(i,j)
        plt.xlabel('x_{}'.format(i));
        plt.ylabel('x_{}'.format(j));