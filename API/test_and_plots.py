#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 15 10:02:53 2019

@author: waylonchen
"""
from scmc import scmc
import matplotlib.pyplot as plt
from constraints import Constraint

file_path = '../example.txt'
const = Constraint(file_path)
constraints_funcs = const.get_functions()

#def scmc(n_dim, size_sample, constraints_funcs, beta_max, seq_size, p_beta=1,p_rw_step=0):
samples = scmc(4, 1000,constraints_funcs, beta_max = 100, seq_size = 20)


sample = samples[-1]

plt.figure()
plt.scatter(sample[:,0], sample[:,1])
plt.xlim(0,1);
plt.ylim(0,1);

plt.figure()
plt.scatter(sample[:,1], sample[:,2])
plt.xlim(0,1);
plt.ylim(0,1);

plt.figure()
plt.scatter(sample[:,1], sample[:,3])
plt.xlim(0,1);
plt.ylim(0,1);

