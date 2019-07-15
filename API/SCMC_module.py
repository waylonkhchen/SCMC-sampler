#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 12 12:56:28 2019

@author: waylonchen
"""
import numpy as np
#from scipy.stats import norm
from constraints import Constraint
from scmc import scmc

#n_dim, size_sample, beta_max, seq_size, p_beta=1,p_rw_step=0
class SCMC():
    
    def __init__(self, input_path, n_results, beta_max = 500, seq_size = 30, p_beta=1,p_rw_step=0):
        const = Constraint(input_path)
        constraints_funcs = const.get_functions()
        
        self.n_dim = const.get_ndim()
        self.n_results = n_results
        self.beta_max = beta_max
        self.seq_size = seq_size
        self.p_beta = p_beta
        self.p_rw_step = p_rw_step 
        
        self.history = scmc(self.n_dim, self.n_results, constraints_funcs, beta_max, seq_size, p_beta,p_rw_step)
        self.output = self.history[-1]
        
#        self.write_ouput(output_path)

    def write_output(self, output_path):
        np.savetxt(output_path, self.output)
        


