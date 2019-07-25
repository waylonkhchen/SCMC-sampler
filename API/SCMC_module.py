#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 12 12:56:28 2019

@author: waylonchen
"""
import numpy as np
#from scipy.stats import norm
from constraints import Constraint
from scmc import scmc, init_sample
import matplotlib.pyplot as plt

#n_dim, size_sample, beta_max, seq_size, p_beta=1,p_rw_step=0
class SCMC():
    """Use sequentially constrained Monte Carlo method to sample given constrained domains uniformly
    
    Methods
    -------
    write_ouput:
        write the final sample in given path
        
    get_correctness:
        return the correctness of the final state
        
    get_history: 
        return a List, the history all samples in the sequential diffusion process
        
    plot_results: 
        scatter plot the results along two arbitrary axes
        params
        ------
            comp1, int: first axis (h)                
            comp2, int: second axis (v)        
            n_iter, int: if want history instead of final state
            
    print_constraints:
        print the original constraints
    
    
        
    
    """
    
    def __init__(self, input_path, n_results, beta_max = 10**4, p_beta=1,p_rw_step=0, track_correctness=False, threshold=.999):
        self.input_path = input_path
        self.constraints = Constraint(input_path)
        
        constraints_funcs = self.constraints.get_functions()
        
        self.n_dim = self.constraints.get_ndim()
        self.n_results = n_results
        self.beta_max = beta_max
        self.p_beta = p_beta
        self.p_rw_step = p_rw_step 
        
        #call sampling method scmc
        i_sample = init_sample(self.n_dim, self.n_results, self.constraints.bounds, self.constraints.get_example())
        self.i_samples = []
        self.i_samples.append(i_sample)
        self.histories=[]
        self.correctnesses=[]
        print(len(constraints_funcs))
        
        for i in range(1, len(constraints_funcs)+1):
            history, correctness =scmc(self.n_dim, self.n_results, self.i_samples[i-1] ,constraints_funcs[0:i], beta_max, self.constraints.bounds,
                                              p_beta,p_rw_step,
                                              track_correctness=track_correctness,
                            given_example = self.constraints.get_example(), threshold = threshold )
            self.i_samples.append(history[-1])
            self.histories.append(history)
            self.correctnesses.append(correctness)
        self.results = self.histories[-1][-1]
            
        
#        self.history, self.correctness = scmc(self.n_dim, self.n_results, i_sample ,constraints_funcs, beta_max, self.constraints.bounds,
#                                              p_beta,p_rw_step,
#                                              track_correctness=track_correctness,
#                            given_example = self.constraints.get_example())
#        self.results = self.history[-1]
#        self.history, self.correctness = scmc(self.n_dim, self.n_results, constraints_funcs, beta_max, self.constraints.bounds,
#                                              p_beta,p_rw_step,
#                                              track_correctness=track_correctness,
#                            given_example = self.constraints.get_example())
#        self.results = self.history[-1]
        
#        self.write_ouput(output_path)

    def write_output(self, output_path):
        """
        save the sample to a output_path
        """
        np.savetxt(output_path, self.results)
        
    def get_correctness(self):
        """
        get correctness
        """
        correctness = [self.constraints.apply(x) for x in self.results]
        correctness = sum(correctness) / self.n_results
        return correctness
    
    def get_history(self):
        return self.history
    
    def get_results(self):
        """
        get the sample
        """
        return self.results
    
    def plot_results(self,comp1, comp2, n_iter =None):
        """
        Plot the sample in x[comp1],x[comp2] 
        """
        if n_iter is None:
            sample = self.results
        else:
            sample = self.history[n_iter-1]
            
        plt.figure()
        plt.scatter(sample[:,comp1], sample[:,comp2])
        plt.xlim(0,1);
        plt.ylim(0,1);
        
    def plot_all_axis(self, n_iter =None):
        
        n_dim = self.n_dim
        for i in range(n_dim ):
            for j in range(i+1,n_dim):
                self.plot_results(i,j,n_iter)
                plt.xlabel('x_{}'.format(i));
                plt.ylabel('x_{}'.format(j));

    def print_constraints(self):
        print(self.constraints.get_exprs_string())


