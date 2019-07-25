#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 15 10:02:53 2019

@author: waylonchen
"""
from SCMC_module import SCMC






#module testing
#input_path = '../example.txt'
#input_path = '../formulation.txt'
input_path = '../alloy.txt'

n_results = 1000
#run the sampler
sampling1 = SCMC(input_path, n_results, track_correctness=True,threshold=.999)


#plot x[0], x[1]
sampling1.plot_results(0,1)

#plot all pair axis
sampling1.plot_all_axis()

#print_constraints()
sampling1.print_constraints()
