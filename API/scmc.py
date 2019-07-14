#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 12 12:56:28 2019

@author: waylonchen
"""
import numpy as np
from scipy.stats import norm
#from scipy import optimize



def uniform_samples(n_dim, size_sample):
    return np.array([np.random.uniform(size = n_dim) for i in range(size_sample)])

def weights_initial(size_sample):
    return np.ones(size_sample)/size_sample




def ESS(importances, beta):
    """
    return a function of that calculates Effective sample size(ESS) from
    unnomalized importnace w
    
    Parameters
    ----------
    w   : np.array, shape ( n_dim, 1 )
    
    Returns
    -------
    ESS :float
    """
    pass
## refer to sigma to define function from sum of an array
#    funcs = [f for f in importances]
#    def sigma(funcs,x): #sum of an array of functions
#        return sum([f(x) for f in funcs]) 
#    def w( t, n )
    
#    return (np.sum(w))**2 / np.sum(w**2)
    

def optimal_next_beta():
    """return optimize.root_scalar(ESS(importances,beta), bracket = ,method=')
    """
    pass

def beta_poly(t, seq_size, p, beta_max):
    """
    generate inv temperature beta with polynomial growth (\propto t^p)
    i.e. beta_poly(t) \porpto t^p. 
    When p=1 beta is linspace(0,beta_max,seq_size)
    
    Parameters    
    ----------
    t: int,  iteration count, also the sequence index in SMC
    
    seq_size: int, the size of the sequence
    
    p: int, degree of t for beta(t)
    
    beta_max: the traget inverse temperature beta in SMC
    """
    return beta_max * (t/seq_size)**p
 

#unnormalized importance weights
def w( be, t, n, samples , constraints , beta):
    """
    function of ``be",
    evaluate the umnormalized importance weights w^t_n point-wise
    Parameters
    ----------
    t: int, iteration count
    
    n: int, index of the point record x in a sample
    
    samples: List, length t+1
        containing np.array of shape (size_sample, n_dim)
    
    constraints: List, length n_constraint
        containing function that evaluate the constraint at a given point record x
        constraint is the algebraic part (g) of the expression: g >= 0
    
    beta: List, length t+1
        containing the past inverse temperature beta
    
        
    Returns
    -------
    float
        
        
    """
    
    res = np.sum([norm.logcdf( be* constraint( samples[t-1][n])) for constraint in constraints] )
    res -= np.sum( [norm.cdf( beta[t-1]* constraint( samples[t-1][n] ) ) for constraint in constraints])
    return np.exp(res)


### testing w
#t=1; n=0; n_dim=2
#samples = [np.array([[.9,.6],[.5,.6],[.1,.6]])]
#constraints = [lambda x: x[0]+x[1],lambda x: x[0]-x[1]]
#beta = [1]
#
#be=1;
#print(w( t, n, samples , constraints , beta, be))

    
    
def importance_resampling(be , samples ,t, beta , constraints):
    size_sample = samples[0].shape(0)
    W = weights_initial(size_sample)
    sample = samples[t-1]
    
    imp_weights = np.array( [w( be, t, n, samples , constraints , beta) for n in range(size_sample) ])
    W = W * imp_weights
    #normalize W
    W = W / np.sum(W)
    
    return sample[np.random.choice(a = len(sample), size = len(sample),p = W)]

def scmc(n_dim, size_sample, beta_max, seq_size, p=1):
    t = 0
    beta = [0]
    #Generate uniform samples in n_dim cube [0,1]^n_dim
    ##samples[t] = W^t_1:n , containing size_sample rows of point in n_dim space
    samples = [uniform_samples(n_dim, size_sample)] 
    W = weights_initial(size_sample)
    #initialize weights
    
    

    
    
    while beta[t] < beta_max:
        t += 1
        
        #next beta
#        be = optimal_next_beta()
        be = beta_poly(t, seq_size, p, beta_max )
        beta.append(be)
        
        #importance sampling
        
        
        
    
    
    
    

    
    