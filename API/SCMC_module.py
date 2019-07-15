#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 12 12:56:28 2019

@author: waylonchen
"""
import numpy as np
from scipy.stats import norm
from constraints import Constraint
#from scipy import optimize

file_path = '../example.txt'
const = Constraint(file_path)

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
    """
    return optimize.root_scalar(ESS(importances,beta), bracket = ,method=')
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
def log_w( be, t, n, samples , constraints_funcs , beta):
    """
    function of ``be",
    evaluate the umnormalized importance weights w^t_n point-wise
    Parameters
    ----------
    t: int, iteration count
    
    n: int, index of the point record x in a sample
    
    samples: List, length t+1
        containing np.array of shape (size_sample, n_dim)
    
    constraints_funcs: List, length n_constraint
        containing function that evaluates the constraint at a given point record x
        i.e. the algebraic part g(x) of the expression: g(x) >= 0
    
    beta: List, length t+1
        containing the past inverse temperature beta
    
        
    Returns
    -------
    float
        
        
    """
    res = np.sum([norm.logcdf( be * g(samples[t-1][n])) for g in constraints_funcs] )
    res -= np.sum( [norm.cdf( beta[t-1]* g( samples[t-1][n] ) ) for g in constraints_funcs])
    return res


### testing w
#t=1; n=0; n_dim=2
#samples = np.array([[[.9,.6],[.5,.6],[.1,.6]],
#                    [[.1,.2],[.3,.4],[.5,.6]]])
#constraints_funcs = [lambda x: x[0]+x[1],lambda x: x[0]-x[1]]
#beta = [1]
#
#be=1;
#print(w( be, t, n, samples , constraints , beta))

    
    
def importance_resampling(be , samples ,t, beta , constraints_funcs):
    """
    Parameters
    ----------
    constraints_funcs: List, length n_constraint
    containing function that evaluates the constraint at a given point record x
    
    """
    sample = samples[t-1]
    size_sample = len(sample)
    W = weights_initial(size_sample)

    
    imp_weights = np.array( [ log_w( be, t, n, samples , constraints_funcs , beta) for n in range(size_sample) ])
    W = np.log(W) + imp_weights
    #normalize W
    W = np.exp(W)
    W = W / np.sum(W)
    return sample[ np.random.choice(a = size_sample, size = size_sample,p = W)]

#Metropolis Random Walk
    
def avoid_leakage_rewalk(x, delta, rw_step, interval=[0,1]):
    n_dim = len(x)
    for i in range(n_dim):
        while (x[i]+delta[i] > interval[1] or x[i]+ delta[i]< interval[0] ):
            delta[i] =  np.random.normal(scale = rw_step[i])
    return delta

def avoid_leakage_rebound(x, delta, rw_step, interval=[0,1]):
    n_dim = len(x)
    for i in range(n_dim):
        while(x[i]+delta[i] > interval[1] or x[i]+delta[i] < interval[0]):
            if x[i]+delta[i] > interval[1]:
                delta[i] =  2*(interval[1]-x[i]) -delta[i]
        
            if x[i]+delta[i] < interval[0] :
                delta[i] =   2*(interval[0]-x[i]) -delta[i]
    return delta
        
def proposal(x, rw_step):
    """
    random walk proposal for each record in the sample
    """
    n_dim = len(x)
    delta = np.random.normal(scale = rw_step, size = n_dim) 
#    xx = x + delta
    
    #avoid leakage
#    delta = avoid_leakage_rewalk(x, delta, rw_step)
    delta = avoid_leakage_rebound(x, delta, rw_step)
    x += delta
    return x



def log_target(be, x, constraints_funcs):
    """
    log (phi(beta * g(x))
    Parameters
    ----------
    x:
        
    
    """
    return np.sum([norm.logcdf( be * g(x)) for g in constraints_funcs] )

def log_accept(be, x, x_, constraints_funcs):
    return log_target(be, x, constraints_funcs ) - log_target(be, x_, constraints_funcs)
    
def adaptive_step(sample, t, p):
    return np.var(sample, axis =0)/t**p

def Metropolis(be, t, sample, proposal ,constraints_funcs,p):
    current = sample
    size_sample = len(current)
    rw_step = adaptive_step(current, t, p)
    print(rw_step)
    proposed = [ proposal(x, rw_step) for x in current ]
    log_acc = [ min( 0, log_accept(be, x, x_, constraints_funcs)  ) for x,x_ in zip(proposed, current)]
    acc = np.exp(log_acc)
    
    p = np.random.uniform(size = size_sample)
    new_sample = [proposed[i] if (p<acc)[i] else current[i] for i in range(size_sample)]
    return np.array(new_sample)
    
    
    


def scmc(n_dim, size_sample, beta_max, seq_size, p_beta=1,p_rw_step=0):
    t = 0
    beta = [0]
    #Generate uniform samples in n_dim cube [0,1]^n_dim
    ##samples[t] = W^t_1:n , containing size_sample rows of point in n_dim space
    samples = [uniform_samples(n_dim, size_sample)]
    
#    #initialize weights
#    W = weights_initial(size_sample)
    
    while beta[t] < beta_max:
        t += 1
        #assign next beta
#        be = optimal_next_beta()
        be = beta_poly(t, seq_size, p_beta, beta_max )
        beta.append(be)
        
        #importance resampling
        constraints_funcs = const.get_functions()
        resample = importance_resampling(be , samples ,t, beta , constraints_funcs)
        
        #Random Walk using Markov Chain kernel
        new_sample = Metropolis(be, t, resample, proposal ,constraints_funcs ,p_rw_step)
        samples.append(new_sample)
    return samples
    
    
    

    
    