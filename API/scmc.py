#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 12 12:56:28 2019

@author: waylonchen
"""
import numpy as np
from scipy.stats import norm
from scipy import optimize
from scipy.special import expit


#file_path = '../example.txt'
#const = Constraint(file_path)

#def uniform_samples(n_dim, size_sample):
#    return np.random.uniform(size = (size_sample, n_dim))


def weights_initial(size_sample):
    return np.ones(size_sample)/size_sample



#
def ESS(be, t, samples , constraints_funcs , beta, indicator_how):
    """
    return a function of ``be" that calculates Effective sample size(ESS) from
    unnomalized importnace w
    
    Parameters
    ----------
    be: float,
        the curent inverse temperature,
    
    Returns
    -------
    ESS :float
    """
    size_sample = len(samples[0])
    w = np.exp([log_w( be, t, n, samples , constraints_funcs , beta, indicator_how) for n in range(size_sample) ])
    ess =np.sum(w)**2/np.sum(w**2)
    return ess

def ESS_p(be, t, samples , constraints_funcs , beta, indicator_how):
    """
    first derivative of ESS(be)
    
    Parameters
    ----------
    be: float,
        the curent inverse temperature,
    
    Returns
    -------
    ESS :float
    """
    size_sample = len(samples[0])
    w = np.exp([log_w( be, t, n, samples , constraints_funcs , beta, indicator_how) for n in range(size_sample) ])
    w_p = np.exp([log_w_p( be, t, n, samples , constraints_funcs , beta, indicator_how) for n in range(size_sample) ])
    
    sum_w = np.sum(w)
    sum_ww = np.sum(w**2)

    ess_p =2* sum_w * np.sum(w_p)* sum_ww- 2* sum_w**2 * np.sum(w * w_p)
    ess_p = ess_p/sum_ww**2
    return ess_p




def toms748(objective, bracket):
    try:
        sol = optimize.root_scalar(
                objective,

                    bracket = bracket,
                    method='toms748')
    except:
#        if bracket[0]<1:
#            return (bracket[0]+10**-4)*1.2
        return bracket[1]
    return sol.root

#this solver wouldn't work...
#def newton(objective, x0, fprime):
#    try:
#        sol = optimize.root_scalar(
#                func = objective,
#                x0 = x0,
#                fprime = fprime,
#                method='newton')
#        return sol.root
#    except:
#        if x0 <1:
#            try:
#                sol = newton(objective, 1, fprime)
#                return sol.root
#            except:
#                return x0*2
#        else:
#            sol = newton(objective, 1, fprime)
#    return sol.root
 

def find_root( objective, x0,fprime, bracket , geomspace, t):
    """
    Using toms748 method for root finding, which requires a bracket that contain the root.
    The more precise precise the bracket is, the faster the convergence.
    Therefore, I partition interval [0, beta_max] into geometric space of 15 partitions.
    The root finding will scan in each partition sequentially
    """
#    interval = np.linspace(0, geomspace[0], 10)
#    if bracket[0] < geomspace[0]:
#        return interval[t]
    i=0
    while i < len(geomspace):
        while bracket[0] < geomspace[i]:
            return toms748(objective,bracket= [bracket[0], geomspace[i] ])
        i+=1
            
    return toms748(objective,bracket= bracket )
#    return newton(objective, x0, fprime)
#    if method == 'toms748':
#        res = toms748(objective,bracket)
#        return res
#    if method =='newton':
#        res = newton(objective, x0, fprime)
#        return res

def optimal_next_beta( t, samples , constraints_funcs , beta, beta_max,indicator_how, a=5, n_partition=15):
    """
    return optimize.root_scalar(ESS(importances,beta), bracket = ,method=')
    """
    size_sample = len(samples[0])
    
    def objective(be):
        return ESS(be, t, samples , constraints_funcs , beta, indicator_how)- size_sample/2
    def fprime(be):
        return ESS_p(be, t, samples , constraints_funcs , beta, indicator_how)
    
    res = find_root(objective,
                    x0 =beta[t-1],
                    fprime=fprime,
                    bracket = [beta[t-1], beta_max],
                    geomspace = np.geomspace(a, beta_max, n_partition),
                    t=t)
    return res
###
#    if method == 'toms748':
#        try:
#            sol = optimize.root_scalar(
#                    objective,
#    #                x0 = beta[t-1],
#    #                x1 = beta[t-1]*1.5,
#                    bracket = [beta[t-1], beta_max],
#                    method='toms748')
#        except:
#            if beta[t-1]<1:
#                return beta[t-1]*1.1
#            return beta_max
#    if method =='newton':
#        try:
#            sol = optimize.root_scalar(
#                    func = objective,
#                    x0 = beta[t-1],
#                    fprime = fprime,
#    #                x0 = beta[t-1],
#
#                    method='newton')
#        except:
#            if beta[t-1]<1:
#                return beta[t-1]*1.1
#            return beta_max        
#    return sol.root
    
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

def constraint_indicator_func(x, indicator_how):
    if indicator_how == "Fermi_Dirac":
        return np.log(expit(x))
    elif indicator_how == "normal":
        return norm.logcdf(x)
    else:
        return norm.logcdf(x)
    
#unnormalized importance weights
def log_w( be, t, n, samples , constraints_funcs , beta,  indicator_how):
    """
    a scaler function of ``be",
    evaluate the log of umnormalized importance weights w^t_n point-wise
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
    float,
        
        
    """
    res = np.sum([constraint_indicator_func( be * g(samples[t-1][n]), indicator_how) for g in constraints_funcs] )
    res -= np.sum( [constraint_indicator_func( beta[t-1]* g( samples[t-1][n] ), indicator_how ) for g in constraints_funcs])
    return res

def log_w_p( be, t, n, samples , constraints_funcs , beta):
    """
    first derivative of w(be), for newton's method
    """
    res = np.sum([g(samples[t-1][n])* norm.logpdf( be * g(samples[t-1][n])) for g in constraints_funcs] )
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

    
    
def importance_resampling(be , samples ,t, beta , constraints_funcs, indicator_how):
    """
    
    Parameters
    ----------
    constraints_funcs: List, length n_constraint
    containing function that evaluates the constraint at a given point record x
    
    """
    sample = samples[t-1]
    size_sample = len(sample)
    W = weights_initial(size_sample)
    
    imp_weights = np.array( [ log_w( be, t, n, samples , constraints_funcs , beta, indicator_how) for n in range(size_sample) ])
    W = np.log(W) + imp_weights

    #normalize W
    W = np.exp(W)
    W = W / np.sum(W)
    return sample[ np.random.choice(a = size_sample, size = size_sample,p = W)]

#Metropolis Random Walk
    
def avoid_leakage_rewalk(x, delta, rw_step, upper, lower):
    n_dim = len(x)
    for i in range(n_dim):
        while (x[i]+delta[i] > upper[i] or x[i]+ delta[i]< lower[i] ):
            delta[i] =  np.random.normal(scale = rw_step[i])
    return delta

def avoid_leakage_rebound(x, delta, rw_step,  upper, lower):
    n_dim = len(x)
    for i in range(n_dim):
        while(x[i]+delta[i] > upper[i] or x[i]+delta[i] < lower[i]):
            #upper rebound
            if x[i]+delta[i] > upper[i]:
                delta[i] =  2*(upper[i]-x[i]) -delta[i]
        
            if x[i]+delta[i] < lower[i] :
                delta[i] =   2*(lower[i]-x[i]) -delta[i]

#        while(x[i]+delta[i] > interval[1] or x[i]+delta[i] < interval[0]):
#            #upper rebound
#            if x[i]+delta[i] > interval[1]:
#                delta[i] =  2*(interval[1]-x[i]) -delta[i]
#        
#            if x[i]+delta[i] < interval[0] :
#                delta[i] =   2*(interval[0]-x[i]) -delta[i]
    return delta
        
def proposal(x, rw_step, upper, lower):
    """
    random walk proposal for each record in the sample
    """
    n_dim = len(x)
    delta = np.random.normal(scale = rw_step, size = n_dim) 
#    xx = x + delta
    
    #avoid leakage
#    delta = avoid_leakage_rewalk(x, delta, rw_step, upper, lower)
    delta = avoid_leakage_rebound(x, delta, rw_step, upper, lower)
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



def Metropolis(be, t, sample, proposal ,constraints_funcs,p, upper, lower):
    
    current = sample
    size_sample = len(current)
    #assign adaptive stride based on current 
    rw_step = adaptive_step(current, t, p)

    proposed = [ proposal(x, rw_step, upper, lower) for x in current ]
###
###work here for some adaptive algorithm that fix the good points in the random walk
###    
    log_acc = [ min( 0, log_accept(be, x, x_, constraints_funcs)  ) for x,x_ in zip(proposed, current)]
    acc = np.exp(log_acc)
    
    q = np.random.uniform(size = size_sample)
    new_sample = [proposed[i] if (q<acc)[i] else current[i] for i in range(size_sample)]
    return np.array(new_sample)
    
    



def get_correctness(sample, constraints_funcs):
    """
    get correctness ratio
    """
    def is_correct(x):
        for f in constraints_funcs:
            if f(x) < 0:
                return False
        return True
    
    correctness = [is_correct(x) for x in sample]
    correctness = sum(correctness) / len(correctness)
    return correctness

def get_count(sample, constraints_funcs):
    """
    get correct count
    """
    def is_correct(x):
        for f in constraints_funcs:
            if f(x) < 0:
                return False
        return True
    
    count = [is_correct(x) for x in sample]
    count = sum(count) 
    return count

def upper_lower(n_dim, bounds):
    """modify the upper and lower bounds of the sampling space if naive constraints are given
    
    Parameters
    ----------
    n_dim: int,
    bounds: List,
        eg. [['+',0, -.5]] meaning x[0] has upper bound at .5
    """
    upper = np.ones(n_dim)
    lower = np.zeros(n_dim)
    if bounds != []:
        for bound in bounds:
            if bound[0] == '-':
                if bound[2]>0:#consistency check, 0< bound[2]<1 must be true
                    upper[bound[1]] = bound[2]
            else:
                if bound[2]<0:#consistency check
                    lower[bound[1]] = -bound[2]
    return upper, lower


def initial_sampling(n_dim, size_sample, given_example, upper, lower):
    """
    Returns:
    --------
    samples, List
        containing np array 
        
    """
    sample = np.random.uniform(low = lower,high =upper, size = (size_sample, n_dim))
    if given_example is not None:
        sample[0] = given_example
    
    return sample


def init_sample(n_dim, size_sample, bounds, given_example=None):
    upper, lower = upper_lower(n_dim, bounds)
    i_sample = initial_sampling(n_dim, size_sample, given_example, upper, lower)
    return i_sample

def scmc(n_dim, size_sample, i_sample, constraints_funcs, beta_max, bounds,
         p_beta=1,p_rw_step=1, verbose=1, track_correctness=True, given_example = None,
         indicator_how = "Fermi-Dirac", threshold=.990):
    
    t = 0
    beta = [0]
    
    #Generate uniform samples in n_dim cube [0,1]^n_dim
    samples=[]
    samples.append(i_sample)
    upper, lower = upper_lower(n_dim, bounds)

    
    #recor correctness

    correctness_history = []
    current_correctness = get_correctness(i_sample, constraints_funcs)
    correctness_history.append(current_correctness)
    
    
    
#    current_count = get_count(sample, constraints_funcs)
#    if current_count <10:
#        pass
        

    print('Sequentially constrained Monte Carlo sampler has started. The iteration terminates when beta > {}'.format(beta_max))
    while (beta[t] < beta_max and current_correctness < threshold) :

        if track_correctness:
            print('t = {:<5} ,beta={:7.2f} ,correctness={:.3f} '.format(t ,beta[t], correctness_history[t]))
        else:
            print('t = {} ,current beta is {} '.format(t ,beta[t]))
            
        t += 1
        
#        if current_correctness <0.01:
#            #assign be
#            be = beta[-1]*2 
#            beta.append(be)
#            #do not do importance resample
#            new_sample = Metropolis(be, t, sample, proposal ,constraints_funcs ,p_rw_step)
#            
#        else:
        #assign next beta
        be = optimal_next_beta( t, samples , constraints_funcs , beta, beta_max, indicator_how)
        #assign with linear
#        be = beta_poly(t, seq_size, p_beta, beta_max )
        beta.append(be)
        
        #importance resampling

        
        resample = importance_resampling(be , samples ,t, beta , constraints_funcs,indicator_how)
        
        #Random Walk using Markov Chain kernel
        new_sample = Metropolis(be, t, resample, proposal ,constraints_funcs ,p_rw_step,upper, lower)
        samples.append(new_sample)
        if track_correctness:
            current_correctness = get_correctness(new_sample, constraints_funcs)
            correctness_history.append(current_correctness)
    
    print('Sampling completed. The final beta {} is achieved.'.format(beta[-1]))
#    if track_correctness:
#        return samples, 
    return samples, correctness_history
    



#def scmc(n_dim, size_sample, constraints_funcs, beta_max, bounds,
#         p_beta=1,p_rw_step=1, verbose=1, track_correctness=True, given_example = None,
#         indicator_how = "Fermi-Dirac"):
#    
#    t = 0
#    beta = [0]
#    
#    #Generate uniform samples in n_dim cube [0,1]^n_dim
#    samples=[]
#    upper, lower = upper_lower(n_dim, bounds)
#    samples.append(initial_sampling(n_dim, size_sample, given_example, upper, lower))
#    
#    #recor correctness
#    sample= samples[0]
#    correctness_history = []
#    current_correctness = get_correctness(sample, constraints_funcs)
#    correctness_history.append(current_correctness)
#    
#    
#    
##    current_count = get_count(sample, constraints_funcs)
##    if current_count <10:
##        pass
#        
#
#    print('Sequentially constrained Monte Carlo sampler has started. The iteration terminates when beta > {}'.format(beta_max))
#    while beta[t] < beta_max:
#
#        if track_correctness:
#            print('t = {:<5} ,beta={:7.2f} ,correctness={:.3f} '.format(t ,beta[t], correctness_history[t]))
#        else:
#            print('t = {} ,current beta is {} '.format(t ,beta[t]))
#            
#        t += 1
#        
##        if current_correctness <0.01:
##            #assign be
##            be = beta[-1]*2 
##            beta.append(be)
##            #do not do importance resample
##            new_sample = Metropolis(be, t, sample, proposal ,constraints_funcs ,p_rw_step)
##            
##        else:
#        #assign next beta
#        be = optimal_next_beta( t, samples , constraints_funcs , beta, beta_max, indicator_how)
#        #assign with linear
##        be = beta_poly(t, seq_size, p_beta, beta_max )
#        beta.append(be)
#        
#        #importance resampling
#
#        
#        resample = importance_resampling(be , samples ,t, beta , constraints_funcs,indicator_how)
#        
#        #Random Walk using Markov Chain kernel
#        new_sample = Metropolis(be, t, resample, proposal ,constraints_funcs ,p_rw_step,upper, lower)
#        samples.append(new_sample)
#        if track_correctness:
#            current_correctness = get_correctness(new_sample, constraints_funcs)
#            correctness_history.append(current_correctness)
#    
#    print('Sampling completed. The final beta {} is achieved.'.format(be))
##    if track_correctness:
##        return samples, 
#    return samples, correctness_history
#    