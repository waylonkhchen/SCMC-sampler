#!/usr/bin/env python
# coding: utf-8

# ## Methodology
# This solution makes reference mainly to this literature: 
# [1]  S. Golchi and J. L. Loeppky, (2015).(https://arxiv.org/abs/1512.07328)
# The method is a Monte Carlo based sampler. We use a combination of methods and th key methods are as follows.
# ### 1. Sequential Monte Carlo (SMC) Sampler 
# #### 1.1 Overview
#    When the distribution $\pi_\chi(x)$  of the target contrained domain of interest $\chi$,   is difficult to sample, SMC can approximate it with the following apprach. We start with $\pi_0(x)$ that is easy to sample, and sample exactly the desired number of particles, say 1,000. ( In our case it's uniformly distributed in the d-dimensional unit hyper cube $[0,1]^d$, or some sub-cube of it when the constraints suggest it, like the case in **alloy.txt**). We then find a family of auxillary ditributions $\{p_t(x)\}_{t=0}^{t=T}$ when $T$ is large enough to connect the two functions, where $p_0(x) = \pi_0(x), p_T(x) \to \pi(x)$. In each SCMC iteration (sequentially) $t \to t+1$, to transition the sample distribution from $p_t(x)$ to $p_{t+1}(x)$, we perform one *biased random walk* for each particle . **In the spirit of MCMC, we only need to know the distribution $p_t(x)$ up to a normalization constant**, say
# \begin{align}
# p_t(x) = \frac{\tilde\Phi_t(x)}{Z_t},
# \end{align}
# where $\tilde\Phi(x)$ is our family of auxillary functions. The property that the Markov chain kernel (random walk) has a unique equlibrium state that distributes the same as the desired distribution is the key for this method to work. Following [1], our choice of auxillary fuction is 
# \begin{align}
# \tilde\Phi_t(x) = \prod_{i=1}^k \Phi(\beta(t)\cdot g_i(x)),
# \end{align}
# where $\Phi(u)$ is the cdf of normal distribution function $\mathcal{N}(0,1)$, and $\{g_i(x)\}_{i=1}^k$ define the constraints $g_i(x) \geq 0$. Note that $\tilde\Phi_t(x) $ is uniform when $\beta(t) = 0$ and $\tilde\Phi_t(x) \propto \mathbb{1}_\chi(x)$, the indicator function of domain $\chi$,  when $\beta \to \infty$. Other function with similar the same asymptotic behavior should yield similar result. For example, the author considers an alternative 
# \begin{align}
# \tilde\theta_t(x) = \prod_{i=1}^k \theta(\beta(t) \cdot g_i(x)) 
# \end{align}
# where $\theta(z) = \frac{1}{1+\exp(-z)}$ is the sigmoid function,
# will also work because it has the same asymptotic behaviors as $\Phi(z)$. The author did not implement this function but it's curious to see how well it would work.
# 
#    The SCMC sampler solution has the following key steps. 
#    1. Initialize random uniformly distributed sample
#    
#    In each iteration:
#    
#    2. Find the optimal next $\beta(t)$
#    3. Resample the particles with importance sampling
#    4. Random walk with Metropolis algorithm performed on each particle in the sample
#    
#    The iteration terminates at a preset maximum $\beta$ value which the author chose to be 10000.
#    The theory and implementation of 3. and 4. can be found in most Baysian Analysis literature. For example, 
#    The author explains 1. and 2. in more detail in the follwing. 
# The SCMC process is implemented in the file "scmc.py".
# 
# 
# #### 1.2 Initialize random distributed sample
#    The initialization is simply uniformly sampling the d-dimensional hypercube $[0,1]^d$, however, with exception when the constraints given are of simple form like 
# \begin{align}
# x_0- 0.5 \geq 0.
# \end{align}
# Constraints of this kind eseentially shrinks the size of the sampling space. to $x_0 \in [0.5,1].$ This observation is crucial to the success in cracking the **alloy.txt** example. Remember that due to the curse of dimensionality, in high dimensional space, the sparsity of the space grows expoenntially with the dimenion. The sampling for the case in alloy.txt would not be possible at all if this observation was not made and implemented. 
# The implementation is at the function **initial_sampling** in scmc.py, and the parsing of the constraints in order to recognize the simple constraints are implemented in **parsing_constraints.py**
# #### 1.3 Find the optimal next $\beta(t)$
#    One tricky step in the sequential method is to choose the sequence of $\beta(t).$ A physical analogy is to view $\beta(t)$ as inverse temperature and our iteration as a cooling process. In order for the system to condense to the ground state, we would like the cooling to be slow enough such that the equalibrium state is apporoximated in each iteration. On the other hand, the limited computer resource would not favor the adiabatic cooling process that would take forever. Fortunately,an optimal equation for the next $\beta(t)$ was given in literature [1] and the references therein. Solving the equation
#    \begin{align}
#    ESS(\beta(t)) = \frac{N}{2}
#    \end{align}
# for $\beta(t)$ will suggest the optimal next $\beta(t)$ in the sequence that we perform importance resampling and the Metropolis random walk. $ESS(\beta(t))$ is the so-called effective sample size in the context of importance sampling (see[1] for detail) and $N$ is the number of particle in our sample.
# 
#    **The above equation can be numerically solved with root finding alogrithm.** The author chose Brent's-like solver for this task. Because we know $\beta(t)$ is increasing, thus it's easier to give an interval that contains the root than to give a good guess. Some tricky implementation is that we partition the interval $[0, \beta_{max}]$ into geometric space to speed up the convergence in root finding. The implementation can be found in optimal_next_beta and find_root functions in scmc.py.
# #### 1.4 Visualization method in the API
# In the API ***SCMC_module.py***, the author has implemented a simple visualization for the results. self.plot_results( i, j, n_iter = None) scatter plot the sample in $x_i, x_j$ at n_iter =t steps. One can make a movie with this function if desired. Also self.plot_all_axis(n_iter=None) will plot the scatter plot of all pairs of axes. One can compare the visualization with the outputs of self.print_constraints() for verification.
# 
# The author also has attached Using_API_exapmly.py to show the methods in the SCMC API.
# 
# 
#     

# In[ ]:




