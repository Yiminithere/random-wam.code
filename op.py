#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 15 20:28:14 2020

@author: yimingxu
"""

import numpy as np
from families import JacobiPolynomials
import pandas as pd
from tqdm.auto import tqdm

def create_Rn_function(n,gamma):
    P = JacobiPolynomials(alpha=gamma,beta=gamma)

    return lambda x: np.sqrt(np.sum(P.eval(x, range(n+1), d=1)**2, axis=1) )

def create_K_function(n,gamma):
    P = JacobiPolynomials(alpha=gamma,beta=gamma)

    return lambda x: np.sqrt(np.sum(P.eval(x, range(n+1), d=0)**2, axis=1) )

def v(n,gamma,c):
    P = JacobiPolynomials(alpha=gamma,beta=gamma)

    return lambda x: sum(c[i]*P.eval(x, i, d=0) for i in range(len(c)))

def mu(gamma,x):
    y = np.array([(1-i**2)**gamma for i in x])
    
    return(y)



# Dimension of V_n
n = 10

# Discretization of D for sampling/evaluation
M_1 = 2*1e3
M_2 = 2*1e3
I_1 = np.linspace(-1,1,int(M_1+1))
I_2 = np.linspace(-1,1,int(M_2+1))

# Use this when gamma<0
#I_1 = np.linspace(-0.999,0.999,int(M_1+1))
#I_2 = np.linspace(-0.999,0.999,int(M_2+1))


# Sampling size
N = 100

# shape parameter
gamma = 0.5

# Create sampling distributions
Rn = create_Rn_function(n-2,gamma)(I_1)
Rn = Rn/sum(Rn)
Un = [1/len(I_1)]*len(I_1)
Mu = mu(gamma,I_1)
Mu = Mu/sum(Mu)
Mu_V = mu(gamma,I_1)*create_K_function(n-1,gamma)(I_1)
Mu_V = Mu_V/sum(Mu_V)

Count_unif = np.zeros([N,100])
Count_Rn = np.zeros([N,100])
Count_mu = np.zeros([N,100])
Count_mu_V = np.zeros([N,100])

for j in tqdm(range(100)):
    print(j)
    # Sampling points using the three different measures
    I_unif = np.random.choice(I_1, size = N, p = Un)
    I_Rn = np.random.choice(I_1, size = N, p = Rn)
    I_mu = np.random.choice(I_1, size = N, p = Mu)
    I_mu_V = np.random.choice(I_1, size = N, p = Mu_V)

    count_unif = np.zeros([N, 1000])
    count_Rn = np.zeros([N, 1000])
    count_mu = np.zeros([N, 1000])
    count_mu_V = np.zeros([N, 1000])

    for i in range(1000):
        c = np.random.normal(0, 1, n)
        c = c/np.sqrt(sum(c**2))
        v_max = max(v(n,gamma,c)(I_2))
        J_unif = abs(v(n,gamma,c)(I_unif))
        J_Rn = abs(v(n,gamma,c)(I_Rn))
        J_mu = abs(v(n,gamma,c)(I_mu))
        J_mu_V = abs(v(n,gamma,c)(I_mu_V))
        
        J_unif = v_max/np.maximum.accumulate(J_unif,axis=0)
        J_Rn = v_max/np.maximum.accumulate(J_Rn,axis=0)
        J_mu = v_max/np.maximum.accumulate(J_mu,axis=0)
        J_mu_V = v_max/np.maximum.accumulate(J_mu_V,axis=0)
        
        count_unif[:,i] = J_unif.squeeze()
        count_Rn[:,i] = J_Rn.squeeze()
        count_mu[:,i] = J_mu.squeeze()
        count_mu_V[:,i] = J_mu_V.squeeze()
     
    count_unif = count_unif.max(axis = 1)
    count_Rn = count_Rn.max(axis = 1)
    count_mu = count_mu.max(axis = 1)
    count_mu_V = count_mu_V.max(axis = 1)
    
    Count_unif[:,j] = count_unif
    Count_Rn[:,j] = count_Rn
    Count_mu[:,j] = count_mu
    Count_mu_V[:,j] = count_mu_V



Count_unif = pd.DataFrame(Count_unif)
Count_Rn = pd.DataFrame(Count_Rn)
Count_mu = pd.DataFrame(Count_mu)
Count_mu_V = pd.DataFrame(Count_mu_V)

Count_unif.to_csv("mydata1.csv")
Count_Rn.to_csv("mydata2.csv")
Count_mu.to_csv("mydata3.csv")
Count_mu_V.to_csv("mydata4.csv")













