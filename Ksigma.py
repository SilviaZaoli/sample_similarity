#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#Module to simulate the null model
import numpy as np
import math
import matplotlib.pyplot as plt
import sympy as sp

def estimate(abdi):
    #This function estimates the parameters K and sigma for all OTUs within a host
    counts=np.sum(abdi.drop(['day'], axis=1).values, axis=1)
    red_abd=abdi.drop(['day','Unrecognized'], axis=1).values 
    relabd= red_abd/counts[:,None] 
    occup=np.sum(relabd>0, axis=0)/len(relabd) 
    mediasq=np.mean(np.divide(red_abd*(red_abd-np.ones(np.shape(red_abd))), (counts*(counts-1))[:,None]  ), axis=0 )  
    meanrelabd= np.mean(relabd, axis=0) 

    temp= 1+ meanrelabd**2/(mediasq-meanrelabd**2)  
    sigma=np.where((temp!=0) & (~np.isnan(temp)), 2/temp, np.nan) 
    K=2*meanrelabd/(2-sigma) 
    ids=np.nonzero((sigma>0) & (occup>0.2)) 
    ids2=np.nonzero(sigma>0) 
    
    return (K,sigma,ids, ids2, meanrelabd)

def sigmaexp(cumsigma):
    #This function estimates the parameter of the exponential distribution of sigma^2
    gm=np.mean(cumsigma**2)
    return gm

def Klogn(cumK,c, mu0=-19,s0=5):
    # This function estimates the parameters (mu, s) of the lognormal distribution of K
    m1=np.mean(np.log(cumK[cumK>c]))
    m2=np.mean(np.log(cumK[cumK>c])**2)
    xmu=sp.symbols('xmu')
    xs=sp.symbols('xs')
    eq1=-m1+xmu + np.sqrt(2/math.pi)*xs*sp.exp(-((np.log(c)-xmu)**2)/2/(xs**2))/(sp.erfc((np.log(c)-xmu)/np.sqrt(2)/xs))
    eq2=-m2+xs**2+m1*xmu+np.log(c)*m1-xmu*np.log(c)
    
    sol=sp.nsolve([eq1,eq2],[xmu,xs],[mu0,s0])
    

    
    return(float(sol[0]),float(sol[1]))

def plotK(Kh,c,s):
    # This function plots the empirical distribution of K, normalized for K>c
    bins1=np.arange(np.log(c),0, step=0.5)
    a=[]
    for i in range(len(bins1)-1):
        counts=(Kh>=bins1[i]) & (Kh<bins1[i+1])
        a.append(sum(counts))

    norm=np.sum(a*np.diff(bins1))
    bins2=np.arange(min(Kh),max(Kh), step=0.5)
    a=[]
    for i in range(len(bins2)-1):
        counts=(Kh>=bins2[i]) & (Kh<bins2[i+1])
        a.append(sum(counts))
    plt.plot(np.log10(np.exp(bins2[:-1]+np.diff(bins2)/2)),a/norm,'-o',markersize=s)