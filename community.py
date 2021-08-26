#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#Module to simulate the null model
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from random import choices, sample
from collections import Counter

def predict(mu, s, N, S, dist, gm,NC1,NC2, rhok=np.arange(0.5,1, 0.05)):
    
    #This function compute the prediction of the null model for the relationships 
    #between beta-diversity measures for pairs of communities with given parameters (mu, s, gm),
    #S species, sigma^2 distributed as 'dist' (if dist is exponential, with average gm), 
    #simulating NC1 pairs of communties within host and NC2 from different hosts
    #The optional parameter rhok specifies the range of correlations of K to use when simulating communties from different hosts
    
    phi=[]
    corr=[]
    jacc=[]
    ovlp=[]
    diss=[]
    bc=[]
    mh=[]
    
    #first, NC1 communities with same K and some correlation in gamma fluctuations
    rho=1; #correlation of K
    rhogamma=np.arange(0,0.6,0.1) # Correlation of abundance fluctuations (gamma distributed)
    for j in rhogamma:
        for k in range(NC1):
            (x1, x2)= generate_comm(mu,s, S, dist, gm,rho, rhogamma=j) #Generate OTU abundances in two samples
            xsamp1=np.random.multinomial(N,x1) # simulate multinomial sampling
            xsamp2=np.random.multinomial(N,x2)
            
            ids=(xsamp1>0) & (xsamp2>0) #OTUs sampled in both
            
            corr.append(np.corrcoef(np.log10(xsamp1[ids]), np.log10(xsamp2[ids]))[0,1]) #pearson correlation
            jacc.append(np.sum(ids)/np.sum((xsamp1>0) | (xsamp2>0))) # Jaccard Dissimilarity (1- Jaccard similarity)
            ovlp.append(np.sum(xsamp1[ids]/np.sum(xsamp1)+xsamp2[ids]/np.sum(xsamp2))/2) # Overlap
            xsamp1hat=xsamp1[ids]/np.sum(xsamp1[ids])
            xsamp2hat=xsamp2[ids]/np.sum(xsamp2[ids])
            m=(xsamp1hat+xsamp2hat)/2
            diss.append(np.sqrt(0.5*np.sum(xsamp1hat*np.log(xsamp1hat/m)+xsamp2hat*np.log(xsamp2hat/m)))) # Dissimilarity
            phi.append(np.nanmean(((xsamp1-xsamp2)**2-(xsamp1+xsamp2))/((xsamp1+xsamp2)**2-(xsamp1+xsamp2)))) #Phi
            bc.append(1-np.sum(np.minimum(xsamp1,xsamp2))/(np.sum(xsamp1)+np.sum(xsamp2))) #Bray-Curtis dissimilarity
            mh.append(1-2*np.sum(xsamp1*xsamp2)/(np.sum(xsamp1**2)+np.sum(xsamp2**2))) # Morisita-Horn dissimilarity
            
    #Then, NC2 communities with correlated but different K
    #rhok=np.arange(0.5,1, 0.05) #correlations of K 
    for j in rhok:
        for k in range(NC2):
            (x1, x2)= generate_comm(mu,s, S, dist, gm,rho=j, rhogamma=0) #Generate OTU abundances in two samples
            xsamp1=np.random.multinomial(N,x1) # simulate multinomial sampling
            xsamp2=np.random.multinomial(N,x2)
            
            ids=(xsamp1>0) & (xsamp2>0) #OTUs sampled in both
            
            corr.append(np.corrcoef(np.log10(xsamp1[ids]), np.log10(xsamp2[ids]))[0,1])  #pearson correlation
            jacc.append(np.sum(ids)/np.sum((xsamp1>0) | (xsamp2>0))) # Jaccard Dissimilarity (1- Jaccard similarity)
            ovlp.append(np.sum(xsamp1[ids]/np.sum(xsamp1)+xsamp2[ids]/np.sum(xsamp2))/2) # Overlap
            xsamp1hat=xsamp1[ids]/np.sum(xsamp1[ids])
            xsamp2hat=xsamp2[ids]/np.sum(xsamp2[ids])
            m=(xsamp1hat+xsamp2hat)/2
            diss.append(np.sqrt(0.5*np.sum(xsamp1hat*np.log(xsamp1hat/m)+xsamp2hat*np.log(xsamp2hat/m)))) # Dissimilarity
            phi.append(np.nanmean(((xsamp1-xsamp2)**2-(xsamp1+xsamp2))/((xsamp1+xsamp2)**2-(xsamp1+xsamp2)))) #Phi
            bc.append(1-np.sum(np.minimum(xsamp1,xsamp2))/(np.sum(xsamp1)+np.sum(xsamp2))) #Bray-Curtis dissimilarity
            mh.append(1-2*np.sum(xsamp1*xsamp2)/(np.sum(xsamp1**2)+np.sum(xsamp2**2))) # Morisita-Horn dissimilarity
            
           
    d = {'pears': corr, 'jacc': jacc, 'ovlp': ovlp, 'diss': diss, 'phi': phi, 'bc':bc, 'mh': mh}
    predictions = pd.DataFrame(data=d)
    return predictions
    
    
def generate_comm(mu,s,S,dist, gm,rho,rhogamma):
    #This function  generates the parameters K and sigma for two communities with given parameters (mu, s), 
    # S species, sigma^2 distributed as 'dist' (if dist is exponential, with average gm), correlation rho between the values of K
    # and correlation rhogamma between the Gamma-distributed fluctuations of abundance
    from scipy.stats import norm
    from scipy.stats import gamma
    
    K=np.exp(np.random.multivariate_normal([mu, mu], [[s, s*rho], [s*rho, s]],S)) #correlated K for the two communities extracted from lognormal dist
    sigmarnd=[]  #Exponentially distributed sigma, common for the two communities
    if dist=='exp':
        for k in range(S):
            tr=100
            while tr>1.95: # Values too close to 2 give numerical problems when extracting from the Gamma distribution
                tr=np.sqrt(np.random.exponential(gm))
       
            sigmarnd.append(tr)
    else:
        if dist=='unif':
            sigmarnd=np.random.uniform(0,1.95,size=S)
        
    
    # Extraction of the who vectors of abundances, distributed according to Gamma distributions with the correlation rhogamma
    Z = np.random.multivariate_normal([0, 0], [[1, rhogamma], [rhogamma, 1]], S)
    U = norm.cdf(Z)
    G = [gamma.ppf(U[:,0],np.divide(2,sigmarnd)-1, scale=sigmarnd*K[:,0]/2), gamma.ppf(U[:,1],np.divide(2,sigmarnd)-1, scale =sigmarnd*K[:,1]/2)]
   
    x1=G[0]
    x2=G[1]
    
    # Normalise, to have relative abundances
    x1=x1/np.sum(x1) 
    x2=x2/np.sum(x2)
    
    return x1, x2

def plot_smooth(x,y,label1,label2):
    #This function plots a scatter of the (x,y) data and a smoothed version obtained performing the average of y in bins of x
    plt.scatter(x, y,s=10,c='k', alpha=0.05, rasterized = True)
    #Compute binned mean 
    bins=np.arange(0,1,0.025)
    binc=bins[0:-1]+np.diff(bins)/2
    bmean=[]
    for j in range(len(bins)-1):
        bmean.append(np.mean(y[(x>=bins[j]) & (x<bins[j+1])]))
        
    plt.plot(binc,bmean,'-k', linewidth=2)
    plt.xlabel(label1)
    plt.ylabel(label2)
    plt.xlim(-0.2, 1)
    plt.tick_params(which='both',direction='in')

def compute(abd,N1,N2, pairs):
    # This function computes the empirical relationship between beta-diversity measures between pairs of empirical communities
    #N1= number of pairs of communties to sample within host
    #N2= number of pairs of communties to sample from different hosts
    phi=[]
    corr=[]
    jacc=[]
    ovlp=[]
    diss=[]
    bc=[]
    mh=[]
    
    ind=choices(np.arange(len(abd)), k=N1) # N1 random sampling with replacement from the set of individuals
    for k in range(N1):
        #sample two days at a distance of at least 30 days
        samp=[1, 2]
        while abs(abd[ind[k]].iloc[samp[0],0]-abd[ind[k]].iloc[samp[1],0])<30: #while they are closer than 30 days
            samp=sample(range(len(abd[ind[k]])),k=2)
        
        n1=abd[ind[k]].iloc[samp[0],1:] #first day
        n2=abd[ind[k]].iloc[samp[1],1:] #second day
        
        ids=(n1>0) & (n2>0) #OTUs sampled in both
            
        corr.append(np.corrcoef(np.log10(n1[ids]), np.log10(n2[ids]))[0,1]) #pearson correlation
        jacc.append(sum(ids)/sum((n1>0) | (n2>0))) # Jaccard Dissimilarity (1- Jaccard similarity)
        ovlp.append(sum(n1[ids]/sum(n1)+n2[ids]/sum(n2))/2) # Overlap
        n1hat=n1[ids]/sum(n1[ids])
        n2hat=n2[ids]/sum(n2[ids])
        m=(n1hat+n2hat)/2
        diss.append(np.sqrt(0.5*sum(n1hat*np.log(n1hat/m)+n2hat*np.log(n2hat/m)))) # Dissimilarity
        bc.append(1-sum(np.minimum(n1,n2))/(sum(n1)+sum(n2))) #Bray-Curtis dissimilarity
        x1=n1/sum(n1)
        x2=n2/sum(n2)
        mh.append(1-2*sum(x1*x2)/(sum(x1**2)+sum(x2**2))) # Morisita-Horn dissimilarity
         
        n1=equalize(n1,n2) #subsample day with more reads, to have equal reads in both
        n2=equalize(n2,n1)
        phi.append(np.nanmean(((n1-n2)**2-(n1+n2))/((n1+n2)**2-(n1+n2))))  #Phi
           
        
    ps=choices(np.arange(len(pairs)), k=N2)
    for k in range(N2):
        if len(pairs)>2:
            samp=sample(range(len(abd[pairs[ps[k]][0]])),k=1) #sample one day from the first individual of the pair
            n1= abd[pairs[ps[k]][0]].iloc[samp[0]-1,1:]
            samp=sample(range(len(abd[pairs[ps[k]][1]])),k=1) #sample one day from the second individual of the pair
            n2= abd[pairs[ps[k]][1]].iloc[samp[0]-1,1:]
        else:
            samp=sample(range(len(abd[0])), k=1) #sample one day from the first individual of the pair
            n1= abd[0].iloc[samp[0]-1,1:]
            samp=sample(range(len(abd[1])), k=1) #sample one day from the second individual of the pair
            n2= abd[1].iloc[samp[0]-1,1:]
        
        
        ids=(n1>0) & (n2>0) #OTUs sampled in both
            
        corr.append(np.corrcoef(np.log10(n1[ids]), np.log10(n2[ids]))[0,1]) #pearson correlation
        jacc.append(sum(ids)/sum((n1>0) | (n2>0))) # Jaccard Similarity
        ovlp.append(sum(n1[ids]/sum(n1)+n2[ids]/sum(n2))/2) # Overlap
        n1hat=n1[ids]/sum(n1[ids])
        n2hat=n2[ids]/sum(n2[ids])
        m=(n1hat+n2hat)/2
        diss.append(np.sqrt(0.5*sum(n1hat*np.log(n1hat/m)+n2hat*np.log(n2hat/m)))) # Dissimilarity
        bc.append(1-sum(np.minimum(n1,n2))/(sum(n1)+sum(n2))) #Bray-Curtis dissimilarity
        mh.append(1-2*sum(n1*n2)/(sum(n1**2)+sum(n2**2))) # Morisita-Horn dissimilarity
         
        n1=equalize(n1,n2) #subsample day with more reads, to have equal reads in both
        n2=equalize(n2,n1)
        phi.append(np.nanmean(((n1-n2)**2-(n1+n2))/((n1+n2)**2-(n1+n2))))  #Phi
        
    d = {'pears': corr, 'jacc': jacc, 'ovlp': ovlp, 'diss': diss, 'phi': phi, 'bc':bc, 'mh': mh}
    data = pd.DataFrame(data=d)
    return data
    
        
def equalize(n1,n2):
    # This function downsamples the sample with more counts between n1 and n2, so that they have the same counts
      n=sum(n2)
      if sum(n1)<=n:
          n_new=n1
      else:
          list=[]
          for i in range(len(n1)):
             list.append(np.tile(i,(n1[i],1)))    #make list of single reads 
          flat_list = [item for sublist2 in list for sublist1 in sublist2 for item in sublist1] #flatten the list
          downsampled=sample(flat_list, k=n)
          ids=(n1==0)
          n_new= pd.Series(dtype=int).reindex_like(n1)
          n_new[ids]=0
          counts=Counter(downsampled)
          for key in counts:
              n_new[key]=counts[key]
          n_new[np.isnan(n_new)]=0
      return n_new
     
         
def plot_binned(data1,data2,label1,label2,binax, rangeax, step=0.05, xmin=0, xmax=1):
    #This function plots a smoothed version of (data1, data2), obtained computing a binned average along the axis 'binax' 
    if binax=='x':
       x=data1
       y=data2
    else:
        x=data2
        y=data1
    #Compute binned mean    
    bins=np.arange(rangeax[0],rangeax[1],step)
    binc=bins[0:-1]+np.diff(bins)/2
    bmean=[]
    for j in range(len(bins)-1):
        bmean.append(np.mean(y[(x>=bins[j]) & (x<bins[j+1])]))
    if binax=='x':
        plt.plot(binc,bmean,'o',markersize=3, markerfacecolor='r', markeredgecolor='r')
    else:
        plt.plot(bmean,binc,'o',markersize=3, markerfacecolor='r', markeredgecolor='r')
    plt.xlabel(label1)
    plt.ylabel(label2)
    plt.xlim(xmin, xmax)
    plt.tick_params(which='both',direction='in')
   