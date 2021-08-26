#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib
matplotlib.rcParams['backend'] = 'SVG'
new_rc_params = {'text.usetex': False,"svg.fonttype": 'none'}
matplotlib.rcParams.update(new_rc_params)
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 7})
import math
import pandas as pd
import scipy.io
from scipy.stats import norm
from itertools import combinations
import community #module to simulate the null modell of the community and compute dissimilarity metrics


#Load OTU abundance data-frames 
# 0 to 9: BIO-ML dataset, gut
# 10-11: Moving Pictures dataset, gut. 10: male, 11: female
# 12 to 15: David dataset, gut. 12: A before travel, 13: A after travel, 14: B before Salmonella, 15: B after Salmonella
# 16-17: Moving Pictures dataset, Left palm. 16: male, 17: female
# 18: Moving Pictures dataset, Right palm, male (female has too low numer of reads)
# 19-20: Moving Picture dataset, Oral. 19: male, 20: female 
abd=[]
for i in range(21):
    abd.append(pd.read_csv("./DataCSV/abd"+str(i)+".csv"))
    
    
##################################################################################
# Analysis for Figure 1: Estimate K and sigma parameters, plot their distribution
##################################################################################

import Ksigma #module to estimate the parameters K and sigma from time-series and fit their distribution

#Estimate K and and sigma for each OTU within each individual time-series
#(NOTE: warnings are produced by those OTU with low counts for which sigma cannot be computed)
K=[]
sigma=[]
ids=[] #IDs of OTU to keep to compute sigma distribution (estimated sigma>0 and occupancy >0.2)
ids2=[]  #IDs of OTU to keep to copute K distribution (estimated sigma>0)
meanrelabd=[]
for i in range(21):
  temp=Ksigma.estimate(abd[i])
  K.append(temp[0])
  sigma.append(temp[1])
  ids.append(temp[2])
  ids2.append(temp[3])

#compute K and sigma for half series
#(NOTE: warnings are produced by those OTU with low counts for which sigma cannot be computed)
K1=[]
K2=[]
sigma1=[]
sigma2=[]
for i in range(21):
    times = abd[i].day #sampling days
    half = times[0] +np.round((times[len(times)-1]-times[0])/2) # day correstponding to the middle of the sampling period
    pos = times>=half # positions of samples in second half
    temp1 = Ksigma.estimate(abd[i][~pos]) #estimates for first half
    temp2 =  Ksigma.estimate(abd[i][pos]) #estimates for second half
    K1.append(temp1[0])
    sigma1.append(temp1[1])
    K2.append(temp2[0])
    sigma2.append(temp2[1])
 
# Fit exponential distribution for sigma^2 and truncated lognormal distribution for K, for each environment
#
#Gut
gm_gut=Ksigma.sigmaexp(np.concatenate([sigma[i][ids[i]] for i in range(16)]))  #parameter of exp dist
c=10**(-4.5); #truncation threshold for the lognormal 
(mu_gut,s_gut)=Ksigma.Klogn(np.concatenate([K[i][ids2[i]] for i in range(16)]), c) #parameters of lognormal dist (mean and std of log(K))

#Palms
gm_palms=Ksigma.sigmaexp(np.concatenate([sigma[i][ids[i]] for i in range(16,19)]))  #parameter of exp dist
c=10**(-4); #truncation threshold for the lognormal 
(mu_palms,s_palms)=Ksigma.Klogn(np.concatenate([K[i][ids2[i]] for i in range(16,19)]), c) #parameters of lognormal dist  (mean and std of log(K))

#Oral
gm_oral=Ksigma.sigmaexp(np.concatenate([sigma[i][ids[i]] for i in range(19, 21)]))  #parameter of exp dist
c=10**(-4); #truncation threshold for the lognormal 
(mu_oral,s_oral)=Ksigma.Klogn(np.concatenate([K[i][ids2[i]] for i in range(19,21)]), c) #parameters of lognormal dist  (mean and std of log(K))

# Make figures
#GUT
c=10**(-4.5);

fig=plt.figure(figsize=(9/2.54, 9/2.54), dpi=300)
#panel1
#ax1=fig.add_subplot(2, 2, 1)
ax1=plt.subplot(2, 2, 1)
for i in range(16):
    Ksigma.plotK(np.log(K[i][ids2[i]]),c,3)    #plots empirical distribution of K, normalized on [log(c),0]
x=np.arange(-14,-1,step=0.1)
#plot truncated lognormal fitted for K>log(c)
ax1.plot(np.log10(np.exp(x)),np.sqrt(2/math.pi)/s_gut *np.exp(-(x-mu_gut)**2 /2/(s_gut**2))/scipy.special.erfc((np.log(c)-mu_gut)/np.sqrt(2)/s_gut),'-k')
ax1.axvline(-4.5,color='k', linestyle='--')
ax1.set_xlabel('K')
ax1.set_ylabel('P(K)')
ax1.set_yscale('log')
ax1.tick_params(which='both',direction='in')
plt.xticks(np.arange(-6,0,2),['$10^{-6}$','$10^{-4}$','$10^{-2}$'])
ax1.set_box_aspect(1)

#panel2
ax2=plt.subplot(2, 2, 2)
for i in range(16):
    hist, bin_edges = np.histogram(sigma[i][ids[i]]**2, density=True)
    ax2.plot(bin_edges[:-1]+np.diff(bin_edges),hist,'-o',markersize=3) #plots empirical distribution of sigma
ax2.plot(np.arange(5),1/gm_gut * np.exp(-np.arange(5)/gm_gut),'-k') #plots exponential fit
ax2.set_xlabel('$\sigma^2$')
ax2.set_ylabel('$P(\sigma^2)$')
ax2.set_yscale('log')
ax2.tick_params(which='both',direction='in')
ax2.set_box_aspect(1)

#panel 3
ax3=plt.subplot(2, 2, 3)
ax3.scatter(K1[0],K2[0],alpha=0.3, c='k',s=3)
ax3.plot([np.nanmin(K1[0]),np.nanmax(K1[0])], [np.nanmin(K1[0]),np.nanmax(K1[0])],'r--')
ax3.set_xscale('log')
ax3.set_yscale('log')
ax3.set_xlabel("K 'ae' (first half) ")
ax3.set_ylabel("K 'ae' (second half)")
ax3.set_aspect(aspect=1)
ax3.tick_params(which='both',direction='in')
ax3.set_box_aspect(1)

#panel4
ax4=plt.subplot(2, 2, 4)
ax4.scatter(K[0],K[1],alpha=0.3, c='k',s=3)
ax4.plot([np.nanmin(K[0]),np.nanmax(K[0])], [np.nanmin(K[0]),np.nanmax(K[0])],'r--')
ax4.set_xscale('log')
ax4.set_yscale('log')
ax4.set_xlabel("K 'ae' ")
ax4.set_ylabel("K 'am' ")
ax4.tick_params(which='both',direction='in')
ax4.set_box_aspect(1)

fig.tight_layout(w_pad=1)

fig.savefig('fig1.svg', format='svg')

#ORAL and PALM
c=10**(-4);

fig=plt.figure(figsize=(14/2.54, 14/2.54), dpi=300)
#panel1
ax1=plt.subplot(2, 2, 1)
for i in range(19, 21):
    Ksigma.plotK(np.log(K[i][ids2[i]]),c,3)    #plots empirical distribution of K, normalized on [log(c),0]
x=np.arange(-14,-1,step=0.1)
#plot truncated lognormal fitted for K>log(c)
ax1.plot(np.log10(np.exp(x)),np.sqrt(2/math.pi)/s_oral *np.exp(-(x-mu_oral)**2 /2/(s_oral**2))/scipy.special.erfc((np.log(c)-mu_oral)/np.sqrt(2)/s_oral),'-k')
ax1.axvline(-4,color='k', linestyle='--')
ax1.set_xlabel('K')
ax1.set_ylabel('P(K)')
ax1.set_yscale('log')
plt.xticks(np.arange(-6,0,2),['$10^{-6}$','$10^{-4}$','$10^{-2}$'])
ax1.tick_params(which='both',direction='in')
ax1.set_box_aspect(1)

#panel2
ax2=plt.subplot(2, 2, 2)
for i in range(19, 21):
    hist, bin_edges = np.histogram(sigma[i][ids[i]]**2, density=True)
    ax2.plot(bin_edges[:-1]+np.diff(bin_edges),hist,'-o',markersize=3) #plots empirical distribution of sigma
ax2.plot(np.arange(5),1/gm_oral * np.exp(-np.arange(5)/gm_oral),'-k') #plots exponential fit
ax2.set_xlabel('$\sigma^2$')
ax2.set_ylabel('$P(\sigma^2)$')
ax2.set_yscale('log')
ax2.tick_params(which='both',direction='in')
ax2.set_box_aspect(1)

#panel 3
ax3=plt.subplot(2, 2, 3)
for i in range(16,19):
    Ksigma.plotK(np.log(K[i][ids2[i]]),c,3)    #plots empirical distribution of K, normalized on [log(c),0]
x=np.arange(-14,-1,step=0.1)
#plot truncated lognormal fitted for K>log(c)
ax3.plot(np.log10(np.exp(x)),np.sqrt(2/math.pi)/s_palms *np.exp(-(x-mu_palms)**2 /2/(s_palms**2))/scipy.special.erfc((np.log(c)-mu_palms)/np.sqrt(2)/s_palms),'-k')
ax3.axvline(-4,color='k', linestyle='--')
ax3.set_xlabel('K')
ax3.set_ylabel('P(K)')
ax3.set_yscale('log')
plt.xticks(np.arange(-6,0,2),['$10^{-6}$','$10^{-4}$','$10^{-2}$'])
ax3.tick_params(which='both',direction='in')
ax3.set_box_aspect(1)

#panel4
ax4=plt.subplot(2, 2, 4)
for i in range(16,19):
    hist, bin_edges = np.histogram(sigma[i][ids[i]]**2, density=True)
    ax4.plot(bin_edges[:-1]+np.diff(bin_edges),hist,'-o',markersize=3) #plots empirical distribution of sigma
#ax2.plot(np.arange(5),1/gm_oral * np.exp(-np.arange(5)/gm_oral),'-k') #plots exponential fit
ax4.set_xlabel('$\sigma^2$')
ax4.set_ylabel('$P(\sigma^2)$')
ax4.set_yscale('log')
ax4.tick_params(which='both',direction='in')
ax4.set_box_aspect(1)

fig.tight_layout(w_pad=4)

fig.savefig('Sfig1.svg', format='svg')


##################################################################################
# Analysis for Figure 2: relation between dissimilarity meaures in the null model
##################################################################################


#Set parameters of the community
S=10**4 # number of species
N=3*10**4 # number of reads
mu=-19 #mean of log(K)
s=5**2 #variance of log(K)
gm=0.9 #mean of sigma;

NC1=100  #pairs of samples to simulate that come from communities with the same values of K (same community at different times)
NC2=100  #pairs of community to simulate that come from communities with correlated but different K
predictions=community.predict(mu, s, N, S,'exp', gm,NC1,NC2)

fig=plt.figure(figsize=(19/2.54, 9/2.54), dpi=300)
ax1=fig.add_subplot(2,3, 1)
community.plot_smooth(predictions.pears, predictions.jacc,"Correlation","Jaccard")
ax2=fig.add_subplot(2,3, 2)
community.plot_smooth(predictions.pears, predictions.bc,"Correlation","Bray-Curtis")
ax3=fig.add_subplot(2,3, 3)
community.plot_smooth(predictions.pears, predictions.mh,"Correlation","Morisita-Horn")
ax4=fig.add_subplot(2,3, 4)
community.plot_smooth(predictions.pears, predictions.ovlp,"Correlation","Overlap")
ax5=fig.add_subplot(2,3, 5)
community.plot_smooth(predictions.pears, predictions.diss,"Correlation","Dissimilarity")
ax6=fig.add_subplot(2,3, 6)
community.plot_smooth(predictions.pears, predictions.phi,"Correlation","$\Phi$")
fig.tight_layout(w_pad=4)

fig.savefig('fig2.svg', format='svg')


##################################################################################
# Analysis for Figure 4A: Dissimilarity-Overlap curve in the null model
##################################################################################
#Set parameters of the community
S=10**4 # number of species
N=3*10**4 # number of reads
mu=-19 #mean of log(K)
s=5**2 #variance of log(K)
gm=0.9 #mean of sigma;

NC1=0  #we only simulate communities with uncorrelated gamma fluctuations
NC2=10

#Simulate communties with rho_k ranging from o to 1
rhok=np.arange(0,1.01, 0.01)
predictions=community.predict(mu, s, N, S,'exp', gm,NC1,NC2, rhok)
     
#Generate examples at different rho_k values  
#High rho_k      
[x1, x2]=community.generate_comm(mu,s,S, "exp", gm ,0.95,0)
xsamp1=np.random.multinomial(N,x1) # simulate multinomial sampling
xsamp2=np.random.multinomial(N,x2)            
ids1=(xsamp1>0) & (xsamp2>0) #OTUs sampled in both
overlap1=np.sum(xsamp1[ids1]/np.sum(xsamp1)+xsamp2[ids1]/np.sum(xsamp2))/2           
xsamp1hat=xsamp1[ids1]/np.sum(xsamp1[ids1])
xsamp2hat=xsamp2[ids1]/np.sum(xsamp2[ids1])
m=(xsamp1hat+xsamp2hat)/2
dissim1=np.sqrt(0.5*np.sum(xsamp1hat*np.log(xsamp1hat/m)+xsamp2hat*np.log(xsamp2hat/m))) # Dissimilarity
#Medium rho_k
[x3, x4]=community.generate_comm(mu,s,S, "exp", gm ,0.6,0)
xsamp3=np.random.multinomial(N,x3) # simulate multinomial sampling
xsamp4=np.random.multinomial(N,x4)         
ids1=(xsamp3>0) & (xsamp4>0) #OTUs sampled in both
overlap2=np.sum(xsamp3[ids1]/np.sum(xsamp3)+xsamp4[ids1]/np.sum(xsamp4))/2           
xsamp3hat=xsamp3[ids1]/np.sum(xsamp3[ids1])
xsamp4hat=xsamp4[ids1]/np.sum(xsamp4[ids1])
m=(xsamp3hat+xsamp4hat)/2
dissim2=np.sqrt(0.5*np.sum(xsamp3hat*np.log(xsamp3hat/m)+xsamp4hat*np.log(xsamp4hat/m))) # Dissimilarity    
#Low rho_k
[x5, x6]=community.generate_comm(mu,s,S, "exp", gm ,0.1,0)
xsamp5=np.random.multinomial(N,x5) # simulate multinomial sampling
xsamp6=np.random.multinomial(N,x6)     
ids1=(xsamp5>0) & (xsamp6>0) #OTUs sampled in both
overlap3=np.sum(xsamp5[ids1]/np.sum(xsamp5)+xsamp6[ids1]/np.sum(xsamp6))/2           
xsamp5hat=xsamp5[ids1]/np.sum(xsamp5[ids1])
xsamp6hat=xsamp6[ids1]/np.sum(xsamp6[ids1])
m=(xsamp5hat+xsamp6hat)/2
dissim3=np.sqrt(0.5*np.sum(xsamp5hat*np.log(xsamp5hat/m)+xsamp6hat*np.log(xsamp6hat/m))) # Dissimilarity    

#Plot Dissimilarity-Overlap curve

rhocol=[]
for i in rhok:
    for j in range(NC2):
        rhocol.append(i)
        
fig=plt.figure(figsize=(9/2.54, 7.5/2.54), dpi=300)
plt.scatter(predictions.ovlp, predictions.diss, s=10, c=rhocol) 
plt.plot(overlap1, dissim1, 'ko')
plt.plot(overlap2, dissim2, 'ko')
#plt.plot(overlap3, dissim3, 'ko')
plt.xlabel('Overlap')
plt.ylabel('Dissimilarity')
plt.colorbar()

fig.savefig('fig4A.svg', format='svg')
#Scatter plot of abundances 
cols=[[0, 0.4470, 0.7410], [0.8500, 0.3250, 0.0980], [0.6350, 0.0780, 0.1840]]
col=np.zeros((S,3))
for i in range(S):
    idx=0*((xsamp1[i]>0) & (xsamp2[i]>0))+1*((xsamp1[i]>0) & (xsamp2[i]==0))|((xsamp1[i]==0) & (xsamp2[i]>0)) +2*((xsamp1[i]==0) & (xsamp2[i]==0))
    col[i,:]= cols[idx]

fig=plt.figure(figsize=(5/2.54, 5/2.54), dpi=300)            
plt.scatter(x1,x2, c=col, s=2) 
plt.axvline(1/N, color='k',linestyle='--') 
plt.axhline(1/N, color='k',linestyle='--')
plt.tick_params(which='both',direction='in')
plt.xlim(10**(-10), 1)
plt.ylim(10**(-10), 1)
plt.yscale('log')
plt.xscale('log')
plt.xlabel('Rel. abd. sample 1')
plt.ylabel('Rel. abd. sample 2')

fig.savefig('fig4A_inset1.svg', format='svg')

cols=[[0, 0.4470, 0.7410], [0.8500, 0.3250, 0.0980], [0.6350, 0.0780, 0.1840]]
col=np.zeros((S,3))
for i in range(S):
    idx=0*((xsamp3[i]>0) & (xsamp4[i]>0))+1*((xsamp3[i]>0) & (xsamp4[i]==0))|((xsamp3[i]==0) & (xsamp4[i]>0)) +2*((xsamp3[i]==0) & (xsamp4[i]==0))
    col[i,:]= cols[idx]

fig=plt.figure(figsize=(5/2.54, 5/2.54), dpi=300)            
plt.scatter(x3,x4, c=col, s=2, rasterized = True) 
plt.axvline(1/N, color='k',linestyle='--') 
plt.axhline(1/N, color='k',linestyle='--')
plt.tick_params(which='both',direction='in')
plt.xlim(10**(-10), 1)
plt.ylim(10**(-10), 1)
plt.yscale('log')
plt.xscale('log')
plt.xlabel('Rel. abd. sample 1')
plt.ylabel('Rel. abd. sample 2')

fig.savefig('fig4A_inset2.svg', format='svg')

cols=[[0, 0.4470, 0.7410], [0.8500, 0.3250, 0.0980], [0.6350, 0.0780, 0.1840]]
col=np.zeros((S,3))
for i in range(S):
    idx=0*((xsamp5[i]>0) & (xsamp6[i]>0))+1*((xsamp5[i]>0) & (xsamp6[i]==0))|((xsamp5[i]==0) & (xsamp6[i]>0)) +2*((xsamp5[i]==0) & (xsamp6[i]==0))
    col[i,:]= cols[idx]

fig=plt.figure(figsize=(5/2.54, 5/2.54), dpi=300)            
plt.scatter(x5,x6, c=col, s=2, rasterized = True) 
plt.axvline(1/N, color='k',linestyle='--') 
plt.axhline(1/N, color='k',linestyle='--')
plt.tick_params(which='both',direction='in')
plt.xlim(10**(-10), 1)
plt.ylim(10**(-10), 1)
plt.yscale('log')
plt.xscale('log')
plt.xlabel('Rel. abd. sample 1')
plt.ylabel('Rel. abd. sample 2')


 
##################################################################################
# Analysis for Figures 3 and 4: comparison between data and null model
##################################################################################

#######---GUT---#########

#Extrapolate the total number of species in a community, from the observed K and knowing the distribution of K
Sreal=[]
for i in range(16):
    Sreal.append(np.sum(K[i]>10**(-4.5))*1/(1-norm.cdf(np.log(10**(-4.5)),mu_gut,s_gut)))
S_gut=round(np.mean(Sreal))

# Predict dissimilarity relations with null modell
N=3*10**4 # Number of reads
NC1=100  #pairs of samples to simulate that come from communities with the same values of K (same community at different times)
NC2=100  #pairs of community to simulate that come from communities with correlated but different K
predictions_gut=community.predict(mu_gut, s_gut**2, N, S_gut, 'exp', gm_gut,NC1,NC2)

#Compute dissimilarity relations from data
N1=100; #pairs of samples from the same individual
N2=200; #pairs of samples from different individuals
pairs=list(combinations(range(9),2))+[ (10,11), (12,14),(12,15),(13,14),(14,15)] #pairs of individuals
data_gut=community.compute(abd[0:16],N1,N2, pairs)


#######---PALMS---#########

#Extrapolate the total number of species in a community, from the observed K and knowing the distribution of K
Sreal=[]
for i in range(16,19):
    Sreal.append(np.sum(K[i]>10**(-4))*1/(1-norm.cdf(np.log(10**(-4)),mu_palms,s_palms)))
S_palms=round(np.mean(Sreal))


## Predict dissimilarity relations with null model
N=2*10**4 # Number of reads
NC1=100  #pairs of samples to simulate that come from communities with the same values of K (same community at different times)
NC2=100  #pairs of community to simulate that come from communities with correlated but different K
predictions_palms=community.predict(mu_palms, s_palms**2, N, S_palms, 'unif', 'nan' ,NC1,NC2)

#We import pre-computed predicition, because the computation is very long due to high number of OTUs
import pickle
f = open('pred_palm.pckl', 'rb')
#pickle.dump(predictions_palms, f)
predictions_palms = pickle.load(f)
f.close()

#Compute dissimilarity relations from data
N1=100; #pairs of samples from the same individual
N2=200; #pairs of samples from different individuals
pairs=(16,17) #pairs of individuals (only for left palm)
data_palms=community.compute(abd[16:18],N1,N2, pairs)

#######---ORAL---#########

#Extrapolate the total number of species in a community, from the observed K and knowing the distribution of K
Sreal=[]
for i in range(19,21):
    Sreal.append(np.sum(K[i]>10**(-4))*1/(1-norm.cdf(np.log(10**(-4)),mu_oral,s_oral)))
S_oral=round(np.mean(Sreal))

# Predict dissimilarity relations with null modell
N=3*10**4 # Number of reads
NC1=100  #pairs of samples to simulate that come from communities with the same values of K (same community at different times)
NC2=100  #pairs of community to simulate that come from communities with correlated but different K
predictions_oral=community.predict(mu_oral, s_oral**2, N, S_oral, 'exp', gm_oral,NC1,NC2)

#Compute dissimilarity relations from data
N1=100; #pairs of samples from the same individual
N2=200; #pairs of samples from different individuals
pairs=(19,20) #pairs of individuals 
data_oral=community.compute(abd[19:21],N1,N2, pairs)

################# Plot Figures ################

#GUT

#Fig3
fig=plt.figure(figsize=(19/2.54, 9/2.54), dpi=300)

ax1=fig.add_subplot(2,3, 1)
x=data_gut.pears
y=data_gut.jacc
ax1.scatter(x.iloc[:N1+1], y.iloc[:N1+1],s=10,color='none',  alpha=0.5, rasterized = True,  edgecolor='k')
ax1.scatter(x.iloc[N1+1:], y.iloc[N1+1:],s=10,c='none', alpha=0.5, rasterized = True, facecolor=None, edgecolor='gray',marker='s')
yrange=(min(y),max(y)) #range for the binned plot of the model predictions
community.plot_binned(predictions_gut.pears, predictions_gut.jacc,"Correlation","Jaccard", 'y', yrange)

ax2=fig.add_subplot(2,3, 2)
x=data_gut.pears
y=data_gut.bc
ax2.scatter(x.iloc[:N1+1], y.iloc[:N1+1],s=10,color='none',  alpha=0.5, rasterized = True,  edgecolor='k')
ax2.scatter(x.iloc[N1+1:], y.iloc[N1+1:],s=10,c='none', alpha=0.5, rasterized = True, facecolor=None, edgecolor='gray',marker='s')
xrange=(min(x),max(x)) #range for the binned plot of the model predictions
community.plot_binned(predictions_gut.pears, predictions_gut.bc,"Correlation","Bray-Curtis",'x', xrange)

ax3=fig.add_subplot(2,3, 3)
x=data_gut.pears
y=data_gut.mh
ax3.scatter(x.iloc[:N1+1], y.iloc[:N1+1],s=10,color='none',  alpha=0.5, rasterized = True,  edgecolor='k')
ax3.scatter(x.iloc[N1+1:], y.iloc[N1+1:],s=10,c='none', alpha=0.5, rasterized = True, facecolor=None, edgecolor='gray',marker='s')
xrange=(min(x),max(x)) #range for the binned plot of the model predictions
community.plot_binned(predictions_gut.pears, predictions_gut.mh,"Correlation","Morisita-Horn",'x', xrange)

ax4=fig.add_subplot(2,3, 4)
x=data_gut.pears
y=data_gut.ovlp
ax4.scatter(x.iloc[:N1+1], y.iloc[:N1+1],s=10,color='none',  alpha=0.5, rasterized = True,  edgecolor='k')
ax4.scatter(x.iloc[N1+1:], y.iloc[N1+1:],s=10,c='none', alpha=0.5, rasterized = True, facecolor=None, edgecolor='gray',marker='s')
xrange=(min(x),max(x)) #range for the binned plot of the model predictions
community.plot_binned(predictions_gut.pears, predictions_gut.ovlp,"Correlation","Overlap",'x', xrange)

ax5=fig.add_subplot(2,3, 5)
x=data_gut.pears
y=data_gut.diss
ax5.scatter(x.iloc[:N1+1], y.iloc[:N1+1],s=10,color='none',  alpha=0.5, rasterized = True,  edgecolor='k')
ax5.scatter(x.iloc[N1+1:], y.iloc[N1+1:],s=10,c='none', alpha=0.5, rasterized = True, facecolor=None, edgecolor='gray',marker='s')
xrange=(min(x),max(x)) #range for the binned plot of the model predictions
community.plot_binned(predictions_gut.pears, predictions_gut.diss,"Correlation","Dissimilarity",'x', xrange)

ax6=fig.add_subplot(2,3, 6)
x=data_gut.pears
y=data_gut.phi
ax6.scatter(x.iloc[:N1+1], y.iloc[:N1+1],s=10,color='none',  alpha=0.5, rasterized = True,  edgecolor='k')
ax6.scatter(x.iloc[N1+1:], y.iloc[N1+1:],s=10,c='none', alpha=0.5, rasterized = True, facecolor=None, edgecolor='gray',marker='s')
yrange=(min(y),max(y)) #range for the binned plot of the model predictions
community.plot_binned(predictions_gut.pears, predictions_gut.phi,"Correlation","$\Phi$",'y',yrange)

fig.tight_layout(w_pad=4)

fig.savefig('fig3.svg', format='svg')

#Fig 4B: relationship between overlap and dissimilarity
fig=plt.figure(figsize=(8/2.54, 8/2.54), dpi=300)
x=data_gut.ovlp
y=data_gut.diss
plt.scatter(x.iloc[:N1+1], y.iloc[:N1+1],s=10,color='none',  alpha=0.5, rasterized = True,  edgecolor='k')
plt.scatter(x.iloc[N1+1:], y.iloc[N1+1:],s=10,c='none', alpha=0.5, rasterized = True, facecolor=None, edgecolor='gray',marker='s')
yrange=(min(y),max(y)) #range for the binned plot of the model predictions
community.plot_binned(predictions_gut.ovlp, predictions_gut.diss,"Overlap","Dissimilarity", 'y', yrange, xmin=0.4, xmax=1)
plt.ylim(0, 0.9)

fig.savefig('fig4.svg', format='svg')

#inset
fig=plt.figure(figsize=(4/2.54, 4/2.54), dpi=300)
x=-np.log(1-data_gut.ovlp)
y=data_gut.diss
plt.scatter(x.iloc[:N1+1], y.iloc[:N1+1],s=10,color='none',  alpha=0.5, rasterized = True,  edgecolor='k')
plt.scatter(x.iloc[N1+1:], y.iloc[N1+1:],s=10,c='none', alpha=0.5, rasterized = True, facecolor=None, edgecolor='gray',marker='s')
xrange=(min(x),max(x)) #range for the binned plot of the model predictions
community.plot_binned(-np.log(1-predictions_gut.ovlp), predictions_gut.diss,"-log(1-Overlap)","Dissimilarity", 'x', xrange, step=0.5, xmin=0.5, xmax=6.2)

fig.savefig('fig4_inset.svg', format='svg')



#PALM

#Fig S2
fig=plt.figure(figsize=(19/2.54, 9/2.54), dpi=300)

ax1=fig.add_subplot(2,3, 1)
x=data_palms.pears
y=data_palms.jacc
ax1.scatter(x.iloc[:N1+1], y.iloc[:N1+1],s=10,color='none',  alpha=0.5, rasterized = True,  edgecolor='k')
ax1.scatter(x.iloc[N1+1:], y.iloc[N1+1:],s=10,c='none', alpha=0.5, rasterized = True, facecolor=None, edgecolor='gray',marker='s')
yrange=(min(y),max(y)) #range for the binned plot of the model predictions
community.plot_binned(predictions_palms.pears, predictions_palms.jacc,"Correlation","Jaccard", 'y', yrange)

ax2=fig.add_subplot(2,3, 2)
x=data_palms.pears
y=data_palms.bc
ax2.scatter(x.iloc[:N1+1], y.iloc[:N1+1],s=10,color='none',  alpha=0.5, rasterized = True,  edgecolor='k')
ax2.scatter(x.iloc[N1+1:], y.iloc[N1+1:],s=10,c='none', alpha=0.5, rasterized = True, facecolor=None, edgecolor='gray',marker='s')
xrange=(min(x),max(x)) #range for the binned plot of the model predictions
community.plot_binned(predictions_palms.pears, predictions_palms.bc,"Correlation","Bray-Curtis",'x', xrange)

ax3=fig.add_subplot(2,3, 3)
x=data_palms.pears
y=data_palms.mh
ax3.scatter(x.iloc[:N1+1], y.iloc[:N1+1],s=10,color='none',  alpha=0.5, rasterized = True,  edgecolor='k')
ax3.scatter(x.iloc[N1+1:], y.iloc[N1+1:],s=10,c='none', alpha=0.5, rasterized = True, facecolor=None, edgecolor='gray',marker='s')
xrange=(min(x),max(x)) #range for the binned plot of the model predictions
community.plot_binned(predictions_palms.pears, predictions_palms.mh,"Correlation","Morisita-Horn",'x', xrange)


ax4=fig.add_subplot(2,3, 4)
x=data_palms.pears
y=data_palms.ovlp
ax4.scatter(x.iloc[:N1+1], y.iloc[:N1+1],s=10,color='none',  alpha=0.5, rasterized = True,  edgecolor='k')
ax4.scatter(x.iloc[N1+1:], y.iloc[N1+1:],s=10,c='none', alpha=0.5, rasterized = True, facecolor=None, edgecolor='gray',marker='s')
xrange=(min(x),max(x)) #range for the binned plot of the model predictions
community.plot_binned(predictions_palms.pears, predictions_palms.ovlp,"Correlation","Overlap",'x', xrange)

ax5=fig.add_subplot(2,3, 5)
x=data_palms.pears
y=data_palms.diss
ax5.scatter(x.iloc[:N1+1], y.iloc[:N1+1],s=10,color='none',  alpha=0.5, rasterized = True,  edgecolor='k')
ax5.scatter(x.iloc[N1+1:], y.iloc[N1+1:],s=10,c='none', alpha=0.5, rasterized = True, facecolor=None, edgecolor='gray',marker='s')
xrange=(min(x),max(x)) #range for the binned plot of the model predictions
community.plot_binned(predictions_palms.pears, predictions_palms.diss,"Correlation","Dissimilarity",'x', xrange)


ax6=fig.add_subplot(2,3, 6)
x=data_palms.pears
y=data_palms.phi
ax6.scatter(x.iloc[:N1+1], y.iloc[:N1+1],s=10,color='none',  alpha=0.5, rasterized = True,  edgecolor='k')
ax6.scatter(x.iloc[N1+1:], y.iloc[N1+1:],s=10,c='none', alpha=0.5, rasterized = True, facecolor=None, edgecolor='gray',marker='s')
yrange=(min(y),max(y)) #range for the binned plot of the model predictions
community.plot_binned(predictions_palms.pears, predictions_palms.phi,"Correlation","$\Phi$",'y',yrange)


fig.tight_layout(w_pad=4)

fig.savefig('Sfig2.svg', format='svg')

#Fig S4 A: relationship between overlap and dissimilarity
fig=plt.figure(figsize=(8/2.54, 8/2.54), dpi=300)
x=data_palms.ovlp
y=data_palms.diss
plt.scatter(x.iloc[:N1+1], y.iloc[:N1+1],s=10,color='none',  alpha=0.5, rasterized = True,  edgecolor='k')
plt.scatter(x.iloc[N1+1:], y.iloc[N1+1:],s=10,c='none', alpha=0.5, rasterized = True, facecolor=None, edgecolor='gray',marker='s')
yrange=(min(y),max(y)) #range for the binned plot of the model predictions
community.plot_binned(predictions_palms.ovlp, predictions_palms.diss,"Overlap","Dissimilarity", 'y', yrange, xmin=0.4, xmax=1)
plt.ylim(0, 0.9)

fig.savefig('figS4A.svg', format='svg')

#inset
fig=plt.figure(figsize=(4/2.54, 4/2.54), dpi=300)
x=-np.log(1-data_palms.ovlp)
y=data_palms.diss
plt.scatter(x.iloc[:N1+1], y.iloc[:N1+1],s=10,color='none',  alpha=0.5, rasterized = True,  edgecolor='k')
plt.scatter(x.iloc[N1+1:], y.iloc[N1+1:],s=10,c='none', alpha=0.5, rasterized = True, facecolor=None, edgecolor='gray',marker='s')
xrange=(min(x),max(x)) #range for the binned plot of the model predictions
community.plot_binned(-np.log(1-predictions_palms.ovlp), predictions_palms.diss,"-log(1-Overlap)","Dissimilarity", 'x', xrange, step=0.5, xmin=0.5, xmax=6.2)

fig.savefig('figS4A_inset.svg', format='svg')

#ORAL

#Fig S3
fig=plt.figure(figsize=(19/2.54, 9/2.54), dpi=300)

ax1=fig.add_subplot(2,3, 1)
x=data_oral.pears
y=data_oral.jacc
ax1.scatter(x.iloc[:N1+1], y.iloc[:N1+1],s=10,color='none',  alpha=0.5, rasterized = True,  edgecolor='k')
ax1.scatter(x.iloc[N1+1:], y.iloc[N1+1:],s=10,c='none', alpha=0.5, rasterized = True, facecolor=None, edgecolor='gray',marker='s')
yrange=(min(y),max(y)) #range for the binned plot of the model predictions
community.plot_binned(predictions_oral.pears, predictions_oral.jacc,"Correlation","Jaccard", 'y',yrange)

ax2=fig.add_subplot(2,3, 2)
x=data_oral.pears
y=data_oral.bc
ax2.scatter(x.iloc[:N1+1], y.iloc[:N1+1],s=10,color='none',  alpha=0.5, rasterized = True,  edgecolor='k')
ax2.scatter(x.iloc[N1+1:], y.iloc[N1+1:],s=10,c='none', alpha=0.5, rasterized = True, facecolor=None, edgecolor='gray',marker='s')
yrange=(min(y),max(y)) #range for the binned plot of the model predictions
community.plot_binned(predictions_oral.pears, predictions_oral.bc,"Correlation","Bray-Curtis",'y', yrange)

ax3=fig.add_subplot(2,3, 3)
x=data_oral.pears
y=data_oral.mh
ax3.scatter(x.iloc[:N1+1], y.iloc[:N1+1],s=10,color='none',  alpha=0.5, rasterized = True,  edgecolor='k')
ax3.scatter(x.iloc[N1+1:], y.iloc[N1+1:],s=10,c='none', alpha=0.5, rasterized = True, facecolor=None, edgecolor='gray',marker='s')
yrange=(min(y),max(y)) #range for the binned plot of the model predictions
community.plot_binned(predictions_oral.pears, predictions_oral.mh,"Correlation","Morisita-Horn",'y',yrange)

ax4=fig.add_subplot(2,3, 4)
x=data_oral.pears
y=data_oral.ovlp
ax4.scatter(x.iloc[:N1+1], y.iloc[:N1+1],s=10,color='none',  alpha=0.5, rasterized = True,  edgecolor='k')
ax4.scatter(x.iloc[N1+1:], y.iloc[N1+1:],s=10,c='none', alpha=0.5, rasterized = True, facecolor=None, edgecolor='gray',marker='s')
xrange=(min(x),max(x)) #range for the binned plot of the model predictions
community.plot_binned(predictions_oral.pears, predictions_oral.ovlp,"Correlation","Overlap",'x', xrange)

ax5=fig.add_subplot(2,3, 5)
x=data_oral.pears
y=data_oral.diss
ax5.scatter(x.iloc[:N1+1], y.iloc[:N1+1],s=10,color='none',  alpha=0.5, rasterized = True,  edgecolor='k')
ax5.scatter(x.iloc[N1+1:], y.iloc[N1+1:],s=10,c='none', alpha=0.5, rasterized = True, facecolor=None, edgecolor='gray',marker='s')
xrange=(min(x),max(x)) #range for the binned plot of the model predictions
community.plot_binned(predictions_oral.pears, predictions_oral.diss,"Correlation","Dissimilarity",'x', xrange)

ax6=fig.add_subplot(2,3, 6)
x=data_oral.pears
y=data_oral.phi
ax6.scatter(x.iloc[:N1+1], y.iloc[:N1+1],s=10,color='none',  alpha=0.5, rasterized = True,  edgecolor='k')
ax6.scatter(x.iloc[N1+1:], y.iloc[N1+1:],s=10,c='none', alpha=0.5, rasterized = True, facecolor=None, edgecolor='gray',marker='s')
yrange=(min(y),max(y)) #range for the binned plot of the model predictions
community.plot_binned(predictions_oral.pears, predictions_oral.phi,"Correlation","$\Phi$",'y', yrange)

fig.tight_layout(w_pad=4)

fig.savefig('Sfig3.svg', format='svg')


#Fig S4 B: relationship between overlap and dissimilarity
fig=plt.figure(figsize=(8/2.54, 8/2.54), dpi=300)
x=data_oral.ovlp
y=data_oral.diss
plt.scatter(x.iloc[:N1+1], y.iloc[:N1+1],s=10,color='none',  alpha=0.5, rasterized = True,  edgecolor='k')
plt.scatter(x.iloc[N1+1:], y.iloc[N1+1:],s=10,c='none', alpha=0.5, rasterized = True, facecolor=None, edgecolor='gray',marker='s')
yrange=(min(y),max(y)) #range for the binned plot of the model predictions
community.plot_binned(predictions_oral.ovlp, predictions_oral.diss,"Overlap","Dissimilarity", 'y', yrange, xmin=0.4, xmax=1)
plt.ylim(0, 0.9)

fig.savefig('figS4B.svg', format='svg')

#inset
fig=plt.figure(figsize=(4/2.54, 4/2.54), dpi=300)
x=-np.log(1-data_oral.ovlp)
y=data_oral.diss
plt.scatter(x.iloc[:N1+1], y.iloc[:N1+1],s=10,color='none',  alpha=0.5, rasterized = True,  edgecolor='k')
plt.scatter(x.iloc[N1+1:], y.iloc[N1+1:],s=10,c='none', alpha=0.5, rasterized = True, facecolor=None, edgecolor='gray',marker='s')
xrange=(min(x),max(x)) #range for the binned plot of the model predictions
community.plot_binned(-np.log(1-predictions_oral.ovlp), predictions_oral.diss,"-log(1-Overlap)","Dissimilarity", 'x', xrange, step=0.5, xmin=0.5, xmax=6.2)

fig.savefig('figS4B_inset.svg', format='svg')


