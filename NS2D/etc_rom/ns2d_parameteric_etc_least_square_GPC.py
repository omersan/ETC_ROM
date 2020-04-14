# -*- coding: utf-8 -*-
"""
Created on Mon Oct  7 17:41:10 2019

@author: Suraj
"""
#%% Import libraries
import numpy as np
import matplotlib.pyplot as plt
from numpy import linalg as LA
from scipy.integrate import simps
import pyfftw

from numpy.random import seed
seed(1)
from tensorflow import set_random_seed
set_random_seed(2)
import pandas as pd
import time as time
import os

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.models import load_model
import keras.backend as K
K.set_floatx('float64')

from sklearn.preprocessing import MinMaxScaler
from sklearn.externals import joblib
from sklearn.model_selection import train_test_split

from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib import ticker

font = {'family' : 'Times New Roman',
        'weight' : 'bold',
        'size'   : 14}    
plt.rc('font', **font)

import matplotlib as mpl
mpl.rcParams['text.usetex'] = True
mpl.rcParams['text.latex.preamble'] = [r'\usepackage{amsmath}']


#%% Define Functions

###############################################################################
#POD Routines
###############################################################################         
def POD(u,R): #Basis Construction
    n,ns = u.shape
    U,S,Vh = LA.svd(u, full_matrices=False)
    Phi = U[:,:R]  
    L = S**2
    #compute RIC (relative inportance index)
    RIC = sum(L[:R])/sum(L)*100   
    return Phi,L,RIC

def PODproj(u,Phi): #Projection
    a = np.dot(u.T,Phi)  # u = Phi * a.T
    return a

def PODrec(a,Phi): #Reconstruction    
    u = np.dot(Phi,a.T)    
    return u


###############################################################################
#Interpolation Routines
###############################################################################  
# Grassmann Interpolation
def GrassInt(Phi,pref,p,pTest):
    # Phi is input basis [training]
    # pref is the reference basis [arbitrarty] for Grassmann interpolation
    # p is the set of training parameters
    # pTest is the testing parameter
    
    nx,nr,nc = Phi.shape
    Phi0 = Phi[:,:,pref] 
    Phi0H = Phi0.T 

    print('Calculating Gammas...')
    Gamma = np.zeros((nx,nr,nc))
    for i in range(nc):
        templ = Phi[:,:,i] - LA.multi_dot([Phi0,Phi0H,Phi[:,:,i]])
        tempr = LA.inv( np.dot(Phi0H,Phi[:,:,i]) )
        temp = np.dot(templ, tempr)
                       
        U, S, Vh = LA.svd(temp, full_matrices=False)
        S = np.diag(S)
        Gamma[:,:,i] = LA.multi_dot([U,np.arctan(S),Vh])
    
    print('Interpolating ...')
    alpha = np.ones(nc)
    GammaL = np.zeros((nx,nr))
    #% Lagrange Interpolation
    for i in range(nc):
        for j in range(nc):
            if (j != i) :
                alpha[i] = alpha[i]*(pTest-p[j])/(p[i]-p[j])
    for i in range(nc):
        GammaL = GammaL + alpha[i] * Gamma[:,:,i]
            
    U, S, Vh = LA.svd(GammaL, full_matrices=False)
    PhiL = LA.multi_dot([ Phi0 , Vh.T ,np.diag(np.cos(S)) ]) + \
           LA.multi_dot([ U , np.diag(np.sin(S)) ])
    PhiL = PhiL.dot(Vh)
    return PhiL

###############################################################################
#LSTM Routines
############################################################################### 
def create_training_data_lstm(training_set, m, n, lookback):
    ytrain = [training_set[i,:] for i in range(lookback,m)]
    ytrain = np.array(ytrain)    
    xtrain = np.zeros((m-lookback,lookback,n))
    for i in range(m-lookback):
        a = training_set[i,:]
        for j in range(1,lookback):
            a = np.vstack((a,training_set[i+j,:]))
        xtrain[i,:,:] = a
    return xtrain , ytrain



def rhs(nr, b_l, b_nl, a): # Right Handside of Galerkin Projection
    r2, r3, r = [np.zeros(nr) for _ in range(3)]
    
    for k in range(nr):
        r2[k] = 0.0
        for i in range(nr):
            r2[k] = r2[k] + b_l[i,k]*a[i]
    
    for k in range(nr):
        r3[k] = 0.0
        for j in range(nr):
            for i in range(nr):
                r3[k] = r3[k] + b_nl[i,j,k]*a[i]*a[j]
    
    r = r2 + r3    
    return r

###############################################################################
# Numerical Routines
###############################################################################
# Thomas algorithm for solving tridiagonal systems:    
def tdma(a, b, c, r, up, s, e):
    for i in range(s+1,e+1):
        b[i] = b[i] - a[i]/b[i-1]*c[i-1]
        r[i] = r[i] - a[i]/b[i-1]*r[i-1]   
    up[e] = r[e]/b[e]   
    for i in range(e-1,s-1,-1):
        up[i] = (r[i]-c[i]*up[i+1])/b[i]

# Computing first derivatives using the fourth order compact scheme:  
def pade4d(u, h, n):
    a, b, c, r = [np.zeros(n+1) for _ in range(4)]
    ud = np.zeros(n+1)
    i = 0
    b[i] = 1.0
    c[i] = 2.0
    r[i] = (-5.0*u[i] + 4.0*u[i+1] + u[i+2])/(2.0*h)
    for i in range(1,n):
        a[i] = 1.0
        b[i] = 4.0
        c[i] = 1.0
        r[i] = 3.0*(u[i+1] - u[i-1])/h
    i = n
    a[i] = 2.0
    b[i] = 1.0
    r[i] = (-5.0*u[i] + 4.0*u[i-1] + u[i-2])/(-2.0*h)
    tdma(a, b, c, r, ud, 0, n)
    return ud
    
# Computing second derivatives using the foruth order compact scheme:  
def pade4dd(u, h, n):
    a, b, c, r = [np.zeros(n+1) for _ in range(4)]
    udd = np.zeros(n+1)
    i = 0
    b[i] = 1.0
    c[i] = 11.0
    r[i] = (13.0*u[i] - 27.0*u[i+1] + 15.0*u[i+2] - u[i+3])/(h*h)
    for i in range(1,n):
        a[i] = 0.1
        b[i] = 1.0
        c[i] = 0.1
        r[i] = 1.2*(u[i+1] - 2.0*u[i] + u[i-1])/(h*h)
    i = n
    a[i] = 11.0
    b[i] = 1.0
    r[i] = (13.0*u[i] - 27.0*u[i-1] + 15.0*u[i-2] - u[i-3])/(h*h)
    
    tdma(a, b, c, r, udd, 0, n)
    return udd


def plot_3d_surface(x,t,field):
    
    fig = plt.figure(figsize=(8,6))
    ax = fig.add_subplot(111, projection='3d')
    
    X, Y = np.meshgrid(t, x)
    
    surf = ax.plot_surface(Y, X, field, cmap=plt.cm.viridis,
                           linewidth=1, antialiased=False,rstride=1,
                            cstride=1)
    
    fig.colorbar(surf, shrink=0.5, aspect=5)
    
    fig.tight_layout()
    plt.show()
    fig.savefig('3d.pdf')

#%% fast poisson solver using second-order central difference scheme
def fpsi(nx, ny, dx, dy, f):
    epsilon = 1.0e-6
    aa = -2.0/(dx*dx) - 2.0/(dy*dy)
    bb = 2.0/(dx*dx)
    cc = 2.0/(dy*dy)
    hx = 2.0*np.pi/np.float64(nx)
    hy = 2.0*np.pi/np.float64(ny)
    
    kx = np.empty(nx)
    ky = np.empty(ny)
    
    kx[:] = hx*np.float64(np.arange(0, nx))

    ky[:] = hy*np.float64(np.arange(0, ny))
    
    kx[0] = epsilon
    ky[0] = epsilon

    kx, ky = np.meshgrid(np.cos(kx), np.cos(ky), indexing='ij')
    
    data = np.empty((nx,ny), dtype='complex128')
    data1 = np.empty((nx,ny), dtype='complex128')
    
    data[:,:] = np.vectorize(complex)(f,0.0)

    a = pyfftw.empty_aligned((nx,ny),dtype= 'complex128')
    b = pyfftw.empty_aligned((nx,ny),dtype= 'complex128')
    
    fft_object = pyfftw.FFTW(a, b, axes = (0,1), direction = 'FFTW_FORWARD')
    fft_object_inv = pyfftw.FFTW(a, b,axes = (0,1), direction = 'FFTW_BACKWARD')
    
    e = fft_object(data)
    #e = pyfftw.interfaces.scipy_fftpack.fft2(data)
    
    e[0,0] = 0.0
    
    data1[:,:] = e[:,:]/(aa + bb*kx[:,:] + cc*ky[:,:])

    ut = np.real(fft_object_inv(data1))
        
    return ut

#%%
def nonlinear_term(nx,ny,dx,dy,wf,sf):
    '''
    this function returns -(Jacobian)
    
    '''
    w = np.zeros((nx+3,ny+3))
    
    w[1:nx+1,1:ny+1] = wf
    
    # periodic
    w[:,ny+1] = w[:,1]
    w[nx+1,:] = w[1,:]
    w[nx+1,ny+1] = w[1,1]
    
    # ghost points
    w[:,0] = w[:,ny]
    w[:,ny+2] = w[:,2]
    w[0,:] = w[nx,:]
    w[nx+2,:] = w[2,:]
    
    s = np.zeros((nx+3,ny+3))
    
    s[1:nx+1,1:ny+1] = sf
    
    # periodic
    s[:,ny+1] = s[:,1]
    s[nx+1,:] = s[1,:]
    s[nx+1,ny+1] = s[1,1]
    
    # ghost points
    s[:,0] = s[:,ny]
    s[:,ny+2] = s[:,2]
    s[0,:] = s[nx,:]
    s[nx+2,:] = s[2,:]
    
    gg = 1.0/(4.0*dx*dy)
    hh = 1.0/3.0
    
    f = np.zeros((nx+1,ny+1))
    
    #Arakawa
    j1 = gg*( (w[2:nx+3,1:ny+2]-w[0:nx+1,1:ny+2])*(s[1:nx+2,2:ny+3]-s[1:nx+2,0:ny+1]) \
             -(w[1:nx+2,2:ny+3]-w[1:nx+2,0:ny+1])*(s[2:nx+3,1:ny+2]-s[0:nx+1,1:ny+2]))

    j2 = gg*( w[2:nx+3,1:ny+2]*(s[2:nx+3,2:ny+3]-s[2:nx+3,0:ny+1]) \
            - w[0:nx+1,1:ny+2]*(s[0:nx+1,2:ny+3]-s[0:nx+1,0:ny+1]) \
            - w[1:nx+2,2:ny+3]*(s[2:nx+3,2:ny+3]-s[0:nx+1,2:ny+3]) \
            + w[1:nx+2,0:ny+1]*(s[2:nx+3,0:ny+1]-s[0:nx+1,0:ny+1]))

    j3 = gg*( w[2:nx+3,2:ny+3]*(s[1:nx+2,2:ny+3]-s[2:nx+3,1:ny+2]) \
            - w[0:nx+1,0:ny+1]*(s[0:nx+1,1:ny+2]-s[1:nx+2,0:ny+1]) \
            - w[0:nx+1,2:ny+3]*(s[1:nx+2,2:ny+3]-s[0:nx+1,1:ny+2]) \
            + w[2:nx+3,0:ny+1]*(s[2:nx+3,1:ny+2]-s[1:nx+2,0:ny+1]) )

    f = -(j1+j2+j3)*hh
                  
    return f[1:nx+1,1:ny+1]

def linear_term(nx,ny,dx,dy,re,f):
    w = np.zeros((nx+3,ny+3))
    
    w[1:nx+1,1:ny+1] = f
    
    # periodic
    w[:,ny+1] = w[:,1]
    w[nx+1,:] = w[1,:]
    w[nx+1,ny+1] = w[1,1]
    
    # ghost points
    w[:,0] = w[:,ny]
    w[:,ny+2] = w[:,2]
    w[0,:] = w[nx,:]
    w[nx+2,:] = w[2,:]
    
    aa = 1.0/(dx*dx)
    bb = 1.0/(dy*dy)
    
    f = np.zeros((nx+1,ny+1))
    
    lap = aa*(w[2:nx+3,1:ny+2]-2.0*w[1:nx+2,1:ny+2]+w[0:nx+1,1:ny+2]) \
        + bb*(w[1:nx+2,2:ny+3]-2.0*w[1:nx+2,1:ny+2]+w[1:nx+2,0:ny+1])
    
    f = lap/re
            
    return f[1:nx+1,1:ny+1]

def pbc(w):
    f = np.zeros((nx+1,ny+1))
    f[:nx,:ny] = w
    f[:,ny] = f[:,0]
    f[nx,:] = f[0,:]
    
    return f

#%% Main program:
# Inputs
nx =  256  #spatial grid number
ny = 256
nc = 4     #number of control parameters (nu)
ns = 200    #number of snapshot per each Parameter 
nr = 8      #number of modes
Re_start = 200.0
Re_final = 800.0
Re  = np.linspace(Re_start, Re_final, nc) #control Reynolds
nu = 1/Re   #control dissipation
lx = 2.0*np.pi
ly = 2.0*np.pi
dx = lx/nx
dy = ly/ny
dt = 1e-1
tm = 20.0

noise = 0.0

ReTest = 1000

training = 'false'

pref = 3 #Reference case in [0:nRe]

#%% Data generation for training
x = np.linspace(0, lx, nx+1)
y = np.linspace(0, ly, ny+1)
t = np.linspace(0, tm, ns+1)

um = np.zeros(((nx)*(ny), ns+1, nc))
up = np.zeros(((nx)*(ny), ns+1, nc))
uo = np.zeros(((nx)*(ny), ns+1, nc))

for p in range(0,nc):
    for n in range(0,ns+1):
        file_input = "./snapshots/Re_"+str(int(Re[p]))+"/w/w_"+str(int(n))+ ".csv"
        w = np.genfromtxt(file_input, delimiter=',')
        
        w1 = w[1:nx+1,1:ny+1]
        
        um[:,n,p] = np.reshape(w1,(nx)*(ny)) #snapshots from unperturbed solution
        up[:,n,p] = noise*um[:,n,p] #perturbation (unknown physics)
        uo[:,n,p] = um[:,n,p] + up[:,n,p] #snapshots from observed solution

#plot_3d_surface(x,t,uo[:,:,-1])

#%% POD basis computation
PHIw = np.zeros(((nx)*(ny),nr,nc))
PHIs = np.zeros(((nx)*(ny),nr,nc))        
       
L = np.zeros((ns+1,nc)) #Eigenvalues      
RIC = np.zeros((nc))    #Relative information content

print('Computing POD basis for vorticity ...')
for p in range(0,nc):
    u = uo[:,:,p]
    PHIw[:,:,p], L[:,p], RIC[p]  = POD(u, nr) 

#PHIw = PHIw/(np.sign(PHIw[0,:,:]))

#%% Calculating true POD coefficients (observed)
at = np.zeros((ns+1,nr,nc))
print('Computing true POD coefficients...')
for p in range(nc):
    at[:,:,p] = PODproj(uo[:,:,p],PHIw[:,:,p])
    PHIw[:,:,p] = PHIw[:,:,p]/(np.sign(at[0,:,p]))
    at[:,:,p] = at[:,:,p]/np.sign(at[0,:,p])

print('Computing POD basis for streamfunction ...')
for p in range(0,nc):
    for i in range(nr):
        phi_w = np.reshape(PHIw[:,i,p],[nx,ny])
        phi_s = fpsi(nx, ny, dx, dy, -phi_w)
        PHIs[:,i,p] = np.reshape(phi_s,(nx)*(ny))

at_modes = np.zeros((nc,ns+1,nr))
phi_basis = np.zeros((nc,nx*ny,nr))

for i in range(nc):
    at_modes[i,:,:] = at[:,:,i]
    phi_basis[i,:,:] = PHIw[:,:,i]

with open("./plotting/all_modes_true.csv", 'w') as outfile:
    outfile.write('# Array shape: {0}\n'.format(at_modes.shape))
    for data_slice in at_modes:
        np.savetxt(outfile, data_slice, delimiter=",")
        outfile.write('# New slice\n')

with open("./plotting/all_basis.csv", 'w') as outfile:
    outfile.write('# Array shape: {0}\n'.format(phi_basis.shape))
    for data_slice in phi_basis:
        np.savetxt(outfile, data_slice, delimiter=",")
        outfile.write('# New slice\n')

#%%
print('Reconstructing with true coefficients')
w = PODrec(at[:,:,1],PHIw[:,:,1])

w = w[:,-1]
w = np.reshape(w,[nx,ny])

fig, ax = plt.subplots(1,1,sharey=True,figsize=(5,4))
cs = ax.contourf(x[:nx],y[:ny],w.T, 120, cmap = 'jet')
ax.set_aspect(1.0)

fig.colorbar(cs,orientation='vertical')
fig.tight_layout() 
plt.show()
fig.savefig("reconstructed.eps", bbox_inches = 'tight')

#%% Galerkin projection [Fully Intrusive]

###############################
# Galerkin projection with nr
###############################
b_l = np.zeros((nr,nr,nc))
b_nl = np.zeros((nr,nr,nr,nc))
linear_phi = np.zeros(((nx)*(ny),nr,nc))
nonlinear_phi = np.zeros(((nx)*(ny),nr,nc))

#%% linear term   
for p in range(nc):
    for i in range(nr):
        phi_w = np.reshape(PHIw[:,i,p],[nx,ny])
        
        lin_term = linear_term(nx,ny,dx,dy,Re[p],phi_w)
        linear_phi[:,i,p] = np.reshape(lin_term,(nx)*(ny))

for p in range(nc):
    for k in range(nr):
        for i in range(nr):
            b_l[i,k,p] = np.dot(linear_phi[:,i,p].T , PHIw[:,k,p]) 
                   
#%% nonlinear term 
for p in range(nc):
    for i in range(nr):
        phi_w = np.reshape(PHIw[:,i,p],[nx,ny])
        for j in range(nr):  
            phi_s = np.reshape(PHIs[:,j,p],[nx,ny])
            nonlin_term = nonlinear_term(nx,ny,dx,dy,phi_w,phi_s)
            jacobian_phi = np.reshape(nonlin_term,(nx)*(ny))
            for k in range(nr):    
                b_nl[i,j,k,p] = np.dot(jacobian_phi.T, PHIw[:,k,p]) 

#%% solving ROM by Adams-Bashforth scheme          
aGP = np.zeros((ns+1,nr,nc))
for p in range(nc):
    aGP[0,:,p] = at[0,:nr,p]
    aGP[1,:,p] = at[1,:nr,p]
    aGP[2,:,p] = at[2,:nr,p]
    for k in range(3,ns+1):
        r1 = rhs(nr, b_l[:,:,p], b_nl[:,:,:,p], aGP[k-1,:,p])
        r2 = rhs(nr, b_l[:,:,p], b_nl[:,:,:,p], aGP[k-2,:,p])
        r3 = rhs(nr, b_l[:,:,p], b_nl[:,:,:,p], aGP[k-3,:,p])
        temp= (23/12) * r1 - (16/12) * r2 + (5/12) * r3
        aGP[k,:,p] = aGP[k-1,:,p] + dt*temp 

#%%
aGP_modes = np.zeros((nc,ns+1,nr))

for i in range(nc):
    aGP_modes[i,:,:] = aGP[:,:,i]
    
with open("./plotting/all_modes_gp.csv", 'w') as outfile:
    outfile.write('# Array shape: {0}\n'.format(aGP_modes.shape))
    for data_slice in aGP_modes:
        np.savetxt(outfile, data_slice, delimiter=",")
        outfile.write('# New slice\n')

np.save('true_modes_train',at)
np.save('gp_modes_train',aGP)
       
#%% plot basis functions
def plot_data_basis(x,y,PHI,filename):
    fig, ax = plt.subplots(nrows=4,ncols=2,figsize=(10,14))
    ax = ax.flat
    nrs = at.shape[1]
    
    for i in range(nrs):
        f = np.zeros((nx+1,ny+1))
        f[:nx,:ny] = np.reshape(PHI[:,i],[nx,ny])
        f[:,ny] = f[:,1]
        f[nx,:] = f[1,:]
        
        cs = ax[i].contourf(x,y,f.T, 10, cmap = 'jet')
        divider = make_axes_locatable(ax[i])
        cax = divider.append_axes('right', size='5%', pad=0.1)
        fig.colorbar(cs,cax=cax,orientation='vertical')
        ax[i].set_aspect(1.0)
        ax[i].set_title(r'$\phi_{'+str(i+1) + '}$')
        
    fig.tight_layout()    
    plt.show()
    fig.savefig(filename)

plot_data_basis(x,y,PHIw[:,:,-1],'basis.pdf')

#%% plot modal coefficients
def plot_data(t,at,aGP,filename):
    fig, ax = plt.subplots(nrows=4,ncols=2,figsize=(12,8))
    ax = ax.flat
    nrs = at.shape[1]
    
    for i in range(nrs):
        #for k in range(at.shape[2]):
        ax[i].plot(t,at[:,i],label=str(k))
        #ax[i].legend(loc=0)
        ax[i].plot(t,aGP[:,i],label=r'Exact Values')
        #ax[i].plot(t,aGPm[:,i],'r-.',label=r'True Values')
        ax[i].set_xlabel(r'$t$',fontsize=14)
        ax[i].set_ylabel(r'$a_{'+str(i+1) +'}(t)$',fontsize=14)    
    
    fig.tight_layout()
    
    fig.subplots_adjust(bottom=0.1)
    line_labels = ["True","Standard GP"]#, "ML-Train", "ML-Test"]
    plt.figlegend( line_labels,  loc = 'lower center', borderaxespad=0.0, ncol=3, labelspacing=0.)
    plt.show()
#    fig.savefig(filename)

plot_data(t,at[:,:,0],aGP[:,:,0],'modes_ns2d.pdf')#,aGP1[-1,:,:])       

#%%
res_proj = at - aGP # difference betwwen true and modified GP

# Create training data for LSTM
lookback = 3 #Number of lookbacks

# use xtrain from here
for p in range(nc):
    xt, yt = create_training_data_lstm(aGP[:,:,p], ns+1, nr, lookback)
    if p == 0:
        xtrain = xt
        ytrain = yt
    else:
        xtrain = np.vstack((xtrain,xt))
        ytrain = np.vstack((ytrain,yt))

data = xtrain # modified GP as the input data

# use ytrain from here
for p in range(nc):
    xt, yt = create_training_data_lstm(res_proj[:,:,p], ns+1, nr, lookback)
    if p == 0:
        xtrain = xt
        ytrain = yt
    else:
        xtrain = np.vstack((xtrain,xt))
        ytrain = np.vstack((ytrain,yt))

labels = ytrain
        
#%%
# Scaling data
p,q,r = data.shape
data2d = data.reshape(p*q,r)

scalerIn = MinMaxScaler(feature_range=(-1,1))
scalerIn = scalerIn.fit(data2d)
data2d = scalerIn.transform(data2d)
data = data2d.reshape(p,q,r)

scalerOut = MinMaxScaler(feature_range=(-1,1))
scalerOut = scalerOut.fit(labels)
labels = scalerOut.transform(labels)

xtrain = data
ytrain = labels

xtrain, xvalid, ytrain, yvalid = train_test_split(data, labels, test_size=0.2 , shuffle= True)

#%%
if training == 'true':
    m,n = ytrain.shape # m is number of training samples, n is number of output features [i.e., n=nr]
    
    # create the LSTM architecture
    model = Sequential()
    #model.add(Dropout(0.2))
    model.add(LSTM(80, input_shape=(lookback, n), return_sequences=True, activation='tanh'))
    #model.add(LSTM(120, input_shape=(lookback, n), return_sequences=True, activation='tanh'))
    #model.add(LSTM(40, input_shape=(lookback, n+1), return_sequences=True, activation='relu', kernel_initializer='uniform'))
    model.add(LSTM(80, input_shape=(lookback, n), activation='tanh'))
    model.add(Dense(n))
    
    # compile model
    #model.compile(loss='mean_absolute_percentage_error', optimizer='adam', metrics=['mean_absolute_percentage_error'])
    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mean_squared_error'])
    
    # run the model
    history = model.fit(xtrain, ytrain, epochs=600, batch_size=32, validation_data= (xvalid,yvalid))
    #history = model.fit(xtrain, ytrain, epochs=600, batch_size=32, validation_split=0.2)
    
    # evaluate the model
    scores = model.evaluate(xtrain, ytrain, verbose=0)
    print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
    
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    
    plt.figure()
    epochs = range(1, len(loss) + 1)
    plt.semilogy(epochs, loss, 'b', label='Training loss')
    plt.semilogy(epochs, val_loss, 'r', label='Validationloss')
    plt.title('Training and validation loss')
    plt.legend()
    filename = 'loss.png'
    plt.savefig(filename, dpi = 400)
    plt.show()
    
    # Save the model
    filename = './plotting/h2_s2_model.hd5'
    model.save(filename)

#%% Testing
# Data generation for testing

uTest = np.zeros((nx*ny, ns+1))
upTest = np.zeros((nx*ny, ns+1))
uoTest = np.zeros((nx*ny, ns+1))
nuTest = 1/ReTest

for n in range(ns+1):
   file_input = "./snapshots/Re_"+str(int(ReTest))+"/w/w_"+str(int(n))+ ".csv"
   w = np.genfromtxt(file_input, delimiter=',')
    
   w1 = w[1:nx+1,1:ny+1]
    
   uTest[:,n] = np.reshape(w1,(nx)*(ny)) #snapshots from unperturbed solution
   upTest[:,n] = noise*uTest[:,n] #perturbation (unknown physics)
   uoTest[:,n] = uTest[:,n] + upTest[:,n] #snapshots from observed solution

w_fom = uoTest[:,-1] # last time step
w_fom = np.reshape(w_fom,[nx,ny])
w_fom = pbc(w_fom)

#% POD basis computation     
print('Computing testing POD basis...')
PHItrue, Ltrue, RICtrue  = POD(uoTest, nr) 

#PHItrue = PHItrue/np.sign(PHItrue[0,:])
  
#% Calculating true POD coefficients
print('Computing testing POD coefficients...')
aTrue = PODproj(uoTest,PHItrue)

PHItrue[:,:] = PHItrue[:,:]/(np.sign(aTrue[0,:]))
aTrue[:,:] = aTrue[:,:]/np.sign(aTrue[0,:])

#%% Basis Interpolation

PHIwtest = GrassInt(PHIw,pref,nu,nuTest)

#PHIwtest = PHIwtest/np.sign(PHIwtest[0,:])

aTest = PODproj(uoTest,PHIwtest)

PHIwtest[:,:] = PHIwtest[:,:]/(np.sign(aTest[0,:]))
aTest[:,:] = aTest[:,:]/np.sign(aTest[0,:])

print('Reconstructing with true coefficients for test Re')
w_test = PODrec(aTest[:,:],PHIwtest[:,:])

bases_test = np.zeros((2,nx*ny,nr))
bases_test[0] = PHItrue
bases_test[1] = PHIwtest

with open("./plotting/bases_"+str(int(ReTest))+".csv", 'w') as outfile:
    outfile.write('# Array shape: {0}\n'.format(bases_test.shape))
    for data_slice in bases_test:
        np.savetxt(outfile, data_slice, delimiter=",")
        outfile.write('# New slice\n')

w_test = w_test[:,-1]
w_test = np.reshape(w_test,[nx,ny])
w_test = pbc(w_test)

PHIstest = np.zeros(((nx)*(ny),nr))

for i in range(nr):
    phi_w = np.reshape(PHIwtest[:,i],[nx,ny])
    phi_s = fpsi(nx, ny, dx, dy, -phi_w)    
    PHIstest[:,i] = np.reshape(phi_s,(nx)*(ny))

#%%
def source(nx,ny,x,y,ts,re):
    source = -0.1*np.cos(3*x)*np.cos(3*y)*np.exp(-ts/(re))
    
    return source[:nx,:ny].reshape(nx*ny)

def source1(nx,ny,x,y):
    source = -0.1*np.cos(3*x)*np.cos(3*y)
    
    return source[:nx,:ny].reshape(nx*ny)
    
nx = 256
ny = 256
x = np.linspace(0.0,2.0*np.pi,nx+1)
y = np.linspace(0.0,2.0*np.pi,ny+1)
x, y = np.meshgrid(x, y, indexing='ij')

#%%
tstart = time.time()
  
b_l = np.zeros((nr,nr))
b_nl = np.zeros((nr,nr,nr))
linear_phi = np.zeros(((nx)*(ny),nr))
nonlinear_phi = np.zeros(((nx)*(ny),nr))
 
for k in range(nr):
    phi_w = np.reshape(PHIwtest[:,k],[nx,ny])
    lin_term = linear_term(nx,ny,dx,dy,ReTest,phi_w)
    linear_phi[:,k] = np.reshape(lin_term,(nx)*(ny))

for k in range(nr):
    for i in range(nr):
        b_l[i,k] = np.dot(linear_phi[:,i].T , PHIwtest[:,k]) 
                   
for i in range(nr):
    phi_w = np.reshape(PHIwtest[:,i],[nx,ny])
    for j in range(nr):  
        phi_s = np.reshape(PHIstest[:,j],[nx,ny])
        nonlin_term = nonlinear_term(nx,ny,dx,dy,phi_w,phi_s)
        jacobian_phi = np.reshape(nonlin_term,(nx)*(ny))
        for k in range(nr):    
            b_nl[i,j,k] = np.dot(jacobian_phi.T, PHIwtest[:,k]) 

aGPtest = np.zeros((ns+1,nr))
aGPtest[0,:] = aTest[0,:nr]
aGPtest[1,:] = aTest[1,:nr]
aGPtest[2,:] = aTest[2,:nr]

for k in range(3,ns+1):
    r1 = rhs(nr, b_l[:,:], b_nl[:,:,:], aGPtest[k-1,:]) 
    r2 = rhs(nr, b_l[:,:], b_nl[:,:,:], aGPtest[k-2,:]) 
    r3 = rhs(nr, b_l[:,:], b_nl[:,:,:], aGPtest[k-3,:]) 
    temp= (23/12) * r1 - (16/12) * r2 + (5/12) * r3
    aGPtest[k,:] = aGPtest[k-1,:] + dt*temp 

gp_time = time.time() - tstart

#%%
tstart = time.time()

b_l = np.zeros((nr,nr))
b_nl = np.zeros((nr,nr,nr))
linear_phi = np.zeros(((nx)*(ny),nr))
nonlinear_phi = np.zeros(((nx)*(ny),nr))
 
for k in range(nr):
    phi_w = np.reshape(PHIwtest[:,k],[nx,ny])
    lin_term = linear_term(nx,ny,dx,dy,ReTest,phi_w)
    linear_phi[:,k] = np.reshape(lin_term,(nx)*(ny))

for k in range(nr):
    for i in range(nr):
        b_l[i,k] = np.dot(linear_phi[:,i].T , PHIwtest[:,k]) 
                   
for i in range(nr):
    phi_w = np.reshape(PHIwtest[:,i],[nx,ny])
    for j in range(nr):  
        phi_s = np.reshape(PHIstest[:,j],[nx,ny])
        nonlin_term = nonlinear_term(nx,ny,dx,dy,phi_w,phi_s)
        jacobian_phi = np.reshape(nonlin_term,(nx)*(ny))
        for k in range(nr):    
            b_nl[i,j,k] = np.dot(jacobian_phi.T, PHIwtest[:,k]) 
            
aGPCtest = np.zeros((ns+1,nr))
aGPCtest[0,:] = aTest[0,:nr]
aGPCtest[1,:] = aTest[1,:nr]
aGPCtest[2,:] = aTest[2,:nr]

re = 1000
s1 = source1(nx,ny,x,y)
ss = np.dot(s1.T , PHIwtest[:,:])

for k in range(3,ns+1):
#    s1 = source(nx,ny,x,y,(k-1)*dt,1000)
#    s2 = source(nx,ny,x,y,(k-2)*dt,1000)
#    s3 = source(nx,ny,x,y,(k-3)*dt,1000)
    r1 = rhs(nr, b_l[:,:], b_nl[:,:,:], aGPCtest[k-1,:]) + ss*np.exp(-(k-1)*dt/(re))#np.dot(s1.T , PHIwtest[:,:]) 
    r2 = rhs(nr, b_l[:,:], b_nl[:,:,:], aGPCtest[k-2,:]) + ss*np.exp(-(k-2)*dt/(re))#np.dot(s2.T , PHIwtest[:,:])
    r3 = rhs(nr, b_l[:,:], b_nl[:,:,:], aGPCtest[k-3,:]) + ss*np.exp(-(k-3)*dt/(re))#np.dot(s3.T , PHIwtest[:,:])
    temp= (23/12) * r1 - (16/12) * r2 + (5/12) * r3
    aGPCtest[k,:] = aGPCtest[k-1,:] + dt*temp 

gpc_time = time.time() - tstart
#%%    
np.save('true_modes_test',aTest)
np.save('gp_modes_test',aGPtest)
np.save('gpc_modes_test',aGPCtest)


print('Reconstructing with GP coefficients for test Re')
w_gp = PODrec(aGPtest[:,:],PHIwtest[:,:])
w_gp = w_gp[:,-1]
w_gp = np.reshape(w_gp,[nx,ny])
w_gp = pbc(w_gp)

print('Reconstructing with GP coefficients for test Re')
w_gpc = PODrec(aGPCtest[:,:],PHIwtest[:,:])
w_gpc = w_gpc[:,-1]
w_gpc = np.reshape(w_gpc,[nx,ny])
w_gpc = pbc(w_gpc)

#%% LSTM [Fully Nonintrusive]
# testing
model = load_model('./plotting/h2_s2_model.hd5')
    
testing_set = aTest
m,n = testing_set.shape
xtest = np.zeros((1,lookback,nr))
rLSTM = np.zeros((ns+1,nr))
aGPmlc = np.zeros((ns+1,nr))
aGPml = np.zeros((ns+1,nr))

#%%
# Initializing
for i in range(lookback):
    temp = testing_set[i,:]
    temp = temp.reshape(1,-1)
    xtest[0,i,:]  = temp
    rLSTM[i, :] = testing_set[i,:] - aGPtest[i,:] 
    aGPmlc[i,:] = testing_set[i,:] # modified GP + correction = True
    aGPml[i,:] = testing_set[i,:]

#%%
'''
iterative prediction with correction = True - standard GP
'''
tstart = time.time()

for i in range(lookback,ns+1):
    xtest_sc = scalerIn.transform(xtest[0])
    xtest_sc = xtest_sc.reshape(1,lookback,nr)
    ytest_sc = model.predict(xtest_sc)
    ytest = scalerOut.inverse_transform(ytest_sc) # residual/ correction
    rLSTM[i, :] = ytest
        
    for k in range(lookback-1):
        xtest[0,k,:] = xtest[0,k+1,:]
    
    r1 = rhs(nr, b_l[:,:], b_nl[:,:,:], aGPml[i-1,:])
    r2 = rhs(nr, b_l[:,:], b_nl[:,:,:], aGPml[i-2,:])
    r3 = rhs(nr, b_l[:,:], b_nl[:,:,:], aGPml[i-3,:])
    temp= (23/12) * r1 - (16/12) * r2 + (5/12) * r3
    
    aGPml[i,:] = aGPml[i-1,:] + dt*temp 
    
    aGPmlc[i,:] = aGPml[i,:] + ytest
            
    xtest[0,lookback-1,:] = aGPml[i,:] 

lstm_time = time.time() - tstart

np.savetxt('cpu_time.csv',[gp_time,gpc_time,lstm_time],delimiter=',')

print('Reconstructing with true coefficients for test Re')
w_ml = PODrec(aGPmlc[:,:],PHIwtest[:,:])

w_ml = w_ml[:,-1]
w_ml = np.reshape(w_ml,[nx,ny])
w_ml = pbc(w_ml)

aLS = np.genfromtxt("./plotting/ls_s2_modes_"+str(int(ReTest))+".csv", delimiter=',') 

w_ls = PODrec(aLS[:,:],PHIwtest[:,:])

w_ls = w_ls[:,-1]
w_ls = np.reshape(w_ls,[nx,ny])
w_ls = pbc(w_ls)

#%%
modal_coeffs = np.hstack((aTest,aGPtest,aGPmlc,aGPCtest))
field = np.zeros((6,w_fom.shape[0],w_fom.shape[1]))
field[0] = w_fom
field[1] = w_test
field[2] = w_gp
field[3] = w_ls
field[4] = w_ml
field[5] = w_gpc

filename = "./plotting/h2_s2_modes_"+str(int(ReTest))+".csv"
np.savetxt(filename, modal_coeffs, delimiter=",")
    
with open("./plotting/h2_s2_field_"+str(int(ReTest))+".csv", 'w') as outfile:
    outfile.write('# Array shape: {0}\n'.format(field.shape))
    for data_slice in field:
        np.savetxt(outfile, data_slice, delimiter=",")
        outfile.write('# New slice\n')

#%%
def plot_data_modes(t,aTrue,aGPtest,aGPCtest,aGPmlc1,filename):
    fig, ax = plt.subplots(nrows=4,ncols=2,figsize=(10,7))
    ax = ax.flat
    nrs = aTrue.shape[1]
    
    for i in range(nrs):
        ax[i].plot(t,aTrue[:,i],'k',label=r'True Values')
        ax[i].plot(t,aGPtest[:,i],'b--',label=r'Exact Values')
        ax[i].plot(t,aGPCtest[:,i],'g--',label=r'Exact Values')
        ax[i].plot(t,aGPmlc1[:,i],'m-.',label=r'Exact Values')
        ax[i].set_xlabel(r'$t$',fontsize=14)
        ax[i].set_ylabel(r'$a_{'+str(i+1) +'}(t)$',fontsize=14)    
        ax[i].set_xlim([0,1.50])
        ax[i].set_xticks(np.arange(min(t), max(t)+0.5, 5))
        
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.12)
    
    line_labels = ["True","GP","GP-C","Hybrid-1"]#, "ML-Train", "ML-Test"]
    plt.figlegend( line_labels,  loc = 'lower center', borderaxespad=0.0, ncol=4, labelspacing=0.)
    plt.show()
#    fig.savefig(filename)

plot_data_modes(t,aTest,aGPtest,aGPCtest,aGPmlc,'h2_s2_modes_'+str(int(ReTest))+'.pdf')

#%%
def plot_final_field(x,y,w_fom, w_true, w_gp, w_ml,filename):
#    fig, ax = plt.subplots(nrows=2,ncols=2,figsize=(7,6))
#    ax = ax.flat
#    nrs = 4
    
    m = [w_fom, w_true, w_gp, w_ml]
    title = ['FOM','True','GP','Hybrid']
    
    k = 0
    
    fig, axes = plt.subplots(nrows=1, ncols=4,figsize=(8,2))
    
    ax = axes.flat
    
    for i in range(4):
        cs = ax[i].contour(x,y,m[i].T, 20, cmap = 'coolwarm', vmin=-0.8, vmax=1.0)
        ax[i].set_aspect(1.0)
        if k<4:
            ax[i].set_title(title[k],fontsize='14')
        ax[i].set_xticks([0,2,4,6])
        ax[i].set_yticks([0,2,4,6])
        k = k+1
        
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.12)
    cbar_ax = fig.add_axes([0.16, -0.1, 0.7, 0.05])
    cbar = fig.colorbar(cs, cax=cbar_ax,orientation='horizontal')
    plt.show()
    fig.savefig(filename,bbox_inches='tight')
    #cbar.ax.set_yticklabels(['{:.1f}'.format(x) for x in [-0.2,0,0.2,0.4,0.6,0.8,1.0]])
    
plot_final_field(x,y, w_fom, w_test, w_gp, w_ml, 'h2_s2_field'+str(ReTest)+'.pdf')     

#%%
def plot_data_allmodes(t,at,aTest,aLSTM,filename):
    fig, ax = plt.subplots(nrows=4,ncols=2,figsize=(12,8))
    ax = ax.flat
    nrs = at.shape[1]
    
    for i in range(nrs):
        #for k in range(at.shape[2]):
        ax[i].plot(t,at[:,i],label=str(k))
        ax[i].plot(t,aTest[:,i],'k--',label=str(k))
        ax[i].plot(t,aLSTM[:,i],'b-.',label=str(k))
        #ax[i].legend(loc=0)
        #ax[i].plot(t,aGP[:,i],label=r'Exact Values')
        #ax[i].plot(t,aGPm[:,i],'r-.',label=r'True Values')
        ax[i].set_xlabel(r'$t$',fontsize=14)
        ax[i].set_ylabel(r'$a_{'+str(i+1) +'}(t)$',fontsize=14)    
    
    fig.tight_layout()
    
    fig.subplots_adjust(bottom=0.1)
    line_labels = ["Re=200","Re=400","Re=600","Re=800","Test="+str(ReTest),"LSTM"]#, "ML-Train", "ML-Test"]
    plt.figlegend( line_labels,  loc = 'lower center', borderaxespad=0.0, ncol=6, labelspacing=0.)
    plt.show()
    fig.savefig(filename)

plot_data_allmodes(t,at[:,:],aTest,aGPmlc,'h2_s2_allmodes'+str(ReTest)+'.pdf')#,aGP1[-1,:,:])

res_proj1 = aTest - aGPtest
#plot_data_allmodes(t[3:],res_proj[3:,:,:],res_proj1[3:,:],rLSTM[3:,:],'h2_s2_residual'+str(ReTest)+'.pdf')

#%%
k = np.linspace(1,ns+1,201)

L_per = np.zeros(L.shape)
Ltrue_per = np.zeros(L.shape[0])
for n in range(L.shape[0]):
    L_per[n,:] = np.sum(L[:n],axis=0,keepdims=True)/np.sum(L,axis=0,keepdims=True)
    Ltrue_per[n] = np.sum(Ltrue[:n],axis=0,keepdims=True)/np.sum(Ltrue,axis=0,keepdims=True)

eigen_history = np.hstack((L,Ltrue.reshape(-1,1),L_per,Ltrue_per.reshape(-1,1)))
filename = "./plotting/eigen_hist_"+str(int(ReTest))+".csv"
np.savetxt(filename, eigen_history, delimiter=",")