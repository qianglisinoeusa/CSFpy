
#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division
from __future__ import print_function 
import os
import sys
import numpy as np

__author__ = "Qiang Li - University of Valencia"
__licence__ = 'GPLv2'

'''
The Standard Spatial Observer Function including achromatic and chromatic.

Version::
    1.0:  6/20-9/20
    2.0:  1/21 ...

If you have any problem, please contact qiang.li@uv.es or liqiangucas@outlook.com  
'''

def csfsso(fs,N):
    
    '''
    
    % function [h,CSFSSO,CSFT,OE]=csfsso(fs,N,g,fm,l,s,w,os)
    % CSFSSO computes the CSF filter of the Standard Spatial Observer
    % It includes the CSF expression of Tyler and the oblique effect.
    %
    %        CSFSSO(fx,fy)=CSFTYL(f)*OE(fx,fy)  
    %
    % where:
    %
    %        CSFTYL(f) = g*(exp(-(f/fm))-l*exp(-(f^2/s^2)))
    %        OE(fx,fy) = 1-w*(4(1-exp(-(f/os)))*fx^2*fy^2)/f^4)
    %
    %        (fx,fy) = 2D spatial frequency vector (in cycl/deg)
    %              f = Modulus of the spatial frequency (in cycl/deg)
    %              g = Overall gain (Recom. value  g=330.74)
    %             fm = Parameter that control the exp. decay of the CSFTyler
    %                  (Recom. value fm=7.28)      
    %              l = Loss at low frequencies (Recom. value l=0.837)
    %              s = Parameter that control the atenuation of the loss factor at 
    %                  high frequencies (Recom. value s=1.809)
    %              w = Weighting of the Oblique Effect (Recom. value w=1)
    %                  (w=0 -> No oblique effect)
    %             os = Oblique Effect scale (controls the attenuation of the 
    %                  effect at high frequencies).
    %                  Recom. value os=6.664).
    % 
    % This program returns the (spatial domain) FIR filter coefficients to be applied 
    % with 'filter2' over the desired image. (These coefficients are similar to the PSF).
    %
    % Currently the filter design method is frequency sampling (see Image Processing Toolbox
    % Tutorial) to (approximately) obtain the desired frequency response, CSFSSO, with the
    % required filter order, N (odd), at a particular sampling frequency, fs (in cycles/deg).
    %
    % USAGE: [h,CSSFO,CSFT,OE]=csfsso(fs,N,g,fm,l,s,w,os);
    %
    % Recomended Values ( Watson&Ramirez,Modelfest OSA Meeting 1999 )
    %
    %        [h,CSSFO,CSFT,OE]=csfsso(fs,N,330.74,7.28,0.837,1.809,1,6.664);
    %
    % Matlab Version: https://isp.uv.es/code/visioncolor/CSF_toolbox.html
    '''
    #Parameters
    g=330.74
    fm=7.28
    l= 0.837
    s = 1.809
    w = 1
    oss = 6.664

    fx, fy=get_grids(N, N)
    
    fx=fx*fs/2
    fy=fy*fs/2  
    
    f = frequency_radius(fx, fy, clean_division=True)
    
    #To avoid singularity at zero frequency
    f=(f>0)*f+0.0001*(f==0)  
    csft=g*(np.exp(-(f/fm))-l*np.exp(-(f**2/s**2)))
    oe=1-w*(4*(1-np.exp(-(f/oss)))*fx**2*fy**2)/(f**4)
    Csfsso=(csft*oe) 
    h=fsamp2(Csfsso)

    return h, Csfsso, csft, oe


def get_grids(N_X, N_Y):
    
    """
        Use that function to define a reference outline for envelopes in Fourier space.
        In general, it is more efficient to define dimensions as powers of 2.
        output is always  of even size.
        A special case is when ``N_frame`` is one, in which case we generate a single frame.
    """
     
    fx, fy = np.mgrid[(-N_X//2):((N_X-1)//2 + 1), (-N_Y//2):((N_Y-1)//2 + 1)]
    fx, fy = fx*1./N_X, fy*1./N_Y
    
    return fx, fy


def frequency_radius(fx, fy, clean_division=False):
    
    """
     Returns the frequency radius. To see the effect of the scaling factor run
     'test_color.py'
    """
    
    f_radius2 = fx**2 + fy**2  # cf . Paul Schrater 00

    if clean_division:
        f_radius2[f_radius2==0.] = np.inf

    return np.sqrt(f_radius2)


def fsamp2(f1):
    
    """
    fsamp2 2-D FIR filter using frequency sampling.
    fsamp2 designs two-dimensional FIR filters based on a desired
    two-dimensional frequency response sampled at points on the
    Cartesian plane.
    """
    
    eps = 2.2204e-16
    hd = f1
    hd = np.rot90(np.fft.fftshift(np.rot90(hd,2)),2)
    h = np.fft.fftshift(np.fft.ifft2(hd))
    h = np.real(h) 
    h = np.rot90(h,2)

    return h


def csf_chrom(N,fs):

    '''
    % CSF_CHROM computes the spatial CSFs for the chromatic channels RG and YB
    % approximately reproducing the data in K. Mullen 85 Vis. Res. paper.
    %
    %  fs = sampling frequency (in cl/deg).
    %  N = size of the square discrete domain (in pixels).
    % 
    '''

    fx, fy=get_grids(N, N)
    fx=fx*fs/2
    fy=fy*fs/2  
    F = frequency_radius(fx, fy, clean_division=True)
    
    csfrg=np.zeros((N,N))
    csfyb=np.zeros((N,N))
    
    for i in range (1, N):
        [iaf_rg,csf_c]=iafrg(F[i,:],0.1,1,[0,0,0])
        csfrg[i,:]=csf_c       
        [iaf_yb,csf_c]=iafyb(F[i,:],0.1,1,[0,0,0])
        csfyb[i,:]=csf_c

    fact_rg=0.75
    fact_yb=0.55 
    max_CSF_achro=201.3

    csfrg=fact_rg*max_CSF_achro*csfrg/np.max(csfrg)
    csfyb=fact_yb*max_CSF_achro*csfyb/np.max(csfyb)

    return csfrg, csfyb

def iafrg(f,C,facfrec,nolin):
    
    '''
    iafrg calculates the values ​​of the information allocation function
    for the RG channel (experim) in the discrete frequency domain and
    contrasts defined by row vectors (f, C) of lengths m and n respectively-
    
    iafrg gives an m * n matrix such that in each row, it contains the values
    of the function for the different contrasts with fixed f
    
    
    nolin should be 2D dimensional. e.g. nolin=[nolin nolin]
    '''
    
    f = facfrec*f
    f=np.reshape(f, (1, f.shape[0]))

    f=f+0.00001*(f==0)
    C=C+0.0000001*(C==0)

    lf=len(f)
    lc=1

    iaf=np.zeros((lf,lc))
    p=[0.0840,0.8345,0.6313,0.2077]

    nolini=nolin
    nolin=nolini[1]

    if ((nolini[0]==0) and (nolini[1]==1)):
        nolin=1
     
    par=[-55.94, 6.64]
    
    if nolin==1:
        for i in range (1, lf):
            y=1/(1+np.exp((f[i]-par[0])/par[1]))
            cu=1/(100*2537.9*y)
            ace[i,:]=umbinc3(C,cu,p(1),p(2),p(3),p(4))
        iaf=1./ace
    else:
        y=1/(1+np.exp((f-par[0])/par[1]))    
        iaf=100*2537.9*y
        iaf=iaf.T*np.ones((1,1))
    csfrg=iaf[:,0].T

    if ((nolini[0]==0) and (nolini[1]==1)):
        s=iaf.shape
        iafc=np.sum(iaf.T).T
        iaf=iafc*np.ones((1,s[1]))
                         
    return iaf, csfrg


def iafyb(f,C,facfrec,nolin):
    '''
    iafyb calculates the values ​​of the information allocation function
    for channel YB (experim) in the discrete frequency domain and
    contrasts defined by row vectors (f, C) of lengths m and n respectively.

    iafyb gives an m * n matrix such that in each row, it contains the values
    of the function for the different contrasts (with fixed f)
    (for the csf given to be correct, the first contrast must be close to 0)
    '''
        
    f=facfrec*f
    f=np.reshape(f, (1, f.shape[0]))

    f=f+0.00001*(f==0)
    C=C+0.0000001*(C==0)

    lf=len(f)
    lc=1

    iaf=np.zeros((lf,lc))
    #p=[0.0840 0.8345 0.6313 0.2077]
    p=[0.1611,1.3354,0.3077,0.7746]
    
    nolini=nolin
    nolin=nolini[1]

    if ((nolini[0]==0) and (nolini[1]==1)):
        nolin=1      

    par=[-31.72, 4.13]
    
    if nolin==1:
        for i in range (1, lf):
            y=1/(1+np.exp((f[i]-par[0])/par[1]))
            cu=1/(100*719.7*y)
            ace[i,:]=umbinc3(C,cu,p(1),p(2),p(3),p(4))
        iaf=1./ace
    else:
        y=1/(1+np.exp((f-par[0])/par[1]))
        iaf=100*719.7*y
        iaf=iaf.T*np.ones((1,1))
        
    csfyb=iaf[:,0].T

    if ((nolini[0]==0) and (nolini[1]==1)):
        s=iaf.shape
        iafc=np.sum(iaf.T).T
        iaf=iafc*np.ones((1,s[1]))
          
    return iaf, csfyb


def CSF_Delay(x, nfreq):
    '''
    Contrast Sensitivity Function implemented with Delay version.
    
    The CSF measures the sensitivity of human visual system to the various frequencies of visual stimuli, 
    Here we apply an adjusted CSF model given by:

    The mathmatic equation of CSF located in my CortexComputing notebook(Random section)

    Input:
        x - Define size of domain, float
        nfreq - Fourier frequency, float

    Output:
        CSF -  Fourier Space of CSF, 2darray
    '''

    params=[0.7, 7.8909, 0.9809, 2.6, 0.0192, 0.114, 1.1]

    N_x=nfreq
    N_x_=np.linspace(0, N_x, x+1)-nfreq/2
    N_x_up=np.ceil(N_x_[:-1])

    [xplane,yplane]=np.meshgrid(N_x_up, N_x_up)
    plane=(xplane+1j*yplane)  
    radfreq=np.abs(plane)	

    s=(1-params[0])/2*np.cos(4*np.angle(plane))+(1+params[0])/2
    radfreq=radfreq/s
    csf = params[3]*(params[4]+params[5]*radfreq)*np.exp(-(params[5]*radfreq)**params[6])
    f = radfreq < params[1]
    csf[f] = params[2]
    return csf


def optical_MTF_Barten1999(u, sigma=0.5):
  u = as_float_array(u)
  sigma = as_float_array(sigma)
  return np.exp(-2 * np.pi**2 * sigma**2 * u**2)

def pupil_diameter_Barten1999(L, X_0):
  return 5 - 3 * np.tanh(0.4 * np.log(L * X_0**2 / 40**2))


def sigma_Barten1999(sigma_0=0.5 / 60, C_ab=0.08 / 60, d=2.5): 
  sigma_0 = as_float_array(sigma_0)
  C_ab = as_float_array(C_ab)
  return np.sqrt((sigma_0)**2 + (C_ab * d)**2)


def retinal_illuminance_Barten1999(L, d=2.5, stiles_crawford_correction=True):
  d = as_float_array(d)
  L = as_float_array(L)
  E = (np.pi * d**2) / 4 * L
  if stiles_crawford_correction:
      E *= (1 - (d / 9.7)**2 + (d / 12.4)**4)
  return E


def function_contrast_sensitivity_Barten1999(u,sigma=sigma_Barten1999(0.5 / 60, 0.08 / 60, 3.0),
                      k=3.0,
                      T=0.1,
                      X_0=60,
                      X_max=12,
                      N_max=15,
                      n=0.03,
                      p=1.2 * 10**6,
                      E=retinal_illuminance_Barten1999(20, 3),
                      phi_0=3 * 10**-8,
                      u_0=7):
    
  u = as_float_array(u)
  k = as_float_array(k)
  T = as_float_array(T)
  X_0 = as_float_array(X_0)
  X_max = as_float_array(X_max)
  N_max = as_float_array(N_max)
  n = as_float_array(n)
  p = as_float_array(p)
  E = as_float_array(E)
  phi_0 = as_float_array(phi_0)
  u_0 = as_float_array(u_0)

  M_opt = optical_MTF_Barten1999(u, sigma)

  S = (M_opt / k) / np.sqrt(2 / T * (1 / X_0**2 + 1 / X_max**2 + u**2 / N_max**2) *
      (1 / (n * p * E) + phi_0 / (1 - np.exp(-(u / u_0)**2))))

  return S

