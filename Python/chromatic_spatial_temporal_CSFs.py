#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division
from __future__ import print_function 
import os
import sys
import numpy as np
import matplotlib.pyplot as plt

__author__ = "Qiang Li - University of Valencia"
__licence__ = 'GPLv2'

'''
The experimental achromatic and chromatic spatial temporal CSFs.

Version::
    1.0:  6/21-9/21
    2.0:  11/21-11/21

If you have any problem, please contact qiang.li@uv.es or liqiangucas@outlook.com  
'''

#Spatial_Temporal WB CSF

fx0 = np.linspace(0.1,15,16)
ft0 = np.linspace(0.1,10,11)

freq_x = [0, 0.5, 2, 4, 8] #cpd
freq_t = [0, 2, 6, 12, 24] #hz

CSF_wb_theor = np.array([[5.7718,	7.0251,	21.4979,	26.0284,	10.3905],
                [12.6428,	13.9682,	30.2892,	28.5677,	8.956],
                [30.123,	25.793,	34.9158,	29.0382,	7.804],
                [31.8934,	22.2669,	24.0124,	16.6067,	4.4373],
                [13.5133,	7.1705,	7.9357,	6.451,	1.9325]])

CSF_wb_theor = CSF_wb_theor.T


CSFwb = np.zeros((len(freq_x),len(freq_t)))
for freq_x_index in range(0, len(freq_x)):
    for freq_t_index in range(0, len(freq_t)):
        CSFwb[freq_x_index,freq_t_index] = CSF_wb_theor[freq_x_index,freq_t_index]

CSF_wb_fliplr = np.concatenate( (np.fliplr(CSFwb[:,:]), CSFwb), axis=1)
CSF_wb_flipup = np.concatenate( (np.flipud(CSF_wb_fliplr[:,:]), CSF_wb_fliplr), axis=0)

xx_a = np.round(sorted(-ft0[0:11]))
xx_b = np.round(ft0[0:11]) 
xx_r = np.concatenate((xx_a, xx_b), axis=0)
yt_a = np.round(sorted(-ft0[:]))
yt_b = np.round(ft0[:]) 
yt_r = np.concatenate((yt_a, yt_b), axis=0)


#Spatial_Temporal RG CSF
fx0 = np.linspace(0.1,15,16)
ft0 = np.linspace(0.1,10,11)

freq_x = [0, 0.5, 2, 4, 8] #cpd
freq_t = [0, 2, 6, 12, 18] #hz

CSF_rg_theor = np.array ([[118.8314,    82.3226, 62.58,   27.0644, 15.6239],
                [104.0772,68.0583, 51.2312, 23.7715, 12.3273],
                [66.1501, 46.0094, 35.3643, 25.8247, 14.1686],
                [52.9703, 37.2209, 40.4204, 27.0733, 16.1121],
                [56.2976, 31.0869, 30.4969, 23.5542, 12.9032]
                ]); 

CSF_rg_theor = CSF_rg_theor.T

CSFrg = np.zeros((len(freq_x),len(freq_t)))
for freq_x_index in range(0, len(freq_x)):
    for freq_t_index in range(0, len(freq_t)):
        CSFrg[freq_x_index,freq_t_index] = CSF_rg_theor[freq_x_index,freq_t_index]

CSF_rg_fliplr = np.concatenate( (np.fliplr(CSFrg[:,:]), CSFrg), axis=1)
CSF_rg_flipup = np.concatenate( (np.flipud(CSF_rg_fliplr[:,:]), CSF_rg_fliplr), axis=0)

xx_a = np.round(sorted(-ft0[0:11]))
xx_b = np.round(ft0[0:11]) 
xx_r = np.concatenate((xx_a, xx_b), axis=0)
yt_a = np.round(sorted(-ft0[:]))
yt_b = np.round(ft0[:]) 
yt_r = np.concatenate((yt_a, yt_b), axis=0)

#Spatial_Temporal WB CSF
fx0 = np.linspace(0.1,15,16)
ft0 = np.linspace(0.1,10,11)

freq_x = [0, 0.5, 2, 4, 8] #cpd
freq_t = [0, 2, 6, 12, 18] #hz

CSF_yb_theor = np.array([[118.8314, 82.3226, 62.58, 27.0644, 15.6239],
                [104.0772, 68.0583, 51.2312, 23.7715, 12.3273],
                [66.1501, 46.0094, 35.3643, 25.8247, 14.1686],
                [52.9703, 37.2209, 40.4204, 27.0733, 16.1121],
                [56.2976, 31.0869, 30.4969, 23.5542, 12.9032]
                ]);


CSF_yb_theor = CSF_yb_theor.T

CSFyb = np.zeros((len(freq_x),len(freq_t)))
for freq_x_index in range(0, len(freq_x)):
    for freq_t_index in range(0, len(freq_t)):
        CSFyb[freq_x_index,freq_t_index] = CSF_yb_theor[freq_x_index,freq_t_index]

CSF_yb_fliplr = np.concatenate( (np.fliplr(CSFyb[:,:]), CSFyb), axis=1)
CSF_yb_flipup = np.concatenate( (np.flipud(CSF_yb_fliplr[:,:]), CSF_yb_fliplr), axis=0)

xx_a = np.round(sorted(-ft0[0:11]))
xx_b = np.round(ft0[0:11]) 
xx_r = np.concatenate((xx_a, xx_b), axis=0)
yt_a = np.round(sorted(-ft0[:]))
yt_b = np.round(ft0[:]) 
yt_r = np.concatenate((yt_a, yt_b), axis=0)
