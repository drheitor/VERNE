#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 24 17:20:50 2019
@author: Heitor
"""

import numpy as np
import matplotlib.pyplot as plt
from astropy.modeling import models
from astropy import units as u
from specutils.spectra import Spectrum1D, SpectralRegion
from specutils.fitting import fit_generic_continuum
from scipy.interpolate import interp1d
import os

#--------------------------------



#INPUT_spec= 'specI_19_spec11nPSF_norm_1800-6_figs.txt'

INPUT_spec= 'specI_22_spec11nPSF_norm_1800-6_figs.txt'

INPUT_spec='rv-Ter9-0072.ascii'
#INPUT_spec= sys.argv[1]

#FILENAME= INPUT_spec[0:31]+'norm.txt'

FILENAME= 'rv-Ter9-0072norm.txt'


try:    
    os.system("mkdir NORM")
except:
    print('...')


wl,fl = np.genfromtxt(INPUT_spec, unpack=True, skip_header=1)

# TRIM LIMITS
lamb1=8405
lamb2=8700

interactive_mode = False

#trim spectrum
mask = (wl > lamb1) & (wl < lamb2)
wl=wl[mask]
fl=fl[mask]

spectrum = Spectrum1D(flux=fl*u.Jy, spectral_axis=wl*u.AA)

#         start   end    start    end
regions=[8489.0, 8563.0, 8642.0, 8697.0 ]


if interactive_mode == False:
    g1_fit = fit_generic_continuum(spectrum, exclude_regions=[SpectralRegion(regions[0]*u.AA, regions[1]*u.AA), 
                                                          SpectralRegion(regions[2]*u.AA, regions[3]*u.AA)])    
    y_continuum_fitted = g1_fit(wl*u.AA)
    
    spec_normalized = spectrum / y_continuum_fitted 
    print(FILENAME+ 'was normalized automaticaly... \n')

#add a spline with a selected dots 

#--------------------------------

if interactive_mode == True:
    f0=plt.figure(figsize=(12,7))
    ax0 = f0.add_subplot(111)

    ax0.plot(wl, fl)
    ax0.set_title('Line select')
    ax0.grid(True)

    point = plt.ginput(n=0, timeout=0, show_clicks=True, mouse_add=1, mouse_pop=3, mouse_stop=2)

    xn=[]
    yn=[]
    n=0
    for i in point:
        xn.append(point[n][0])
        yn.append(point[n][1])
        n=n+1
      

#--------------------------------

    f = interp1d(xn,yn, kind='cubic')
    xnew = np.linspace(wl[0], wl[-1], num=len(wl)*10, endpoint=True)
    

#g1_fit = fit_generic_continuum(spectrum)
#continum = g1_fit(wl*u.AA)

#--------------------------------


    f1=plt.figure(figsize=(12,7))
    ax1 = f1.add_subplot(111)

    ax1.plot(wl, fl)
    ax1.set_title('Continuum Fit')
    #ax1.plot(wl, y_continuum_fitted)
    #ax1.plot(spec_normalized.spectral_axis, spec_normalized.flux)
    ax1.plot(wl,f(wl))
    f1.savefig('fit-'+FILENAME+'.pdf')

    continum = f(wl)
    spec_normalized = spectrum / continum 
    
    print(FILENAME+ 'was normalized... \n')
#--------------------------------



f2=plt.figure(figsize=(12,7))
ax2 = f2.add_subplot(111)
ax2.plot(spec_normalized.spectral_axis, spec_normalized.flux)
ax2.set_title('Continuum normalized spectrum')
ax2.grid('on')

if interactive_mode == False:
    ax2.axvspan(regions[0], regions[1], alpha=0.2, color='red')
    ax2.axvspan(regions[2], regions[3], alpha=0.2, color='red')
    
f2.savefig('./NORM/normspec-'+FILENAME+'.pdf')

pfil = open('./NORM/'+FILENAME, 'w')
pfil.write('# Wavelenght Flux \n')
  
n=0   
for i in spec_normalized.flux.value:
    flu  = float(i)
    wave = float(spec_normalized.spectral_axis[n].value)
    pfil.write('%7.4f %10.5f\n'%(wave,flu))     
    n=n+1



print(' ------ norm.py code ran uneventifuly ------ \n')


#--------------------------------