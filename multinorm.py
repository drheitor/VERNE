#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 29 14:46:23 2020

@author: Heitor
"""

import numpy as np 
from astropy.io import fits
import os 
#import time
#import datetime


# read header to use the spectrum name before run each step


#list of directories ( I_mag_spec1...) format
os.system('ls -d *nPSF_norm_1800-6 > list')


#list of spectra format: spec1 name.spec.txt
NAMES = np.genfromtxt('list', dtype=str, unpack=True)



for name in NAMES:
    n=1
            
    command = 'python3 norm.py '+ name
    print(name+ '\n')
    #print(command+ '\n')

    os.system(command)
    print('Spectrum'+ str(n)+ 'Done! \n')
    n=n+1
    


print('*------------DONE------------*')
    
    





















