#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 12 14:15:21 2020

@author: Heitor
"""

#-------------------------------------------
import numpy as np
import matplotlib.pyplot as plt
from astropy.modeling import models, fitting
from astropy.modeling.models import Voigt1D
from astropy.modeling.models import Chebyshev1D
import pyfiglet
import sys
import os
from astropy import units as u
from specutils import Spectrum1D, SpectralRegion
from specutils.analysis import centroid
from specutils.analysis import gaussian_sigma_width, gaussian_fwhm, fwhm, fwzi
from specutils.analysis import line_flux
from specutils.analysis import equivalent_width
from specutils.fitting import fit_generic_continuum
from astropy.convolution import convolve_models
from specutils.fitting import find_lines_derivative
from scipy.special import comb
from scipy.signal import argrelextrema
from lmfit import Model
import seaborn as sns

from lmfit.models import ExponentialModel, GaussianModel, VoigtModel, PolynomialModel, PseudoVoigtModel

#-------------------------------------------
# =====================================================================================

def DER_SNR(flux):
   
# =====================================================================================
   """
   DESCRIPTION This function computes the signal to noise ratio DER_SNR following the
               definition set forth by the Spectral Container Working Group of ST-ECF,
	       MAST and CADC. 

               signal = median(flux)      
               noise  = 1.482602 / sqrt(6) median(abs(2 flux_i - flux_i-2 - flux_i+2))
	       snr    = signal / noise
               values with padded zeros are skipped

   USAGE       snr = DER_SNR(flux)
   PARAMETERS  none
   INPUT       flux (the computation is unit independent)
   OUTPUT      the estimated signal-to-noise ratio [dimensionless]
   USES        numpy      
   NOTES       The DER_SNR algorithm is an unbiased estimator describing the spectrum 
	       as a whole as long as
               * the noise is uncorrelated in wavelength bins spaced two pixels apart
               * the noise is Normal distributed
               * for large wavelength regions, the signal over the scale of 5 or
	         more pixels can be approximated by a straight line
 
               For most spectra, these conditions are met.

   REFERENCES  * ST-ECF Newsletter, Issue #42:
               www.spacetelescope.org/about/further_information/newsletters/html/newsletter_42.html
               * Software:
	       www.stecf.org/software/ASTROsoft/DER_SNR/
   AUTHOR      Felix Stoehr, ST-ECF
               24.05.2007, fst, initial import
               01.01.2007, fst, added more help text
               28.04.2010, fst, return value is a float now instead of a numpy.float64
   """
   from numpy import array, where, median, abs 

   flux = array(flux)

   # Values that are exactly zero (padded) are skipped
   flux = array(flux[where(flux != 0.0)])
   n    = len(flux)      

   # For spectra shorter than this, no value can be returned
   if (n>4):
      signal = median(flux)

      noise  = 0.6052697 * median(abs(2.0 * flux[2:n-2] - flux[0:n-4] - flux[4:n]))

      return float(signal / noise)  

   else:

      return 0.0

# end DER_SNR -------------------------------------------------------------------------

def rv(centre,centre2):
    dwl=centre-centre2
    wl=centre2
    rv= (dwl*299792.458)/wl
    return rv 


def corr_mask(wl,rv):
    dwl=(rv/299792.458)*wl
    wlcorr=wl-dwl
    return wlcorr

def corrv(wl,flux,rv):
    wlnewl=[]
    for wlp in wl:
        dwl=-(rv/299792.458)*wlp
        wlnew=wlp-dwl
        wlnewl.append(wlnew)
    return [wlnewl,flux]


def line(spec,wave1,wave2):
    # finding the centorid and deriving guesses parameters
    centre=centroid(spec, SpectralRegion(wave1*u.AA, wave2*u.AA))  
    centre=float(centre/(1. * u.AA))
    FWHM=fwhm(spec) 
    FWHM=float(FWHM/(1. * u.AA))
    A=line_flux(spec, SpectralRegion(lamb1*u.AA, lamb2*u.AA))
    a=1* u.Unit('J cm-2 s-1 AA-1') 
    A=float(A/(1. * u.AA*a))
    # PARAMETERS
    return [centre,A,FWHM]



def EW_VHB(lamb,flux,name):

# defining the pace 
    dw=lamb[7]-lamb[6]
    print('-----------'+name+'------------')
#line A
    #mask_A = (lamb > xn[0]) & (lamb < xn[1])
    #EWa =np.sum(((1-che_A[mask_A]-y[mask_A])/1-che_A[mask_A]) * dw)
    #print('EW A line: '+str(EWa))
#line B
    mask_B = (lamb > xn[2]) & (lamb < xn[3])
    #EWb_test=np.sum(((che_A[mask_CATb_t]-y[mask_CATb_t])/che_A[mask_CATb_t]) * dw)
    #usar um for para o dw 
    EWb =np.sum(((1-che_A[mask_B]-flux[mask_B])/1-che_A[mask_B]) * dw)
    print('EW B line: '+str(EWb))
#line C
    mask_C = (lamb > xn[4]) & (lamb < xn[5])
    EWc =np.sum(((1-che_A[mask_C]-flux[mask_C])/1-che_A[mask_C]) * dw)
    print('EW C line: '+str(EWc))
#open log file 
    
    #nonlinear to metal-poor
    
    EWbc= (EWb+EWc)
    EWbc= float(EWbc)
    
    EWp = (EWbc)**(-1.5) 
    
    #nonlinear to metal-poor
    #Wl = float(EWb / (1. * u.AA)) + float(EWc / (1. * u.AA)) + (0.64 * V_VHB)
    #FeH= -2.81 + 0.44*Wl
    # FeH constants to V-VHB
    
    a=-2.87
    b=0.195
    c=0.458
    d=-0.913
    e=0.0155
    
    a2=-2.62
    b2=0.195
    c2=0.457
    d2=-0.908
    e2=0.0146
    
    #float all
    
    FeH = a + b * V_VHB + c * EWbc + d * EWp + e * EWbc * V_VHB 
    CaH = a2 + b2 * V_VHB  + c2 * EWbc + d2 * EWp + e2 * EWbc * V_VHB  

    
    print('[Fe/H]: '+str(FeH))
    print('[Ca/H]: '+str(CaH))

    
    return [EWb,EWc,FeH,CaH]


def EW_MV(lamb,flux,name):

# defining the pace 
    dw=lamb[7]-lamb[6]
    print('-----------'+name+'------------')
#line A
    #mask_A = (lamb > xn[0]) & (lamb < xn[1])
    #EWa =np.sum(((1-che_A[mask_A]-y[mask_A])/1-che_A[mask_A]) * dw)
    #print('EW A line: '+str(EWa))
#line B
    mask_B = (lamb > xn[2]) & (lamb < xn[3])
    #EWb_test=np.sum(((che_A[mask_CATb_t]-y[mask_CATb_t])/che_A[mask_CATb_t]) * dw)
    #usar um for para o dw 
    EWb =np.sum(((1-che_A[mask_B]-flux[mask_B])/1-che_A[mask_B]) * dw)
    print('EW B line: '+str(EWb))
#line C
    mask_C = (lamb > xn[4]) & (lamb < xn[5])
    EWc =np.sum(((1-che_A[mask_C]-flux[mask_C])/1-che_A[mask_C]) * dw)
    print('EW C line: '+str(EWc))
#open log file 
    
    #nonlinear to metal-poor
    
    EWbc= (EWb+EWc)
    EWbc= float(EWbc)
    
    EWp = (EWbc)**(-1.5) 
    
    #nonlinear to metal-poor
    #Wl = float(EWb / (1. * u.AA)) + float(EWc / (1. * u.AA)) + (0.64 * V_VHB)
    #FeH= -2.81 + 0.44*Wl
    # FeH constants to MV
    
    a=-2.90
    b=0.187
    c=0.422
    d=-0.882
    e=0.0133
    
    a2=-2.65
    b2=0.185
    c2=0.422
    d2=-0.876
    e2=0.0137
    
    #float all
    
    FeH = a + b * MV + c * EWbc + d * EWp + e * EWbc * MV 
    CaH = a2 + b2 * MV + c2 * EWbc + d2 * EWp + e2 * EWbc * MV 

    
    print('[Fe/H]: '+str(FeH))
    print('[Ca/H]: '+str(CaH))
    
    return [EWb,EWc,FeH,CaH]

def EW_MI(lamb,flux,name):
    
    # TESTE 
    Del1=4.5
    Del2=5
    Del3=5
    xn=[float(lineAcentre/(1. * u.AA))-Del1, float(lineAcentre/(1. * u.AA))+Del1, 
        float(lineBcentre/(1. * u.AA))-Del2, float(lineBcentre/(1. * u.AA))+Del2,
        float(lineCcentre/(1. * u.AA))-Del3, float(lineCcentre/(1. * u.AA))+Del3]

# defining the pace 
    dw=lamb[7]-lamb[6]
    print('-----------'+name+'------------')
#line A
    #mask_A = (lamb > xn[0]) & (lamb < xn[1])
    #EWa =np.sum(((1-che_A[mask_A]-y[mask_A])/1-che_A[mask_A]) * dw)
    #print('EW A line: '+str(EWa))
#line B
    mask_B = (lamb > xn[2]) & (lamb < xn[3])
    #EWb_test=np.sum(((che_A[mask_CATb_t]-y[mask_CATb_t])/che_A[mask_CATb_t]) * dw)
    #usar um for para o dw 
    EWb =np.sum(((1-che_A[mask_B]-flux[mask_B])/1-che_A[mask_B]) * dw)
    print('EW B line: '+str(EWb))
#line C
    mask_C = (lamb > xn[4]) & (lamb < xn[5])
    EWc =np.sum(((1-che_A[mask_C]-flux[mask_C])/1-che_A[mask_C]) * dw)
    print('EW C line: '+str(EWc))
#open log file 
    
    #nonlinear to metal-poor

    
    EWbc= (EWb+EWc)
    EWbc= float(EWbc)
    
    EWp = (EWbc)**(-1.5) 
    
    #nonlinear to metal-poor
    #Wl = float(EWb / (1. * u.AA)) + float(EWc / (1. * u.AA)) + (0.64 * V_VHB)
    #FeH= -2.81 + 0.44*Wl
    # FeH constants to MI
    
    a=-2.78
    b=0.193
    c=0.442
    d=-0.834
    e=0.0017
    
    a2=-2.53
    b2=0.193
    c2=0.439
    d2=-0.825
    e2=0.0013
    
    #float all
    
    FeH = a + b * MI + c * EWbc + d * EWp + e * EWbc * MI 
    CaH = a2 + b2 * MI + c2 * EWbc + d2 * EWp + e2 * EWbc * MI 

    
    print('[Fe/H]: '+str(FeH))
    print('[Ca/H]: '+str(CaH))
    
    return [EWb,EWc,FeH,CaH]
 


def Mmag(D,m):

    M= m + 5 -5*(np.log10(D))
    return(M)



#------------------------------

def CaT_voigt(x, che_1, che_2, amplitude_A, amplitude_B, amplitude_C, fraction, sigma, z):   

    continuum = np.polynomial.chebyshev.chebval(x=x,c=(che_1,che_2))

    CaTA= 8498.018
    CaTB= 8542.089
    CaTC= 8662.140
    sigma=2.41
    #c = 299792.458 # speed of light in km/s
    #v=280
    #z=v/c   
    #x_fit=x*(1+z)
    x_fit=x
     
    sigma_g = sigma/(np.sqrt(2*np.log(2)))
    
    F_A = ((1-fraction)*amplitude_A)/(sigma_g*np.sqrt(2*np.pi))* np.exp((-(x_fit-((1+z)*CaTA))**2) /(2*sigma_g**2)) + ((fraction*amplitude_A)/(np.pi))*((sigma)/((x_fit-((1+z)*CaTA))**2 + sigma**2)) 
    F_B = ((1-fraction)*amplitude_B)/(sigma_g*np.sqrt(2*np.pi))* np.exp((-(x_fit-((1+z)*CaTB))**2) /(2*sigma_g**2)) + ((fraction*amplitude_B)/(np.pi))*((sigma)/((x_fit-((1+z)*CaTB))**2 + sigma**2)) 
    F_C = ((1-fraction)*amplitude_C)/(sigma_g*np.sqrt(2*np.pi))* np.exp((-(x_fit-((1+z)*CaTC))**2) /(2*sigma_g**2)) + ((fraction*amplitude_C)/(np.pi))*((sigma)/((x_fit-((1+z)*CaTC))**2 + sigma**2)) 
   
    
    F = np.array(F_A) + np.array(F_B) + np.array(F_C) + np.array(continuum)


    return F


#------------------------------



#------------------------------------------
#========================================================================
#------------------------------------------------------------------------
#========================================================================

#file names
print('\n')
print("*******************************************************\n")


ascii_banner = pyfiglet.figlet_format("VERNE")
print(ascii_banner)


print("*******************************************************\n")
print('\n')
print('\n')
print('READING ALL FILES ... \n')


#-------------------------------------------

sns.set_style("white")
sns.set_context("paper", font_scale=2.0, rc={"lines.linewidth": 2.5})

try:    
    os.system("mkdir VERNEresults")
except:
    print('...')


#INPUT FILES

#INPUT= sys.argv[1]

INPUT='RGB_stacked.txt'


print('Input file: '+str(INPUT))

#-------------------------------------------
#Reading spectrum 

#normalized spectrum without noise column
#x,y= np.genfromtxt(INPUT, dtype=float, skip_header=1, unpack=True)
#normalized spectrum with noise 
x,y,sigma= np.genfromtxt(INPUT, dtype=float, skip_header=1, unpack=True)

#-------------------------------------------
#=======================================================================

#------------ MODEL FOR 3 LINES ------------

interactive_mode = False

#To guess a Radial Velocity 
#interactive_mode = 'Guess'
rv_init= 320

#trim lambda
lamb1=8470
lamb2=8800


#name files
NAME = INPUT[0:-4]


#NIST
LINES={'CaIIa':8498.018,'CaIIb':8542.089,'CaIIc':8662.140}


#VHB mode using Starkenburg  et al. 2010 calibrations

#VHB = 20.35
#V = 17
#V_VHB = V - VHB

#Magnitude 
# in parsec/ works for V and I
D=1.62e6


#RGB
m=21.70

#absolute mag conversion
MI = Mmag(D,m)
#MV = Mmag(D,m)


#-------------
#Selecting the passband

#Rutledge et al. 1997a Excluded regions based on the continuum bandpasses
# RV DEPENDANT
#         start   end    start    end
#regions_ex=[8489.0, 8563.0, 8642.0, 8697.0 ]

# Rutledge 1997a
#Del1=7
#Del2=8
#Del3=9

#Carrera et al. 2007 Excluded regions based on the continuum bandpasses
# RV DEPENDANT
#            start   end    start    end
#regions_ex=[8484.0, 8562.0, 8642.0, 8682.0 ]

# Carrera 2007
Del1=4.5
Del2=20
Del3=20


#Vasquez et al. 2018 Excluded regions based on the continuum bandpasses
# RV DEPENDANT
#              start   end    start    end
regions_ex=[ 8491.0, 8505.0, 8461.0, 8568.0, 8650.0, 8700.0 ]

# Vasquez 2018
Del1=7
Del2=8
Del3=9

#-------------

#add or correct RV 
#if necessary 
rvs = 0.0
x,y = corrv(x,y,rvs)
x=np.array(x)


#automatically find the min flux of CaT-b
_flux_threshold_ = 0.6



#=====================================================================
#-------------------------------------------
# Delta lamb of the 3 lines

DlambTa=LINES[list(LINES.keys())[0]] - LINES[list(LINES.keys())[1]]
DlambT=LINES[list(LINES.keys())[2]] - LINES[list(LINES.keys())[1]]


#trim spectrum
mask = (x > lamb1) & (x < lamb2)

x=x[mask]
y=y[mask]
sigma=sigma[mask]


#-------------------------------------------
#INTERACTIVE MODE
#Selecting lines as well the bandpass in the order CaT-A, B, C


if interactive_mode == True:
    f0=plt.figure(figsize=(12,7))
    ax0 = f0.add_subplot(111)

    ax0.plot(x, y, 'k')
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
 
y=-y+1.0    

 
#--------------------------------      
#Creating spectrums to fit the pseudovoigt function

y_guess=y
y = y * u.Unit('J cm-2 s-1 AA-1') 
# reading spec
spec = Spectrum1D(spectral_axis=x*u.AA, flux=y)


#--------------------------------
#LINES LIMITS
#derive the line centre. 
if interactive_mode == False:
    
    lines = find_lines_derivative(spec, flux_threshold= 1 - _flux_threshold_)

    if len(lines)>3:
        print('Check _flux_threshold_' )
    
    try:
        lineAcentre=lines[0][0]
        lineBcentre=lines[1][0]
        lineCcentre=lines[2][0]
    # if the code was only able to find the CaT-b line    
    except:
        lineAcentre=lines[0][0]+DlambTa*u.AA
        lineBcentre=lines[0][0]
        lineCcentre=lines[0][0]+DlambT*u.AA     
    
    
    print(str(len(lines))+' Lines \n')

    #Tied cores based in the readial velocity 
    #xn=[float(lineAcentre/(1. * u.AA))-Del, float(lineAcentre/(1. * u.AA))+Del, 
    #    float(lineAcentre/(1. * u.AA))-Del+DlambT, float(lineAcentre/(1. * u.AA))+Del+DlambT]

    #untied cores
    xn=[float(lineAcentre/(1. * u.AA))-Del1, float(lineAcentre/(1. * u.AA))+Del1, 
        float(lineBcentre/(1. * u.AA))-Del2, float(lineBcentre/(1. * u.AA))+Del2,
        float(lineCcentre/(1. * u.AA))-Del3, float(lineCcentre/(1. * u.AA))+Del3]

#-----------
    
    
#--------------------------------
#LINES LIMITS
#derive the line centre. 
if interactive_mode == 'Guess':
            
    wl_cat,y_cat = corrv([LINES[list(LINES.keys())[0]],LINES[list(LINES.keys())[1]],LINES[list(LINES.keys())[2]]],[1,1,1],rv_init)
    
    #14A because it is a range between +-200km/s
    lim_A=[wl_cat[0]-10,wl_cat[0]+10]
    lim_B=[wl_cat[1]-10,wl_cat[1]+10]
    lim_C=[wl_cat[2]-10,wl_cat[2]+10]
    
    mask_1 = (x > lim_A[0]) & (x < lim_A[1])
    mask_2 = (x > lim_B[0]) & (x < lim_B[1])
    mask_3 = (x > lim_C[0]) & (x < lim_C[1])

    x_1=x[mask_1]
    y_1=y_guess[mask_1]
    
    x_2=x[mask_2]
    y_2=y_guess[mask_2]
    
    x_3=x[mask_3]
    y_3=y_guess[mask_3]
    
      
    lineAcentre=x_1[list(y_1).index(min(y_1))]* u.AA
    lineBcentre=x_2[list(y_2).index(min(y_2))]* u.AA
    lineCcentre=x_3[list(y_3).index(min(y_3))]* u.AA

    #untied cores
    xn=[float(lineAcentre/(1. * u.AA))-Del1, float(lineAcentre/(1. * u.AA))+Del1, 
        float(lineBcentre/(1. * u.AA))-Del2, float(lineBcentre/(1. * u.AA))+Del2,
        float(lineCcentre/(1. * u.AA))-Del3, float(lineCcentre/(1. * u.AA))+Del3]
    
    
    


#-----------

lineAlims=[xn[0], xn[1]]
lineBlims=[xn[2], xn[3]]
lineClims=[xn[4], xn[5]]


#-------------------------------------------
#spec finding LINES PARAMETERS

lineA= line(spec,lineAlims[0], lineAlims[1])

lineB= line(spec,lineBlims[0], lineBlims[1])

lineC= line(spec,lineClims[0], lineClims[1])


#-----------------------------------------------------------------------------
#MODELS AND FIT
#-------------------------------------------
# Continuum FIT

#MASK lines continum and SNR 

#define mask SNR 
mask_CATa = (x > regions_ex[0]) & (x < regions_ex[1])
mask_CATb = (x > regions_ex[2]) & (x < regions_ex[3])
mask_CATc = (x > regions_ex[4]) & (x < regions_ex[5])

mask_SNR = [any(tup) for tup in zip(mask_CATa,mask_CATb, mask_CATc)]
Imask_SNR = np.invert(mask_SNR)


#correct offset excluded regions
#lineBcentre=8546.0 * u.AA
try:
    Corr_rv= rv(LINES[list(LINES.keys())[1]],float(lineBcentre/(1. * u.AA)))
except:
    Corr_rv= rv(LINES[list(LINES.keys())[1]],float(lineBcentre))


Che_model=fit_generic_continuum(spec, model=Chebyshev1D(2), exclude_regions=[SpectralRegion(corr_mask(regions_ex[0],Corr_rv)*u.AA, corr_mask(regions_ex[1],Corr_rv)*u.AA),
                                                          SpectralRegion(corr_mask(regions_ex[2],Corr_rv)*u.AA, corr_mask(regions_ex[3],Corr_rv)*u.AA), 
                                                          SpectralRegion(corr_mask(regions_ex[4],Corr_rv)*u.AA, corr_mask(regions_ex[5],Corr_rv)*u.AA),
                                                          SpectralRegion(corr_mask(8409,Corr_rv)*u.AA, corr_mask(8415,Corr_rv)*u.AA), 
                                                          SpectralRegion(corr_mask(8415,Corr_rv)*u.AA, corr_mask(8422,Corr_rv)*u.AA),
                                                          SpectralRegion(corr_mask(8422,Corr_rv)*u.AA, corr_mask(8428,Corr_rv)*u.AA),
                                                          SpectralRegion(corr_mask(8431,Corr_rv)*u.AA, corr_mask(8442,Corr_rv)*u.AA),
                                                          SpectralRegion(corr_mask(8465,Corr_rv)*u.AA, corr_mask(8471,Corr_rv)*u.AA),
                                                          SpectralRegion(corr_mask(8579,Corr_rv)*u.AA, corr_mask(8585,Corr_rv)*u.AA),
                                                          SpectralRegion(corr_mask(8595,Corr_rv)*u.AA, corr_mask(8600,Corr_rv)*u.AA),
                                                          SpectralRegion(corr_mask(8610,Corr_rv)*u.AA, corr_mask(8630,Corr_rv)*u.AA),
                                                          SpectralRegion(corr_mask(8329,Corr_rv)*u.AA, corr_mask(8354,Corr_rv)*u.AA),
                                                          SpectralRegion(corr_mask(8252,Corr_rv)*u.AA, corr_mask(8256,Corr_rv)*u.AA),
                                                          ])
    


#Chebyshev parameters 
C0=float(Che_model.c0.value)
C1=float(Che_model.c1.value)
C2=float(Che_model.c2.value)
#C3=float(Che_model.c3.value)

che_A=np.array(Che_model(x*u.AA))


#--------------------------------
#FIT

x=np.array(x)
y=np.array(y)


#------
c =299792.458 #km/s
Z= 270/c


SUPER_CAT = Model(CaT_voigt)


#-------MOD---------

mod = SUPER_CAT 

pars = mod.make_params(che_1=0.0, che_2=1.0, amplitude_A=0.0, amplitude_B=0.0, amplitude_C=0.0, fraction=0.0, sigma=2.41, z=Z)

try:
    weights = 1/sigma
    #best fit
    FIT = mod.fit(y, pars, x=x, method= 'leastsq', nan_policy='omit', weights=weights)
except:
    print('invalid noise...')
    print('Fitting withoutnoise weight.')
    #best fit
    FIT = mod.fit(y, pars, x=x, method= 'leastsq', nan_policy='omit')




#-------MOD---------

init = mod.eval( pars, x=x)

# Variable reports
print(FIT.fit_report(min_correl=1.5))

#-------------------------------------------
#EW

y= -np.array(y)+1.0
init= -np.array(init)+1.0
FIT.best_fit= -np.array(FIT.best_fit)+1.0


#EW [Fe/H]

print('===========')

try:
    #EWs=EW_VHB(x,V_fit,'FIT')
    EWs=EW_MI(x,FIT.best_fit,'FIT')
    #EWs=EW_MV(x,V_fit,'FIT')
except:
    print('Check the input mag')


print('===========')


#-------------------------------------------
#RADIAL VELOCITY


RV= FIT.params['z'].value * c
print('RV: '+str(RV))

print('\n')

rvmean=RV

print('Mean RV: '+ str(rvmean) +'\n')


#-----------------------------------------------------------------------------
#Creating files and plots
#-------------------------------------------
#LOG

print('Wrinting Log file... \n')

LOG = open('./VERNEresults/LOG-'+NAME, 'w')
LOG.write('Log file of '+ INPUT+' \n \n')
LOG.write('Input Spectrum:   '+ INPUT +' \n \n')
LOG.write('RV :             '+ str(RV) +' \n') 
LOG.write('EW A line:             '+ str(EWs[0]) +' \n') 
LOG.write('EW B line:             '+ str(EWs[1]) +' \n') 
LOG.write('[Fe/H]_CaT:            '+ str(EWs[2]) +' \n') 
LOG.write('[Ca/H]_CaT:            '+ str(EWs[3]) +' \n') 
LOG.write('S/N:                   '+ str(int(DER_SNR(y[Imask_SNR]))) +' \n') 
LOG.write('\n') 
LOG.write(str(FIT.fit_report(min_correl=1.5))) 
LOG.write('\n') 
#LOG.write(str(FIT_fit[0]))
#LOG.write('\n') 
#LOG.write(str(FIT_fit[1]))  


#-------------------------------------------

#SPEC and MODEL

print('Wrinting Model file... \n')

mod = open('./VERNEresults/MODEL-'+NAME, 'w')
mod.write('Lambda Flux_spec Flux_model \n')
n=0   
while n < len(x):
    xsp  = x[n]
    fluxsp = y[n]
    fluxmod = FIT.best_fit[n]
    mod.write('%7.3f %7.4f %7.4f\n'%(xsp,fluxsp,fluxmod))
    n=n+1  



#-------------------------------------------
#PLOT

f0 = plt.figure(figsize=(12,7))
ax0 = f0.add_axes((.1,.3,.8,.6))

ax0.plot(x, y, 'k')
#ax0.plot(x, init, 'b--', label='initial fit', alpha=0.6)
ax0.plot(x, FIT.best_fit, 'r-', label='best fit')
ax0.plot(x, sigma, 'k-', alpha=0.2)
ax0.set_ylim([min(y)-0.1,max(y)+0.1])

ax0.axvspan(lineAlims[0], lineAlims[1], alpha=0.4, color='gray')
ax0.axvspan(lineBlims[0], lineBlims[1], alpha=0.4, color='gray')
ax0.axvspan(lineClims[0], lineClims[1], alpha=0.4, color='gray')

_flux_threshold_array = np.zeros( (len(x),) ) + _flux_threshold_ 
ax0.plot(x, _flux_threshold_array, 'k:', alpha=0.1)


#ax0.set_xlabel('Wavelength ($\AA$)')
ax0.set_ylabel('Flux')

ax0.set_title(NAME)

#residual plot
difference = y - FIT.best_fit 

ax02=f0.add_axes((.1,.1,.8,.2), sharex = ax0)        
ax02.plot(x,difference,'.k', )
ax02.plot(x,np.zeros((len(x),)), color='r', linewidth=3.0)
ax02.set_xlabel('Wavelength ($\AA$)')
plt.grid()



print('S/N: '+ str(int(DER_SNR(y[Imask_SNR])))+'\n')

ax0.text(8710, min(y)+0.14, 'RV: '    +str("{0:.2f}".format(rvmean)) , fontsize=20, color='black') 
ax0.text(8710, min(y)+0.07, '[Fe/H]~ '+str("{0:.2f}".format(EWs[2])) , fontsize=20, color='black') 
#ax0.text(min(x), min(y)+0.02, 'S/N: '+ str(int(DER_SNR(y[Imask_SNR])))    , fontsize=15, color='black') 


#plt.tight_layout()

# INTERATIVE MODE 
if interactive_mode == True:
    plt.show()
    


plt.savefig('./VERNEresults/'+NAME+'LM_FIT.pdf')








print('**********DONE**********')
