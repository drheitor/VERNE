#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 25 12:12:23 2020

@author: Heitor
"""

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

def rv1(centre,centre2):
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
        dwl=(rv/299792.458)*wlp
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


def EWmet(EWb,EWc):
    #nonlinear to metal-poor
    V_VHB = -2.0
    
    EWbc= (EWb+EWc)
    EWbc= float(EWbc/(1. * u.AA))
    
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
    
    
    FeH = a + b * V_VHB + c * EWbc + d * EWp + e * EWbc * V_VHB 
    
    return FeH

def EW(lamb,flux,name):

    flux = flux * u.Unit('J cm-2 s-1 AA-1') 
    #flux = flux * u.Unit('erg cm-2 s-1 AA-1') 
    lamb= lamb * u.AA 
    spec = Spectrum1D(spectral_axis=lamb, flux=flux) 
    #cont_norm_spec = spec / fit_generic_continuum(spec)(spec.spectral_axis) 
    cont_norm_spec = spec 
    print('-----------'+name+'------------')
#line A
    #EWa = equivalent_width(cont_norm_spec, regions=SpectralRegion(8493*u.AA, 8502*u.AA))
    #FWHMa = fwhm(cont_norm_spec, regions=SpectralRegion(8493*u.AA, 8502*u.AA))
    #print('EW A line: '+str(EWa))
#line B
    EWb = equivalent_width(cont_norm_spec, regions=SpectralRegion(lineBlims[0]*u.AA, lineBlims[1]*u.AA))
    print('EW B line: '+str(EWb))
#line C
    EWc = equivalent_width(cont_norm_spec, regions=SpectralRegion(lineClims[0]*u.AA, lineClims[1]*u.AA))
    print('EW C line: '+str(EWc))
#open log file 
    
    #nonlinear to metal-poor
    V_VHB = -2.0
    
    EWbc= (EWb+EWc)
    EWbc= float(EWbc/(1. * u.AA))
    
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
    
    #float all
    
    FeH = a + b * V_VHB + c * EWbc + d * EWp + e * EWbc * V_VHB 
     
    
    print('[Fe/H]: '+str(FeH))
    
    return [EWb,EWc,FeH]


def EW2(lamb,flux,name):

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
    EWb =np.sum(((1-che_A[mask_B]-y[mask_B])/1-che_A[mask_B]) * dw)
    print('EW B line: '+str(EWb))
#line C
    mask_C = (lamb > xn[4]) & (lamb < xn[5])
    EWc =np.sum(((1-che_A[mask_C]-y[mask_C])/1-che_A[mask_C]) * dw)
    print('EW C line: '+str(EWc))
#open log file 
    
    #nonlinear to metal-poor
    V_VHB = -2.0
    
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
    
    #float all
    
    FeH = a + b * V_VHB + c * EWbc + d * EWp + e * EWbc * V_VHB 
     
    
    print('[Fe/H]: '+str(FeH))
    
    return [EWb,EWc,FeH]






#__author__ = 'Jens-Kristian Krogager'
# ==== VOIGT PROFILE ===============
def H(a, x):
    """Voigt Profile Approximation from T. Tepper-Garcia 2006, 2007."""
    P = x**2
    H0 = np.exp(-x**2)
    Q = 1.5/x**2
    return H0 - a/np.sqrt(np.pi)/P * (H0*H0*(4.*P*P + 7.*P + 4. + Q) - Q - 1)


def Voigt(wl, l0, f, N, b, gam, z=0):
    """
    Calculate the optical depth Voigt profile.

    Parameters
    ----------
    wl : array_like, shape (N)
        Wavelength grid in Angstroms at which to evaluate the optical depth.

    l0 : float
        Rest frame transition wavelength in Angstroms.

    f : float
        Oscillator strength.

    N : float
        Column density in units of cm^-2.

    b : float
        Velocity width of the Voigt profile in cm/s.

    gam : float
        Radiation damping constant, or Einstein constant (A_ul)

    z : float
        The redshift of the observed wavelength grid `l`.

    Returns
    -------
    tau : array_like, shape (N)
        Optical depth array evaluated at the input grid wavelengths `l`.
    """
    # ==== PARAMETERS ==================

    c = 2.99792e10        # cm/s
    m_e = 9.1094e-28       # g
    e = 4.8032e-10        # cgs units

    # ==================================
    # Calculate Profile

    C_a = np.sqrt(np.pi)*e**2*f*l0*1.e-8/m_e/c/b
    a = l0*1.e-8*gam/(4.*np.pi*b)

    dl_D = b/c*l0
    wl = wl/(z+1.)
    x = (wl - l0)/dl_D + 0.00001

    tau = np.float64(C_a) * N * H(a, x)
    return tau   
    
#plt.plot(wl,-np.exp(-tau))
    

def smoothstep(x, x_min=0, x_max=1, N=1):
    x = np.clip((x - x_min) / (x_max - x_min), 0, 1)

    result = 0
    for n in range(0, N + 1):
         result += comb(N + n, n) * comb(2 * N + 1, N - n) * (-x) ** n

    result *= x ** (N + 1)

    return result

#plt.plot(x+20, smoothstep(x, N=5))
#plt.plot(x, -smoothstep(x, N=5)+1)   

def tie_RVa(model):
    x_0 = model.x_0_0 + DlambTa
    return x_0



def tie_RV(model):
    x_0 = model.x_0_0 + DlambT
    return x_0



def CaTbflux_min(x,y):
    K = argrelextrema(y, np.less)
    indexlist=[]
    for index in K[0]:
  
        if x[index] > (8400.0) and x[index] < (8775.0):
            indexlist.append(index)
             
    return(min(y[indexlist]))

 
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

try:    
    os.system("mkdir VERNEresults")
except:
    print('...')


#FILES
#PFANT 'fluxSUN-CaT.norm.nulbad.0.200'

#INPUT='rv-Ter9_0072cbnorm.txt'
#INPUT='fluxSUN-V.norm.nulbad.0.200'
#INPUT='specI_19_spec11nPSF_norm_180norm.txt'

#INPUT='specI_22_spec11nPSF_norm_180norm.txt'
    
#spec11
#INPUT='s4500g1.00m1.0z-3.00t2.0a0.40_3000_9000.spec-norm.txt'
#INPUT='specRESAMPLEDI_19_spec11nPSF_norm_1800-6_figs.txt'
#spec7
#INPUT='s4500g1.00m1.0z-1.00t2.0a0.40_3000_9000.spec-norm.txt'
INPUT='specRESAMPLEDI_19_spec7nPSF_norm_1800-6_figs.txt'

INPUT='rv-Ter9-0072norm.txt'

#INPUT='specI_19_spec7nPSF_norm_180norm.txt'

#INPUT= sys.argv[1]



print('Input file:'+str(INPUT))


#dataset.add_component('FeII', 1.794060, 20, 15.0, var_z=1)
#dataset.add_component('FeII', 1.794282, 20, 14.3, var_z=1)

#-------------------------------------------
#normalized spectrum
x,y= np.genfromtxt(INPUT, dtype=float, skip_header=1, unpack=True)

#lamb1=min(x)
#lamb2=max(x)

#-------------------------------------------
#=======================================================================

#------------ MODEL FOR 3 LINES ------------

interactive_mode = False

#trim lambdas
lamb1=8405
lamb2=8700


#NIST
LINES={'CaIIa':8498.018,'CaIIb':8542.089,'CaIIc':8662.140}


#Rutledge et al. 1997a Excluded regions based on the continuum bandpasses
# RV DEPENDANT
#         start   end    start    end
regions_ex=[8489.0, 8563.0, 8642.0, 8697.0 ]


#Rutledge et al. 1997a  Bandpasses ( SIMETRIC )
#             start   end    start    end     start    end
regions_CaT=[8490.0, 8506.0, 8532.0, 8552.0, 8653.0, 8671.0]

# add RV
rv= 0.0
x,y = corrv(x,y,rv)
x=np.array(x)


#automatically find the min flux of CaT-b
_flux_threshold_ = CaTbflux_min(x,y) + 0.05



# Vasquez 2018
Del1=7
Del2=8
Del3=9

#=====================================================================
#-------------------------------------------
# Delta lamb of the 3 lines

DlambTa=LINES[list(LINES.keys())[0]] - LINES[list(LINES.keys())[1]]
DlambT=LINES[list(LINES.keys())[2]] - LINES[list(LINES.keys())[1]]


#trim spectrum
mask = (x > lamb1) & (x < lamb2)

x=x[mask]
y=y[mask]


#-------------------------------------------
#INTERACTIVE MODE

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
      
#--------------------------------      
#Creating spectrums to fit 


y=-y+1.0
y = y * u.Unit('J cm-2 s-1 AA-1') 
# reading spec
spec = Spectrum1D(spectral_axis=x*u.AA, flux=y)


#--------------------------------
#LINES LIMITS
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
    
    #max in min \AA
    #Del1=  LINES[list(LINES.keys())[0]] -  regions_CaT[0]
    #Del2=  LINES[list(LINES.keys())[1]] -  regions_CaT[2]
    #Del3=  LINES[list(LINES.keys())[2]] -  regions_CaT[4]
    
    print(str(len(lines))+' Lines \n')

    #Tied cores based in the readial velocity 
    #xn=[float(lineAcentre/(1. * u.AA))-Del, float(lineAcentre/(1. * u.AA))+Del, 
    #    float(lineAcentre/(1. * u.AA))-Del+DlambT, float(lineAcentre/(1. * u.AA))+Del+DlambT]

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


#-------------------------------------------
# Continuum FIT

#Rutledge et al. 1997a Excluded regions based on the continuum bandpasses
#         start   end    start    end
#regions=[8489.0, 8563.0, 8642.0, 8697.0 ]


#MASK lines continum and SNR 


#define mask SNR 
mask_CATb = (x > regions_ex[0]) & (x < regions_ex[1])
mask_CATc = (x > regions_ex[2]) & (x < regions_ex[3])

mask_SNR = [any(tup) for tup in zip(mask_CATb, mask_CATc)]
Imask_SNR = np.invert(mask_SNR)


#corr offset
Corr_rv= rv1(LINES[list(LINES.keys())[1]],float(lineBcentre/(1. * u.AA)))


Che_model=fit_generic_continuum(spec, exclude_regions=[SpectralRegion(corr_mask(regions_ex[0],Corr_rv)*u.AA, corr_mask(regions_ex[1],Corr_rv)*u.AA), 
                                                          SpectralRegion(corr_mask(regions_ex[2],Corr_rv)*u.AA, corr_mask(regions_ex[3],Corr_rv)*u.AA),
                                                          SpectralRegion(corr_mask(8409,Corr_rv)*u.AA, corr_mask(8415,Corr_rv)*u.AA), 
                                                          SpectralRegion(corr_mask(8415,Corr_rv)*u.AA, corr_mask(8422,Corr_rv)*u.AA),
                                                          SpectralRegion(corr_mask(8422,Corr_rv)*u.AA, corr_mask(8428,Corr_rv)*u.AA),
                                                          SpectralRegion(corr_mask(8431,Corr_rv)*u.AA, corr_mask(8442,Corr_rv)*u.AA),
                                                          SpectralRegion(corr_mask(8465,Corr_rv)*u.AA, corr_mask(8471,Corr_rv)*u.AA),
                                                          SpectralRegion(corr_mask(8579,Corr_rv)*u.AA, corr_mask(8585,Corr_rv)*u.AA),
                                                          SpectralRegion(corr_mask(8595,Corr_rv)*u.AA, corr_mask(8600,Corr_rv)*u.AA),
                                                          SpectralRegion(corr_mask(8610,Corr_rv)*u.AA, corr_mask(8623,Corr_rv)*u.AA),
                                                          ])


#Che_model=fit_generic_continuum(spec, exclude_regions=[SpectralRegion(regions_ex[0]*u.AA, regions_ex[1]*u.AA), 
#                                                          SpectralRegion(regions_ex[2]*u.AA, regions_ex[3]*u.AA)])


C0=float(Che_model.c0.value)
C1=float(Che_model.c1.value)
C2=float(Che_model.c2.value)
C3=float(Che_model.c3.value)

che_A=np.array(Che_model(x*u.AA))


#--------------------------------
#FIT

x=np.array(x)
y=np.array(y)

# Now to fit the data create a new superposition with initial
# guesses for the parameters:

#tied_parameters = {'c0': C0 , 'c1': C1 , 'c2': C2 , 'c3': C3 }
#tied_parameters = {'c0_2': tie_che }


tied_RVa = {'x_0': tie_RVa }
tied_RV = {'x_0': tie_RV }


#need good guess

#gg_init =( Voigt1D(x_0=lineB[0], amplitude_L=lineB[1], fwhm_L=lineB[2], fwhm_G=lineB[2] ) + 
#           Voigt1D(x_0=0, amplitude_L=lineC[1], fwhm_L=lineC[2], fwhm_G=lineC[2], tied=tied_RV) + 
#           Chebyshev1D(3, c0=C0, c1=C1, c2=C2, c3=C3, fixed={'c0': True, 'c1': True, 'c2': True, 'c3': True})   )




#  3 LINES MODEL CAT
gg_init =( Voigt1D(x_0=lineB[0], amplitude_L=lineB[1], fwhm_L=lineB[2], fwhm_G=lineB[2] ,bounds={"amplitude_L": (-0.1, 50)} ) + 
           Voigt1D(x_0=0, amplitude_L=lineC[1], fwhm_L=lineC[2], fwhm_G=lineC[2], tied=tied_RV, bounds={"amplitude_L": (-0.1, 50)}) + 
           Chebyshev1D(3, c0=C0, c1=C1, c2=C2, c3=C3, fixed={'c0': True, 'c1': True, 'c2': True, 'c3': True}) +
           Voigt1D(x_0=lineA[0], amplitude_L=lineA[1], fwhm_L=lineA[2], fwhm_G=lineA[2], tied=tied_RVa, bounds={"amplitude_L": (-0.1, 50)} )   
           )



#Levenberg-Marquardt algorithm
#SLSQPLSQFitter() does not work for strong lines  sequential quadratic programming
#fitter = fitting.SLSQPLSQFitter() 
fitter = fitting.LevMarLSQFitter()
gg_fit = fitter(gg_init, x ,y)


#gg_fit.mean.tied = tie_center

print(gg_fit)


#VMaster = convolve_models(gg_fit,  Che_fit)
#VMaster= gg_fit + Che_fit
#-------------------------------------------

#EW
y= -np.array(y)+1.0
V_fit= -np.array(gg_fit(x))+1.0
#V_fit= -np.array(gg_fit(x))+1.0 - ( 1 - Che_fit(x))
#V_fit= -np.array(gg_fit(x))+1.0 * A
#V_fit= np.array(VMaster(x))


flux_fit = V_fit
flux_fit = flux_fit * u.Unit('J cm-2 s-1 AA-1') 
spec_FIT = Spectrum1D(spectral_axis=x*u.AA, flux=flux_fit)


EWs=EW(x,flux_fit,'FIT')

print('===========')

EWs=EW2(x,flux_fit,'FIT-2')

print('===========')

EWsb=EW(x,y,'FIT-spec')




#-------------------------------------------
#RADIAL VELOCITY

DlambOBS= gg_fit[0].x_0.value - gg_fit[1].x_0.value


#gg_fit[0].param_names
#a = gg_fit[0].x_0.value


#LINES A
#RV=rv1(linemodA[0],lineA[0])
#print('RVA: '+str(RV))
RV0=rv1(LINES[list(LINES.keys())[0]],gg_fit[3].x_0.value)
print('\nRVa: '+str(RV0))

#LINES B
#RV=rv1(linemodA[0],lineA[0])
#print('RVA: '+str(RV))
RV=rv1(LINES[list(LINES.keys())[1]],gg_fit[0].x_0.value)
print('RVb: '+str(RV))

#LINES C
#RV2=rv1(linemodB[0],lineB[0])
#print('RVB: '+str(RV2))
RV2=rv1(LINES[list(LINES.keys())[2]],gg_fit[1].x_0.value)
print('RVc: '+str(RV2))

print('\n')

print('Mean RV: '+ str(np.mean([RV,RV2,RV0])) +'\n')

#-------------------------------------------

#LOG

print('Wrinting Log file... \n')

LOG = open('./VERNEresults/LOG-'+INPUT, 'w')
LOG.write('Log file of '+ INPUT+' \n \n')
LOG.write('Input Spectrum:   '+ INPUT +' \n \n')
LOG.write('RV A line:             '+ str(RV) +' \n') 
LOG.write('RV B line:             '+ str(RV2) +' \n') 
LOG.write('RV mean:               '+ str(np.mean([RV,RV2,RV0])) +' \n') 
LOG.write('EW A line:             '+ str(EWs[0]) +' \n') 
LOG.write('EW B line:             '+ str(EWs[1]) +' \n') 
LOG.write('[Fe/H]_CaT:            '+ str(EWs[2]) +' \n') 
LOG.write('\n') 
LOG.write(str(gg_fit)) 
LOG.write('\n') 
LOG.write(str(gg_fit[0]))
LOG.write('\n') 
LOG.write(str(gg_fit[1]))  


#-------------------------------------------

#SPEC and MODEL

print('Wrinting Model file... \n')

mod = open('./VERNEresults/MODEL-'+INPUT, 'w')
mod.write('Lambda Flux_spec Flux_model \n')
n=0   
while n < len(x):
    xsp  = x[n]
    fluxsp = y[n]
    fluxmod = V_fit[n]
    mod.write('%7.3f %7.4f %7.4f\n'%(xsp,fluxsp,fluxmod))
    n=n+1  



#-------------------------------------------

# Plot the data with the best-fit model
f1 = plt.figure(figsize=(10,5))
ax = f1.add_axes((.1,.3,.8,.6))

ax.plot(x, y, 'k-')
ax.plot(x, V_fit, color= 'firebrick', label='Voigt')
#ax.plot(x, -gg_fitL(x), color= 'green', label='Lorentz')



ax.axvspan(lineAlims[0], lineAlims[1], alpha=0.4, color='gray')
ax.axvspan(lineBlims[0], lineBlims[1], alpha=0.4, color='gray')
ax.axvspan(lineClims[0], lineClims[1], alpha=0.4, color='gray')

_flux_threshold_array = np.zeros( (len(x),) ) + _flux_threshold_ 
ax.plot(x, _flux_threshold_array, 'k:', alpha=0.1)

plt.legend(loc='lower right', ncol=1)
ax.set_xlabel('Wavelength ($\AA$)')
ax.set_ylabel('Flux')




#test mask new 
ax.axvspan( corr_mask(8409,Corr_rv), corr_mask(8415,Corr_rv), alpha=0.4, color='red')
ax.axvspan( corr_mask(8415,Corr_rv), corr_mask(8422,Corr_rv), alpha=0.4, color='red')
ax.axvspan( corr_mask(8422,Corr_rv), corr_mask(8428,Corr_rv), alpha=0.4, color='red')
ax.axvspan( corr_mask(8431,Corr_rv), corr_mask(8442,Corr_rv), alpha=0.4, color='red')
ax.axvspan( corr_mask(8465,Corr_rv), corr_mask(8471,Corr_rv), alpha=0.4, color='red')
ax.axvspan( corr_mask(8579,Corr_rv), corr_mask(8585,Corr_rv), alpha=0.4, color='red')
ax.axvspan( corr_mask(8595,Corr_rv), corr_mask(8600,Corr_rv), alpha=0.4, color='red')
ax.axvspan( corr_mask(8610,Corr_rv), corr_mask(8623,Corr_rv), alpha=0.4, color='red')


#residual plot
difference = y - V_fit 
ax2=f1.add_axes((.1,.1,.8,.2), sharex = ax)        
ax2.plot(x,difference,'.k', )
ax2.plot(x,np.zeros((len(x),)), color='firebrick', linewidth=3.0)
ax2.set_xlabel('Wavelength ($\AA$)')
plt.grid()




#excluded regions 
#ax.axvspan(regions_ex[0], regions_ex[0], alpha=0.05, color='red')
#ax.axvspan(regions_ex[2], regions_ex[3], alpha=0.05, color='red')
#continuum regions 
ax.axvspan(lamb1        , regions_ex[0], alpha=0.1, color='gray')
ax.axvspan(regions_ex[1], regions_ex[2], alpha=0.1, color='gray')

ax.text(min(x), min(y)+0.14, 'RV: '    +str("{0:.2f}".format(np.mean([RV,RV2,RV0]))) , fontsize=20, color='black') 
ax.text(min(x), min(y)+0.07, '[Fe/H]~ '+str("{0:.2f}".format(EWs[2])) , fontsize=20, color='black') 
if INPUT[9:15] == 'spec7':
    met=-1.00
    ax.text(min(x), min(y), '$\sigma$ ~ '+str("{0:.2f}".format(EWs[2]-met)) , fontsize=20, color='black') 
if INPUT[9:15] == 'spec8':
    met=-1.50
    ax.text(min(x), min(y), '$\sigma$ ~ '+str("{0:.2f}".format(EWs[2]-met)) , fontsize=20, color='black') 
if INPUT[9:15] == 'spec9':
    met=-2.00
    ax.text(min(x), min(y), '$\sigma$ ~ '+str("{0:.2f}".format(EWs[2]-met)) , fontsize=20, color='black') 
if INPUT[9:15] == 'spec10':
    met=-2.50
    ax.text(min(x), min(y), '$\sigma$ ~ '+str("{0:.2f}".format(EWs[2]-met)) , fontsize=20, color='black') 
if INPUT[9:15] == 'spec11':
    met=-3.00
    ax.text(min(x), min(y), '$\sigma$ ~ '+str("{0:.2f}".format(EWs[2]-met)) , fontsize=20, color='black') 
if INPUT[9:15] == 'spec12':
    met=-4.00
    ax.text(min(x), min(y), '$\sigma$ ~ '+str("{0:.2f}".format(EWs[2]-met)) , fontsize=20, color='black') 






print('SNR: '+ str(int(DER_SNR(y[Imask_SNR])))+'\n')


# INTERATIVE MODE 
#plt.show()
    
plt.tight_layout()
plt.savefig('./VERNEresults/'+INPUT+'_FIT.pdf')




print('**********DONE**********')









































#-------------------------------------------
