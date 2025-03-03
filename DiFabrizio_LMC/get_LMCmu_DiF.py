#! /usr/bin/env python

#from __future__ import division
import numpy as np                 
from astropy.io import ascii
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import matplotlib.cm as cm
import matplotlib as mpl

Rg = 3.518
Rr = 2.617
Ri = 1.971

mu_true = 18.477

def get_LMCmu(lp, mag, feh, zp, slope, met, title_in):

    M = zp + slope*lp + met*feh
    delta_mu = (mag - M) - mu_true
    N = len(lp)

    dmean = np.mean(delta_mu)
    dmedian = np.median(delta_mu)
    print("%s\t: %2.3f\t%2.3f" % (title_in, dmean, dmedian))
    
    plt.minorticks_on()
    plt.tick_params(which='both', bottom='on', top='on', left='on', right='on', direction='inout')

    plt.hist(delta_mu, bins=50, color='skyblue', edgecolor='black')
    #plt.hist(delta_mu, bins=int(N/10)+8, color='skyblue', edgecolor='black')
    plt.axvline(0.0, c='r', ls='-')
    plt.axvline(dmean, c='r', ls='--')
    plt.ylabel("Count", fontsize=13)
    plt.xlabel(r"$\mu_j-\mu_{LMC}^0$", fontsize=15)
    plt.title(str(title_in))
    plt.tight_layout()
    plt.show()
    

# ========================================================

indat = ascii.read("DiF_in.csv", format='csv')

mi = indat['Imag'] < 99.999
mab = indat['rtype'] == 'ab'
mc = indat['rtype'] == 'c'

logP = np.log10(indat['period'])
FeH = indat['FeH']

BV = indat['Bmag'] - indat['Vmag']

gmag = indat['Bmag'] - 0.108 - 0.485*BV - 0.032*BV*BV
rmag = indat['Vmag'] + 0.082 - 0.462*BV + 0.041*BV*BV
imag = indat['Imag'] + 0.341 + 0.154*BV - 0.025*BV*BV

g0 = gmag - Rg*indat['EBV']
r0 = rmag - Rr*indat['EBV']
i0 = imag - Ri*indat['EBV']

ilogP = logP[mi]
iFeH = FeH[mi]
ii0 = i0[mi]
iab = mab[mi]
ic = mc[mi]

# my PLZ
get_LMCmu(logP[mab], g0[mab], FeH[mab], 0.649, -0.302, 0.159, "Ngeow g-band ab-PLZ")
get_LMCmu(logP[mab], r0[mab], FeH[mab], 0.337, -1.090, 0.139, "Ngeow r-band ab-PLZ")
get_LMCmu(ilogP[iab], ii0[iab], iFeH[iab], 0.243, -1.432, 0.144, "Ngeow i-band ab-PLZ")

# Narloch PLZ
P0 = -0.25
Z0 = -1.5

get_LMCmu(logP[mab]-P0, g0[mab], FeH[mab]-Z0, 0.794, -0.527, 0.264, "Narloch g-band ab-PLZ(1)")
get_LMCmu(logP[mab]-P0, r0[mab], FeH[mab]-Z0, 0.651, -1.230, 0.205, "Narloch r-band ab-PLZ(1)")
get_LMCmu(ilogP[iab]-P0, ii0[iab], iFeH[iab]-Z0, 0.635, -1.682, 0.174, "Narloch i-band ab-PLZ(1)")

get_LMCmu(logP[mab]-P0, g0[mab], FeH[mab]-Z0, 0.798, -0.503, 0.266, "Narloch g-band ab-PLZ(2)")
get_LMCmu(logP[mab]-P0, r0[mab], FeH[mab]-Z0, 0.655, -1.227, 0.204, "Narloch r-band ab-PLZ(2)")
get_LMCmu(ilogP[iab]-P0, ii0[iab], iFeH[iab]-Z0, 0.638, -1.673, 0.175, "Narloch i-band ab-PLZ(2)")

get_LMCmu(logP[mab]-P0, g0[mab], FeH[mab]-Z0, 0.792, -0.389, 0.267, "Narloch g-band ab-PLZ(3)")
get_LMCmu(logP[mab]-P0, r0[mab], FeH[mab]-Z0, 0.650, -1.092, 0.208, "Narloch r-band ab-PLZ(3)")
get_LMCmu(ilogP[iab]-P0, ii0[iab], iFeH[iab]-Z0, 0.633, -1.544, 0.177, "Narloch i-band ab-PLZ(3)")

get_LMCmu(logP[mab]-P0, g0[mab], FeH[mab]-Z0, 0.791, -0.473, 0.257, "Narloch g-band ab-PLZ(4)")
get_LMCmu(logP[mab]-P0, r0[mab], FeH[mab]-Z0, 0.650, -1.177, 0.198, "Narloch r-band ab-PLZ(4)")
get_LMCmu(ilogP[iab]-P0, ii0[iab], iFeH[iab]-Z0, 0.633, -1.629, 0.168, "Narloch i-band ab-PLZ(4)")
