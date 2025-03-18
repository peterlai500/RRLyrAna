import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import ascii

Ng22_PWZcoeffs = {
    'Wgr': {'RR0':     {'a': -0.644, 'b': -3.324, 'c': 0.052}, 
            'RR1':     {'a': -1.327, 'b': -3.680, 'c': 0.013},
            'RR0+RR1': {'a': -0.727, 'b': -3.286, 'c': 0.010}},
    'Wri': {'RR0':     {'a':  0.093, 'b': -2.600, 'c': 0.193}, 
            'RR1':     {'a': -0.884, 'b': -3.503, 'c': 0.067},
            'RR0+RR1': {'a':  0.010, 'b': -2.756, 'c': 0.149}},
    'Wgi': {'RR0':     {'a': -0.198, 'b': -2.908, 'c': 0.142}, 
            'RR1':     {'a': -1.086, 'b': -3.696, 'c': 0.055},
            'RR0+RR1': {'a': -0.288, 'b': -3.066, 'c': 0.101}}
}

Na24_PWZcoeffs01 = { # Parallax
    'Wgr': {'RR0':     {'a':  -3.061, 'b': 0.584, 'c': 0.082}, 
            'RR0+RR1': {'a':  -2.848, 'b': 0.583, 'c': 0.105}},
    'Wri': {'RR0':     {'a':  -3.273, 'b': 0.239, 'c': 0.033}, 
            'RR0+RR1': {'a':  -3.148, 'b': 0.238, 'c': 0.050}},
    'Wgi': {'RR0':     {'a':  -3.154, 'b': 0.433, 'c': 0.060}, 
            'RR0+RR1': {'a':  -2.979, 'b': 0.432, 'c': 0.081}}
}
Na24_PWZcoeffs02 = { # ABL
    'Wgr': {'RR0':     {'a':  -3.027, 'b': 0.587, 'c': 0.084}, 
            'RR0+RR1': {'a':  -2.837, 'b': 0.585, 'c': 0.107}},
    'Wri': {'RR0':     {'a':  -3.347, 'b': 0.241, 'c': 0.019}, 
            'RR0+RR1': {'a':  -3.178, 'b': 0.240, 'c': 0.042}},
    'Wgi': {'RR0':     {'a':  -3.169, 'b': 0.435, 'c': 0.055}, 
            'RR0+RR1': {'a':  -2.987, 'b': 0.433, 'c': 0.078}}
}
Na24_PWZcoeffs03 = { # Geometric Distance
    'Wgr': {'RR0':     {'a':  -2.923, 'b': 0.583, 'c': 0.085}, 
            'RR0+RR1': {'a':  -2.824, 'b': 0.580, 'c': 0.096}},
    'Wri': {'RR0':     {'a':  -3.135, 'b': 0.237, 'c': 0.036}, 
            'RR0+RR1': {'a':  -3.124, 'b': 0.236, 'c': 0.040}},
    'Wgi': {'RR0':     {'a':  -3.016, 'b': 0.431, 'c': 0.063}, 
            'RR0+RR1': {'a':  -2.955, 'b': 0.429, 'c': 0.071}}
}
Na24_PWZcoeffs04 = { # Geometric Distance
    'Wgr': {'RR0':     {'a':  -3.008, 'b': 0.582, 'c': 0.075}, 
            'RR0+RR1': {'a':  -2.835, 'b': 0.580, 'c': 0.095}},
    'Wri': {'RR0':     {'a':  -3.220, 'b': 0.236, 'c': 0.026}, 
            'RR0+RR1': {'a':  -3.135, 'b': 0.236, 'c': 0.039}},
    'Wgi': {'RR0':     {'a':  -3.101, 'b': 0.430, 'c': 0.054}, 
            'RR0+RR1': {'a':  -2.967, 'b': 0.430, 'c': 0.070}}
}

He25_PWZcoeffs = {
    'Wgr': {'RR0': {'a':  -2.678, 'b': 0.007, 'c': -0.610}, 
            'RR1': {'a':  -3.437, 'b': 0.019, 'c': -1.209}},
    'Wri': {'RR0': {'a':  -2.671, 'b': 0.056, 'c': -0.266}, 
            'RR1': {'a':  -3.186, 'b': 0.107, 'c': -0.740}},
    'Wgi': {'RR0': {'a':  -2.512, 'b': 0.095, 'c': -0.006}, 
            'RR1': {'a':  -3.136, 'b': 0.137, 'c': -0.544}}
}

def PWZrelation(logp, feh, passband, rrtype, coeffs, model):
    a, b, c = coeffs[passband][rrtype].values()

    if rrtype == 'RR0+RR1':
        type0, type1 = w_masks[rrtype][passband]
        mask = np.logical_or(type0, type1)
        p0 = logp[type0]
        p1 = logp[type1] + 0.127
        p = np.concatenate((p0, p1), axis=0)
    else:
        mask = w_masks[rrtype][passband]
        p = logp[mask]

    if model == 'Ng22':
        a, b, c = coeffs[passband][rrtype].values()
        return a + b * p + c * feh[mask]
    elif model == 'Na24':
        P0 = P0_pivot[rrtype]
        feh0 = -1.5
        return a * (logp[mask] - P0) + b + c * (feh[mask] - feh0)
    elif model == 'He25':
        a, b, c = coeffs[passband][rrtype].values()
        return a * p + b * feh[mask] + c

def plot_PWZhist(passband, rrtype, coeffs, model, mu_true, bins_factor=10):
    if rrtype == 'RR0+RR1':
        type0, type1 = w_masks[rrtype][passband]
        m0 = data[f'{passband}'][type0]
        m1 = data[f'{passband}'][type1]
        m = np.concatenate((m0, m1), axis=0)
    else:
        mask = w_masks[rrtype][passband]
        m = data[f'{passband}'][mask]

    M = PWZrelation(logP, m_feh, passband, rrtype, coeffs, model)

    delta_mu = (m - M) - mu_true

    dmean   = np.mean(delta_mu)
    dmedian = np.median(delta_mu)
    N = len(m)

    print(f"Data counts: {N}, Mean: {dmean:.3f}, Median: {dmedian:.3f}")
    plt.figure(figsize=(7, 5))
    plt.minorticks_on()
    plt.tick_params(which='both', bottom=True, top=True, left=True, right=True, direction='inout')
    plt.hist(delta_mu, bins=int(N / bins_factor) + 8, color='skyblue', edgecolor='black')
    plt.axvline(0.0, c='r', ls='-')  # True distance modulus
    plt.axvline(dmean, c='r', ls='--', label=f"Mean = {dmean:.3f}")
    plt.ylabel("Count", fontsize=13)
    plt.xlabel(r"$\mu_j-\mu_{LMC}^0$", fontsize=13)
    plt.title(f"# of Distance Modulus Residuals ({passband}, {rrtype})", fontsize=14)
    plt.legend()
    plt.tight_layout()




data = ascii.read("DiF_in.csv", format='csv')

mu_true = 18.477

Rg = 3.518
Rr = 2.617
Ri = 1.971

logP = np.log10(data['period'])
m_feh = data['FeH']
BV = data['Bmag'] - data['Vmag']

data['g'] = data['Bmag'] - 0.108 - 0.485*BV - 0.032*BV*BV
data['r'] = data['Vmag'] + 0.082 - 0.462*BV + 0.041*BV*BV
data['i'] = data['Imag'] + 0.341 + 0.154*BV - 0.025*BV*BV

data['Mg'] = gmag - Rg*data['EBV'] - mu
data['Mr'] = rmag - Rr*data['EBV'] - mu
data['Mi'] = imag - Ri*data['EBV'] - mu

mab = data['rtype'] == 'ab'
mc  = data['rtype'] == 'c'

mi = data['i'] < 99

data['Wgr'] = data['r'] - 2.905 * (data['g'] - data['r'])
data['Wri'] = data['r'] - 4.051 * (data['r'] - data['i'])
data['Wgi'] = data['g'] - 2.274 * (data['g'] - data['i'])

ri = data['Wri'] < 99
gi = data['Wgi'] < 99

w_masks = {
    'RR0': {
        'Wgr': mab,
        'Wri': mab & ri,
        'Wgi': mab & gi
    },
    'RR1': {
        'Wgr': mc,
        'Wri': mc & ri,
        'Wgi': mc & gi
    },
    'RR0+RR1': {
        'Wgr': [mab, mc],
        'Wri': [mab & ri, mc & ri],
        'Wgi': [mab & gi, mc & gi]
    }
}


##################################################
########## RR0 PWZ ###############################
# Wgr
plot_PWZhist("Wgr", "RR0", Ng22_PWZcoeffs, 'Ng22', mu_true)
#plt.savefig(f"figures/dmu-Ng22PLZ_RR0-Wgr-DiFa", dpi=300)
plot_PWZhist("Wgr", "RR0", He25_PWZcoeffs, 'He25', mu_true)
#plt.savefig(f"figures/dmu-He25PLZ_RR0-Wgr-DiFa", dpi=300)
plot_PWZhist("Wgr", "RR0", Na24_PWZcoeffs01, 'Na24', mu_true)
#plt.savefig(f"figures/dmu-Na24PLZ01_RR0-Wgr-DiFa", dpi=300)
plot_PWZhist("Wgr", "RR0", Na24_PWZcoeffs02, 'Na24', mu_true)
#plt.savefig(f"figures/dmu-Na24PLZ02_RR0-Wgr-DiFa", dpi=300)
plot_PWZhist("Wgr", "RR0", Na24_PWZcoeffs03, 'Na24', mu_true)
#plt.savefig(f"figures/dmu-Na24PLZ03_RR0-Wgr-DiFa", dpi=300)
plot_PWZhist("Wgr", "RR0", Na24_PWZcoeffs04, 'Na24', mu_true)
#plt.savefig(f"figures/dmu-Na24PLZ04_RR0-Wgr-DiFa", dpi=300)

# Wgi
plot_PWZhist("Wgi", "RR0", Ng22_PWZcoeffs, 'Ng22', mu_true)
#plt.savefig(f"figures/dmu-Ng22PLZ_RR0-Wgi-DiFa", dpi=300)
plot_PWZhist("Wgi", "RR0", He25_PWZcoeffs, 'He25', mu_true)
#plt.savefig(f"figures/dmu-He25PLZ_RR0-Wgi-DiFa", dpi=300)
plot_PWZhist("Wgi", "RR0", Na24_PWZcoeffs01, 'Na24', mu_true)
#plt.savefig(f"figures/dmu-Na24PLZ01_RR0-Wgi-DiFa", dpi=300)
plot_PWZhist("Wgi", "RR0", Na24_PWZcoeffs02, 'Na24', mu_true)
#plt.savefig(f"figures/dmu-Na24PLZ02_RR0-Wgi-DiFa", dpi=300)
plot_PWZhist("Wgi", "RR0", Na24_PWZcoeffs03, 'Na24', mu_true)
#plt.savefig(f"figures/dmu-Na24PLZ03_RR0-Wgi-DiFa", dpi=300)
plot_PWZhist("Wgi", "RR0", Na24_PWZcoeffs04, 'Na24', mu_true)
#plt.savefig(f"figures/dmu-Na24PLZ04_RR0-Wgi-DiFa", dpi=300)

# Wri
plot_PWZhist("Wri", "RR0", Ng22_PWZcoeffs, 'Ng22', mu_true)
#plt.savefig(f"figures/dmu-Ng22PLZ_RR0-Wri-DiFa", dpi=300)
plot_PWZhist("Wri", "RR0", He25_PWZcoeffs, 'He25', mu_true)
#plt.savefig(f"figures/dmu-He25PLZ_RR0-Wri-DiFa", dpi=300)
plot_PWZhist("Wri", "RR0", Na24_PWZcoeffs01, 'Na24', mu_true)
#plt.savefig(f"figures/dmu-Na24PLZ01_RR0-Wri-DiFa", dpi=300)
plot_PWZhist("Wri", "RR0", Na24_PWZcoeffs02, 'Na24', mu_true)
#plt.savefig(f"figures/dmu-Na24PLZ02_RR0-Wri-DiFa", dpi=300)
plot_PWZhist("Wri", "RR0", Na24_PWZcoeffs03, 'Na24', mu_true)
#plt.savefig(f"figures/dmu-Na24PLZ03_RR0-Wri-DiFa", dpi=300)
plot_PWZhist("Wri", "RR0", Na24_PWZcoeffs04, 'Na24', mu_true)
#plt.savefig(f"figures/dmu-Na24PLZ04_RR0-Wri-DiFa", dpi=300)


##################################################
########## RR1 PWZ ###############################
# Wgr
plot_PWZhist("Wgr", "RR1", Ng22_PWZcoeffs, 'Ng22', mu_true)
# plt.savefig(f"figures/dmu-Ng22PWZ_RR1-Wgr-DiFa", dpi=300)
plot_PWZhist("Wgr", "RR1", He25_PWZcoeffs, 'He25', mu_true)
# plt.savefig(f"figures/dmu-He25PWZ_RR1-Wgr-DiFa", dpi=300)

# Wri
plot_PWZhist("Wri", "RR1", Ng22_PWZcoeffs, 'Ng22', mu_true)
# plt.savefig(f"figures/dmu-Ng22PWZ_RR1-Wri-DiFa", dpi=300)
plot_PWZhist("Wri", "RR1", He25_PWZcoeffs, 'He25', mu_true)
# plt.savefig(f"figures/dmu-He25PWZ_RR1-Wri-DiFa", dpi=300)

# Wgi
plot_PWZhist("Wgi", "RR1", Ng22_PWZcoeffs, 'Ng22', mu_true)
# plt.savefig(f"figures/dmu-Ng22PWZ_RR1-Wgi-DiFa", dpi=300)
plot_PWZhist("Wgi", "RR1", He25_PWZcoeffs, 'He25', mu_true)
# plt.savefig(f"figures/dmu-He25PWZ_RR1-Wgi-DiFa", dpi=300)


##################################################
########## RR0+RR1 PWZ ###########################
# Wgr
plot_PWZhist("Wgr", "RR0+RR1", Ng22_PWZcoeffs, 'Ng22', mu_true)
# plt.savefig(f"figures/dmu-Ng22PWZ_RR01-Wgr-DiFa", dpi=300)
plot_PWZhist("Wgr", "RR0+RR1", Na24_PWZcoeffs01, 'Na24', mu_true)
# plt.savefig(f"figures/dmu-Na24PWZ01_RR01-Wgr-DiFa", dpi=300)
plot_PWZhist("Wgr", "RR0+RR1", Na24_PWZcoeffs02, 'Na24', mu_true)
# plt.savefig(f"figures/dmu-Na24PWZ02_RR01-Wgr-DiFa", dpi=300)
plot_PWZhist("Wgr", "RR0+RR1", Na24_PWZcoeffs03, 'Na24', mu_true)
# plt.savefig(f"figures/dmu-Na24PWZ03_RR01-Wgr-DiFa", dpi=300)
plot_PWZhist("Wgr", "RR0+RR1", Na24_PWZcoeffs04, 'Na24', mu_true)
# plt.savefig(f"figures/dmu-Na24PWZ04_RR01-Wgr-DiFa", dpi=300)

# Wri
plot_PWZhist("Wri", "RR0+RR1", Ng22_PWZcoeffs, 'Ng22', mu_true)
# plt.savefig(f"figures/dmu-Ng22PWZ_RR01-Wri-DiFa", dpi=300)
plot_PWZhist("Wri", "RR0+RR1", Na24_PWZcoeffs01, 'Na24', mu_true)
# plt.savefig(f"figures/dmu-Na24PWZ01_RR01-Wri-DiFa", dpi=300)
plot_PWZhist("Wri", "RR0+RR1", Na24_PWZcoeffs02, 'Na24', mu_true)
# plt.savefig(f"figures/dmu-Na24PWZ02_RR01-Wri-DiFa", dpi=300)
plot_PWZhist("Wri", "RR0+RR1", Na24_PWZcoeffs03, 'Na24', mu_true)
# plt.savefig(f"figures/dmu-Na24PWZ03_RR01-Wri-DiFa", dpi=300)
plot_PWZhist("Wri", "RR0+RR1", Na24_PWZcoeffs04, 'Na24', mu_true)
# plt.savefig(f"figures/dmu-Na24PWZ04_RR01-Wri-DiFa", dpi=300)

# Wgi
plot_PWZhist("Wgi", "RR0+RR1", Ng22_PWZcoeffs, 'Ng22', mu_true)
# plt.savefig(f"figures/dmu-Ng22PWZ_RR01-Wgi-DiFa", dpi=300)
plot_PWZhist("Wgi", "RR0+RR1", Na24_PWZcoeffs01, 'Na24', mu_true)
# plt.savefig(f"figures/dmu-Na24PWZ01_RR01-Wgi-DiFa", dpi=300)
plot_PWZhist("Wgi", "RR0+RR1", Na24_PWZcoeffs02, 'Na24', mu_true)
# plt.savefig(f"figures/dmu-Na24PWZ02_RR01-Wgi-DiFa", dpi=300)
plot_PWZhist("Wgi", "RR0+RR1", Na24_PWZcoeffs03, 'Na24', mu_true)
# plt.savefig(f"figures/dmu-Na24PWZ03_RR01-Wgi-DiFa", dpi=300)
plot_PWZhist("Wgi", "RR0+RR1", Na24_PWZcoeffs04, 'Na24', mu_true)
# plt.savefig(f"figures/dmu-Na24PWZ04_RR01-Wgi-DiFa", dpi=300)
