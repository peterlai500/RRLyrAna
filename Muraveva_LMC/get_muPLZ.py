import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

Ng22_PLZcoeffs = {
    'g': {'RR0':     {'a':  0.649, 'b': -0.302, 'c': 0.159},
          'RR1':     {'a':  0.411, 'b': -0.342, 'c': 0.092},
          'RR0+RR1': {'a':  0.801, 'b': -0.032, 'c': 0.190}},
    'r': {'RR0':     {'a':  0.337, 'b': -1.090, 'c': 0.139},
          'RR1':     {'a': -0.082, 'b': -1.393, 'c': 0.091},
          'RR0+RR1': {'a':  0.432, 'b': -0.874, 'c': 0.154}},
    'i': {'RR0':     {'a':  0.243, 'b': -1.432, 'c': 0.144},
          'RR1':     {'a': -0.205, 'b': -1.706, 'c': 0.077},
          'RR0+RR1': {'a':  0.249, 'b': -1.362, 'c': 0.115}}
}

Na24_PLZcoeffs01 = { # Parallax
    'g': {'RR0':     {'a':  -0.527, 'b': 0.794, 'c': 0.264},
          'RR0+RR1': {'a':  -0.284, 'b': 0.791, 'c': 0.289}},
    'r': {'RR0':     {'a':  -1.230, 'b': 0.651, 'c': 0.205},
          'RR0+RR1': {'a':  -1.017, 'b': 0.650, 'c': 0.228}},
    'i': {'RR0':     {'a':  -1.682, 'b': 0.635, 'c': 0.174},
          'RR0+RR1': {'a':  -1.469, 'b': 0.633, 'c': 0.198}}
}

Na24_PLZcoeffs02 = { # ABL
    'g': {'RR0':     {'a':  -0.503, 'b': 0.798, 'c': 0.266},
          'RR0+RR1': {'a':  -0.290, 'b': 0.795, 'c': 0.290}},
    'r': {'RR0':     {'a':  -1.227, 'b': 0.655, 'c': 0.204},
          'RR0+RR1': {'a':  -1.031, 'b': 0.652, 'c': 0.227}},
    'i': {'RR0':     {'a':  -1.673, 'b': 0.638, 'c': 0.175},
          'RR0+RR1': {'a':  -1.479, 'b': 0.635, 'c': 0.197}}
}

Na24_PLZcoeffs03 = { # Geometric Distance
    'g': {'RR0':     {'a':  -0.389, 'b': 0.792, 'c': 0.267},
          'RR0+RR1': {'a':  -0.217, 'b': 0.790, 'c': 0.285}},
    'r': {'RR0':     {'a':  -1.092, 'b': 0.650, 'c': 0.208},
          'RR0+RR1': {'a':  -0.950, 'b': 0.648, 'c': 0.223}},
    'i': {'RR0':     {'a':  -1.544, 'b': 0.633, 'c': 0.177},
          'RR0+RR1': {'a':  -1.402, 'b': 0.632, 'c': 0.193}}
}

Na24_PLZcoeffs04 = { # Geometric Distance
    'g': {'RR0':     {'a':  -0.473, 'b': 0.791, 'c': 0.257},
          'RR0+RR1': {'a':  -0.260, 'b': 0.789, 'c': 0.280}},
    'r': {'RR0':     {'a':  -1.177, 'b': 0.650, 'c': 0.198},
          'RR0+RR1': {'a':  -0.993, 'b': 0.647, 'c': 0.218}},
    'i': {'RR0':     {'a':  -1.629, 'b': 0.633, 'c': 0.168},
          'RR0+RR1': {'a':  -1.445, 'b': 0.631, 'c': 0.188}}
}

P0_pivot = {'RR0':     -0.25,
            'RR1':     -0.45,
            'RR0+RR1': -0.25}

He25_PLZcoeffs = {
    'g': {'RR0': {'a':  -0.829, 'b': 0.233, 'c': 0.976},
          'RR1': {'a':  -0.898, 'b': 0.148, 'c': 0.509}},
    'r': {'RR0': {'a':  -1.432, 'b': 0.180, 'c': 0.561},
          'RR1': {'a':  -1.395, 'b': 0.139, 'c': 0.202}},
    'i': {'RR0': {'a':  -1.639, 'b': 0.169, 'c': 0.442},
          'RR1': {'a':  -1.551, 'b': 0.135, 'c': 0.120}}
}

def PLZrelation(logp, feh, passband, rrtype, coeffs, model):
    a, b, c = coeffs[passband][rrtype].values()

    if rrtype == 'RR0+RR1':
        type0, type1 = mag_masks[rrtype][passband]
        mask = np.logical_or(type0, type1)
        p0 = logp[type0].values
        p1 = logp[type1].values + 0.127
        p = np.concatenate((p0, p1), axis=0)
    else:
        mask = mag_masks[rrtype][passband]
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

def plot_PLZhist(passband, rrtype, coeffs, model, mu_true, bins_factor=10):
    if rrtype == 'RR0+RR1':
        type0, type1 = mag_masks[rrtype][passband]
        m0 = data['mag_0'][type0].values
        m1 = data['mag_0'][type1].values
        m = np.concatenate((m0, m1), axis=0)
    else:
        mask = mag_masks[rrtype][passband]
        m = data['mag_0'][mask].values

    M = PLZrelation(logP, FeH, passband, rrtype, coeffs, model)

    delta_mu = (m - M) - mu_true

    dmean   = np.mean(delta_mu)
    dmedian = np.median(delta_mu)
    N = len(m)

    print(f"Mean: {dmean:.3f}, Median: {dmedian:.3f}")
    plt.figure(figsize=(7, 5))
    plt.minorticks_on()
    plt.tick_params(which='both', bottom=True, top=True, left=True, right=True, direction='inout')
    plt.hist(delta_mu, bins=int(N / bins_factor) + 8, color='skyblue', edgecolor='black')
    plt.axvline(0.0, c='r', ls='-')  # True distance modulus
    plt.axvline(dmean, c='r', ls='--', label=f"Mean = {dmean:.3f}")
    plt.ylabel("Count", fontsize=13)
    plt.xlabel(r"$\mu_j-\mu_{LMC}^0$", fontsize=13)
    plt.title(f"# of Distance Modulus Residuals ({passband}-band, {rrtype})", fontsize=14)
    plt.legend()
    plt.tight_layout()

mu_true = 18.477

data = pd.read_csv("light_curve_results.csv")

R = {"g": 3.518, "r": 2.617, "i": 1.971}

mg = data['band'] == 'g'
mr = data['band'] == 'r'
mi = data['band'] == 'i'

mab = data['type'] == 'RRab'
mc  = data['type'] == 'RRc'

logP = np.log10(data['period'])
FeH  = data['[Fe/H]'].values

data['mag_0'] = data['mean_mag'] - data['band'].map(R) * data['E(B-V)']
data['Mag'] = data['mag_0'] - data['Distance_modulus']

good    = data['err_mag'] < 0.5
img_N   = data['N_image'].values
d_free  = img_N - 3
chi_squ = data['chi_squared'].values
reduced_chi = chi_squ / d_free
well_fit    = (reduced_chi<2) & (reduced_chi>0.5)

mag_masks = {
    "RR0": {
        "g": mab & mg & good & well_fit,
        "r": mab & mr & good & well_fit,
        "i": mab & mi & good & well_fit
    },
    "RR1": {
        "g": mc & mg,
        "r": mc & mr,
        "i": mc & mi
    },
    "RR0+RR1": {
        "g": [mab & mg & good & well_fit, mc & mg & good & well_fit] ,
        "r": [mab & mr & good & well_fit, mc & mr & good & well_fit],
        "i": [mab & mi & good & well_fit, mc & mi & good & well_fit]
    }
}

##################################################
########## RR0 PLZ ###############################
# g band
plot_PLZhist("g", "RR0", Ng22_PLZcoeffs, 'Ng22', mu_true)
#plt.savefig(f"figures/dmu-Ng22PLZ_RR0-g", dpi=300)
plot_PLZhist("g", "RR0", He25_PLZcoeffs, 'He25', mu_true)
#plt.savefig(f"figures/dmu-He25PLZ_RR0-g", dpi=300)
plot_PLZhist("g", "RR0", Na24_PLZcoeffs01, 'Na24', mu_true)
#plt.savefig(f"figures/dmu-Na24PLZ01_RR0-g", dpi=300)
plot_PLZhist("g", "RR0", Na24_PLZcoeffs02, 'Na24', mu_true)
#plt.savefig(f"figures/dmu-Na24PLZ02_RR0-g", dpi=300)
plot_PLZhist("g", "RR0", Na24_PLZcoeffs03, 'Na24', mu_true)
#plt.savefig(f"figures/dmu-Na24PLZ03_RR0-g", dpi=300)
plot_PLZhist("g", "RR0", Na24_PLZcoeffs04, 'Na24', mu_true)
#plt.savefig(f"figures/dmu-Na24PLZ04_RR0-g", dpi=300)

# r band
plot_PLZhist("r", "RR0", Ng22_PLZcoeffs, 'Ng22', mu_true)
# plt.savefig(f"figures/dmu-Ng22PLZ_RR0-r", dpi=300)
plot_PLZhist("r", "RR0", He25_PLZcoeffs, 'He25', mu_true)
# plt.savefig(f"figures/dmu-He25PLZ_RR0-r", dpi=300)
plot_PLZhist("r", "RR0", Na24_PLZcoeffs01, 'Na24', mu_true)
# plt.savefig(f"figures/dmu-Na24PLZ01_RR0-r", dpi=300)
plot_PLZhist("r", "RR0", Na24_PLZcoeffs02, 'Na24', mu_true)
# plt.savefig(f"figures/dmu-Na24PLZ02_RR0-r", dpi=300)
plot_PLZhist("r", "RR0", Na24_PLZcoeffs03, 'Na24', mu_true)
# plt.savefig(f"figures/dmu-Na24PLZ03_RR0-r", dpi=300)
plot_PLZhist("r", "RR0", Na24_PLZcoeffs04, 'Na24', mu_true)
# plt.savefig(f"figures/dmu-Na24PLZ04_RR0-r", dpi=300)

# i band
plot_PLZhist("i", "RR0", Ng22_PLZcoeffs, 'Ng22', mu_true)
# plt.savefig(f"figures/dmu-Ng22PLZ_RR0-i", dpi=300)
plot_PLZhist("i", "RR0", He25_PLZcoeffs, 'He25', mu_true)
# plt.savefig(f"figures/dmu-He25PLZ_RR0-i", dpi=300)
plot_PLZhist("i", "RR0", Na24_PLZcoeffs01, 'Na24', mu_true)
# plt.savefig(f"figures/dmu-Na24PLZ01_RR0-i", dpi=300)
plot_PLZhist("i", "RR0", Na24_PLZcoeffs02, 'Na24', mu_true)
# plt.savefig(f"figures/dmu-Na24PLZ02_RR0-i", dpi=300)
plot_PLZhist("i", "RR0", Na24_PLZcoeffs03, 'Na24', mu_true)
# plt.savefig(f"figures/dmu-Na24PLZ03_RR0-i", dpi=300)
plot_PLZhist("i", "RR0", Na24_PLZcoeffs04, 'Na24', mu_true)
# plt.savefig(f"figures/dmu-Na24PLZ04_RR0-i", dpi=300)


'''
##################################################
########## RR0 PLZ ###############################
# g band
plot_PLZhist("g", "RR1", Ng22_PLZcoeffs, 'Ng22', mu_true)
# plt.savefig(f"figures/dmu-Ng22PLZ_RR1-g", dpi=300)
plot_PLZhist("g", "RR1", He25_PLZcoeffs, 'He25', mu_true)
# plt.savefig(f"figures/dmu-He25PLZ_RR1-g", dpi=300)

# r band
plot_PLZhist("r", "RR1", Ng22_PLZcoeffs, 'Ng22', mu_true)
# plt.savefig(f"figures/dmu-Ng22PLZ_RR1-r", dpi=300)
plot_PLZhist("r", "RR1", He25_PLZcoeffs, 'He25', mu_true)
# plt.savefig(f"figures/dmu-He25PLZ_RR1-r", dpi=300)

# i band
plot_PLZhist("i", "RR1", Ng22_PLZcoeffs, 'Ng22', mu_true)
# plt.savefig(f"figures/dmu-Ng22PLZ_RR1-i", dpi=300)
plot_PLZhist("i", "RR1", He25_PLZcoeffs, 'He25', mu_true)
# plt.savefig(f"figures/dmu-He25PLZ_RR1-i", dpi=300)

##################################################
########## RR0+RR1 PLZ ###########################
# g band
plot_PLZhist("g", "RR0+RR1", Ng22_PLZcoeffs, 'Ng22', mu_true)
# plt.savefig(f"figures/dmu-Ng22PLZ_RR01-g", dpi=300)
plot_PLZhist("g", "RR0+RR1", Na24_PLZcoeffs01, 'Na24', mu_true)
# plt.savefig(f"figures/dmu-Na24PLZ01_RR01-g", dpi=300)
plot_PLZhist("g", "RR0+RR1", Na24_PLZcoeffs02, 'Na24', mu_true)
# plt.savefig(f"figures/dmu-Na24PLZ02_RR01-g", dpi=300)
plot_PLZhist("g", "RR0+RR1", Na24_PLZcoeffs03, 'Na24', mu_true)
# plt.savefig(f"figures/dmu-Na24PLZ03_RR01-g", dpi=300)
plot_PLZhist("g", "RR0+RR1", Na24_PLZcoeffs04, 'Na24', mu_true)
# plt.savefig(f"figures/dmu-Na24PLZ04_RR01-g", dpi=300)

# r band
plot_PLZhist("r", "RR0+RR1", Ng22_PLZcoeffs, 'Ng22', mu_true)
# plt.savefig(f"figures/dmu-Ng22PLZ_RR01-r", dpi=300)
plot_PLZhist("r", "RR0+RR1", Na24_PLZcoeffs01, 'Na24', mu_true)
# plt.savefig(f"figures/dmu-Na24PLZ01_RR01-r", dpi=300)
plot_PLZhist("r", "RR0+RR1", Na24_PLZcoeffs02, 'Na24', mu_true)
# plt.savefig(f"figures/dmu-Na24PLZ02_RR01-r", dpi=300)
plot_PLZhist("r", "RR0+RR1", Na24_PLZcoeffs03, 'Na24', mu_true)
# plt.savefig(f"figures/dmu-Na24PLZ03_RR01-r", dpi=300)
plot_PLZhist("r", "RR0+RR1", Na24_PLZcoeffs04, 'Na24', mu_true)
# plt.savefig(f"figures/dmu-Na24PLZ04_RR01-r", dpi=300)

# i band
plot_PLZhist("i", "RR0+RR1", Ng22_PLZcoeffs, 'Ng22', mu_true)
# plt.savefig(f"figures/dmu-Ng22PLZ_RR01-i", dpi=300)
plot_PLZhist("i", "RR0+RR1", Na24_PLZcoeffs01, 'Na24', mu_true)
# plt.savefig(f"figures/dmu-Na24PLZ01_RR01-i", dpi=300)
plot_PLZhist("i", "RR0+RR1", Na24_PLZcoeffs02, 'Na24', mu_true)
# plt.savefig(f"figures/dmu-Na24PLZ02_RR01-i", dpi=300)
plot_PLZhist("i", "RR0+RR1", Na24_PLZcoeffs03, 'Na24', mu_true)
# plt.savefig(f"figures/dmu-Na24PLZ03_RR01-i", dpi=300)
plot_PLZhist("i", "RR0+RR1", Na24_PLZcoeffs04, 'Na24', mu_true)
# plt.savefig(f"figures/dmu-Na24PLZ04_RR01-i", dpi=300)
'''
