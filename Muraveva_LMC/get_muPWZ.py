import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


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
        type0, type1 = w_mask[rrtype][passband]
        mask = np.logical_or(type0, type1)
        p0 = logp[type0].values
        p1 = logp[type1].values + 0.127
        p = np.concatenate((p0, p1), axis=0)
    else:
        mask = w_mask[rrtype][passband]
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
        type0, type1 = w_mask[rrtype][passband]
        w0 = w_df[passband][type0].values
        w1 = w_df[passband][type1].values
        w = np.concatenate((w0, w1), axis=0)
    else:
        mask = w_mask[rrtype][passband]
        w = w_df[passband][mask]
        
    W = PWZrelation(w_logP, w_feh, passband, rrtype, coeffs, model)
    
    delta_mu = (w - W) - mu_true

    dmean   = np.mean(delta_mu)
    dmedian = np.median(delta_mu)
    N = len(w)
    
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


mu_true = 18.477

data = pd.read_csv("light_curve_results.csv")

R = {"g": 3.518, "r": 2.617, "i": 1.971}

mg = data['band'] == 'g'
mr = data['band'] == 'r'
mi = data['band'] == 'i'

mab = data['type'] == 'RRab'
mc  = data['type'] == 'RRc'

logP = np.log10(data['period'])
m_feh  = data['[Fe/H]'].values

data['mag_0'] = data['mean_mag'] - data['band'].map(R) * data['E(B-V)']
data['Mag'] = data['mag_0'] - data['Distance_modulus']

all_id = np.unique(data["object_id"])

wesenheit_dict = {"object_id": [], "type":[], "logP": [], "Wgr": [], "Wgi": [], "Wri": [], "FeH": []}

for objid in all_id:
    phot = data[data.object_id == objid]
    if len(phot) > 1:
        bands = phot.band.values
        p = phot.period.values[0]
        p = np.log10(p)
        feh = phot['[Fe/H]'].values[0]
        rrtype = phot.type.values[0]
    
        Wgr = Wgi = Wri = np.nan
        
        if set(bands)  == {'g', 'r', 'i'}:
            g = phot.mag_0[bands == 'g'].values[0]
            r = phot.mag_0[bands == 'r'].values[0]
            i = phot.mag_0[bands == 'i'].values[0]
            Wgr = r - 2.905 * (g - r)
            Wri = r - 4.051 * (r - i)
            Wgi = g - 2.274 * (g - i)
            # print(f'Wgr: {Wgr}\t Wgi: {Wri}\t Wri: {Wgi}')
        elif set(bands)  == {'g', 'r'}:
            g = phot.mag_0[bands == 'g'].values[0]
            r = phot.mag_0[bands == 'r'].values[0]
            Wgr = r - 2.905 * (g - r)
            # print('Wgr')
        elif set(bands)  == {'i', 'r'}:
            r = phot.mag_0[bands == 'r'].values[0]
            i = phot.mag_0[bands == 'i'].values[0]
            Wri = r - 4.051 * (r - i)
            # print('Wri')
        elif set(bands)  == {'g', 'i'}:
            g = phot.mag_0[bands == 'g'].values[0]
            i = phot.mag_0[bands == 'i'].values[0]
            Wgi = g - 2.274 * (g - i)
            # print('Wgi')
            
        wesenheit_dict["object_id"].append(objid)
        wesenheit_dict["type"].append(rrtype)
        wesenheit_dict["logP"].append(p)
        wesenheit_dict["Wgr"].append(Wgr)
        wesenheit_dict["Wgi"].append(Wgi)
        wesenheit_dict["Wri"].append(Wri)
        wesenheit_dict["FeH"].append(feh)

w_df = pd.DataFrame(wesenheit_dict)

wab = w_df.type == 'RRab'
wc = w_df.type == 'RRc'

gr = w_df.Wgr > 0
ri = w_df.Wri > 0
gi = w_df.Wgi > 0

w_feh  = w_df.FeH
w_logP = w_df.logP

##################################################
########## RR0 PWZ ###############################
# Wgr
plot_PWZhist("Wgr", "RR0", Ng22_PWZcoeffs, 'Ng22', mu_true)
# plt.savefig(f"figures/dmu-Ng22PWZ_RR0-Wgr", dpi=300)
plot_PWZhist("Wgr", "RR0", He25_PWZcoeffs, 'He25', mu_true)
# plt.savefig(f"figures/dmu-He25PWZ_RR0-Wgr", dpi=300)
plot_PWZhist("Wgr", "RR0", Na24_PWZcoeffs01, 'Na24', mu_true)
# plt.savefig(f"figures/dmu-Na24PWZ01_RR0-Wgr", dpi=300)
plot_PWZhist("Wgr", "RR0", Na24_PWZcoeffs02, 'Na24', mu_true)
# plt.savefig(f"figures/dmu-Na24PWZ02_RR0-Wgr", dpi=300)
plot_PWZhist("Wgr", "RR0", Na24_PWZcoeffs03, 'Na24', mu_true)
# plt.savefig(f"figures/dmu-Na24PWZ03_RR0-Wgr", dpi=300)
plot_PWZhist("Wgr", "RR0", Na24_PWZcoeffs04, 'Na24', mu_true)
# plt.savefig(f"figures/dmu-Na24PWZ04_RR0-Wgr", dpi=300)

# Wri
plot_PWZhist("Wri", "RR0", Ng22_PWZcoeffs, 'Ng22', mu_true)
# plt.savefig(f"figures/dmu-Ng22PWZ_RR0-Wri", dpi=300)
plot_PWZhist("Wri", "RR0", He25_PWZcoeffs, 'He25', mu_true)
# plt.savefig(f"figures/dmu-He25PWZ_RR0-Wri", dpi=300)
plot_PWZhist("Wri", "RR0", Na24_PWZcoeffs01, 'Na24', mu_true)
# plt.savefig(f"figures/dmu-Na24PWZ01_RR0-Wri", dpi=300)
plot_PWZhist("Wri", "RR0", Na24_PWZcoeffs02, 'Na24', mu_true)
# plt.savefig(f"figures/dmu-Na24PWZ02_RR0-Wri", dpi=300)
plot_PWZhist("Wri", "RR0", Na24_PWZcoeffs03, 'Na24', mu_true)
# plt.savefig(f"figures/dmu-Na24PWZ03_RR0-Wri", dpi=300)
plot_PWZhist("Wri", "RR0", Na24_PWZcoeffs04, 'Na24', mu_true)
# plt.savefig(f"figures/dmu-Na24PWZ04_RR0-Wri", dpi=300)

# Wgi
plot_PWZhist("Wgi", "RR0", Ng22_PWZcoeffs, 'Ng22', mu_true)
#plt.savefig(f"figures/dmu-Ng22PWZ_RR0-Wgi", dpi=300)
plot_PWZhist("Wgi", "RR0", He25_PWZcoeffs, 'He25', mu_true)
#plt.savefig(f"figures/dmu-He25PWZ_RR0-Wgi", dpi=300)
plot_PWZhist("Wgi", "RR0", Na24_PWZcoeffs01, 'Na24', mu_true)
#plt.savefig(f"figures/dmu-Na24PWZ01_RR0-Wgi", dpi=300)
plot_PWZhist("Wgi", "RR0", Na24_PWZcoeffs02, 'Na24', mu_true)
#plt.savefig(f"figures/dmu-Na24PWZ02_RR0-Wgi", dpi=300)
plot_PWZhist("Wgi", "RR0", Na24_PWZcoeffs03, 'Na24', mu_true)
#plt.savefig(f"figures/dmu-Na24PWZ03_RR0-Wgi", dpi=300)
plot_PWZhist("Wgi", "RR0", Na24_PWZcoeffs04, 'Na24', mu_true)
#plt.savefig(f"figures/dmu-Na24PWZ04_RR0-Wgi", dpi=300)


##################################################
########## RR1 PWZ ###############################
# Wgr
plot_PWZhist("Wgr", "RR1", Ng22_PWZcoeffs, 'Ng22', mu_true)
# plt.savefig(f"figures/dmu-Ng22PWZ_RR1-Wgr", dpi=300)
plot_PWZhist("Wgr", "RR1", He25_PWZcoeffs, 'He25', mu_true)
# plt.savefig(f"figures/dmu-He25PWZ_RR1-Wgr", dpi=300)

# Wri
plot_PWZhist("Wri", "RR1", Ng22_PWZcoeffs, 'Ng22', mu_true)
# plt.savefig(f"figures/dmu-Ng22PWZ_RR1-Wri", dpi=300)
plot_PWZhist("Wri", "RR1", He25_PWZcoeffs, 'He25', mu_true)
# plt.savefig(f"figures/dmu-He25PWZ_RR1-Wri", dpi=300)

# Wgi
plot_PWZhist("Wgi", "RR1", Ng22_PWZcoeffs, 'Ng22', mu_true)
# plt.savefig(f"figures/dmu-Ng22PWZ_RR1-Wgi", dpi=300)
plot_PWZhist("Wgi", "RR1", He25_PWZcoeffs, 'He25', mu_true)
# plt.savefig(f"figures/dmu-He25PWZ_RR1-Wgi", dpi=300)


##################################################
########## RR0+RR1 PWZ ###########################
# Wgr
plot_PWZhist("Wgr", "RR0+RR1", Ng22_PWZcoeffs, 'Ng22', mu_true)
# plt.savefig(f"figures/dmu-Ng22PWZ_RR01-Wgr", dpi=300)
plot_PWZhist("Wgr", "RR0+RR1", Na24_PWZcoeffs01, 'Na24', mu_true)
# plt.savefig(f"figures/dmu-Na24PWZ01_RR01-Wgr", dpi=300)
plot_PWZhist("Wgr", "RR0+RR1", Na24_PWZcoeffs02, 'Na24', mu_true)
# plt.savefig(f"figures/dmu-Na24PWZ02_RR01-Wgr", dpi=300)
plot_PWZhist("Wgr", "RR0+RR1", Na24_PWZcoeffs03, 'Na24', mu_true)
# plt.savefig(f"figures/dmu-Na24PWZ03_RR01-Wgr", dpi=300)
plot_PWZhist("Wgr", "RR0+RR1", Na24_PWZcoeffs04, 'Na24', mu_true)
# plt.savefig(f"figures/dmu-Na24PWZ04_RR01-Wgr", dpi=300)

# Wgi
plot_PWZhist("Wri", "RR0+RR1", Ng22_PWZcoeffs, 'Ng22', mu_true)
# plt.savefig(f"figures/dmu-Ng22PWZ_RR01-Wri", dpi=300)
plot_PWZhist("Wri", "RR0+RR1", Na24_PWZcoeffs01, 'Na24', mu_true)
# plt.savefig(f"figures/dmu-Na24PWZ01_RR01-Wri", dpi=300)
plot_PWZhist("Wri", "RR0+RR1", Na24_PWZcoeffs02, 'Na24', mu_true)
# plt.savefig(f"figures/dmu-Na24PWZ02_RR01-Wri", dpi=300)
plot_PWZhist("Wri", "RR0+RR1", Na24_PWZcoeffs03, 'Na24', mu_true)
# plt.savefig(f"figures/dmu-Na24PWZ03_RR01-Wri", dpi=300)
plot_PWZhist("Wri", "RR0+RR1", Na24_PWZcoeffs04, 'Na24', mu_true)
# plt.savefig(f"figures/dmu-Na24PWZ04_RR01-Wri", dpi=300)

# Wri
plot_PWZhist("Wgi", "RR0+RR1", Ng22_PWZcoeffs, 'Ng22', mu_true)
# plt.savefig(f"figures/dmu-Ng22PWZ_RR01-Wgi", dpi=300)
plot_PWZhist("Wgi", "RR0+RR1", Na24_PWZcoeffs01, 'Na24', mu_true)
# plt.savefig(f"figures/dmu-Na24PWZ01_RR01-Wgi", dpi=300)
plot_PWZhist("Wgi", "RR0+RR1", Na24_PWZcoeffs02, 'Na24', mu_true)
# plt.savefig(f"figures/dmu-Na24PWZ02_RR01-Wgi", dpi=300)
plot_PWZhist("Wgi", "RR0+RR1", Na24_PWZcoeffs03, 'Na24', mu_true)
# plt.savefig(f"figures/dmu-Na24PWZ03_RR01-Wgi", dpi=300)
plot_PWZhist("Wgi", "RR0+RR1", Na24_PWZcoeffs04, 'Na24', mu_true)
# plt.savefig(f"figures/dmu-Na24PWZ04_RR01-Wgi", dpi=300)
