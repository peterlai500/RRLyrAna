import os
import sys

import pandas as pd
import numpy as np
import multiprocessing as mp
import timeit

from astropy.coordinates import SkyCoord, Angle, EarthLocation, get_sun
from astropy.io import ascii
from astropy.table import join, Table, vstack
from astropy.time import Time
import astropy.units as u

import apply_ugrizy_templates

def mag_filter(image_ls):
    return len(image_ls) > 3  # Returns True if good, False if bad

def mjd_to_hjd(mjd, ra, dec, observatory='greenwich'):
    """
    Convert MJD to Heliocentric Julian Date (HJD).

    Parameters:
    - mjd: float, Modified Julian Date
    - ra: float, Right Ascension in degrees
    - dec: float, Declination in degrees
    - observatory: str, Name of the observing location (default: Greenwich)

    Returns:
    - HJD: float, Heliocentric Julian Date
    """
    # Convert MJD to JD
    time = Time(mjd, format='mjd', scale='utc')

    # Define the observatory location (default Greenwich, override if needed)
    location = EarthLocation.of_site(observatory)

    # Define the celestial target coordinates
    target = SkyCoord(ra=ra * u.deg, dec=dec * u.deg)

    # Get the Sun's position
    light_time_correction = time.light_travel_time(target, 'heliocentric', location=location)

    # Apply correction to get HJD
    hjd = time.tdb + light_time_correction
    return hjd.jd

def type_band(rr_type, band):
    if rr_type == 'RRab':
        pulsation_type = 0
    else:
        pulsation_type = 1

    if band == 'g':
        passband = 2
    elif band == 'r':
        passband = 3
    elif band == 'i':
        passband = 4

    if pulsation_type == 0:
        amplmaxs = {2: 1.8, 3: 1.5, 4: 1.1, 5: 1.1}
        amplmins = {2: .2, 3: .15, 4: .1, 5: .1}
        amplratio_ogle_to_use = 'Ampl(x)/Ampl(I)_ab'
    else:
        amplmaxs = {2: .9, 3: .7, 4: .55, 5: 1.1}
        amplmins = {2: .1, 3: .05, 4: 0., 5: 0.}
        amplratio_ogle_to_use = 'Ampl(x)/Ampl(I)_c'

    amplmax = amplmaxs[passband]
    amplmin = amplmins[passband]

    return pulsation_type, passband, amplmax, amplmin

def apply_light_curve_fit(data, file_coeff, t0=0):
    obj, pulsation_period, rr_type, ra, dec, band, feh, feh_err, EBV, EBV_err, mu, mu_err = data[['object_id', 'Period', 'Type', 'RA', 'DEC','filter',
                                                          "[Fe/H]", "[Fe/H]_error", "E(B-V)", "E(B-V)_error",
                                                          "Distance_modulus", "Distance_modulus_error"]].iloc[0]
    pulsation_type, passband, ampl_max, ampl_min = type_band(rr_type, band)

    mag  = np.asarray(data['mag_psf'])
    err  = np.asarray(data['e_mag_psf'])
    date = np.asarray(data['date'])

    hjd = mjd_to_hjd(date, ra, dec)
    try:
        templatebin_int = apply_ugrizy_templates.find_templatebin(pulsation_type,
                                                                  pulsation_period,
                                                                  passband)[0]

        result_resample = apply_ugrizy_templates.apply_templatefit(hjd, mag, err,
                                                    pulsation_type, pulsation_period, t0, passband,
                                                    file_coeff, ampl=(ampl_max+ampl_min)/2,
                                                    amplmax=ampl_max, amplmin=ampl_min, figure_out=f'{folder}/plot-lcfit/{obj}-{band}.pdf')
        return {
            "object_id": obj,
            "period": pulsation_period,
            "type": rr_type,
            "band": band,
            "mean_mag": result_resample['mag_mean'],
            "err_mag": result_resample['errmag_mean'],
            "chi_squared": result_resample['chisq'],
            "[Fe/H]": feh,
            "[Fe/H]_error": feh_err,
            "E(B-V)": EBV,
            "E(B-V)_error": EBV_err,
            "Distance_modulus": mu,
            "Distance_modulus_error": mu_err,
        }

    except Exception as e:
        return None


################################################################################
########## Merge the csv tables################################################
muraveva_df = pd.read_csv("Muraveva_LMC-gaia_ID.csv", usecols=["source_id", "RA", "DEC", "Type",
                                                               "Period", "[Fe/H]", "[Fe/H]_error",
                                                               "E(B-V)", "E(B-V)_error",
                                                               "Distance_modulus", "Distance_modulus_error"])
objid_df    = pd.read_csv("Muraveva_LMC-smss_ID.csv", usecols=["object_id", "gaia_dr3_id1"])
imageid_df  = pd.read_csv("Muraveva_LMC-smss_photometry.csv")

# Rename columns while reading (avoiding extra reassignment)
muraveva_df.rename(columns={'source_id': 'gaia_dr3_id1'}, inplace=True)

# Merge tables efficiently
merged_df = objid_df.merge(muraveva_df, on='gaia_dr3_id1', how="inner") \
                    .merge(imageid_df, on='object_id', how="inner")


################################################################################
########## Select object with sufficient photmetries (>=4) #####################
objid = np.unique(merged_df['object_id'])
sample_dict = {f: {"good": [], "bad" : []} for f in ["g", "r", "i"]}

for obj in objid:
    current_target = merged_df[merged_df['object_id'] == obj]

    # Process all filters in one loop
    for band in ["g", "r", "i"]:
        image_data = current_target[current_target['filter'] == band]

        # Classify sample into good or bad
        category = "good" if mag_filter(image_data) else "bad"
        sample_dict[band][category].append(image_data)


################################################################################
########## Apply light-curve template fitting ##################################
folder = os.getcwd()+'/' #To be changed by the user
file_coeff = folder+'templates_analytics_230901.csv'

# Prepare the list of tasks
tasks = []
for band in ["g", "r", "i"]:
    tasks.extend([(data, file_coeff) for data in sample_dict[band]["good"]])
    
start = timeit.default_timer()

# Use multiprocessing to run tasks in parallel
if __name__ == "__main__":
    with mp.Pool(processes=mp.cpu_count()) as pool:  # Use all available CPU cores
        results_list = pool.starmap(apply_light_curve_fit, tasks)

    # Filter out None values (errors)
    results_list = [res for res in results_list if res is not None]

    # Convert results to a DataFrame and save to CSV
    df_results = pd.DataFrame(results_list)
    df_results.to_csv("light_curve_results.csv", index=False, float_format="%.6f")

    print("CSV export complete: light_curve_results.csv")

stop = timeit.default_timer()
print(f'Total number of objects:{len(objid)}')
print(f'Totally fitted {len(df_results)} light curves')
print('Fitting time:', stop - start)
