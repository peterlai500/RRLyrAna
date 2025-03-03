import numpy as np

from astropy.io import ascii
from astropy.coordinates import SkyCoord, Angle
import astropy.units as u
from astropy.table import vstack, Table

from astroquery.utils.tap.core import TapPlus

import timeit

########## select data ##########
Table_A2 = ascii.read("Table_A2.txt")

upper_mu = 19.5
lower_mu = 18
mu_select = Table_A2[
    (Table_A2["Distance_modulus"] >= lower_mu) &
    (Table_A2["Distance_modulus"] <= upper_mu)
]

lmc_center = SkyCoord(ra=80.894167*u.degree, dec=-69.756111*u.degree, frame='icrs')
radius = 5* u.degree

data_coords = SkyCoord(ra=mu_select["RA"]*u.degree, dec=mu_select["DEC"]*u.degree, frame='icrs')

sep = data_coords.separation(lmc_center)

mu_cut_data = mu_select[sep<radius]

print(f"Total {len(mu_cut_data)} sources selected")

mu_cut_data.write('Muraveva_LMC-gaia_ID.csv', format='csv', overwrite=True)


########## Query SMSS object ID  ##########
gaia_Table = ascii.read("Muraveva_LMC-gaia_ID.csv")
SkyMapper = TapPlus(url="https://api.skymapper.nci.org.au/public/tap/")
print("Begin query object id")
objectid_table = Table(names=["object_id", "gaia_dr3_id1", "raj2000", "dej2000"], dtype=["int64", "int64", "float64", "float64"])
start = timeit.default_timer()

for (gaia_id, ra, dec) in zip(gaia_Table["source_id"], gaia_Table["RA"], gaia_Table["DEC"]):
    prompt = f"""
    SELECT
        object_id, gaia_dr3_id1, raj2000, dej2000
        FROM
            dr4.master 
        WHERE 
            1=CONTAINS(POINT('ICRS', raj2000, dej2000),
                       CIRCLE('ICRS', {ra}, {dec}, 0.1 ))
            AND gaia_dr3_id1={gaia_id}
    """
    job = SkyMapper.launch_job(prompt)
    result = job.get_data()

    if len(result) == 1:
        objectid_table = vstack([objectid_table, result])

stop = timeit.default_timer()
print("Finish query SMSS object ID")
print('Time: ', stop - start)
print(len(id_table["object_id"]), "ID found from GAIA ID")

objectid_table.write('Muraveva_LMC-smss_ID.csv', format='csv', overwrite=True)

########## Query photometry ##########
id_table = ascii.read("Muraveva_LMC-smss_ID.csv")

photometry = Table(names=["object_id", "image_id", "ra_img", "decl_img", "filter", "mag_psf", "e_mag_psf", "date", "exp_time", "flags"], dtype=["int64", "int64", "float64", "float64", "str", "float32", "float32", "float64", "float32", "int16"])

print("Begin query SMSS photometry")
start = timeit.default_timer()
for ID in id_table['object_id']:
    prompt = f"""
    SELECT
        object_id, image_id, ra_img, decl_img, f.filter, mag_psf, e_mag_psf, date, exp_time, flags
        FROM
            dr4.photometry f
        JOIN
            dr4.images USING (image_id)
        WHERE
            object_id={ID}
            AND f.filter IN ('g', 'r', 'i')
    """
    job = SkyMapper.launch_job(prompt)
    result = job.get_results()

    photometry = vstack([photometry, result])

stop = timeit.default_timer()
print("Finish photometry query")
print('Query time:', stop - start)

photometry.write('Muraveva_LMC-smss_photometry.csv', format='csv', overwrite=True)
