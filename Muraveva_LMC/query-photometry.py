import os
import sys

import pandas as pd
import numpy as np
import multiprocessing as mp

from astropy.io import ascii
import astropy.units as u
from astropy.table import Table, vstack

from astroquery.utils.tap.core import TapPlus

import timeit

id_table = ascii.read("Muraveva_LMC-smss_ID.csv")

SkyMapper = TapPlus(url="https://api.skymapper.nci.org.au/public/tap/")

image_id = Table(names=["object_id", "image_id", "ra_img",  "decl_img", "filter", "mag_psf", "e_mag_psf", "date", "exp_time", "flags"], dtype=["int64", "int64", "float64", "float64", "str", "float32", "float32",	"float64", "float32", "int16"])

print("Begin query SMSS photometry")
start = timeit.default_timer()


# Function to query SkyMapper photometry for a single object_id
def query_photometry(object_id):
    prompt = f"""
    SELECT
        object_id, image_id, ra_img, decl_img, f.filter, mag_psf, e_mag_psf, date, exp_time, flags
    FROM
        dr4.photometry f
    JOIN
        dr4.images USING (image_id)
    WHERE
        object_id={object_id}
        AND f.filter IN ('g', 'r', 'i')
    """
    try:
        job = SkyMapper.launch_job(prompt)
        result = job.get_results()
        if len(result) > 0:
            return result  # Return results only if there's data
    except Exception as e:
        print(f"Query failed for object_id {object_id}: {e}")
    return None  # Return None if the query fails

# Set up multiprocessing
num_workers = mp.cpu_count()  # Use all available CPU cores
with mp.Pool(processes=num_workers) as pool:
    results = pool.map(query_photometry, id_table['object_id'])

# Filter out None results and merge tables
valid_results = [r for r in results if r is not None]
if valid_results:
    image_id = vstack(valid_results)


stop = timeit.default_timer()

print("Done SMSS image ID query")
print('Query time:', stop - start)
print(f'Total {len(image_id)} photometric entries retrieved.')

image_id.write('Muraveva_LMC-smss_photometry.csv', format='csv', overwrite=True)
print("Results saved to 'Muraveva_LMC-smss_photometry.csv'")
