import os
import sys

import pandas as pd
import numpy as np
import multiprocessing as mp

from astropy.io import ascii
from astropy.coordinates import SkyCoord, Angle
import astropy.units as u
from astropy.table import Table, vstack

from astroquery.utils.tap.core import TapPlus

import timeit

gaia_Table = ascii.read("Muraveva_LMC-gaia_ID.csv")
SkyMapper = TapPlus(url="https://api.skymapper.nci.org.au/public/tap/")
objectid_table = Table(names=["object_id", "gaia_dr3_id1", "raj2000", "dej2000"], dtype=["int64", "int64", "float64", "float64"])

def query_skymapper(gaia_id, ra, dec):
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
    try:
        job = SkyMapper.launch_job(prompt)
        result = job.get_data()
        if len(result) == 1:
            return result
    except Exception as e:
        print(f"Query failed for GAIA ID {gaia_id}: {e}")
    return None

print("Begin query SMSS object ID")
start = timeit.default_timer()

num_workers = mp.cpu_count()  # Use all available CPU cores
with mp.Pool(processes=num_workers) as pool:
    results = pool.starmap(query_skymapper, zip(gaia_Table["source_id"], gaia_Table["RA"], gaia_Table["DEC"]))

# Filter out None results and merge tables
valid_results = [r for r in results if r is not None]
if valid_results:
    id_table = vstack(valid_results)

stop = timeit.default_timer()
print("Finish query SMSS object ID")
print('Query time: ', stop - start)
print(len(id_table["object_id"]), "ID found from GAIA ID")

id_table.write('Muraveva_LMC-smss_ID.csv', format='csv', overwrite=True)
print("Results saved to 'Muraveva_LMC-smss_ID.csv'")
