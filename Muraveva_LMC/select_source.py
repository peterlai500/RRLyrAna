from astropy.io import ascii
from astropy.coordinates import SkyCoord, Angle
import astropy.units as u

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
