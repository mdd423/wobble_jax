import astropy.table as at
import astropy.units as u
import astropy.coordinates as coord
import astropy.constants as const
import astropy.time as atime

import numpy as np
import jax.numpy as jnp

def velocities(shifts):
    expon = jnp.exp(2*shifts)
    vel = const.c * (expon-1)/(1 + expon)
    return vel

def delta_x(R):
    return jnp.log(1+1/R)

def shifts(vel):
    return jnp.log(np.sqrt((1 + vel/(const.c))/(1 - vel/(const.c))))

def get_star_velocity(BJD,star_name,observatory_name,parse=False):
    star = coord.SkyCoord.from_name(star_name,parse=parse)
    loc      = coord.EarthLocation.of_site(observatory_name)
    ts       = atime.Time(BJD, format='jd', scale='tdb')
    bc       = star.radial_velocity_correction(obstime=ts, location=loc).to(u.km/u.s)
    return bc
