import numpy as np
# import matplotlib.pyplot as plt
import astropy.table as at
import astropy.units as u
import astropy.coordinates as coord
import astropy.constants as const
import astropy.time as atime
import scipy.ndimage as ndimage

import numpy.polynomial as polynomial

# import jabble.model as wobble_model
import jax.numpy as jnp

def find_nearest(array,value):
    array = np.asarray(array)
    idx   = (np.abs(array-value)).argmin()
    return idx

def velocityfromshift(shifts):
    expon = np.exp(2*shifts)
    vel = const.c * (expon-1)/(1 + expon)
    return vel

def get_loss_array(shift_grid,model,xs,ys,yerr,loss,*args):
    # so so so shit
    if len(xs.shape) == 1:
        xs = np.expand_dims(xs,axis=0)

    if len(shift_grid.shape) == 1:
        loss_arr = np.empty((xs.shape[0],shift_grid.shape[0]))
        for i in range(xs.shape[0]):
            for j,shift in enumerate(shift_grid):
                loss_arr[i,j] = loss(model.p,xs[i,:]+shift,ys[i,:],yerr[i,:],i,model,*args)
    if len(shift_grid.shape) == 2:
        loss_arr = np.empty((xs.shape[0],shift_grid.shape[1]))
        for i in range(xs.shape[0]):
            for j,shift in enumerate(shift_grid[i,:]):
                loss_arr[i,j] = loss(model.p,xs[i,:]+shift,ys[i,:],yerr[i,:],i,model,*args)

    return loss_arr

def get_parabolic_min(loss_array,grid,return_all=False):
    epoches = loss_array.shape[0]
    grid_min = np.empty(epoches)

    xss = np.empty((epoches,3))
    yss = np.empty((epoches,3))
    polys = []

    for n in range(epoches):
        idx = loss_array[n,:].argmin()
        print("epch {}: min {}".format(n,idx))
        if idx == 0:
            print("minimum likely out of range")
            idx = 1
        if idx == grid.shape[1]-1:
            print("minimum likely out of range")
            idx -= 1
        # else:

        xs = grid[n,idx-1:idx+2]
        xss[n,:] = xs
        ys = loss_array[n,idx-1:idx+2]
        yss[n,:] = ys

        poly = np.polyfit(xs,ys,deg=2)
        polys.append(poly)
        deriv = np.polyder(poly)

        x_min = np.roots(deriv)
        x_min = x_min[x_min.imag==0].real
        y_min = np.polyval(poly,x_min)

        grid_min[n] = x_min

    if (return_all):
        return grid_min, xss, yss, polys
    else:
        return grid_min

def zplusone(vel):
    return np.sqrt((1 + vel/(const.c))/(1 - vel/(const.c)))

def shifts(vel):
    return np.log(zplusone(vel))

def get_star_velocity(BJD,star_name,observatory_name,parse=False):
    hatp20_c = coord.SkyCoord.from_name(star_name,parse=parse)
    loc      = coord.EarthLocation.of_site(observatory_name)
    ts       = atime.Time(BJD, format='jd', scale='tdb')
    bc       = hatp20_c.radial_velocity_correction(obstime=ts, location=loc).to(u.km/u.s)
    return bc

def interpolate_mask(flux,mask):
    new_flux = np.zeros(flux.shape)
    new_flux = flux
    for j,mask_row in enumerate(mask):
        cnt = 0
        for i, mask_ele in enumerate(mask_row):
            if mask_ele != 0:
                cnt += 1
            if mask_ele == 0 and cnt != 0:
                new_flux[j,i-cnt:i] = np.linspace(flux[j,i-cnt-1],flux[j,i],cnt+2)[1:-1]
                cnt = 0
    return new_flux

def gauss_filter(flux,sigma):
    filtered_flux = ndimage.gaussian_filter1d(flux,sigma)
    return filtered_flux

def normalize_flux(flux,sigma):
    return flux/gauss_filter(flux,sigma)

def convert_xy(lamb,flux,ferr):
    y    = np.log(flux)
    x    = np.log(lamb)
    yerr = ferr/flux
    return x, y, yerr

def set_masked(y,yerr,mask,y_const=0.0,err_const=10.0):
    y[mask]    = y_const
    yerr[mask] = err_const
    return y, yerr

class WobbleDataset:
    def __init__(self,wavelength,flux,flux_error,mask,sigma=80):
        flux       = interpolate_mask(flux,mask)
        flux_norm  = normalize_flux(flux,sigma)
        self.xs, self.ys, self.yerr = np.log(wavelength/u.Angstrom), np.log(flux_norm), flux_error/flux
        self.ys, self.yerr    = set_masked(self.ys,self.yerr,mask)
        self.epoches    = self.ys.shape[0]
