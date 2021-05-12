import numpy as np
# import matplotlib.pyplot as plt
import astropy.table as at
import astropy.units as u
import astropy.coordinates as coord
import scipy.constants as const
import astropy.time as atime
import scipy.ndimage as ndimage

import numpy.polynomial as polynomial

import model as wobble_model
import jax.numpy as jnp

def get_loss_array(shift_grid,model,xs,ys,yerr,loss,*args):
    if len(xs.shape) == 1:
        xs = np.expand_dims(xs,axis=0)
    loss_arr = np.empty((xs.shape[0],shift_grid.shape[0]))
    for i in range(xs.shape[0]):
        for j,shift in enumerate(shift_grid):
            loss_arr[i,j] = loss(model.params,ys[i,:],yerr[i,:],xs[i,:]+shift,None,model,*args)
    return loss_arr

def get_parabolic_min(loss_array,grid,return_all=False):
    epoches = loss_array.shape[0]
    grid_min = np.empty(epoches)

    for n in range(epoches):
        idx = loss_array[n,:].argmin()
        # print("epch {}: min {}".format(n,idx))
        xs = grid[idx-1:idx+2]
        ys = loss_array[n,idx-1:idx+2]

        poly = np.polyfit(xs,ys,deg=2)
        deriv = np.polyder(poly)

        x_min = np.roots(deriv)
        x_min = x_min[x_min.imag==0].real
        y_min = np.polyval(poly,x_min)
        grid_min[n] = x_min
    if (return_all):
        return grid_min, xs, ys, poly
    else:
        return grid_min

def zplusone(vel):
    return np.sqrt((1 + vel/(const.c*u.m/u.s))/(1 - vel/(const.c*u.m/u.s)))

def shifts(vel):
    return np.log(zplusone(vel))

def getInitXShift(BJD,star_name,observatory_name,parse=False):
    hatp20_c = coord.SkyCoord.from_name(star_name,parse=parse)
    loc      = coord.EarthLocation.of_site(observatory_name)
    ts       = atime.Time(BJD, format='jd', scale='tdb')
    bc       = hatp20_c.radial_velocity_correction(obstime=ts, location=loc).to(u.km/u.s)
    x_shifts = shifts(bc)
    return x_shifts

class AstroDataset():
    def __init__(self,flux,lamb,mask,ferr):
        self.flux = flux
        self.lamb = lamb#/u.Angstrom
        self.mask = mask
        self.ferr = ferr

    def interpolate_mask(self):
        new_flux = np.zeros(self.flux.shape)
        new_flux = self.flux
        for j,mask_row in enumerate(self.mask):
            cnt = 0
            for i, mask_ele in enumerate(mask_row):
                if mask_ele != 0:
                    cnt += 1
                if mask_ele == 0 and cnt != 0:
                    new_flux[j,i-cnt:i] = np.linspace(self.flux[j,i-cnt-1],self.flux[j,i],cnt+2)[1:-1]
                    cnt = 0
        self.flux = new_flux

    def gauss_filter(self,sigma):
        filtered_flux = ndimage.gaussian_filter1d(self.flux,sigma)
        return filtered_flux

    def get_xy(self,filtered,y_const=0.0,err_const=10):
        # start, end = subset

        y     = np.log(self.flux/filtered)
        x     = np.log(self.lamb)
        y_err = (self.ferr)
        y_err /= self.flux

        y[self.mask]   = y_const
        y_err[self.mask] = err_const

        return jnp.array(x,dtype=np.float32), jnp.array(y,dtype=np.float32), jnp.array(y_err,dtype=np.float32)
