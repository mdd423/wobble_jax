
# Third Party
import numpy as np
import numpy.polynomial as polynomial

import scipy.ndimage

import jax.numpy as jnp

# not really important
def find_nearest(array,value):
    array = np.asarray(array)
    idx   = (np.abs(array-value)).argmin()
    return idx

# important for grid search
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

# kinda not important function
# important interpolate function
def interpolate_mask(flux,mask):
    # assumes even spacing in wavelength of flux
    new_flux = np.zeros(flux.shape)
    new_flux = flux
    for j,mask_row in enumerate(mask):
        cnt = 0
        for i, mask_ele in enumerate(mask_row):
            if mask_ele != 0:
                cnt += 1
            if mask_ele == 0 and cnt != 0:
                # right here is that assumption in the linspace wort wavelength
                new_flux[j,i-cnt:i] = np.linspace(flux[j,i-cnt-1],flux[j,i],cnt+2)[1:-1]
                cnt = 0
    return new_flux

class Dataset:
    def __init__(self,xs,ys,yerr,mask):
        self.xs = xs
        self.ys = ys
        self.yerr = yerr
        self.mask = mask

        self.yivar = 1/self.yerr**2

    def set_mask(self,y_val,yerr_val):
        self.ys[self.mask] = y_val
        self.yerr[self.mask] = yerr_val

        self.yivar = 1/self.yerr**2

    def __getitem__(self,i):
        return Dataset(self.xs[i,:],self.ys[i,:],self.yerr[i,:],self.mask[i,:])

    def from_flux(wave,flux,ferr,mask,normalize=None,nargs=[]):
        if normalize is None:
            nargs = [80]
            normalize = scipy.ndimage.gaussian_filter
        xs = np.log(wave.to(u.Angstrom).value)
        flux_interp = interpolate_mask(flux,mask)

        flux_norm = np.empty(flux.shape)
        for i in range(flux.shape[0]):
            flux_norm[i,:] = normalize(flux_interp[i,:],*nargs)
        ys = np.log(flux_interp/flux_norm)
        yerr = ferr/flux_interp
        return Dataset(xs, ys, yerr, mask)
