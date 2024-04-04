from dataclasses import dataclass
# Third Party
import numpy as np
import numpy.polynomial as polynomial

import scipy.ndimage

import jax.numpy as jnp
import jax

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

@dataclass
class Data:
    """Temporary Data Type"""
    dataframes: list

    def __getitem__(self,i):
        return self.dataframes[i]

    @property
    def yerr(self):
        return 1/np.sqrt(self.yivar)

    @property
    def xs(self):
        return [dataframe.xs for dataframe in self.dataframes]

    @property
    def ys(self):
        return [dataframe.ys for dataframe in self.dataframes]

    @property
    def yivar(self):
        return [dataframe.yivar for dataframe in self.dataframes]

    @property
    def mask(self):
        return [dataframe.mask for dataframe in self.dataframes]

    def __len__(self):
        return len(self.dataframes)

    def from_lists(xs,ys,yivar,ma):
        frames = []
        for iii in range(len(xs)):
            frames.append(DataFrame(xs[iii],ys[iii],yivar[iii],ma[iii]))
        return Data(frames)

    def to_device(self,device):
        for dataframe in self.dataframes:
            dataframe.to_device(device)

    def blockify(data,device):
        max_ind = np.max([len(dataframe.xs) for dataframe in data])
        xs    = np.zeros((len(data),max_ind))
        ys    = np.zeros((len(data),max_ind))
        yivar = np.zeros((len(data),max_ind))
        mask  = np.ones((len(data),max_ind))
    
        for i,dataframe in enumerate(data):
            frame_size = len(dataframe.xs)
            xs[i,:frame_size]    = dataframe.xs
            ys[i,:frame_size]    = dataframe.ys
            yivar[i,:frame_size] = dataframe.yivar
            mask[i,:frame_size]  = dataframe.mask

        xs    = jax.device_put(jnp.array(xs),device)
        ys    = jax.device_put(jnp.array(ys),device)
        yivar = jax.device_put(jnp.array(yivar),device)
        mask  = jax.device_put(jnp.array(mask,dtype=bool),device)

        return xs, ys, yivar, mask


class DataFrame:
    def __init__(self, xs: jnp.array,ys: jnp.array,yivar: jnp.array, mask: jnp.array):
        self.xs = jnp.array(xs)
        self.ys = jnp.array(ys)
        self.yivar = jnp.array(yivar)
        self.mask = jnp.array(mask)
    

    def to_device(self,device):
        self.xs = jax.device_put(jnp.array(self.xs),device)
        self.ys = jax.device_put(jnp.array(self.ys),device)
        self.yivar = jax.device_put(jnp.array(self.yivar),device)
        self.mask = jax.device_put(jnp.array(self.mask),device)