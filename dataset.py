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

def get_loss_array(shift_grid,model,loss):
    loss_arr = np.empty((model.epoches,shift_grid.shape[0]))
    for i in range(model.epoches):
        for j,shift in enumerate(shift_grid):
            loss_arr[i,j] = loss(model.params,model.ys[i,:],model.xs[i,:]+shift,None,model)
    return loss_arr

def get_parabolic_min(loss_array,grid,return_all=False):
    epoches = loss_array.shape[0]
    grid_min = np.empty(epoches)

    for n in range(epoches):
        idx = loss_array[n,:].argmin()
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

def getInitXShift(BJD,star_name,observatory_name):
    hatp20_c = coord.SkyCoord.from_name(star_name)
    loc      = coord.EarthLocation.of_site(observatory_name)
    ts       = atime.Time(BJD, format='jd', scale='tdb')
    bc       = hatp20_c.radial_velocity_correction(obstime=ts, location=loc).to(u.km/u.s)
    x_shifts = shifts(bc)
    return x_shifts

class AstroDataset():
    def __init__(self,flux,lamb,mask,ferr):
        self.flux = flux
        self.lamb = lamb/u.Angstrom
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
        self.sigma = sigma
        self.filtered_flux = ndimage.gaussian_filter1d(self.flux,sigma)

    # def plot_epoch_one(self,x,y,i,y_err=None,xlims=None):
    #     fig = plt.figure(figsize=[12.8,9.6])
    #     # plt.legend(['filtered','unfiltered masked','unfiltered unmasked'])
    #     # fig.axes[0].xaxis.set_visible(False)
    #     # fig.axes[0].yaxis.set_visible(False)
    #     plt.xlabel('x (log lambda)')
    #     plt.ylabel('y (log flux)')
    #     plt.title('gauss filtered lin interp corrected data w/ sigma {}'.format(self.sigma))
    #     plt.ylim(-1,0.25)
    #     if xlims is not None:
    #         plt.xlim(np.log(xlims[0]),np.log(xlims[1]))
    #     plt.plot(x[i,self.mask[i,:]] ,y[i,self.mask[i,:]] ,'bo',alpha=0.5)
    #     if y_err is not None:
    #         plt.errorbar(x[i,~self.mask[i,:]],y[i,~self.mask[i,:]],yerr=y_err[i,~self.mask[i,:]],fmt='.k',alpha=0.5)
    #     else:
    #         plt.plot(x[i,~self.mask[i,:]],y[i,~self.mask[i,:]],'.k',alpha=0.5)
    #
    # def plot_epoches(self,x,y,y_err=None,xlims=None):
    #
    #     size_x, size_y = wobble_model.getPlotSize(x.shape[0])
    #
    #     fig = plt.figure(figsize=[12.8,9.6])
    #     # plt.legend(['filtered','unfiltered masked','unfiltered unmasked'])
    #     # fig.axes[0].xaxis.set_visible(False)
    #     # fig.axes[0].yaxis.set_visible(False)
    #     plt.xlabel('x (log lambda)')
    #     plt.ylabel('y (log flux)')
    #     plt.title('gauss filtered lin interp corrected data w/ sigma {}'.format(self.sigma))
    #
    #     for i,x_row in enumerate(x):
    #         ax = fig.add_subplot(size_x,size_y,i+1)
    #         ax.axes.xaxis.set_visible(False)
    #         ax.axes.yaxis.set_visible(False)
    #         if xlims is not None:
    #             plt.xlim(np.log(xlims[0]),np.log(xlims[1]))
    #         plt.ylim(-1,0.25)
    #         if y_err is not None:
    #             plt.errorbar(x_row[~self.mask[i,:]],y[i,~self.mask[i,:]],yerr=y_err[i,~self.mask[i,:]],fmt='.k',alpha=0.5)
    #         else:
    #             plt.plot(x_row[~self.mask[i,:]],y[i,~self.mask[i,:]],'.k',alpha=0.5)
    #         plt.plot(x_row[self.mask[i,:]] ,y[i,self.mask[i,:]] ,'bo',alpha=0.5)

    def get_xy(self,subset=None):
        if self.filtered_flux is None:
            print('please filter data first')
            return
        if subset is None:
            y     = np.log(self.flux/self.filtered_flux)
            x     = np.log(self.lamb)
            y_err = np.log(self.ferr)
        else:
            start, end = subset
            # hackky and gross
            self.mask = self.mask[:,start:end]
            # print(start,end)
            y     = np.log(self.flux[:,start:end]/self.filtered_flux[:,start:end])
            x     = np.log(self.lamb[:,start:end])
            y_err = np.log(self.ferr[:,start:end])
        return jnp.array(x,dtype=np.float32), jnp.array(y,dtype=np.float32), jnp.array(y_err,dtype=np.float32)
