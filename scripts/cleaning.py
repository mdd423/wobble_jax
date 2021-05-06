import numpy as np
import matplotlib.pyplot as plt
import astropy.table as at
import astropy.units as u
import astropy.coordinates as coord
import scipy.constants as const
from astropy.time import Time
import scipy.ndimage as ndimage

def dataInterp(masks,flux):
    new_flux = np.zeros(flux.shape)
    new_flux = flux
    for j,msk in enumerate(masks):
        cnt = 0

        for i, mask in enumerate(msk):
            if mask != 0:
                cnt += 1
            if mask == 0 and cnt != 0:
                new_flux[j,i-cnt:i] = np.linspace(flux[j,i-cnt-1],flux[j,i],cnt+2)[1:-1]
                cnt = 0

    return new_flux

def zplusone(vel):
    return np.sqrt((1 + vel/(const.c*u.m/u.s))/(1 - vel/(const.c*u.m/u.s)))

def plot_normed_fluxes(wavelengths,fluxes,flux_errors,maskes,gauss_datas,gauss_ratioes,sigma):
    size_x, size_y = wobble_model.getPlotSize(wavelengths.shape[0])

    fig = plt.figure(figsize=[12.8,9.6])
    plt.xlabel('wavelength (A)')
    plt.ylabel('flux ratio')
    plt.legend(['masked','unmasked'])
    plt.title('ratio of data w/ sigma {}'.format(sigma))

    for i,wavelength in enumerate(wavelengths):
        ax = fig.add_subplot(size_x,size_y,i+1)
        plt.ylim(0,1.2)
        # plt.figure(figsize=[12.8,9.6])
        plt.plot(wavelength,gauss_ratio[i,:],color='blue',alpha=0.5)

def plot_epoches(x,y,mask,sigma):

    size_x, size_y = wobble_model.getPlotSize(x.shape[0])

    fig = plt.figure(figsize=[12.8,9.6])
    # plt.legend(['filtered','unfiltered masked','unfiltered unmasked'])
    plt.xlabel('x (log lambda)')
    plt.ylabel('y (log flux)')
    plt.title('gauss filtered lin interp corrected data w/ sigma {}'.format(sigma))

    for i,x_row in enumerate(x):
        ax = fig.add_subplot(size_x,size_y,i+1)
        plt.plot(x_row[mask[i,:]] ,y[i,mask[i,:]] ,'bo',alpha=0.5)
        plt.plot(x_row[~mask[i,:]],y[i,~mask[i,:]],'.k',alpha=0.5)

def plot_epoches_flux(wavelengths,fluxes,flux_errors,maskes,gauss_datas,sigma):
    size_x, size_y = wobble_model.getPlotSize(wavelengths.shape[0])

    fig = plt.figure(figsize=[12.8,9.6])
    plt.legend(['filtered','unfiltered masked','unfiltered unmasked'])
    plt.xlabel('wavelength (A)')
    plt.ylabel('flux')
    plt.title('gauss filtered lin interp corrected data w/ sigma {}'.format(sigma))

    for i,wavelength in enumerate(wavelengths):
        ax = fig.add_subplot(size_x,size_y,i+1)
        # plt.figure(figsize=[12.8,9.6])
        plt.plot(wavelength                  ,gauss_datas[i,:]                 ,color='red',alpha=0.5)
        plt.errorbar(wavelength[~maskes[i,:]],fluxes[i,~maskes[i,:]],yerr=flux_errors[i,~maskes[i,:]],fmt='.k',alpha=0.5)
        plt.plot(wavelength[maskes[i,:]]     ,fluxes[i,maskes[i,:]] ,'bo',alpha=0.5)

import loss as wobble_loss
import simulator as wobble_sim
import model as wobble_model
import jax.numpy as jnp

if __name__ == '__main__':
    tbl = at.QTable.read('data/hat-p-20.fits')

    hatp20_c = coord.SkyCoord.from_name('HAT-P-20')
    loc      = coord.EarthLocation.of_site('APO')
    ts       = Time(tbl['BJD'], format='jd', scale='tdb')
    bc       = hatp20_c.radial_velocity_correction(obstime=ts, location=loc).to(u.km/u.s)
    x_shifts = np.log(zplusone(bc))

    flux = np.array(tbl['flux'])
    lamb = np.array(tbl['wavelength']/u.Angstrom)
    mask = np.array(tbl['mask'])
    ferr = np.array(tbl['flux_err'])

    flux_to_model = flux[:,1800:2000]
    lamb_to_model = lamb[:,1800:2000]
    mask_to_model = mask[:,1800:2000]
    ferr_to_model = ferr[:,1800:2000]

    plt.plot(x_shifts)
    plt.show()

    data_corrected = dataInterp(mask_to_model,flux_to_model)
    sigma = 80.
    gauss_data = ndimage.gaussian_filter1d(data_corrected,sigma)

    y     = np.log(flux_to_model/gauss_data)
    x     = np.log(lamb_to_model)
    y_err = np.log(ferr_to_model)

    mask  = jnp.array(mask_to_model)
    y     = jnp.array(y,dtype=np.float32)
    x     = jnp.array(x,dtype=np.float32)
    y_err = jnp.array(y_err,dtype=np.float32)

    plot_epoches(x,y,mask,sigma)
    # plot_normed_fluxes(x,y,y_err,mask,gauss_data,gauss_ratio,sigma)
    plt.show()

    # loss_1 = wobble_model.LossFunc('L2Loss')
    # loss_2 = 1 * wobble_model.LossFunc('L2Reg')
    # loss   = loss_1 + loss_2
    #
    # num_params = 80
    # str_model  = wobble_model.LinModel(num_params,flux_to_model,lamb_to_model,lamb_shift)
    # str_model.optimize(loss)
    # str_model.plot(fler_to_model)
    # plt.show()
