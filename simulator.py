import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt
import astropy.units as u
import dataset as wobble_data
# import model as wobble_model

import scipy.signal as signal
import numpy.random as random

def average_difference(x,axis):

    return np.mean([t - s for s, t in zip(x, x[1:])])

def get_mus_stds_hgts(n,xmin,xmax,width,ymin=0.6,ymax=1.0):
    mus  = (random.rand(n)*(xmax-xmin)) + xmin
    stds = (random.rand(n))*width
    hgts = (random.rand(n)*(ymax-ymin)) + ymin
    return mus, stds, hgts

def create_freq_dist(mus,stds,heights,x):
    y = np.zeros(x.shape)
    for i in range(len(mus)):
        for j in range(x.shape[0]):
            y[j,:] -= heights[i] * np.exp(-np.power(x[j,:] - mus[i], 2.) / (2 * np.power(stds[i], 2.)))
    return y

def generate_shift_grid(x,epoches):
    # first index of x should be epoches
    pixel_difference = average_difference(x[0,:])
    dels = pixel_difference * random.rand(epoches)
    for i in range(epoches):
        x[i,:] += dels[i]
    return x, dels

def generate_gas_cell(x,n_lines):
    width = average_difference(x[0,:])

    xmin, xmax = x[0,0], x[0,-1]
    mus,stds,heights = get_mus_stds_hgts(n_lines,xmin,xmax,width)
    y = create_freq_dist(mus,stds,heights,x)
    return y, mus

def generate_telluric(x,n,epoches):
    airmass = random.rand(epoches)

    xmin, xmax = x[0,0], x[0,-1]
    mus,stds,heights = get_mus_stds_hgts(n_lines,xmin,xmax,width)
    y = create_freq_dist(mus,stds,heights,x)
    for i,a in enumerate(airmass):
        y[i,:] *= a
    return y, mus, airmass

def generate_stellar(x,n,epoches,vel_width=30*u.km/u.s):
    vels = rand.random(epoches) * vel_width
    deltas = wobble_data.shifts(vels)
    for delta in deltas:
        x[i,:] += delta

    xmin, xmax = x[0,0], x[0,-1]
    mus,stds,heights = get_mus_stds_hgts(n_lines,xmin,xmax,width)
    y = create_freq_dist(mus,stds,heights,x)

    return y, deltas, mus

def generate_noise(epoches,size,scale=1.0):
    return random.normal(scale=scale,size=(epoches,size))

def convolve(y,psf=None,n=5):
    if psf is None:
        psf = np.ones(n)
    else:
        n = psf.shape[0]
    y = signal.convolve(y,psf,mode='same')
    return y, psf

class GasCellEnv:
    def __init__(self,n_lines,lines,widths):
        return 0.0

class FreqEnv:
    def __init__(self,minimum,maximum,epoches=16,num_peaks=4,num_tells=2,size=128,noise=0.015,vel_scale=1.0,std_atm_scale=1.,hgt_atm_scale=0.5,std_str_scale=1.,hgt_str_scale=0.5):
        self.vel_scale = vel_scale
        self.noise = noise

        self.epoches = epoches

        self.size = size

        self.lambdas = np.linspace(minimum,maximum,self.size)
        # spectra of sky
        self.mu_atm     = ((maximum - minimum) * np.random.rand(num_tells)) - minimum
        self.std_atm    = (2 * (maximum - minimum)/size) * np.ones(num_tells)#1/resolution * np.ones(num_tells) #std_atm_scale * np.random.rand(num_tells)
        self.height_atm = hgt_atm_scale * np.random.rand(num_tells)
        # spectra of star
        self.mus     = ((maximum - minimum) * np.random.rand(num_peaks)) - minimum
        self.stds    = (2 * (maximum - minimum)/size) * np.ones(num_peaks)#1/resolution * np.ones(num_peaks) #std_str_scale * np.random.rand(num_peaks)
        self.heights = hgt_str_scale * np.random.rand(num_peaks)
        # true shift of each epoch
        self.real_vel = np.array([0.0])
        # velocities will range from -vel_scale to +vel_scale
        self.real_vel = jnp.append(self.real_vel,self.generate_new_vel(epoches-1))

    def get_stellar_flux(self):
        # initialize flux array and fill
        fluxes = np.ones(self.size)
        fluxes -= create_freq_dist(self.mus,self.stds,self.heights,self.lambdas)
        return fluxes

    def get_telluric_flux(self):
        # initialize flux array and fill
        fluxes = np.ones(self.size)
        fluxes -= create_freq_dist(self.mu_atm,self.std_atm,self.height_atm,self.lambdas)
        return fluxes

    def get_flux(self):
        # initialize flux array and fill
        fluxes = np.ones([self.epoches,self.size])
        # Shifted Epoches
        for i,vel in enumerate(self.real_vel):
            # create star flux THIS PART GETS SHIFTED
            fluxes[i,:], dummy = self.generate_new_epoch(vel=vel)
        return fluxes

    def plot_stellar_specter(self,model=None,log_space=False):
        if model is not None:
            plt.plot(model.x,model.y,'.r',linestyle='solid',linewidth=.8,zorder=2,alpha=0.5,ms=6)
        if log_space:
            plt.plot(self.lambdas,np.log(self.get_stellar_flux()))
        else:
            plt.plot(self.lambdas,self.get_stellar_flux())

    def generate_new_vel(self,size):
        return self.vel_scale * (2 * np.random.random(size) - 1)

    def generate_new_epoch(self,vel=None):
        if vel is None:
            new_vel = self.generate_new_vel(1)[0]
        else:
            new_vel = vel
        flux    = jnp.ones(self.size)
        flux   -= create_freq_dist(self.mus,self.stds,self.heights,self.lambdas + new_vel)
        # add atmosphere flux
        flux   -= create_freq_dist(self.mu_atm,self.std_atm,self.height_atm,self.lambdas)
        # add noise
        if self.noise is not 0:
            flux += np.random.normal(0.0,scale=self.noise,size=self.size)
        return flux, new_vel
