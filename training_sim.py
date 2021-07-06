import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import astropy.table as at
import jax.numpy as jnp
import h5py

import loss      as wobble_loss
import simulator as wobble_sim
import model     as wobble_model
import dataset   as wobble_data
#import plottings as wobble_plot

import argparse
import os.path as path
import os

import astropy.units as u
import astropy.constants as const

parser = argparse.ArgumentParser()
parser.add_argument('-l',action='store',default=0,type=int)
parser.add_argument('-r',action='store',default=200,type=int)
# parser.add_argument('-n',action='store',default=256,type=int)
parser.add_argument('--sigma',action='store',default=80.0,type=float)
parser.add_argument('--maxiter',action='store',default=4,type=int)
parser.add_argument('--maxiter2',action='store',default=4,type=int)
parser.add_argument('-d',action='store',default='out/',type=str)
args   = parser.parse_args()

# @profile
def main():
    allvisits_True = True
    filename  = 'data/sim_hr100000_lr20000_x9.53-9.82_s20.0_e30_sn100_tn100_gn0_y0.0-0.9_v200.0_ep0.0_g1.0_w0.0.h5'
    # size of grid search, and velocity +/- to extend in this search in km/s
    vel_width = 300 * u.km/u.s
    vel_padding = 300 * u.km/u.s
    resolving_constant = 20000 # c/resolving_constant is the step size to grid search at midpoint

    file_tail     = path.split(filename)[1][:-4]
    outname = '{}_l{}_r{}_mI{}_2mI{}_s{}_rs{}'.format(file_tail,args.l,args.r,args.maxiter,args.maxiter2,vel_width.value,resolving_constant)
    outdir = args.d
    # os.mkdir(args.d,exist_ok=True)

    loss_name     = path.join(outdir,'loss{}_l{}_r{}_mI{}_chi_s{}_rs{}.pt'.format(file_tail,args.l,args.r,args.maxiter,vel_width.value,resolving_constant))
    x_name        = path.join(outdir,'xs{}_l{}_r{}_mI{}_chi_s{}_rs{}.xs'.format(file_tail,args.l,args.r,args.maxiter,vel_width.value,resolving_constant))
    model_name    = path.join(outdir,'model{}_l{}_r{}_mI{}_chi_s{}_rs{}.pt'.format(file_tail,args.l,args.r,args.maxiter,vel_width.value,resolving_constant))
    model_tell_name = path.join(outdir,'model{}_l{}_r{}_mI{}_chi_s{}_rs{}.pt'.format(file_tail,args.l,args.r,args.maxiter,args.maxiter2,vel_width.value,resolving_constant))

    model_tail = path.split(model_name)[1][:-3]

    # Read-In Data, and Clean it up
    ############################################################################################
    df = h5py.File(filename,"r")
    x = np.log(jnp.array(df["samples"]["wavelength"]),dtype=np.float32)
    y = np.log(jnp.array(df["samples"]["flux"]),dtype=np.float32)
    y_err = jnp.array(np.array(df["samples"]["flux_error"])/np.array(df["samples"]["flux"]),dtype=np.float32)
    epoches = np.array(df["constants"]["delta"]).shape
    x = np.expand_dims(x,axis=0)
    x = np.repeat(x,repeats=epoches,axis=0)

    loss = wobble_loss.ChiSquare()

    # First Round of Training
    #############################################################################################)
    x_grid = wobble_model.get_lin_spaced_grid(x[:,args.l:args.r],padding=wobble_data.shifts(vel_padding),step=wobble_data.shifts(const.c/resolving_constant))

    x_shifts = np.array(df["constants"]["delta"])
    model  = wobble_model.JaxLinear(x_grid,x_shifts)
    maxiter = 4
    res, callback = model.optimize(loss,x[:,args.l:args.r],y[:,args.l:args.r],y_err[:,args.l:args.r],maxiter)
    # wobble_model.save(model_name,model)

    maxiter = 4
    airmass = np.array(df["constants"]["airmass"])
    model_tell = wobble_model.TelluricModel(x_grid,airmass)
    res, callback = model_tell.optimize(loss,x[:,args.l:args.r],y[:,args.l:args.r],y_err[:,args.l:args.r],maxiter)
    # wobble_model.save(model_tell_name,model_tell)

    maxiter = 8
    model_tot = model + model_tell
    res, callback = model_tot.optimize(loss,x[:,args.l:args.r],y[:,args.l:args.r],y_err[:,args.l:args.r],maxiter)
    wobble_model.save(model_name,model_tot)

    # Grid Searching using Trained Model
    ############################################################################################
    # velocity_kern = np.arange(-vel_width.to(u.km/u.s).value,vel_width.to(u.km/u.s).value,const.c.to(u.km/u.s).value/resolving_constant)*u.km/u.s
    # velocity_grid = np.add.outer(init_vels,velocity_kern)
    #
    # shift_grid    = wobble_data.shifts(velocity_grid)
    # loss_array    = wobble_data.get_loss_array(shift_grid,model,x[:,args.l:args.r],y[:,args.l:args.r],y_err[:,args.l:args.r],loss)
    # stacked = np.stack((shift_grid,loss_array))
    # wobble_model.save(loss_name,stacked)
    # x_min   = wobble_data.get_parabolic_min(loss_array,shift_grid)
    # wobble_model.save(x_name,x_min)

    # Second Round of Training
    ############################################################################################
    # model_2 = wobble_model.JaxVelLinear(args.n,x_grid,x_min,model.p)
    # model_2 += model_tell
    # results = model_2.optimize(loss,x[:,args.l:args.r],y[:,args.l:args.r],y_err[:,args.l:args.r],args.maxiter2)
    # wobble_model.save(model_2_name,model_2)

if __name__ == '__main__':
    main()
