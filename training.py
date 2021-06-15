import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import astropy.table as at
import jax.numpy as jnp

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
parser.add_argument('-n',action='store',default=256,type=int)
parser.add_argument('--sigma',action='store',default=80.0,type=float)
parser.add_argument('--maxiter',action='store',default=4,type=int)
parser.add_argument('--maxiter2',action='store',default=4,type=int)
parser.add_argument('-d',action='store',default='out/',type=str)
args   = parser.parse_args()

def find_nearest(array,value):
    array = np.asarray(array)
    idx   = (np.abs(array-value)).argmin()
    return idx

# @profile
def main():
    allvisits_True = True
    filename  = 'data/2M14371943+5754143.fits'
    star_name = '2MASS J14371943+5754143'
    obs_name  = 'APO'
    # size of grid search, and velocity +/- to extend in this search in km/s
    vel_width = 300 * u.km/u.s
    resolving_constant = 40000 # c/resolving_constant is the step size to grid search at midpoint

    file_tail     = path.split(filename)[1][:-4]
    outname = '{}_n{}_l{}_r{}_mI{}_2mI{}_s{}_rs{}'.format(file_tail,args.n,args.l,args.r,args.maxiter,args.maxiter2,vel_width.value,resolving_constant)
    outdir = args.d
    print(args.d)
    os.mkdir(args.d)
    
    loss_name     = path.join(outdir,'loss{}n{}_l{}_r{}_mI{}_chi_s{}_rs{}.pt'.format(file_tail,args.n,args.l,args.r,args.maxiter,vel_width.value,resolving_constant))
    x_name        = path.join(outdir,'xs{}n{}_l{}_r{}_mI{}_chi_s{}_rs{}.xs'.format(file_tail,args.n,args.l,args.r,args.maxiter,vel_width.value,resolving_constant))
    model_name    = path.join(outdir,'model{}n{}_l{}_r{}_mI{}_chi_s{}_rs{}.pt'.format(file_tail,args.n,args.l,args.r,args.maxiter,vel_width.value,resolving_constant))
    model_2_name = path.join(outdir,'model{}n{}_l{}_r{}_mI{}_2mI{}_chi_s{}_rs{}.pt'.format(file_tail,args.n,args.l,args.r,args.maxiter,args.maxiter2,vel_width.value,resolving_constant))
    
    model_tail = path.split(model_name)[1][:-3]

    # Read-In Data, and Clean it up
    ############################################################################################
    
    tbl         = at.QTable.read(filename)
    dataset     = wobble_data.AstroDataset(tbl['flux'],tbl['wavelength'],tbl['mask'],tbl['flux_err'])
    dataset.interpolate_mask()
    filtered    = dataset.get_gauss_filter(sigma=args.sigma)
    x, y, y_err = dataset.get_xy(filtered)

    #y, y_err    = dataset.set_masked_equal_to(y,y_err,0.0,10.0)

    if allvisits_True:
        
        visits  = at.QTable.read('data/allvisits-2M14371943+5754143.fits')
        indices = []
        for value in tbl['BJD']:
            indices.append(find_nearest(visits['JD'],value))
        x_shifts = wobble_data.shifts( visits['VHELIO'][indices]*u.km/u.s + wobble_data.get_star_velocity(tbl['BJD'],star_name,obs_name))

    else:
        x_shifts   = wobble_data.getInitXShift(tbl['BJD'],star_name,obs_name)
    init_vels  = wobble_data.velocityfromshift(x_shifts)
    
    loss_1 = wobble_loss.ChiSquare()
    
    # First Round of Training
    #############################################################################################)
    epoch_mask = np.arange(0,x.shape[0],dtype=int)
    epoch_mask  = np.delete(epoch_mask,4)
    
    
    x_grid = wobble_model.get_lin_spaced_grid(x[:,args.l:args.r],x_shifts,args.n)
    model  = wobble_model.JaxLinear(args.n,x_grid,x_shifts)
    res, callback = model.optimize(loss_1,x[:,args.l:args.r],y[:,args.l:args.r],y_err[:,args.l:args.r],args.maxiter)

    wobble_model.save(model_name,model)

    # Grid Searching using Trained Model
    ############################################################################################
    velocity_kern = np.arange(-vel_width.to(u.km/u.s).value,vel_width.to(u.km/u.s).value,const.c.to(u.km/u.s).value/resolving_constant)*u.km/u.s
    velocity_grid = np.add.outer(init_vels,velocity_kern)
    
    shift_grid    = wobble_data.shifts(velocity_grid)
    loss_array    = wobble_data.get_loss_array(shift_grid,model,x[:,args.l:args.r],y[:,args.l:args.r],y_err[:,args.l:args.r],loss_1)
    stacked = np.stack((shift_grid,loss_array))
    wobble_model.save(loss_name,stacked)
    x_min   = wobble_data.get_parabolic_min(loss_array,shift_grid)
    wobble_model.save(x_name,x_min)

    # Second Round of Training
    ############################################################################################
    model_2 = wobble_model.JaxVelLinear(args.n,x_grid,x_min,model.params)
    results = model_2.optimize(loss_1,x[:,args.l:args.r],y[:,args.l:args.r],y_err[:,args.l:args.r],args.maxiter2)
    wobble_model.save(model_2_name,model_2)

if __name__ == '__main__':
    main()
