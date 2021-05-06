import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import astropy.table as at
import jax.numpy as jnp

import loss      as wobble_loss
import simulator as wobble_sim
import model     as wobble_model
import dataset   as wobble_data
import plottings as wobble_plot

import argparse
import os.path as path
import os

import astropy.units as u

parser = argparse.ArgumentParser()
parser.add_argument('-l',action='store',default=0,type=int)
parser.add_argument('-r',action='store',default=200,type=int)
parser.add_argument('-n',action='store',default=256,type=int)
parser.add_argument('--sigma',action='store',default=80.0,type=float)
parser.add_argument('--maxiter',action='store',default=4,type=int)
parser.add_argument('--maxiter2',action='store',default=4,type=int)
parser.add_argument('-f',action='store',required=True)
# parser.add_argument('--dir',action='store',required=True)
args   = parser.parse_args()

# @profile
def main():

    file_tail    = path.split(args.f)[1][:-4]
    model_name   = 'out/model{}n{}_l{}_r{}f_mI{}.pt'.format(file_tail,args.n,args.l,args.r,args.maxiter)
    model_2_name = 'out/model{}n{}_l{}_r{}f_mI{}_2mI{}.pt'.format(file_tail,args.n,args.l,args.r,args.maxiter,args.maxiter2)
    fig_1_name   = 'out/fig1n{}_l{}_r{}f_mI{}.png'.format(file_tail,args.n,args.l,args.r,args.maxiter)
    fig_2_name   = 'out/fig2n{}_l{}_r{}f_mI{}_2mI{}.png'.format(file_tail,args.n,args.l,args.r,args.maxiter,args.maxiter2)

    model_tail = path.split(model_name)[1][:-3]

    tbl         = at.QTable.read(args.f)
    dataset     = wobble_data.AstroDataset(tbl['flux'],tbl['wavelength']/u.Angstrom,tbl['mask'],tbl['flux_err'])
    dataset.interpolate_mask()
    filtered    = dataset.gauss_filter(sigma=args.sigma)
    x, y, y_err = dataset.get_xy(filtered)

    x_shifts = wobble_data.getInitXShift(tbl['BJD'],'HAT-P-20','APO')

    loss = wobble_loss.ChiSquare()

    x_grid = wobble_model.get_lin_spaced_grid(x[:,args.l:args.r],x_shifts,args.n)
    # y[args.l:args.r],x[args.l:args.r],y_err[args.l:args.r]
    model  = wobble_model.JnpLin(args.n,x_grid,x_shifts)
    res, callback = model.optimize(loss,x[:,args.l:args.r],y[:,args.l:args.r],args.maxiter,0,y_err[:,args.l:args.r])

    wobble_plot.plot_linear_model(model.x,model.params,model.shifted,x[:,args.l:args.r],y[:,args.l:args.r],y_err[:,args.l:args.r])
    plt.savefig(fig_1_name)
    wobble_model.save_model(model_name,model)

    velocity_grid = np.linspace(-300,300,100) * u.km/u.s
    shift_grid    = wobble_data.shifts(velocity_grid)
    loss_array    = wobble_data.get_loss_array(shift_grid,model,x[:,args.l:args.r],y[:,args.l:args.r],loss,y_err[:,args.l:args.r])
    x_min   = wobble_data.get_parabolic_min(loss_array,shift_grid)

    model_2 = wobble_model.JnpVelLin(args.n,x_grid,x_min,model.params)
    results = model_2.optimize(loss,x,y,args.maxiter2,0,y_err)

    # model_2.plot(xlim=(9.455,9.456))
    wobble_plot.plot_linear_model(model_2.x,model_2.params[:-model_2.epoches],model_2.params[-model_2.epoches:],x,y,y_err)
    plt.savefig(fig_2_name)
    wobble_model.save_model(model_2_name,model_2)

if __name__ == '__main__':
    main()
