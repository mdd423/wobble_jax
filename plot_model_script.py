import numpy as np
import matplotlib.pyplot as plt
import astropy.table as at
import astropy.units as u
import jax.numpy as jnp

import loss      as wobble_loss
import simulator as wobble_sim
import model     as wobble_model
import dataset   as wobble_data
import plottings as wobble_plot

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-l',action='store',default=0,type=int)
parser.add_argument('-r',action='store',default=200,type=int)
parser.add_argument('-n',action='store',default=256,type=int)
parser.add_argument('--sigma',action='store',default=80.0,type=float)
parser.add_argument('--maxiter',action='store',default=4,type=int)
parser.add_argument('--maxiter2',action='store',default=4,type=int)
args   = parser.parse_args()

# @profile
def main():
    xlim = (0,256)
    boundaries = (9.736,9.7365)
    fig1_name = "out/hatp20{}_{}_n{}_l{}_r{}_mI{}.png".format(xlim[0],xlim[1],args.n,args.l,args.r,args.maxiter)
    fig2_name = "out/hatp20{}_{}_n{}_l{}_r{}_mI{}_rd2mI{}.png".format(xlim[0],xlim[1],args.n,args.l,args.r,args.maxiter,args.maxiter2)

    tbl     = at.QTable.read('data/hat-p-20.fits')
    dataset = wobble_data.AstroDataset(tbl['flux'],tbl['wavelength'],tbl['mask'],tbl['flux_err'])
    dataset.interpolate_mask()
    filter      = dataset.gauss_filter(sigma=args.sigma)
    dataset.mask_flux_error()
    x, y, y_err = dataset.get_xy(filter,subset=(args.l,args.r))
    wobble_plot.plot_error(x,y_err**2)
    plt.show()

    x_shifts = wobble_data.getInitXShift(tbl['BJD'],'HAT-P-20','APO')

    loss = wobble_loss.ChiSquare()

    model  = wobble_model.JnpLinErr(args.n,y,x,y_err,x_shifts)
    model.optimize(loss,maxiter=args.maxiter)

    wobble_plot.plot_linear(model,model.params,model.shifted,noise=model.y_err,xlim=boundaries)
    plt.savefig(fig1_name)

    # wobble_modåel.save_model()

    velocity_grid = np.linspace(-300,300,10) * u.km/u.s
    shift_grid    = wobble_data.shifts(velocity_grid)
    loss_array    = wobble_data.get_loss_array(shift_grid,model,loss)

    # print(loss_array)
    wobble_plot.plot_loss_array(loss_array,shift_grid,x_shifts,epsilon=1e-3)
    plt.show()

    x_min   = wobble_data.get_parabolic_min(loss_array,shift_grid)

    model_2 = wobble_model.JnpVelLinErr(args.n,y,x,y_err,x_min,pretrained=model)
    results = model_2.optimize(loss,maxiter=args.maxiter2)

    epoches_to_plot = [0,2,57,58,59]
    wobble_plot.plot_linear(model_2,model_2.params[:-model.epoches],model_2.params[-model.epoches:],epoches_to_plot=epoches_to_plot,noise=model.y_err,xlim=boundaries)
    plt.savefig(fig2_name)
    # model_2.save_model()

if __name__ == '__main__':
    main()
