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
    fig1_name = "out/hatp20{}_{}_n{}_l{}_r{}_mI{}.png".format(xlim[0],xlim[1],args.n,args.l,args.r,args.maxiter)
    fig2_name = "out/hatp20{}_{}_n{}_l{}_r{}_mI{}_rd2mI{}.png".format(xlim[0],xlim[1],args.n,args.l,args.r,args.maxiter,args.maxiter2)

    tbl     = at.QTable.read('data/hat-p-20.fits')
    dataset = wobble_data.AstroDataset(tbl['flux'],tbl['wavelength'],tbl['mask'],tbl['flux_err'])
    dataset.interpolate_mask()
    dataset.gauss_filter(sigma=args.sigma)
    x, y, y_err = dataset.get_xy(subset=(args.l,args.r))

    x_shifts = wobble_data.getInitXShift(tbl['BJD'],'HAT-P-20','APO')

    loss = wobble_loss.L2Loss()

    model  = wobble_model.JnpLin(args.n,y,x,x_shifts)
    model.optimize(loss,maxiter=args.maxiter)

    wobble_plot.plot_linear(model,model.params,model.shifted)
    plt.savefig(fig1_name)
    # wobble_modåel.save_model()

    velocity_grid = np.linspace(-300,300,100) * u.km/u.s
    shift_grid    = wobble_data.shifts(velocity_grid)
    loss_array    = wobble_data.get_loss_array(velocity_grid,model,loss)

    x_min   = wobble_data.get_parabolic_min(loss_array,shift_grid)

    model_2 = wobble_model.JnpVelLin(args.n,y,x,x_min,pretrained=model)
    results = model_2.optimize(loss,maxiter=args.maxiter2)

    wobble_plot.plot_linear(model_2,model_2.params[:-model.epoches],model_2.params[-model.epoches:])
    plt.savefig(fig2_name)
    # model_2.save_model()

if __name__ == '__main__':
    main()
