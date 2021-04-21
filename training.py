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

parser = argparse.ArgumentParser()
parser.add_argument('-l',action='store',default=0,type=int)
parser.add_argument('-r',action='store',default=200,type=int)
parser.add_argument('-n',action='store',default=256,type=int)
parser.add_argument('--sigma',action='store',default=80.0,type=float)
parser.add_argument('--maxiter',action='store',default=4,type=int)
parser.add_argument('-f',action='store',required=True)
parser.add_argument('--dir',action='store',required=True)
args   = parser.parse_args()

# @profile
def main():
    outdir = args.dir
    os.mkdir(outdir)
    
    file_tail = path.split(args.f)[1][:-4]
    model_name = '/home/mdd423/wobble_jax/out/model{}n{}_l{}_r{}f.pt'.format(file_tail,args.n,args.l,args.r)
    model_tail = path.split(model_name)[1][:-3]
    
    tbl     = at.QTable.read(args.f)
    dataset = wobble_data.AstroDataset(tbl['flux'],tbl['wavelength'],tbl['mask'],tbl['flux_err'])
    dataset.interpolate_mask()
    dataset.gauss_filter(sigma=args.sigma)
    x, y, y_err = dataset.get_xy(subset=(args.l,args.r))

    x_shifts = wobble_data.getInitXShift(tbl['BJD'],'HAT-P-20','APO')

    loss = wobble_loss.L2Loss()
    
    model  = wobble_model.JnpLin(args.n,y,x,x_shifts)
    model.optimize(loss,maxiter=args.maxiter)
    
    model.plot(xlim=(9.455,9.456))
    plt.savefig(outdir + )
    
    wobble_model.save_model('/home/mdd423/wobble_jax/models/model{}n{}_l{}_r{}_mI{}f.pt'.format(file_tail,args.n,args.l,args.r,args.maxiter),model)

    velocity_grid = np.linspace(-300,300,100) * u.km/u.s
    shift_grid    = wobble_data.shifts(velocity_grid)
    loss_array    = wobble_plot.get_loss_array(velocity_grid,model,loss)

    x_min   = wobble_plot.get_parabolic_min(loss_array,shift_grid)
    
    model_2 = wobble_model.JnpVelLin(args.c,y,x,x_min,pretrained=model)
    results = model_2.optimize(loss,maxiter=4)
    
    model_2.plot(xlim=(9.455,9.456))
    plt.savefig('/home/mdd423/wobble_jax/out/{}')
    model_2.save_model(model_2_name)

    
    
if __name__ == '__main__':
    main()
