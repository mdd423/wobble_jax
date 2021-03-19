import numpy as np
import matplotlib.pyplot as plt
import astropy.table as at
import jax.numpy as jnp

import loss      as wobble_loss
import simulator as wobble_sim
import model     as wobble_model
import dataset   as wobble_data

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-l',action='store',default=0,type=int)
parser.add_argument('-r',action='store',default=200,type=int)
parser.add_argument('-n',action='store',default=256,type=int)
parser.add_argument('--sigma',action='store',default=80.0,type=float)
parser.add_argument('--maxiter',action='store',default=4,type=int)
args   = parser.parse_args()

# @profile
def main():
    tbl     = at.QTable.read('data/hat-p-20.fits')
    dataset = wobble_data.AstroDataset(tbl['flux'],tbl['wavelength'],tbl['mask'],tbl['flux_err'])
    dataset.interpolate_mask()
    dataset.gauss_filter(sigma=args.sigma)
    x, y, y_err = dataset.get_xy(subset=(args.l,args.r))

    x_shifts = wobble_data.getInitXShift(tbl['BJD'],'HAT-P-20','APO')

    loss_1 = wobble_model.LossFunc('L2Loss')
    model  = wobble_model.JnpLin(args.n,y,x,x_shifts)

    model.optimize(loss_1,maxiter=args.maxiter)
    model.plot()
    plt.savefig('out/hatp20jnpL{}_R{}_N{}f.png'.format(args.l,args.r,args.n))
    model.save_model('modeln{}_l{}_r{}_mI{}f.pt'.format(args.n,args.l,args.r,args.maxiter))

    # model2 = wobble_model.JnpLin(args.n,y,x,x_shifts)
    # model2.load_model('modeln{}_l{}_r{}_mI{}.pt'.format(args.n,args.l,args.r,args.maxiter))

    # plt.show()
if __name__ == '__main__':
    main()
