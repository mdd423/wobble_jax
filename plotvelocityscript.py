import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import astropy.table as at
import jax.numpy as jnp

import loss      as wobble_loss
import simulator as wobble_sim
import model     as wobble_model
import dataset   as wobble_data

import scipy.constants as const
import astropy.units as u
import os.path as path

def plot(loss_array,shifts,real_vels,model):
    epoches = real_vels.shape[0]
    size_x, size_y = wobble_model.getPlotSize(epoches)

    fig = plt.figure(figsize=[12.8,9.6])
    # Once again we apply the shift to the xvalues of the model when we plot it
    for i in range(epoches):
        ax = fig.add_subplot(size_x,size_y,i+1)
        ax.set_title('epoch %i: vel %.2f' % (i, model.shifted[i]))

        plt.plot(shifts,loss_array[i,:],'.k',zorder=1,alpha=0.9,ms=6)
        plt.vlines(real_vels[i],ymin=-1000,ymax=1000)
        plt.ylim(min(loss_array[0,:]),max(loss_array[0,:]))

def getloss(shift_grid,model,loss):
    loss_arr = np.empty((model.epoches,shift_grid.shape[0]))
    for i in range(model.epoches):
        for j,shift in enumerate(shift_grid):
            loss_arr[i,j] = loss(model.params,model.ys[i,:],model.xs[i,:]+shift,None,model)
    return loss_arr


def main():
    velocities = np.linspace(-300,300,100) * u.km/u.s
    shifts     = wobble_data.shifts(velocities)

    loss  = wobble_loss.L2Loss()
    model_name  = '/home/mdd423/wobble_jax/models/modeln7000_l0_r6000_mI32f.pt'
    model_tail = path.split(model_name)[1]
    print(path.split(model_name))
    model = wobble_model.load_model(model_name)
    loss_array = getloss(shifts,model,loss)
    plot(loss_array,shifts,model.shifted,model)
    plt.savefig('/home/mdd423/wobble_jax/out/velvslossplot_{}.pdf'.format(model_tail))

if __name__ == '__main__':
    main()
