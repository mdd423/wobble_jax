import numpy as np
import matplotlib.pyplot as plt
import astropy.table as at
import jax.numpy as jnp

import loss      as wobble_loss
import simulator as wobble_sim
import model     as wobble_model
import dataset   as wobble_data

import scipy.constants as const
import astropy.units as u

def plot_model(model,i):
    plt.plot(model.x - model.shifted[i],model.params,'.r',linestyle='solid',linewidth=.8,zorder=2,alpha=0.5,ms=6)


def plot(loss_array,shifts,real_vels):
    size_x, size_y = wobble_data.getPlotSize(model.epoches)

    fig = plt.figure(figsize=[12.8,9.6])
    # Once again we apply the shift to the xvalues of the model when we plot it
    for i in range(model.epoches):
        ax = fig.add_subplot(size_x,size_y,i+1)
        ax.set_title('epoch %i: vel %.2f' % (i, model.shifted[i]))

        # plot_model(model,i)
        plt.plot(shifts,loss_array[i,:],'.k',zorder=1,alpha=0.9,ms=6)
        plt.vlines(real_vel[i])
        if xlim is None:
            plt.xlim(min(model.xs[i,:]),max(model.xs[i,:]))
        else:
            plt.xlim(xlim[0],xlim[1])
        plt.ylim(-0.8,0.2)

def plotloss(shifts,model,loss):
    loss_arr = np.empty((model.epoches,shifts.shape[0]))
    for i in range(model.epoches):
        for j in range(shifts.shape[0]):
            loss_arr[i,j] = loss(model.params,model.ys[i,:],model.xs[i,:]+shifts[j],None,model)
    return loss_arr


def main():
    velocities = np.linspace(-const.c/10,const.c/10,4500)/1e3 * u.km/u.s
    shifts     = wobble_data.shifts(velocities)

    loss  = wobble_loss.L2Loss()
    model = wobble_model.load_model('models/modeln512_l0_r400_mI4f.pt')
    loss_array = plotloss(shifts,model,loss)
    plot(loss_array,shifts,model.shifted)
    plt.show()

if __name__ == '__main__':
    main()
