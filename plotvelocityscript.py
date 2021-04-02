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



def plot(loss_array,shifts,real_vels,epsilon):
    epoches = real_vels.shape[0]
    size_x, size_y = wobble_model.getPlotSize(epoches)

    fig,axs = plt.subplots(size_x,size_y,figsize=[12.8,9.6],sharey=False)
    # Once again we apply the shift to the xvalues of the model when we plot it
    for idx in range(epoches):
        i, j = (idx%size_x,idx//size_x)
        #ax.set_title('epoch %i: vel %.2f' % (i, model.shifted[i]))
        axs[i][j].get_yaxis().set_visible(False)
        for tick in axs[i][j].xaxis.get_major_ticks():
            tick.label.set_fontsize(6)
            tick.label.set_rotation('vertical')
        axs[i][j].plot(shifts,loss_array[idx,:],'.k',zorder=1,alpha=0.9,ms=6)
        minimum = min(loss_array[idx,:])
        maximum = max(loss_array[idx,:])
        #print(i, i % size_x, i // size_x)
        axs[i][j].set_xlim(real_vels[idx]-epsilon,real_vels[idx]+epsilon)
        axs[i][j].vlines(real_vels[idx],ymin=minimum,ymax=maximum)
        plt.ylim(minimum,maximum)

def getloss(shift_grid,model,loss):
    loss_arr = np.empty((model.epoches,shift_grid.shape[0]))
    for i in range(model.epoches):
        for j,shift in enumerate(shift_grid):
            loss_arr[i,j] = loss(model.params,model.ys[i,:],model.xs[i,:]+shift,None,model)
    return loss_arr

import numpy.polynomial as polynomial

def plotpoly(loss_array,shifts,deg,real_vels):
    epoches = real_vels.shape[0]
    size_x, size_y = wobble_model.getPlotSize(epoches)

    vel_min = np.empty(epoches)
    fig,axs = plt.subplots(size_x,size_y,figsize=[12.8,9.6],sharey=False)
    for n in range(epoches):
        
        i, j = (n%size_x,n//size_x)
        axs[i][j].get_yaxis().set_visible(False)
        for tick in axs[i][j].xaxis.get_major_ticks():
            tick.label.set_fontsize(6)
            tick.label.set_rotation('vertical')
            
        idx = loss_array[n,:].argmin()
        xs = shifts[idx-1:idx+2]
        ys = loss_array[n,idx-1:idx+2]
        
        xs2 = shifts[idx-2:idx+3]
        ys2 = loss_array[n,idx-2:idx+3]
        
        poly = np.polyfit(xs,ys,deg=2)
        deriv = np.polyder(poly)
        
        x_min = np.roots(deriv)
        x_min = x_min[x_min.imag==0].real
        y_min = np.polyval(poly,x_min)
        axs[i][j].plot(x_min,y_min,'o')
        axs[i][j].plot(xs,ys,'*k')
        xc = np.linspace(xs[0],xs[-1],1000)
        yc = np.polyval(poly,xc)
        axs[i][j].plot(xc,yc,'r',alpha=0.3)
        minimum = min(yc)
        maximum = max(yc)

        axs[i][j].vlines(real_vels[n],ymin=minimum,ymax=maximum)
        vel_min[n] = x_min[0]
    return vel_min
        
import pickle
def save_loss(filename,loss_array):
    with open(filename,'wb') as output:
        pickle.dump(loss_array,output,pickle.HIGHEST_PROTOCOL)

def load_loss(filename):
    with open(filename,'rb') as input:
        loss = pickle.load(input)
        return loss

def main():
    velocities = np.linspace(-300,300,100) * u.km/u.s
    shifts     = wobble_data.shifts(velocities)
    epsilon = 0.0001
    loss  = wobble_loss.L2Loss()
    model_name  = '/home/mdd423/wobble_jax/models/modeln7000_l0_r6000_mI32f.pt'
    
    model_tail = path.split(model_name)[1][:-3]
    loss_name = '/home/mdd423/wobble_jax/models/lossvelgrid{}.lg'.format(model_tail)
    #print(path.split(model_name))
    model = wobble_model.load_model(model_name)
    loss_array = load_loss(loss_name)#getloss(shifts,model,loss)
    #save_loss(loss_name,loss_array)
    vel_min = plotpoly(loss_array,shifts,deg=2,real_vels=model.shifted)
    plt.savefig('/home/mdd423/wobble_jax/out/polyplot_deg2{}.pdf'.format(model_tail))
    model_2_name = '/home/mdd423/wobble_jax/models/{}_rd2_mI{}.pt'.format(model_tail,maxiter)
    model_2 = wobble_model.load(model_2_name)
    #model_2 = wobble_model.JnpVelLin(num_params=None,x=None,y=None,pretrained=model,vel_shifts=vel_min)
    #maxiter= 4
    #model_2.optimize(loss,maxiter=maxiter)
    lb, ub = (9.65,9.655)
    model_2.plot(xlim=(lb,ub))
    plt.savefig('/home/mdd423/wobble_jax/out/hatp20min_{}_l{}_r{}.pdf'.format(model_tail,lb,ub))
    wobble_model.save_model(,model_2)

if __name__ == '__main__':
    main()
