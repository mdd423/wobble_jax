import numpy as np
import matplotlib; #matplotlib.use("Agg")
import matplotlib.pyplot as plt
import astropy.table as at
import jax.numpy as jnp
import pickle

import model     as wobble_model
import dataset   as wobble_data

import scipy.constants as const
import astropy.units as u
import os.path as path

def save(filename,array):
    with open(filename,'wb') as output:
        pickle.dump(array,output,pickle.HIGHEST_PROTOCOL)

def load(filename):
    with open(filename,'rb') as input:
        loss = pickle.load(input)
        return loss

def getDivisor(n):
    # print(n)
    array = np.arange(int(np.sqrt(n)),0,-1,dtype=int)
    for x in array:
        if n % x == 0:
            break
    # print(x,n//x)
    return x, n//x

def getPlotSize(epoches):
    size_x = np.floor(np.sqrt(epoches))
    size_y = epoches//size_x
    while epoches % size_y != 0:
        size_y = epoches//size_x
        size_x -= 1
    else:
        size_x += 1
    size_x = int(size_x)
    size_y = int(size_y)
    return size_x, size_y

def plot_error(x,y_err,mask):
    size_x, size_y = getDivisor(y_err.shape[0])
    epoches = range(y_err.shape[0])
    fig,axs = plt.subplots(size_x,size_y,figsize=[12.8,9.6],sharey=False)

    for iteration,n in enumerate(epoches):

        i, j = (iteration%size_x,iteration//size_x)
        for tick in axs[i][j].xaxis.get_major_ticks():
            tick.label.set_fontsize(6)
            tick.label.set_rotation('vertical')
        # x,y = self.plot_model(n)
        axs[i][j].plot(x[n,~mask[n,:]],y_err[n,~mask[n,:]],'.k',zorder=2,alpha=0.7,ms=6)
        axs[i][j].plot(x[n,mask[n,:]],y_err[n,mask[n,:]],'.r',zorder=2,alpha=0.7,ms=6)
        axs[i][j].set_ylim([0,5e-2])

def plot_loss_array(loss_array,shifts,real_vels,epsilon):
    epoches = real_vels.shape[0]
    size_x, size_y = getPlotSize(epoches)

    fig,axs = plt.subplots(size_x,size_y,figsize=[12.8,9.6],sharey=False)
    # Once again we apply the shift to the xvalues of the model when we plot it
    for idx in range(epoches):
        i, j = (idx%size_x,idx//size_x)
        #ax.set_title('epoch %i: vel %.2f' % (i, model.shifted[i]))
        axs[i][j].get_yaxis().set_visible(False)
        for tick in axs[i][j].xaxis.get_major_ticks():
            tick.label.set_fontsize(6)
            tick.label.set_rotation('vertical')
        axs[i][j].plot(shifts,loss_array[idx,:],'*r',zorder=1,alpha=0.9,ms=6)
        minimum = min(loss_array[idx,:])
        maximum = max(loss_array[idx,:])
        #print(i, i % size_x, i // size_x)
        axs[i][j].set_xlim(real_vels[idx]-epsilon,real_vels[idx]+epsilon)
        axs[i][j].vlines(real_vels[idx],ymin=minimum,ymax=maximum)
        plt.ylim(minimum,maximum)

def plot_parabola(loss_array,grid,real_vels,second_velocity=None):
    epoches = real_vels.shape[0]
    size_x, size_y = getPlotSize(epoches)


    vel_min,xs,ys,poly = get_parabolic_min(loss_array,grid,epoches,return_all=True)
    fig,axs = plt.subplots(size_x,size_y,figsize=[12.8,9.6],sharey=False)
    for n in range(epoches):

        i, j = (n%size_x,n//size_x)
        axs[i][j].get_yaxis().set_visible(False)
        for tick in axs[i][j].xaxis.get_major_ticks():
            tick.label.set_fontsize(6)
            tick.label.set_rotation('vertical')

        y_min = np.polyval(poly,vel_min)
        axs[i][j].plot(vel_min,y_min,'o')
        axs[i][j].plot(xs,ys,'*k')
        xc = np.linspace(xs[0],xs[-1],1000)
        yc = np.polyval(poly,xc)
        axs[i][j].plot(xc,yc,'r',alpha=0.3)
        minimum = min(yc)
        maximum = max(yc)

        if second_velocity is not None:
            axs[i][j].vlines(second_velocity[n],ymin=minimum,ymax=maximum,colors='red')

        axs[i][j].vlines(real_vels[n],ymin=minimum,ymax=maximum,label='blue')
        vel_min[n] = x_min[0]
    return vel_min

def plot_RV_time(velocity_array,dates):
    iter_size = velocity_array.shape[1]
    size_x, size_y = getPlotSize(iter_size)

    #print(iter_size,size_x,size_y)
    fig,axs = plt.subplots(size_x,size_y,figsize=[12.8,9.6],sharey=False)
    for n in range(iter_size):
        i, j = (n%size_x,n//size_x)
        print(n,i,j)
        axs[i][j].get_yaxis().set_visible(False)
        for tick in axs[i][j].xaxis.get_major_ticks():
            tick.label.set_fontsize(6)
            tick.label.set_rotation('vertical')

        axs[i][j].plot(dates,velocity_array[:,n],'*b')

def plot_linear(model,params,shifts,noise=None,xlim=None,epoches_to_plot=None):
    if epoches_to_plot is None:
        epoches = np.arange(model.epoches,dtype=int)
    else:
        epoches = np.array(epoches_to_plot)
    size_x, size_y = getDivisor(epoches.shape[0])

    fig,axs = plt.subplots(size_x,size_y,figsize=[12.8,9.6],sharey=False)
    if len(axs.shape) == 1:
        axs = np.expand_dims(axs,axis=0)
    print(axs.shape)
    # Once again we apply the shift to the xvalues of the model when we plot it
    for iteration,n in enumerate(epoches):

        i, j = (iteration%size_x,iteration//size_x)
        #ax.set_title('epoch %i: vel %.2f' % (i, self.shifted[i]))

        for tick in axs[i][j].xaxis.get_major_ticks():
            tick.label.set_fontsize(6)
            tick.label.set_rotation('vertical')
        # x,y = self.plot_model(n)
        axs[i][j].plot(model.x-shifts[n],params,'.r',linestyle='solid',linewidth=.8,zorder=2,alpha=0.5,ms=6)
        if noise is not None:
            axs[i][j].errorbar(model.xs[n,:],model.ys[n,:],yerr=noise[n,:],fmt='.k',zorder=1,alpha=0.9,ms=6)
        else:
            axs[i][j].plot(model.xs[n,:],model.ys[n,:],'.k',zorder=1,alpha=0.9,ms=6)
        if xlim is None:
            axs[i][j].set_xlim(min(model.xs[n,:]),max(model.xs[n,:]))
        else:
            axs[i][j].set_xlim(xlim[0],xlim[1])
        axs[i][j].set_ylim(-0.8,0.2)

def plot_data(lamb,flux,error,filtered=None,xinds=(3200,3300),ypadding=0.1):
    size_x, size_y = getPlotSize(lamb.shape[0])

    fig,axs = plt.subplots(size_x,size_y,figsize=[12.8,9.6],sharey=False)
    if len(axs.shape) == 1:
        axs = np.expand_dims(axs,axis=0)
    # fig = plt.figure(figsize=[12.8,9.6])
    # plt.legend(['filtered','unfiltered masked','unfiltered unmasked'])
    # plt.xlabel('wavelength (A)')
    # plt.ylabel('flux')
    # plt.title('gauss filtered lin interp corrected data w/ sigma {}'.format(self.sigma))

    xmin = lamb[0,xinds[0]]
    xmax = lamb[0,xinds[1]]


    for iteration,wavelength in enumerate(lamb):

        i, j = (iteration%size_x,iteration//size_x)
        #ax.set_title('epoch %i: vel %.2f' % (i, self.shifted[i]))

        for tick in axs[i][j].xaxis.get_major_ticks():
            tick.label.set_fontsize(6)
            tick.label.set_rotation('vertical')

        # ax = fig.add_subplot(size_x,size_y,i+1)
        # if xlims is not None:
        ymin = np.amin(flux[iteration,:])
        ymax = np.amax(flux[iteration,:])
        axs[i][j].set_xlim(xmin,xmax)
        axs[i][j].set_ylim(ymin-ypadding,ymax+ypadding)
        # if ylims is not None:
        #     plt.ylim(100,2500)
        # if dataset.filtered_flux is not None:
        #     plt.plot(wavelength,dataset.filtered_flux[i,:],color='red',alpha=0.5)
        axs[i][j].errorbar(wavelength,flux[iteration,:],error[iteration,:],fmt='.k',alpha=0.5)
        if filtered is not None:
            axs[i][j].plot(wavelength,filtered[iteration,:] ,color='red',alpha=0.5)
