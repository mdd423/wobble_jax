import numpy as np
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import scipy.optimize
import sys

import pickle

import simulator as wobble_sim
import loss as wobble_loss

def getCellArray(x,xs):

    if xs[0]  < x[0]:
        print('error xs datapoints do not fit within the model')
        return None
    if xs[-1] > x[-1]:
        print('error xs datapoints do not fit within the model')
        return None

    cell_array = np.zeros(len(xs),dtype=int)
    j     = 1
    x_val = x[j]
    for i, xss in enumerate(xs):
        while x_val < xss:
            j    += 1
            x_val = x[j]
        cell_array[i] = int(j)
    return cell_array

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

def save_model(filename,model):
    with open(filename, 'wb') as output:  # Overwrites any existing file.
        pickle.dump(model, output, pickle.HIGHEST_PROTOCOL)

def load_model(filename):
    with open(filename, 'rb') as input:  # Overwrites any existing file.
        model = pickle.load(input)
        return model

    # you need to generalize this to whatever function occurs in the forward pass
    # like you said you shouldn't have to define this function two places
    # one for parameter running and one for prediction
    # they should be in the same place

    # not written in v generalizable way,see plt.plot(...,self.params) and not a __call__(sel.fparams)

    def cross_correlation(self,flux,lambdas,size=1000):

        shifts = np.linspace(-self.padding+0.01,self.padding-0.01,size)
        ccs = np.zeros(size)
        for i,shift in enumerate(shifts):
            ccs[i] = np.dot(self(lambdas + shift),flux)
        return ccs, shifts

class LinModel:
    def __init__(self,num_params,y,x,vel_shifts):
        self.epoches = len(vel_shifts)
        # when defining ones own model, need to include inputs as xs, outputs as ys
        # and __call__ function that gets ya ther, and params (1d ndarray MUST BE BY SCIPY) to be fit
        # also assumes epoches of data that is shifted between
        self.xs = x
        self.ys = y

        self.size = x.shape[1]
        # print(self.size)

        self.shifted = vel_shifts

        self.padding = abs(self.shifted).max()

        minimum = self.xs.min()
        maximum = self.xs.max()
        self.x = np.linspace(minimum-self.padding,maximum+self.padding,num_params)

        # the model x's must be shifted appropriately
        # this might have to be moved to the forward section if velocity is fit bt evals of ys

        # given x cells are shifted, the cell arrays contain the information for
        # which data points are in which cells
        self.cell_array = np.zeros([self.epoches,self.size],dtype=int)
        for i in range(self.epoches):
            # print((self.x - self.shifted[i]).shape)
            # print(self.xs.shape)
            # added the shifted to the freq dist so subtract shift from model
            # print(type(self.x - self.shifted[i]),type(self.xs[i,:]))
            self.cell_array[i,:] = getCellArray(self.x - self.shifted[i],self.xs[i,:])
        self.params = np.zeros(num_params)

    def __call__(self,params,input,epoch_idx,*args):
        # print(type(self),type(x),type(i))
        # can only be used once model is optimized
        # i = args[0]
        # cell_array = self.cell_array[epoch_idx,:] #getCellArray(self.x + self.shifted[epoch_idx],input)
        cell_array = self.cell_array[epoch_idx,:] #getCellArray(self.x + self.shifted[epoch_idx],input)
        # if cell_array is None:
        #     cell_array = getCellArray(self.x,x)
        x = input + self.shifted[epoch_idx]
        y = params
        # the x values for the model need to be shifted here but only for the intercept
        m  = (y[cell_array] - y[cell_array-1])/(self.x[cell_array] - self.x[cell_array-1])
        ys = y[cell_array-1] + m * (x - self.x[cell_array-1])
        return jnp.array(ys)

    def plot_model(self,i):
        plt.plot(self.x - self.shifted[i],self.params,'.r',linestyle='solid',linewidth=.8,zorder=2,alpha=0.5,ms=6)

    def plot(self,noise=None,xlim=None):
        size_x, size_y = getPlotSize(self.epoches)

        fig,axs = plt.subplots(size_x,size_y,figsize=[12.8,9.6],sharey=False)
        # Once again we apply the shift to the xvalues of the model when we plot it
        for n in range(self.epoches):
            
            i, j = (n%size_x,n//size_x)
            #ax.set_title('epoch %i: vel %.2f' % (i, self.shifted[i]))

            for tick in axs[i][j].xaxis.get_major_ticks():
                tick.label.set_fontsize(6)
                tick.label.set_rotation('vertical')
            x,y = self.plot_model(n)
            axs[i][j].plot(x,y,'.r',linestyle='solid',linewidth=.8,zorder=2,alpha=0.5,ms=6)
            if noise is not None:
                axs[i][j].errorbar(self.xs[n,:],self.ys[n,:],yerr=noise[i,:],fmt='.k',zorder=1,alpha=0.9,ms=6)
            else:
                axs[i][j].plot(self.xs[n,:],self.ys[n,:],'.k',zorder=1,alpha=0.9,ms=6)
            if xlim is None:
                axs[i][j].set_xlim(min(self.xs[n,:]),max(self.xs[n,:]))
            else:
                axs[i][j].set_xlim(xlim[0],xlim[1])
            axs[i][j].set_ylim(-0.8,0.2)

    def optimize(self,loss,maxiter,iprint=0,*args):
        # Train model
        func_grad = jax.value_and_grad(loss.train, argnums=0)
        def whatevershit(p,*args):
            val, grad = func_grad(p,*args)
            return np.array(val,dtype='f8'),np.array(grad,dtype='f8')
        res = scipy.optimize.minimize(whatevershit, self.params, jac=True,
               method='L-BFGS-B',
               args=(self.ys,self.xs,self,*args),
               options={'maxiter':maxiter,
                        'iprint':iprint
               })

        self.params = res.x
        return res
# foo = jax.numpy.interp(xs, x - shifts, params)
# res = scipy.optimize.minimize(lamdba(): (ys - foo(xs))**2, params, args=(self,*args), method='BFGS', jac=jax.grad(loss),
#        options={'disp': True})

class JnpLin(LinModel):

    def __call__(self,p,input,i=None,*args):
        if i == None:
            ys = jax.numpy.interp(input, self.x, p)
        else:
            ys = jax.numpy.interp(input, self.x - self.shifted[i], p)
        return ys

class JnpVelLin(LinModel):
    def __init__(self,num_params,y,x,vel_shifts,pretrained=None):
        
        if pretrained is not None:
            super(JnpVelLin,self).__init__(pretrained.x.shape[0],pretrained.ys,pretrained.xs,vel_shifts)
            self.params = pretrained.params

        else:
            super(JnpVelLin,self).__init__(num_params,y,x,vel_shifts)
        print(self.params.shape,self.shifted.shape)
        self.params = np.concatenate((self.params,self.shifted))

    def __call__(self,params,input,epoch_idx,*args):
        ys = jax.numpy.interp(input, self.x - params[-self.epoches+epoch_idx], params[:-self.epoches])
        return ys

    def plot_model(self,i):
        x = self.x - self.params[-self.epoches+i]
        y = self.params[:-self.epoches],
        #plt.plot(,'.r',linestyle='solid',linewidth=.8,zorder=2,alpha=0.5,ms=6)
        return x,y[0][:]



#for future
class FourierModel(LinModel):
    def __init__(self,num_params,y,x,shifts):
        self.epoches = y.shape[0]
        self.ys = y
        self.xs = x

        self.base_freq = (x.min() - x.max())/2

        self.shifted = shifts

        self.params = np.zeros(num_params)

    def __call__(self,params,input,epoch_idx,*args):
        out = 0
        for j, param in enumerate(params):
            if j % 2 == 0:
                out += param * np.cos((self.base_freq * np.floor(j/2)) * (input + self.shifted[epoch_idx]))
            if j % 2 == 1:
                out += param * np.sin((self.base_freq * np.floor(j/2)) * (input + self.shifted[epoch_idx]))
        return  out

    def plot_model(self,i):
        plt.plot(self.xs[0,:],self.predict(self.xs[0,:]))
