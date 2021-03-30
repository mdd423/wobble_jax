import numpy as np
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import scipy.optimize
import sys

import pickle

import simulator as wobble_sim
import loss as wobble_loss

# you can pass whatever args you want to your function after the 0th which is the model
# pass these after the loss function in the optimize call

# args can also be passed to the forward pass of the model
# if you define a model to take on additional arguments in the forward pass
# idk how this will work with multiple update steps

class LossFunc():
    def __init__(self,loss_func,loss_parms=1.0):
        if   type(loss_func) == type('str'):
            self.func_list = np.array([loss_func])
        elif type(loss_func) == type(np.array([])):
            self.func_list = loss_func
        else:
            sys.exit('loss_func parameter not correct type: str or list')
        if   type(loss_parms) == type(1.0):
            self.params = np.array([loss_parms])
        elif type(loss_parms) == type(np.array([])):
            self.params = loss_parms
            assert len(self.params) == len(self.func_list)
        else:
            sys.exit('loss_func parameter not correct type: float or list')

    def __add__(self,x):

        return LossFunc(loss_func=np.append(self.func_list, x.func_list ) ,loss_parms=np.append(self.params, x.params ))

    def __mul__(self,x):
        return LossFunc(loss_func=self.func_list,loss_parms=x * self.params)

    def __rmul__(self,x):
        return LossFunc(loss_func=self.func_list,loss_parms=x * self.params)

    def __call__(self,params,*args):
        output = 0.0
        for i,loss in enumerate(self.func_list):
            output += self.params[i] * wobble_loss.loss_dict[loss](params,*args)
        return output

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

class LinModel():
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

    # def plot_epoch(self,noise,en)

    def plot(self,noise=None,env=None,atm_model=None,xlim=None):
        size_x, size_y = getPlotSize(self.epoches)

        fig = plt.figure(figsize=[12.8,9.6])
        # Once again we apply the shift to the xvalues of the model when we plot it
        for i in range(self.epoches):
            ax = fig.add_subplot(size_x,size_y,i+1)
            ax.set_title('epoch %i: vel %.2f' % (i, self.shifted[i]))

            if atm_model is not None:
                plt.plot(atm_model.x - atm_model.shifted[i],atm_model.params,'.g',linestyle='solid',linewidth=.8,zorder=2,alpha=0.5,ms=6)

            self.plot_model(i)
            if noise is not None:
                plt.errorbar(self.xs[i,:],self.ys[i,:],yerr=noise[i,:],fmt='.k',zorder=1,alpha=0.9,ms=6)
            else:
                plt.plot(self.xs[i,:],self.ys[i,:],'.k',zorder=1,alpha=0.9,ms=6)
            if xlim is None:
                plt.xlim(min(self.xs[i,:]),max(self.xs[i,:]))
            else:
                plt.xlim(xlim[0],xlim[1])
            plt.ylim(-0.8,0.2)
            if env is not None:
                plt.plot(env.lambdas - self.shifted[i],env.get_stellar_flux(),color='red', alpha=0.4)

    def optimize(self,loss,maxiter,*args):
        # Train model
        # def func_grad(loss):
        #      = jax.value_and_grad(loss, argnums=0)
        #     return
        func_grad = jax.value_and_grad(loss, argnums=0)
        def whatevershit(p,*args):
            val, grad = func_grad(p,*args)
            return np.array(val,dtype='f8'),np.array(grad,dtype='f8')
        res = scipy.optimize.minimize(whatevershit, self.params, jac=True,
               method='L-BFGS-B',
               args=(self,*args),
               options={'maxiter':maxiter})

        # res = scipy.optimize.minimize(loss, self.params, args=(self,*args), method='BFGS', jac=jax.grad(loss),
        #        options={'disp': True, 'maxiter': maxiter})
        self.params = res.x
        return res

    # gives all predicted data from the input data in the model given the parameters
    def forward(self,params,*args):
        # input = args[0]
        # organization of params here is the same as in __init__
        # TO DO: determine cell array with each forward pass if we are to fit velocity shift

        # preds should be of the same shape as the out of __call__
        # EXCEPT that it has an additional axis per epoch
        # assume the same input for every epoch (?)
        input = self.xs[0,:]
        preds = jnp.expand_dims(self(params,input,0,*args),axis=0)

        for i in range(1,self.epoches):
            input = self.xs[i,:]
            # temp_x = self.x+self.shifted[i]
            # subtracted shift from model
            # cell_array = self.cell_array[i,:] #getCellArray(temp_x,self.xs)
            # # the x values for the model need to be shifted here but only for the intercept
            # m   = (y[cell_array] - y[cell_array-1])/(self.x[cell_array] - self.x[cell_array-1])
            # ys2 = y[cell_array-1] + m * (input - self.x[cell_array-1] + self.shifted[i])

            # decide on official
            ys    = jnp.expand_dims(self(params,input,i,*args),axis=0)
            preds = jnp.append(preds,ys,axis=0)
        return preds

    def predict(self,input,*args):
        return self(self.params,input,*args)

    def save_model(self, filename):
        with open(filename, 'wb') as output:  # Overwrites any existing file.
            pickle.dump(self, output, pickle.HIGHEST_PROTOCOL)

    def load_model(self, filename):
        with open(filename, 'wb') as input:  # Overwrites any existing file.
            self = pickle.load(input)

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

# foo = jax.numpy.interp(xs, x - shifts, params)
# res = scipy.optimize.minimize(lamdba(): (ys - foo(xs))**2, params, args=(self,*args), method='BFGS', jac=jax.grad(loss),
#        options={'disp': True})
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

class JnpLin(LinModel):

    def __call__(self,params,input,epoch_idx,*args):
        ys = jax.numpy.interp(input, self.x - self.shifted[epoch_idx], params)
        return ys

def JnpVelLin(LinModel):
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
        self.params = np.concatenate(self.params,self.shifted)

    def __call__(self,params,input,epoch_idx,*args):
        ys = jax.numpy.interp(input, self.x - params[self.epoches:], params[:-self.epoches])
        return ys

    # def plot_model(self,i):
    #     plt.plot(self.x - self.shifted[i],self.params,'.r',linestyle='solid',linewidth=.8,zorder=2,alpha=0.5,ms=6)
