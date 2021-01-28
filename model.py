import numpy as np
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import scipy.optimize
import sys

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
            self.func_list = [loss_func]
        elif type(loss_func) == type(['list']):
            self.func_list = loss_func
        else:
            sys.exit('loss_func parameter not correct type: str or list')
        if   type(loss_parms) == type(1.0):
            self.params = [loss_parms]
        elif type(loss_parms) == type(['list']):
            self.params = loss_parms
            assert len(self.params) == len(self.func_list)
        else:
            sys.exit('loss_func parameter not correct type: float or list')

    def __add__(self,x):

        return LossFunc(loss_func=self.func_list + x.func_list,loss_parms=self.params + x.params)

    def __call__(self,params,*args):
        output = 0.0
        for i,loss in enumerate(self.func_list):
            output += self.params[i] * wobble_loss.loss_dict[loss](params,*args)
        return output

def getCellArray(x,xs):
    cell_array = np.zeros(len(xs),dtype=int)
    x_val = x[0]
    j = 0
    for i, xss in enumerate(xs):
        while x_val < xss:
            j += 1
            x_val = x[j]
        cell_array[i] = int(j)
    return cell_array

def getPlotSize(model):
    size_x = np.floor(np.sqrt(model.epoches))
    size_y = model.epoches//size_x
    while model.epoches % size_y != 0:
        size_y = model.epoches//size_x
        size_x -= 1
    else:
        size_x += 1
    size_x = int(size_x)
    size_y = int(size_y)
    return size_x, size_y

class LinModel():
    def __init__(self,num_params,fluxes,lambdas,epoches,vel_shifts,size):
        self.epoches = epoches
        self.xs = lambdas
        self.ys = fluxes

        self.shifted = vel_shifts

        self.padding = abs(self.shifted).max()

        minimum = self.xs.min()
        maximum = self.xs.max()
        self.x = np.linspace(minimum-self.padding,maximum+self.padding,num_params)

        # the model x's must be shifted appropriately
        # this might have to be moved to the forward section if velocity is fit bt evals of ys

        # given x cells are shifted, the cell arrays contain the information for
        # which data points are in which cells
        self.cell_array = np.zeros([self.epoches,size],dtype=int)
        for i in range(self.epoches):
            # added the shifted to the freq dist so subtract shift from model
            self.cell_array[i,:] = getCellArray(self.x - self.shifted[i],self.xs)
        self.y = np.ones(num_params)

    def optimize(self,loss,*args):
        # Train model
        res = scipy.optimize.minimize(loss, self.y, args=(self,*args), method='BFGS', jac=jax.grad(loss),
               options={'disp': True})
        self.y = res.x
        return res

    # gives all predicted data from the input data in the model given the parameters
    def forward(self,params,*args):
        y = params
        # TO DO: determine cell array with each forward pass if we are to fit velocity shift
        ys = jnp.array([])
        for i in range(self.epoches):
            # temp_x = self.x+self.shifted[i]
            # subtracted shift from model
            cell_array = self.cell_array[i,:] #getCellArray(temp_x,self.xs)
            # the x values for the model need to be shifted here but only for the intercept
            m   = (y[cell_array] - y[cell_array-1])/(self.x[cell_array] - self.x[cell_array-1])
            ys2 = y[cell_array-1] + m * (self.xs - self.x[cell_array-1] + self.shifted[i])
            ys  = jnp.append(ys,ys2)
        return ys

    def __call__(self,x):
        # can only be used once model is optimized
        cell_array = getCellArray(self.x,x)
        # the x values for the model need to be shifted here but only for the intercept
        m  = (self.y[cell_array] - self.y[cell_array-1])/(self.x[cell_array] - self.x[cell_array-1])
        ys = self.y[cell_array-1] + m * (x - self.x[cell_array-1])
        return ys

    def plot(self,noise,env=None):
        size_x, size_y = getPlotSize(self)

        fig = plt.figure(figsize=[12.8,9.6])
        # Once again we apply the shift to the xvalues of the model when we plot it
        for i in range(self.epoches):
            ax = fig.add_subplot(size_x,size_y,i+1)
            ax.set_title('epoch %i: vel %.2f' % (i, self.shifted[i]))

            plt.plot(self.x - self.shifted[i],self.y,'.r',linestyle='solid',linewidth=.8,zorder=2,alpha=0.5,ms=6)

            plt.errorbar(self.xs,self.ys[i,:],yerr=noise,fmt='.k',zorder=1,alpha=0.9,ms=6)

            plt.xlim(min(self.xs),max(self.xs))

            plt.ylim(-0.8,0.2)
            if env is not None:
                plt.plot(env.lambdas - self.shifted[i],env.get_stellar_flux(),color='red', alpha=0.4)

    def cross_correlation(self,flux,lambdas,size=1000):

        shifts = np.linspace(-self.padding+0.01,self.padding-0.01,size)
        ccs = np.zeros(size)
        for i,shift in enumerate(shifts):
            ccs[i] = np.dot(self(lambdas + shift),flux)
        return ccs, shifts
