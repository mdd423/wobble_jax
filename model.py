import numpy as np
import jax
import jax.numpy as jnp
# import matplotlib.pyplot as plt
import scipy.optimize
import sys

import pickle5 as pickle

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

def save(filename,model):
    with open(filename, 'wb') as output:  # Overwrites any existing file.
        pickle.dump(model, output, pickle.HIGHEST_PROTOCOL)

def load(filename):
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

def get_lin_spaced_grid(xs,shifts,n):
    padding = abs(shifts).max()
    minimum = xs.min()
    maximum = xs.max()
    return np.linspace(minimum-padding,maximum+padding,n)

class Model:
    def optimize(self,loss,xs,ys,yerr,maxiter,iprint=0,*args):
        # Train model
        func_grad = jax.value_and_grad(loss.train, argnums=0)
        # callback = Callback()
        def callback(p):
            func_eval = loss.train(p,xs,ys,yerr,self,*args)
            self.func_evals.append(func_eval)
            print(func_eval)

        def whatevershit(p,*args):
            val, grad = func_grad(p,*args)
            return np.array(val,dtype='f8'),np.array(grad,dtype='f8')
        res = scipy.optimize.minimize(whatevershit, self.params, jac=True,
               method='L-BFGS-B',
               callback=callback,
               args=(xs,ys,yerr,self,*args),
               options={'maxiter':maxiter,
                        'iprint':iprint
               })

        self.results = res
        self.params = res.x
        return res, callback

    def __add__(self,x):
        return AdditiveModel(models=[self,x],parameters_per_model=[self.params.shape[0],x.params.shape])

    def __radd__(self,x):
        return AdditiveModel(models=[self,x],parameters_per_model=[self.params.shape[0],x.params.shape])

    def composite(self,x):
        return CompositeModel(models=[self,x],parameters_per_model=[self.params.shape[0],x.params.shape[0]])

class CompostieModel(Model):
    def __init__(self,models,parameters_per_model):
        self.models = []
        self.parameters_per_model = []

    def __call__(self,params,input,epoch_idx,*args):

        for model in self.models:
            input = model(self.params,input,epoch_idx,*args)
        return input

class AdditiveModel(Model):

    def __init__(self,models,parameters_per_model):
        self.models = models
        self.parameters_per_model = parameters_per_model

    def __call__(self,params,input,epoch_idx,*args):
        output = 0.0
        for i,model in enumerate(self.models):
            indices = np.arange(np.sum(self.parameters_per_model[:i]),np.sum(self.parameters_per_model[:i+1]))
            output += model(params[indices],input,epoch_idx,*args)
        return output

    def __add__(self,x):
        if isinstance(x,CompositeModel):
            return AdditiveModel(models=[*self.models,*x.models],parameters_per_model=[*self.parameters_per_model,*x.parameters_per_model])
        else:
            return AdditiveModel(models=[*self.models,x],parameters_per_model=[*self.parameters_per_model,x.params.shape[0]])

    def __radd__(self,x):
        if isinstance(x,CompositeModel):
            return AdditiveModel(models=[*self.models,*x.models],parameters_per_model=[*self.parameters_per_model,*x.parameters_per_model])
        else:
            return AdditiveModel(models=[*self.models,x],parameters_per_model=[*self.parameters_per_model,x.params.shape[0]])

class LinearModel(Model):
    def __init__(self,n,x_grid,shifts):
        self.epoches = len(shifts)
        # when defining ones own model, need to include inputs as xs, outputs as ys
        # and __call__ function that gets ya ther, and params (1d ndarray MUST BE BY SCIPY) to be fit
        # also assumes epoches of data that is shifted between
        # self.xs = x
        # self.ys = y

        self.params = np.zeros(n)
        self.shifted = shifts

        self.x = x_grid

        self.func_evals = []

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

class DataCalibration(Model):
    def __init__(self,n):
        self.n = n
        self.params = np.ones(n)

    def __call__(self,p,x,i,*args):
        x_out = 0.0
        for i in range(self.n):
            x_out = self.params[i]*x**i
        return x_out


# foo = jax.numpy.interp(xs, x - shifts, params)
# res = scipy.optimize.minimize(lamdba(): (ys - foo(xs))**2, params, args=(self,*args), method='BFGS', jac=jax.grad(loss),
#        options={'disp': True})

class TelluricModel(Model):
    def __init__(self,n,x_grid,airmass):
        self.n = n
        self.x = x_grid
        self.params = np.zeros(n)
        self.airmass = airmass

    def __call__(self,p,input,epoch_idx,*args):
        output = self.airmass[epoch_idx] * jax.numpy.interp(input, self.x, p)
        return output

class GasCellModel(Model):
    def __init__(self,lines):
        self.n          = lines.shape[0]
        self.lines      = lines
        self.widths     = np.ones(self.n)
        self.amplitudes = np.ones(self.n)

        self.p      = np.concatenate((self.amplitude))

    def __call__(self,p,input,epoch_idx,*args):

        output = 0.0
        for i,amp in enumerate(p):
            output += amp * np.exp(-np.power(input - self.lines[i], 2.) / (2 * np.power(self.widths[i], 2.)))
        return output

class JnpLin(LinearModel):

    def __call__(self,p,input,i=None,*args):
        if i == None:
            ys = jax.numpy.interp(input, self.x, p)
        else:
            ys = jax.numpy.interp(input, self.x - self.shifted[i], p)
        return ys

    def evaluate(self,p,x):
        return jax.numpy.interp(x,self.x,self.params)

class JnpVelLin(LinearModel):
    def __init__(self,n,x_grid,shifts,params=None):
        super(JnpVelLin,self).__init__(n,x_grid,shifts)
        if params is not None:
            self.params = params
        self.params = np.concatenate((self.params,self.shifted))

    def __call__(self,p,input,i,*args):
        if i is None:
            ys = jax.numpy.interp(input, self.x, p[:-self.epoches])
        else:
            ys = jax.numpy.interp(input, self.x - p[-self.epoches+i], p[:-self.epoches])
        return ys

    def evaluate(self,p,x):
        return jax.numpy.interp(x,self.x,self.params[:-self.epoches])

#for future
class FourierModel(Model):
    def __init__(self,n,y,x,shifts):
        self.epoches = y.shape[0]
        self.ys = y
        # self.xs = x

        self.base_freq = (x.min() - x.max())/2

        self.shifted = shifts

        self.params = np.zeros(n)

    def __call__(self,params,input,epoch_idx,*args):
        out = 0
        for j, param in enumerate(params):
            if j % 2 == 0:
                out += param * np.cos((self.base_freq * np.floor(j/2)) * (input + self.shifted[epoch_idx]))
            if j % 2 == 1:
                out += param * np.sin((self.base_freq * np.floor(j/2)) * (input + self.shifted[epoch_idx]))
        return  out

    # def plot_model(self,i):
    #     plt.plot(self.xs[0,:],self.predict(self.xs[0,:]))
