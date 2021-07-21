import numpy as np
import jax
import jax.numpy as jnp
# import matplotlib.pyplot as plt
import scipy.optimize
import scipy.signal as signal
import sys

import pickle5 as pickle

import simulator as wobble_sim
import loss as wobble_loss

def spacing_from_res(R):
    return np.log(1+1/R)

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

# def cross_correlation(self,flux,lambdas,size=1000):

    shifts = np.linspace(-self.padding+0.01,self.padding-0.01,size)
    ccs = np.zeros(size)
    for i,shift in enumerate(shifts):
        ccs[i] = np.dot(self(lambdas + shift),flux)
    return ccs, shifts

# make function like this but in terms of resolution not n
# plus padding
def get_lin_spaced_grid(xs,padding,step):
    # padding = abs(shifts).max()
    minimum = xs.min()
    maximum = xs.max()
    return np.arange(minimum-padding,maximum+padding,step=step)

class Model:
    def optimize(self,loss,xs,ys,yerr,maxiter,iprint=0,method='L-BFGS-B',*args):
        # Train model
        func_grad = jax.value_and_grad(loss.loop, argnums=0)

        def callback(p):
            func_eval = loss.loop(p,xs,ys,yerr,self,*args)
            self.func_evals.append(func_eval)
            print(func_eval)

        def whatevershit(p,*args):
            val, grad = func_grad(p,*args)
            return np.array(val,dtype='f8'),np.array(grad,dtype='f8')

        res = scipy.optimize.minimize(whatevershit, self.p, jac=True,
               method=method,
               callback=callback,
               args=(xs,ys,yerr,self,*args),
               options={'maxiter':maxiter,
                        'iprint':iprint
               })

        self.results = res
        # parameters need to be replaced in all submodels
        # so that they can be plot using variable names
        # not some indices of p, unpack function is for user
        # to know how to plot parameters being fit
        # either way the models parameters are put back into 1d p
        try:
            self.unpack(res.x)
        except NameError:
            print('cannot unpack, setting ps tho')
            pass
        self.p = res.x
        return res, callback

    def __add__(self,x):
        return AdditiveModel(models=[self,x])

    def __radd__(self,x):
        return AdditiveModel(models=[self,x])

    def composite(self,x):
        return CompositeModel(models=[self,x])

    def evaluate(self,x):
        return self(self.p,x,i=None)

    def fix_parameters(self):
        self.p = np.empty(shape=(0))
        self.bool = False

    def fit_parameters(self,parameters):
        self.bool = True
        if self.p.shape[0] == 0:
            self.p = parameters
    
class CompositeModel(Model):
    def __init__(self,models):
        self.models = models
        self.parameters_per_model = np.array([model.p.shape[0] for model in models])
        self.p = np.concatenate([model.p for model in models])
        self.func_evals = []

    def __call__(self,p,x,i,*args):

        for k,model in enumerate(self.models):
            indices = np.arange(np.sum(self.parameters_per_model[:k]),np.sum(self.parameters_per_model[:k+1]),dtype=int)
            input = model(p[indices],x,i,*args)
        return input

    def __getitem__(self,idx):
        return self.models[idx]

    def composite(self,x):
        if isinstance(x,CompositeModel):
            return CompositeModel(models=[*self.models,*x.models])
        else:
            return CompositeModel(models=[*self.models,x])

    def unpack(self,p):

        for k,model in enumerate(self.models):
            indices = np.arange(np.sum(self.parameters_per_model[:k]),np.sum(self.parameters_per_model[:k+1]),dtype=int)
            try:
                model.unpack(p[indices])
            except NameError:
                pass
            model.p = p[indices]

    def fix_parameters(self,model_index):
        self.models[model_index].fix_parameters()
        self.p = np.concatenate([model.p for model in self.models])
        self.parameters_per_model[model_index] = 0

    def fit_parameters(self,model_index,parameters):
        self.models[model_index].fit_parameters(parameters)
        self.p = np.concatenate([model.p for model in self.models])
        self.parameters_per_model[model_index] = parameters.shape[0]

class AdditiveModel(Model):

    def __init__(self,models):
        self.models = models
        self.parameters_per_model = np.array([model.p.shape[0] for model in models])
        self.p = np.concatenate([model.p for model in models])
        self.func_evals = []

    def __call__(self,p,x,i,*args):
        output = 0.0
        # PARALLELIZABLE
        for k,model in enumerate(self.models):
            indices = np.arange(np.sum(self.parameters_per_model[:k]),np.sum(self.parameters_per_model[:k+1]),dtype=int)
            output += model(p[indices],x,i,*args)
        return output

    def __add__(self,x):
        if isinstance(x,AdditiveModel):
            return AdditiveModel(models=[*self.models,*x.models])
        else:
            return AdditiveModel(models=[*self.models,x])

    def __radd__(self,x):
        if isinstance(x,CompositeModel):
            return AdditiveModel(models=[*self.models,*x.models])
        else:
            return AdditiveModel(models=[*self.models,x])

    def __getitem__(self,idx):
        return self.models[idx]

    def unpack(self,p):

        for k,model in enumerate(self.models):
            indices = np.arange(np.sum(self.parameters_per_model[:k]),np.sum(self.parameters_per_model[:k+1]),dtype=int)
            try:
                model.unpack(p[indices])
            except NameError:
                pass
            model.p = p[indices]

    def fix_parameters(self,model_index):
        self.models[model_index].fix_parameters()
        self.parameters_per_model[model_index] = 0

    def fit_parameters(self,model_index,parameters):
        self.models[model_index].fit_parameters(parameters)
        self.parameters_per_model[model_index] = parameters.shape[0]

# class DataCalibration(Model):
    def __init__(self,n):
        self.n   = n
        self.delta = np.zeros(n)
        self.func_evals = []

        self.bool = False

    def __call__(self,p,x,i,*args):
        p[~self.bool] = self.p[~self.bool]
        y = x + p[i]
        return y

    def unpack(self,p):
        self.delta = p

class ConvolutionalModel(Model):
    def __init__(self,n):
        self.omega = np.ones(n)
        self.bool = False
        self.p = np.empty(shape=(0))

        self.func_evals = []

    def __call__(self,p,x,i):
        if self.bool:
            y = signal.convolve(x,p,mode='same')
        else:
            y = signal.convolve(x,self.omega,mode='same')
        return y

    def unpack(self,p):
        if self.bool:
            self.omega = p

class ShiftingModel(Model):
    def __init__(self,deltas):
        self.epoches = deltas.shape[0]
        self.deltas = deltas
        self.bool = False
        self.p = np.empty(shape=(0))

    def __call__(self,p,x,i):
        if self.bool:
            return p[i] + x
        else:
            return self.deltas[i] + x

    def unpack(self,p):
        if self.bool:
            self.deltas = p

class StretchingModel(Model):
    def __init__(self,m=None,epoches=0):
        self.epoches = stretches.shape[0]
        if m is None:
            self.m = np.ones((epoches))
        else:
            self.m = m
        self.bool = False
        self.p = np.empty(shape=(0))

    def __call__(self,p,x,i):
        if self.bool:
            return p[i] * x
        else:
            return self.m[i] * x

    def unpack(self,p):
        if self.bool:
            self.m = p

class JaxLinear(Model):
    def __init__(self,xs):
        self.n       = len(xs)
        # when defining ones own model, need to include inputs as xs, outputs as ys
        # and __call__ function that gets ya ther, and params (1d ndarray MUST BE BY SCIPY) to be fit
        # also assumes epoches of data that is shifted between
        self.theta = np.zeros(self.n)
        self.xs    = xs

        self.func_evals = []
        self.bool = False
        self.p = np.empty(shape=(0))

    def __call__(self,p,x,i):
        if self.bool:
            y = jax.numpy.interp(x, self.xs, p)
        else:
            y = jax.numpy.interp(x, self.xs, self.theta)
        return y

    def unpack(self,p):
        if self.bool:
            self.theta = p

# foo = jax.numpy.interp(xs, x - shifts, params)
# res = scipy.optimize.minimize(lamdba(): (ys - foo(xs))**2, params, args=(self,*args), method='BFGS', jac=jax.grad(loss),
#        options={'disp': True})

# class TelluricModel(Model):
#     def __init__(self,x_grid,airmass):
#         self.n = x_grid.shape[0]
#         self.x = x_grid
#         self.omega = np.zeros(self.n)
#         self.airmass = airmass
#
#         self.p = self.omega
#         self.bool = np.ones(self.p.shape,dtype='bool')
#         self.func_evals = []
#
#     def __call__(self,p,x,i,*args):
#         p[~self.bool] = self.p[~self.bool]
#         y = self.airmass[i] * jax.numpy.interp(x, self.x, p)
#         return y
#
#     def unpack(self,p):
#         self.omega = p
#
# class GasCellModel(Model):
#     def __init__(self,lines,widths):
#         self.n      = lines.shape[0]
#         self.lines  = lines
#         self.widths = widths
#         self.b      = np.ones(self.n)
#
#         self.func_evals = []
#         self.p = self.b
#         self.bool = np.ones(self.p.shape,dtype='bool')
#
#     def __call__(self,p,x,i,*args):
#         p[~self.bool] = self.p[~self.bool]
#         y = np.zeros(x.shape)
#         for k,amp in enumerate(p):
#             y -= amp * np.exp(-np.power(x - self.lines[k], 2.) / (2 * np.power(self.widths[k], 2.)))
#         return y
#
#     def unpack(self,p):
#         self.b = p
#
# class LinearModel(Model):
#     def __init__(self,x_grid,delta):
#         self.epoches = len(delta)
#         self.n = len(x_grid)
#         # when defining ones own model, need to include inputs as xs, outputs as ys
#         # and __call__ function that gets ya ther, and params (1d ndarray MUST BE BY SCIPY) to be fit
#         # also assumes epoches of data that is shifted between
#         self.omega = np.zeros(self.n)
#         self.delta = delta
#         self.x     = x_grid
#
#         self.func_evals = []
#         self.p     = self.omega
#         self.bool = np.ones(self.p.shape,dtype='bool')
#
#     def __call__(self,p,x,i,*args):
#         # print(type(self),type(x),type(i))
#         # can only be used once model is optimized
#         # i = args[0]
#         # cell_array = self.cell_array[epoch_idx,:] #getCellArray(self.x + self.shifted[epoch_idx],input)
#         cell_array = self.cell_array[i,:] #getCellArray(self.x + self.shifted[epoch_idx],input)
#         # if cell_array is None:
#         #     cell_array = getCellArray(self.x,x)
#         x = x + self.delta[i]
#         y = p
#         # the x values for the model need to be shifted here but only for the intercept
#         m  = (y[cell_array] - y[cell_array-1])/(self.x[cell_array] - self.x[cell_array-1])
#         ys = y[cell_array-1] + m * (x - self.x[cell_array-1])
#         return jnp.array(ys)
#
#     def unpack(self,p):
#         self.omega = p
#
# class StellarModel(LinearModel):
#     def __init__(self,x_grid,delta,p=None):
#         super(StellarModel,self).__init__(x_grid,delta)
#         if p is not None:
#             self.p = p
#         self.p = np.concatenate((self.p,self.delta))
#         self.bool = np.ones(self.p.shape,dtype='bool')
#
#     def __call__(self,p,x,i=None,*args):
#
#         if i is None:
#             y = jax.numpy.interp(x, self.x, p[:-self.epoches])
#         else:
#             y = jax.numpy.interp(x, self.x - p[-self.epoches+i], p[:-self.epoches])
#         return y
#
#     def unpack(self,p):
#         self.omega = p[:-self.epoches]
#         self.delta = p[-self.epoches:]
#
# #for future
# class FourierModel(Model):
    # def __init__(self,n,y,x,shifts):
    #     self.epoches = y.shape[0]
    #     self.ys = y
    #     # self.xs = x
    #
    #     self.base_freq = (x.min() - x.max())/2
    #
    #     self.shifted = shifts
    #
    #     self.p = np.zeros(n)
    #
    # def __call__(self,p,input,epoch_idx,*args):
        # out = 0
        # for j, param in enumerate(p):
        #     if j % 2 == 0:
        #         out += param * np.cos((self.base_freq * np.floor(j/2)) * (input + self.shifted[epoch_idx]))
        #     if j % 2 == 1:
        #         out += param * np.sin((self.base_freq * np.floor(j/2)) * (input + self.shifted[epoch_idx]))
        # return  out
