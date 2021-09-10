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

# make function like this but in terms of resolution not n
# plus padding
def get_lin_spaced_grid(xs,padding,step):
    '''Generates grid of x points for JaxLinear model, using the minimum of the
    observations, some padding amount, and the step size'''
    # padding = abs(shifts).max()
    minimum = xs.min()
    maximum = xs.max()
    return np.arange(minimum-padding,maximum+padding,step=step)


class Model:
    '''General model class of Jabble:
    contains methods for optimizing, calling'''
    def __init__(self):
        self.fitting = jnp.array([])
        self.func_evals = []

    def _call(self,p,*args):

        if len(p) == 0:
            return self(self.p,*args)
        else:
            return self(p,*args)

    def optimize(self,loss,data,maxiter,iprint=0,method='L-BFGS-B',*args):
        # Train model
        func_grad = jax.value_and_grad(loss.loop, argnums=0)

        def callback(p):
            func_eval = loss.loop(p,data.xs,data.ys,data.yerr,self._call,*args)
            self.func_evals.append(func_eval)
            print(func_eval)

        def val_gradient_function(p,*args):
            val, grad = func_grad(p,*args)
            return np.array(val,dtype='f8'),np.array(grad,dtype='f8')

        res = scipy.optimize.minimize(val_gradient_function, self.get_parameters(), jac=True,
               method=method,
               callback=callback,
               args=(data.xs,data.ys,data.yerr,self._call,*args),
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
        # self.p = res.x
        return res, callback

    def __add__(self,x):
        return AdditiveModel(models=[self,x])

    def __radd__(self,x):
        return AdditiveModel(models=[self,x])

    def composite(self,x):
        return CompositeModel(models=[self,x])

    def evaluate(self,x):
        return self(self.p,x,i=None)

    def fix(self):

        self.fitting = jnp.array([])

    def fit(self):

        self.fitting = self.p

    def get_parameters(self):
        return self.fitting

    def unpack(self,p):
        if len(p) != 0:
            self.fitting = p
            self.p = p


class ContainerModel(Model):
    def __init__(self,models):
        super(ContainerModel,self).__init__()
        self.models = models
        self.parameters_per_model = np.zeros((len(models)))

    def __getitem__(self,idx):
        return self.models[idx]

    def unpack(self,p):

        for k,model in enumerate(self.models):
            indices = np.arange(np.sum(self.parameters_per_model[:k]),
                                np.sum(self.parameters_per_model[:k+1]),dtype=int)
            model.unpack(p[indices])

    def get_parameters(self):
        x = jnp.array([])
        # self.parameters_per_model = np.array([])
        for i,model in enumerate(self.models):
            # adding parameters_per_model should be done when put in fitting mode
            x = jnp.concatenate((x,model.get_parameters()))

        return x

    def fit(self,i,*args):
        self[i].fit(*args)
        self.parameters_per_model[i] = self[i].get_parameters().shape[0]

    def fix(self,i,*args):
        self[i].fix(*args)
        self.parameters_per_model[i] = 0


class CompositeModel(ContainerModel):
    def __call__(self,p,x,i,*args):
        # print(self.parameters_per_model)
        for k,model in enumerate(self.models):
            indices = np.arange(np.sum(self.parameters_per_model[:k]),
                                np.sum(self.parameters_per_model[:k+1]),dtype=int)
            # print(indices)
            x = model._call(p[indices],x,i,*args)
        return x

    def composite(self,x):
        if isinstance(x,CompositeModel):
            return CompositeModel(models=[*self.models,*x.models])
        else:
            return CompositeModel(models=[*self.models,x])


class AdditiveModel(ContainerModel):
    def __call__(self,p,x,i,*args):
        output = 0.0
        # PARALLELIZABLE
        for k,model in enumerate(self.models):
            indices = np.arange(np.sum(self.parameters_per_model[:k]),
                                np.sum(self.parameters_per_model[:k+1]),dtype=int)
            output += model._call(p[indices],x,i,*args)
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


class ConvolutionalModel(Model):
    def __init__(self,n,p=None):
        super(ConvolutionalModel,self).__init__()
        if p is None:
            self.p = np.array([0,1,0])
        else:
            self.p = p

    def __call__(self,p,x,i):
        y = signal.convolve(x,p,mode='same')
        return y


class ShiftingModel(Model):
    def __init__(self,deltas):
        super(ShiftingModel,self).__init__()
        self.epoches = deltas.shape[0]
        self.p       = deltas

    def __call__(self,p,x,i):

        return p[i] + x

    # cant do this generically because unlike most parameters in models
    # these are independent of one another and loss of each combination epoch is just the sum
    # each of the individuals
    def grid_search(self,shift_grid,loss,model,xs,ys,yerr,index):
        # put all submodels in fixed mode except the shiftingmodel
        # to be searched then take loss of each epoch
        # that we hand the loss a slice of the shift array
        # since at __call__ itll on take the shift_grid[i,j] element
        model.fix(True)
        # index is the index of the submodel to grid search this is redundant
        model.fit(index)
        # this is called because this resets the parameters per model
        # array
        # I want to have this be done when a submodel is put into fix or fix mode
        model.get_parameters()
        loss_arr = np.empty(shift_grid.shape)
        for i in range(loss_arr.shape[0]):
            for j in range(loss_arr.shape[1]):
                loss_arr[i,j] = loss(shift_grid[:,j],xs[i,:],ys[i,:],yerr[i,:],i,model)
        return loss_arr


class StretchingModel(Model):
    def __init__(self,m=None,epoches=0):
        super(StretchingModel,self).__init__()
        self.epoches = stretches.shape[0]
        if m is None:
            self.p = np.ones((epoches))
        else:
            self.p = m

    def __call__(self,p,x,i):

        return p[i] * x


class JaxLinear(Model):
    def __init__(self,xs):
        super(JaxLinear,self).__init__()
        self.n       = len(xs)
        # when defining ones own model, need to include inputs as xs, outputs as ys
        # and __call__ function that gets ya ther, and params (1d ndarray MUST BE BY SCIPY) to be fit
        # also assumes epoches of data that is shifted between
        self.xs    = xs
        self.p = np.zeros(self.n)

    def __call__(self,p,x,i):
        # print()
        # print(p.shape)
        y = jax.numpy.interp(x, self.xs, p)
        return y

# foo = jax.numpy.interp(xs, x - shifts, params)
# res = scipy.optimize.minimize(lamdba(): (ys - foo(xs))**2, params, args=(self,*args), method='BFGS', jac=jax.grad(loss),
#        options={'disp': True})
