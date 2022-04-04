import numpy as np
import jax
import jax.numpy as jnp
# import matplotlib.pyplot as plt
import scipy.optimize
import scipy.signal as signal
import sys

import copy

import astropy.constants as const
import logging

import pickle#5 as pickle
import jabble.dataset

# import simulator as wobble_sim
# import loss as wobble_loss
def save(filename,model):
    with open(filename, 'wb') as output:  # Overwrites any existing file.
        pickle.dump(model, output, pickle.HIGHEST_PROTOCOL)

def load(filename):
    with open(filename, 'rb') as input:  # Overwrites any existing file.
        model = pickle.load(input)
        return model

# make function like this but in terms of resolution not n
# plus padding
def create_x_grid(xs,vel_padding,resolution):
    x_min = xs.min()
    x_max = xs.max()
    step  = jabble.dataset.shifts(const.c/resolution)
    x_padding = jabble.dataset.shifts(vel_padding)
    return np.arange(x_min-x_padding,x_max+x_padding,step)

def stellar_model(shifts,x_grid):
    return CompositeModel([ShiftingModel(shifts),JaxLinear(x_grid)])

def tellurics_model(airmass,x_grid):
    return CompositeModel([JaxLinear(x_grid),StretchingModel(airmass)])


class Model:
    '''General model class of Jabble:
    contains methods for optimizing, calling'''
    def __init__(self):
        self._fit    = False
        self.func_evals = []
        self.history = []
        self.save_history = False

        self.loss_history = []
        self.save_loss = []

    def __call__(self,p,*args):
        # if there are no parameters coming in, then use the stored parameters
        if len(p) == 0:
            return self.call(self.p,*args)
        else:
            return self.call(p,*args)

    def optimize(self,loss,data,method='L-BFGS-B',verbose=False,margs={},save_history=False,save_loss=False,*args):
        # Fits the Model
        self.save_history = save_history
        if loss is None:
            loss_ind = np.arange(data.shape[0])

        func_grad = jax.value_and_grad(loss.loss_all, argnums=0)
        def val_gradient_function(p,*args):
            val, grad = func_grad(p,*args)
            self.func_evals.append(val)
            if verbose:
                print('\r[ Value: {:+3.2e} Grad: {:+3.2e} ]'.format(val,np.inner(grad,grad)))

            if self.save_history:
                self.history.append(np.array(p))

            if self.save_loss:
                initialize = loss(p,data,0,self)
                tmp = np.zeros((data.ys.shape[0],*initilize.shape))
                tmp[0,...] = initilize
                for i in range(1,data.ys.shape[0]):
                    tmp[i,...] = loss(p,data,i,self)
                self.loss_history.append(tmp)

            return np.array(val,dtype='f8'),np.array(grad,dtype='f8')

        res = scipy.optimize.minimize(val_gradient_function, self.get_parameters(), jac=True,
               method=method,
               args=(data,self,*args),
               options=margs
               )
        try:
            self.results.append(res)
        except AttributeError:
            self.results = [res]

        try:
            self.unpack(res.x)
        except NameError:
            pass
        return res

    def __add__(self,x):
        return AdditiveModel(models=[self,x])

    def __radd__(self,x):
        return AdditiveModel(models=[self,x])

    def composite(self,x):
        return CompositeModel(models=[self,x])

    def evaluate(self,x,i):
        return self(self.p,x,i)
    # make this a property of model
    def fix(self):
        # add inds here as possibility, then add to the call, that the correct
        # indices for fixed parameters are pulled and combined with variable
        # parameters in the function call
        self._fit = False

    def fit(self):

        self._fit = True

    def get_parameters(self):
        if self._fit:
            return jnp.array(self.p)
        else:
            return jnp.array([])

    def unpack(self,p):
        if len(p) != 0:
            self.p = p

    def display(self,string=''):
        out = string+'-'+self.__class__.__name__+'-'
        out += '-----------------------------------'
        params = str(len(self.get_parameters()))
        while len(out + params) % 17 != 0:
            # print(len())
            out += '-'

        out += params
        print(out)

    def split_p(self,p):
        return p

    def copy(self):
        return copy.deepcopy(self)


class ContainerModel(Model):
    def __init__(self,models):
        super(ContainerModel,self).__init__()
        self.models = models
        self.parameters_per_model = np.empty((len(models)))
        for i,model in enumerate(models):
            self.parameters_per_model[i] = len(model.get_parameters())

    def append(self,model):
        self.models.append(model)
        self.parameters_per_model = np.concatenate((self.parameters_per_model,len(model.get_parameters())))

    def __call__(self,p,*args):
        # if there are no parameters coming in, then use the stored parameters
        if len(p) == 0:
            return self.call(np.array([]),*args)
        else:
            return self.call(p,*args)

    def __getitem__(self,i):
        return self.models[i]

    def unpack(self,p):

        for k,model in enumerate(self.models):
            indices = np.arange(np.sum(self.parameters_per_model[:k]),
                                np.sum(self.parameters_per_model[:k+1]),dtype=int)
            model.unpack(p[indices])

    def get_parameters(self):
        x = jnp.array([])
        # self.parameters_per_model = np.array([])
        for i,model in enumerate(self.models):
            params = model.get_parameters()
            self.parameters_per_model[i] = params.shape[0]
            # adding parameters_per_model should be done when put in fitting mode
            x = jnp.concatenate((x,params))

        return x

    def fit(self,i=None,*args):
        if i is None:
            for j,model in enumerate(self.models):
                model.fit()
                self.parameters_per_model[j] = model.get_parameters().shape[0]
        else:
            self[i].fit(*args)
            self.parameters_per_model[i] = self[i].get_parameters().shape[0]

    def fix(self,i=None,*args):
        if i is None:
            for j,model in enumerate(self.models):
                model.fix()
                self.parameters_per_model[j] = 0
        else:
            self[i].fix(*args)
            self.parameters_per_model[i] = 0

    def p():
        doc = "The p property."
        def fget(self):
            out = np.array([])
            for model in self.models:
                out = np.concatenate((out,model.p))
            return out
        return locals()
    p = property(*p())

    def display(self,string=''):
        self.get_parameters()
        super(ContainerModel,self).display(string)
        for i,model in enumerate(self.models):
            # print(model)
            tab = '  {}'.format(i)
            string += tab
            model.display(string)
            string = string[:-len(tab)]

    def split_p(self,p):
        p_list = [self.models[k].split_p(p[jnp.arange(jnp.sum(self.parameters_per_model[:k]), \
                                                      jnp.sum(self.parameters_per_model[:k+1]),dtype=int)]) \
                                                      for k in range(len(self.parameters_per_model))]
        return p_list

    def copy(self):
        return self.__class__(models=copy.deepcopy(self.models))

    def save_history(self,p):
        for i,model in enumerate(self.models):
            model.save_history(p[np.arange(np.sum(self.parameters_per_model[:i]).astype(int), \
                                            np.sum(self.parameters_per_model[:i+1]).astype(int),dtype=int)])


class CompositeModel(ContainerModel):
    def call(self,p,x,i,*args):
        # prstringt(self.parameters_per_model)
        for k,model in enumerate(self.models):
            indices = np.arange(np.sum(self.parameters_per_model[:k]),
                                np.sum(self.parameters_per_model[:k+1]),dtype=int)
            # print(indices)
            x = model(p[indices],x,i,*args)
        return x

    def composite(self,x):
        if isinstance(x,CompositeModel):
            return CompositeModel(models=[*self.models,*x.models])
        else:
            return CompositeModel(models=[*self.models,x])


class AdditiveModel(ContainerModel):
    def call(self,p,x,i,*args):
        output = 0.0
        # PARALLELIZABLE
        for k,model in enumerate(self.models):
            indices = np.arange(np.sum(self.parameters_per_model[:k]),
                                np.sum(self.parameters_per_model[:k+1]),dtype=int)
            output += model(p[indices],x,i,*args)
        return output

    def __add__(self,x):
        if isinstance(x,AdditiveModel):
            return AdditiveModel(models=[*self.models,*x.models])
        else:
            return AdditiveModel(models=[*self.models,x])

    def __radd__(self,x):
        if isinstance(x,AdditiveModel):
            return AdditiveModel(models=[*self.models,*x.models])
        else:
            return AdditiveModel(models=[*self.models,x])


class EnvelopModel(Model):
    def __init__(self,model):
        super(EnvelopModel,self).__init__()
        self.model = model

    def __call__(self,p,*args):
        # if there are no parameters coming in, then use the stored parameters
        if len(p) == 0:
            return self.call(np.array([]),*args)
        else:
            return self.call(p,*args)

    def __getitem__(self,i):
        return self.model[i]

    def fit(self,*args):
        self.model.fit(*args)

    def fix(self,*args):
        self.model.fix(*args)

    def unpack(self,p):
        self.model.unpack(p)

    def get_parameters(self):
        return self.model.get_parameters()

    def display(self,string=''):
        super(EnvelopModel,self).display(string)
        tab = '  '
        string += tab
        self.model.display(string)
        string = string[:-len(tab)]

    def split_p(self,p):
        return self.model.split_p(p)


class JaxEnvLinearModel(EnvelopModel):
    def __init__(self,xs,model,p=None):
        super(JaxEnvLinearModel,self).__init__(model)
        # when defining ones own model, need to include inputs as xs, outputs as ys
        # and __call__ function that gets ya ther, and params (1d ndarray MUST BE BY SCIPY) to be fit
        # also assumes epoches of data that is shifted between
        self.xs = xs
        if p is not None:
            if p.shape == self.xs.shape:
                self.p = p
            else:
                logging.error('p {} must be the same shape as x_grid {}'.format(p.shape,xs.shape))
        else:
            self.p = np.zeros(xs.shape)

    def call(self,p,x,i,*args):
        ys = self.model(p,self.xs,i,*args)
        y = jax.numpy.interp(x, self.xs, ys)
        return y


class ConvolutionalModel(Model):
    def __init__(self,p=None):
        super(ConvolutionalModel,self).__init__()
        if p is None:
            self.p = np.array([0,1,0])
        else:
            self.p = p

    def call(self,p,x,i):
        y = jnp.convolve(x,p,mode='same')
        return y


class EpochSpecificModel(Model):
    def __init__(self,epoches):
        super(EpochSpecificModel,self).__init__()
        self.n = epoches
        self._epoches = slice(0,epoches)

    def __call__(self,p,*args):
        # if there are no parameters coming in, then use the stored parameters
        if len(p) == 0:
            return self.call(self.p[self._epoches],*args)
        else:
            return self.call(p,*args)

    def grid_search(self,grid,loss,model,data,epoches=None):
        if epoches is None:
            epoches = slice(0,self.n)
        # put all submodels in fixed mode except the shiftingmodel
        # to be searched then take loss of each epoch
        # that we hand the loss a slice of the shift array
        # since at __call__ itll on take the shift_grid[i,j] element
        model.fix()
        # index is the index of the submodel to grid search this is redundant
        self.fit(epoches=epoches)
        if isinstance(model,ContainerModel):
            model.get_parameters()
        # this is called because this resets the parameters per model
        # array
        # I want to have this be done when a submodel is put into fix or fix mode
        loss_arr = np.empty(grid.shape)
        for i in range(grid.shape[0]):
            for j in range(grid.shape[1]):
                # print(shift_grid[:,j].shape)
                loss_arr[i,j] = loss(grid[:,j],data,i,model)
        return loss_arr

    def add_component(self,value=0.0,n=1):

        self.p = np.concatenate((self.p,value*np.ones(n)))

    def fix(self,epoches=None):

        self._fit = False
        if epoches is not None:
            self._epoches = epoches

    def fit(self,epoches=None):

        self._fit = True
        if epoches is not None:
            self._epoches = epoches

    def get_parameters(self):
        if self._fit:
            return jnp.array(self.p[self._epoches])
        else:
            return jnp.array([])

    def unpack(self,p):
        if len(p) != 0:
            self.p[self._epoches] = p


class ShiftingModel(EpochSpecificModel):
    def __init__(self,p=None,epoches=0):
        if p is None:
            self.p = np.zeros(epoches)
        else:
            self.p = np.array(p)
            epoches = len(p)
        super(ShiftingModel,self).__init__(epoches)

    def call(self,p,x,i):

        return p[i] + x


class StretchingModel(EpochSpecificModel):
    def __init__(self,p=None,epoches=0):
        if p is None:
            self.p = np.ones((epoches))
        else:
            self.p = p
            epoches = len(p)
        super(StretchingModel,self).__init__(epoches)


    def call(self,p,x,i):

        return p[i] * x


class JaxLinear(Model):
    def __init__(self,xs,p=None):
        super(JaxLinear,self).__init__()
        # when defining ones own model, need to include inputs as xs, outputs as ys
        # and __call__ function that gets ya ther, and params (1d ndarray MUST BE BY SCIPY) to be fit
        # also assumes epoches of data that is shifted between
        self.xs = xs
        if p is not None:
            if p.shape == self.xs.shape:
                self.p = p
            else:
                logging.error('p {} must be the same shape as x_grid {}'.format(p.shape,xs.shape))
        else:
            self.p = np.zeros(xs.shape)

    def call(self,p,x,i):
        # print()
        # print(p.shape)
        y = jax.numpy.interp(x, self.xs, p)
        return y

# foo = jax.numpy.interp(xs, x - shifts, params)
# res = scipy.optimize.minimize(lamdba(): (ys - foo(xs))**2, params, args=(self,*args), method='BFGS', jac=jax.grad(loss),
#        options={'disp': True})
