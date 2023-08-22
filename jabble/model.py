import numpy as np
import jax
import jax.numpy as jnp
import jax.experimental.sparse
# import matplotlib.pyplot as plt
import scipy.optimize
import scipy.signal as signal
import sys
from functools import partial
from jax import jit

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

def _parameters_indices_check(p_i):


    return

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

        self.results = []

    def __call__(self,p,*args):
        # if there are no parameters coming in, then use the stored parameters
        if len(p) == 0:
            assert self._fit == False
            return self.call(self.p,*args)
        else:
            assert self._fit == True
            # if self._fit_indices is not None:
                # return jnp.where(self._fit_indices,p,self.p)
            return self.call(p,*args)

    def optimize(self,loss,data,method='L-BFGS-B',bounds=None,verbose=False,options={},save_history=False,save_loss=False,*args):
        # Fits the Model
        self.save_history = save_history
        self.save_loss    = save_loss
        # if loss is None:
        #     loss_ind = np.arange(data.shape[0])

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
                tmp        = np.zeros((data.ys.shape[0],*initialize.shape))
                tmp[0,...] = initialize
                for i in range(1,data.ys.shape[0]):
                    tmp[i,...] = loss(p,data,i,self)
                self.loss_history.append(tmp)

            return np.array(val,dtype='f8'),np.array(grad,dtype='f8')

        res = scipy.optimize.minimize(val_gradient_function, self.get_parameters(), jac=True,
               method=method,
               args=(data,self,*args),
               options=options,
               bounds=bounds
               )
        self.results.append(res)
        self.unpack(res.x)
        return res
    
    def new_lbfgsb(self,loss,data,verbose=False,save_history=False,save_loss=False,**options):
        # Fits the Model
#         self.save_history = save_history
#         self.save_loss    = save_loss
        # if loss is None:
        #     loss_ind = np.arange(data.shape[0])

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
                tmp        = np.zeros((data.ys.shape[0],*initialize.shape))
                tmp[0,...] = initialize
                for i in range(1,data.ys.shape[0]):
                    tmp[i,...] = loss(p,data,i,self)
                self.loss_history.append(tmp)

            return np.array(val,dtype='f8'),np.array(grad,dtype='f8')

        x, f, d = scipy.optimize.fmin_l_bfgs_b(val_gradient_function, self.get_parameters(), None, (data, self))
        self.results.append(d)
        self.unpack(x)
        return d

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

    def fit_p(self,p_i):

        # self._fit_indices = p_i
        self._fit = True

    def get_parameters(self):
        if self._fit:
            # if self._fit_indices is not None:
            #     return jnp.array(self.p[self._fit_indices])
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
        if self._fit:
            # if self._fit_indices is not None:
            #     return jnp.array(p[self._fit_indices])
            return jnp.array(p)
        else:
            return jnp.array([])

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
        p_list = [p[jnp.arange(jnp.sum(self.parameters_per_model[:k]), \
                                    jnp.sum(self.parameters_per_model[:k+1]),dtype=int) ] \
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
            return self.call(self.p,*args)
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
            return jnp.array(self.p)
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

    def call(self,p,x,i,*arg):

        return x - p[i]
    
class OrderShiftingModel(EpochSpecificModel):
    def __init__(self,p=None,epoches=0):
        if p is None:
            assert epoches != 0
            self.p = np.zeros(epoches)

        else:
            self.p  = np.array(p)
            epoches = len(p)
        super(OrderShiftingModel,self).__init__(epoches)

    def call(self,p,x,i,j,*args):

        return x - p[j]


class StretchingModel(EpochSpecificModel):
    def __init__(self,p=None,epoches=0):
        if p is None:
            self.p = np.ones((epoches))
        else:
            self.p = p
            epoches = len(p)
        super(StretchingModel,self).__init__(epoches)


    def call(self,p,x,i,*args):

        return p[i] * x
# every factual statement needs references

class ScipySpline(Model):
    def __init__(self,xs,p=None):
        super(ScipySpline,self).__init__()
        import scipy.interpolate
        self.xs = xs
        if p is not None:
            if p.shape == self.xs.shape:
                self.p = p
            else:
                logging.error('p {} must be the same shape as x_grid {}'.format(p.shape,xs.shape))
        else:
            self.p = np.zeros(xs.shape)

    def call(self,p,x,i,*arg):
        f = scipy.interpolate.CubicSpline(self.xs,p)
        return f(x)

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

    def call(self,p,x,i,*arg):
        # print()
        # print(p.shape)
        y = jax.numpy.interp(x, self.xs, p)
        return y

def _alpha_recursion(i,j,p):
    # fixed values
    if i < 0 or i > p:
        return 0
    if j < 0 or j > p:
        return 0
    if p == 0:
        return 1
    # recursion
    return  (j/p) * _alpha_recursion(i,j,p-1) +\
            (1/p) * _alpha_recursion(i-1,j,p-1) +\
            ((p+1-j)/p) * _alpha_recursion(i,j-1,p-1) -\
            (1/p) * _alpha_recursion(i-1,j-1,p-1)

class BSpline:
    def __init__(self,p):
        # calculate coefficients of basis spline functions(piecewise polynomial)
        # p piecewise functions with p+1 terms
#         p = p.astype(int)
        p = int(p)
        self.p = p
        self.alphas = np.zeros((p+1,p+1))
        for i in range(p+1):
            for j in range(p+1):
                self.alphas[i,j] = _alpha_recursion(i,j,p)
        self.alphas = jnp.array(self.alphas)

    @partial(jit,static_argnums=[0])
    def __call__(self,x,*args):
        # Recentering
        i = jnp.floor(x + (self.p+1) / 2).astype(int)
        cond1 = i >= 0
        cond2 = i <= self.p
        f = jnp.where((cond1 * cond2), \
                      jnp.polyval(self.alphas[::-1,i], (x + (self.p+1) / 2) % 1), \
                      0.0)
        return f

def _sparse_design_matrix(x,xp,basis,a):

    '''
        Internal Function for general_interp_simple
        to do:
        make sparse using 'a' and fast
        choose fast sparse encoding
        the fastest for lstsq solve
        time all
    '''
    # get difference between cardinal splines
    dx     = xp[1] - xp[0]
    # get distance between each element and the closest cardinal basis to its left
    inputs = ((x - xp[0]) / dx) % 1
    # get index of that cardinal basis spline
    index  = (x - xp[0]) // dx
    # print((x - xp[0]) / dx, index, inputs)
    # create range of of basis vectors that each datapoint touches bc each basis spans from -a to a from its center
    arange = jnp.arange(-a-1,a+2,step=1.0)
    # get indices of these basis vectors
    ainds  = jnp.floor(arange)
    # print(arange, ainds)
    # use indices, and a indices to get all ms associated with each datapoint
    ms = (index[:,None] + ainds[None,:]).flatten().astype(int)
    # use indices of datapoints and a indices to get js
    js = (np.arange(0,len(x),dtype=int)[:,None] * np.ones(ainds[None,:].shape)).flatten().astype(int)
    # compute nonzero x values
    x_tilde = (inputs[:,None] - ainds[None,:]).flatten()
    # restrict boundary conditions
    cond1 = (ms >= 0)
    cond2 = (ms < xp.shape[0])
    data    = jnp.where((cond1*cond2), basis(x_tilde), 0.0)
    indices = jnp.concatenate((ms[:,None],js[:,None]),axis=1)
    # create sparse matrix using these ms,js indices and basis evaluation
    out  = jax.experimental.sparse.BCOO((data,indices),shape=(xp.shape[0],x.shape[0]))
    return out

@partial(jit,static_argnums=[3,4])
def cardinal_basis_sparse(x, xp, ap, basis, a):
    '''XP must be equally spaced
        deal boundary conditions 0D, 0N
        padding points
        with user inputs values

        for future test for a, where basis function goes to zero
    '''
#     a = int((p+1)//2)
    # GET EXACT SPACING from XP
#     assert jnp.allclose(xp[1:] - xp[:-1],dx) # require uniform spacing
#     X    = _sparse_design_matrix(xp,xp,dx,basis,a)

    # This is a toeplitz matrix solve, may be faster also sparse
    # make sparse scipy jax function maybe
#     alphas,res,rank,s = jnp.linalg.lstsq(X,fp)


    # This is to ensure the multiplication with the sparse mat works
    ap = jnp.array(ap)
    design = _sparse_design_matrix(x,xp,basis,a)
    # design = jax.experimental.sparse.BCOO.fromdense(_full_design_matrix(x,xp,basis))

    check = (ap[:,None] * design).sum(axis=0)
    # print(np.array(design.todense()))
    if isinstance(check, jax.experimental.sparse.bcoo.BCOO):
        return check.todense()

    return check

def _full_design_matrix(x,xp,dx,basis):
    from jax.experimental import sparse
    '''
        Internal Function for general_interp_simple
        to do:
        make sparse using 'a' and fast
        choose fast sparse encoding
        the fastest for lstsq solve
        time all
    '''
    dx = xp[1] - xp[0]
    input = (x[None,:] - xp[:,None])/dx
    # cond1 = jnp.floor(input) < -a
    # cond2 = jnp.floor(input) >  a
    # input[(cond1 + cond2).astype(bool)] = 0.0
    # spinput = sparse.BCOO.fromdense(input)

    return basis(input)

@partial(jit,static_argnums=[3,4])
def general_interp_loose(x, xp, ap, basis):
    '''XP must be equally spaced
        deal boundary conditions 0D, 0N
        padding points
        with user inputs values

        for future test for a, where basis function goes to zero
    '''
    dx = xp[1] - xp[0]
    # a = int((p+1)//2)
    # GET EXACT SPACING from XP
#     assert jnp.allclose(xp[1:] - xp[:-1],dx) # require uniform spacing
#     X    = _sparse_design_matrix(xp,xp,dx,basis,a)

    # This is a toeplitz matrix solve, may be faster also sparse
    # make sparse scipy jax function maybe
#     alphas,res,rank,s = jnp.linalg.lstsq(X,fp)

    return (ap[:,None] * _full_design_matrix(x,xp,dx,basis)).sum(axis=0)

class BSplineModel(Model):
    def __init__(self,xs,p_val=2,p=None):
        super(BSplineModel,self).__init__()
        # when defining ones own model, need to include inputs as xs, outputs as ys
        # and __call__ function that gets ya ther, and params (1d ndarray MUST BE BY SCIPY) to be fit
        # also assumes epoches of data that is shifted between
        self.spline = BSpline(p_val)
        self.p_val = p_val
        self.xs = xs
        if p is not None:
            if p.shape == self.xs.shape:
                self.p = p
            else:
                logging.error('p {} must be the same shape as x_grid {}'.format(p.shape,xs.shape))
        else:
            self.p = np.zeros(xs.shape)

    def call(self,p,x,*args):
        # print()
        # print(p.shape)
        a = (self.p_val+1)/2
        y = cardinal_basis_sparse(x, self.xs, p, self.spline,a)
        return y

# foo = jax.numpy.interp(xs, x - shifts, params)
# res = scipy.optimize.minimize(lamdba(): (ys - foo(xs))**2, params, args=(self,*args), method='BFGS', jac=jax.grad(loss),
#        options={'disp': True})
