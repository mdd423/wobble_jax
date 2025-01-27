import numpy as np
import math
import jax
import jax.numpy as jnp
import jax.experimental.sparse

# import jaxopt

import scipy.optimize
import scipy.signal as signal
import scipy.constants
import sys
from functools import partial
from jax import jit
import h5py

import copy

import astropy.constants as const
import logging

import pickle  # 5 as pickle
import jabble.dataset
import jabble.physics


def save(filename, model):
    with open(filename, "wb") as output:  # Overwrites any existing file.
        pickle.dump(model, output, pickle.HIGHEST_PROTOCOL)


def load(filename):
    with open(filename, "rb") as input:  # Overwrites any existing file.
        model = pickle.load(input)
        return model


def create_x_grid(xs, vel_padding, resolution):
    """
    Get grid in log wavelength space with equally spaced steps at speed of light over the resolution,
    padded by velocity padding on each side

    Parameters
    ----------
    xs : `np.ndarray`
        the log wavelength grid or grids of the datasets to be modeled
    vel_padding: `float`
        velocity padding on either side of the min and max of the above xs grid in m/s
    resolution: `float`
        instrument resolution that sets grid spacing

    Returns
    -------
    sp : `nd.array`
        Evenly spaced numpy.ndarray
    """

    x_min = xs.min()
    x_max = xs.max()
    step = jabble.physics.shifts(scipy.constants.c / resolution)
    x_padding = jabble.physics.shifts(vel_padding)
    model_grid = np.arange(x_min - x_padding, x_max + x_padding, step)
    return model_grid


class Model:
    """
    Base Model class with scipy optimizer using jax.

    Attributes
    ----------
    results : `list`
       List of results objects produced each call to optimize
    """

    def __init__(self, *args, **kwargs):
        self._fit = False
        self.func_evals = []
        self.history = []
        self.save_history = False

        self.loss_history = []
        self.save_loss = []

        self.results = np.empty(shape=(0),dtype=[('task', 'U64'), ('nit', int), ('funcalls',int),('warnflag',int),\
                                                ('value',np.double),('loss','U64')])

        self.metadata = {}

    def __call__(self, p, *args):
        """
        Call wrapper function. Checks if there are incoming parameters, if not uses fixed parameters.
        Then sends to self.call.

        Parameters
        ----------
        p : `(jnp.ndarray, [])`
            parameters of model, or empty list if using fixed parameters


        Returns
        ----------
        self.call : `jnp.ndarray`
            Returns array from call.
        """
        if len(p) == 0:
            assert self._fit == False
            return self.call(self.p, *args)
        else:
            assert self._fit == True
            return self.call(p, *args)

    def gaussnewton(self, data, *args):
        """
        maybe defunct optimizer using residuals to fit parameters using jaxopt.GaussNewton.
        Fits using y residuals times y information

        Parameters
        ----------
        data : `jabble.Dataset`
            jabble.Dataset, that is handed to the Loss function during optimization

        Returns
        ----------
        gn_sol : `OptStep`
            JaxOpt.OptStep object with results from optimization loop
        """

        def chi_1(p):
            residual = jnp.zeros(data.xs.shape)
            for i in range(data.xs.shape[0]):

                residual = residual.at[i, :].set(
                    (data.ys[i, :] - self(p, data.xs[i, :], i)) * data.yivar[i, :]
                )
            return residual

        gn = jax.GaussNewton(residual_fun=chi_1)
        gn_sol = gn.run(self.get_parameters())

        self.results.append(gn_sol)
        self._unpack(gn_sol.params)
        return gn_sol

    def optimize(self, loss, data, device_store, device_op, batch_size, options={}):
        """
        Choosen optimizer for jabble is scipy.fmin_l_bfgs_b.
        optimizes all parameters in fit mode with respect to the loss function using jabble.Dataset

        Parameters
        ----------
        loss : `jabble.Loss`
            jabble.loss object,
        data : `jabble.Dataset`
            jabble.Dataset, that is handed to the Loss function during optimization
        verbose : `bool`
            if true prints, loss, grad dot grad at every function
        save_history : `bool`
            if true, saves values of parameters at every function call
        save_loss : `bool`
            if true, saves loss array every function call of optimization
        options :
            additional keyword options to be passed to scipy.fmin_l_bfgs_b


        Returns
        ----------
        d : `dict`
            Results from scipy.fmin_l_bgs_b call
        """

        func_grad = jax.value_and_grad(loss.loss_all, argnums=0)

        def val_gradient_function(p, *args):
            val, grad = func_grad(p, *args)

            return np.array(val, dtype="f8"), np.array(grad, dtype="f8")

        # blockify dataset
        datablock, metablock = data.blockify(device_store)

        ##########################################################
        loss.ready_indices(self)
        x, f, d = scipy.optimize.fmin_l_bfgs_b(
            val_gradient_function,
            self.get_parameters(),
            None,
            (datablock, metablock, self, device_op, batch_size),
            **options,
        )

        self.results = np.append(self.results, np.array([(d['task'],d['nit'],d['funcalls'],d['warnflag'],f,repr(loss)),],\
                                                        dtype=[('task', 'U64'), ('nit', int), ('funcalls',int),('warnflag',int),('value',np.double),('loss','U64')]), axis=0) #{'task': d['task'], 'nit':d['nit'], 'funcalls':d['funcalls'],'warnflag':d['warnflag'],\'value':f,'loss':repr(loss)}
        # self.results.append({"out": d, "value": f, "loss": repr(loss)})
        self._unpack(jax.device_put(jnp.array(x), device_op))
        return d

    def __add__(self, x):
        if isinstance(x, AdditiveModel):
            return AdditiveModel(models=[self, *x.models])
        return AdditiveModel(models=[self, x])

    def __radd__(self, x):
        return AdditiveModel(models=[self, x])

    def composite(self, x):
        return CompositeModel(models=[self, x])

    def evaluate(self, x, i):
        self.fit()
        return self(self.p, x, i)

    def fix(self):
        """
        Sets model into fixed mode. None of the parameters will be varied at optimization call.
        """
        self._fit = False

    def fit(self):
        """
        Sets model into fitting model. All parameters will be varied during next optimization call.
        """
        self._fit = True

    def to_device(self, device):
        """
        Move all parameters to given device
        """
        self.p = jax.device_put(self.p, device)

    def get_parameters(self):
        """
        Gets parameters from the model that are in fitting mode.

        Returns
        ----------
        parameters : `jnp.ndarray`
            Returns jax array of all parameters that are to be fit.
        """
        if self._fit:
            return jnp.array(self.p)
        else:
            return jnp.array([])

    def _unpack(self, p):
        if len(p) != 0:
            self.p = p

    def display(self, string=""):
        """
        Prints model and submodels if self is ContainerModel, and the number of parameters in fit mode.
        """
        out = string + "-" + self.__class__.__name__ + "-"
        out += "-----------------------------------"
        params = str(len(self.get_parameters()))
        while len(out + params) % 17 != 0:
            # print(len())
            out += "-"

        out += params
        print(out)

    def copy(self):
        return copy.deepcopy(self)

    def save(self,filename: str, mode: str) -> None:
        
        if mode == "hdf":
            with h5py.File(filename + "." + mode,'w') as file:
                self.save_hdf(file)
        elif mode == "pkl":
            with open(filename + "." + mode, "wb") as output:  # Overwrites any existing file.
                pickle.dump(self, output, pickle.HIGHEST_PROTOCOL)

    def save_hdf(self,file,index=[]):
        ind_str = ""
        for x in index:
            ind_str += '[{}]'.format(x)
        # print(ind_str + self.__class__.__name__)
        group = file.create_group(ind_str + self.__class__.__name__)
        # for key in self.__dict__:
        #     if key[0] != "_":
                # print(key)
        if len(self.results) > 0:
            res_group = file.create_group('results')
            for item in self.results.dtype.names:
                if self.results[item].dtype == '<U64':
                    res_group.attrs[item] = np.array(self.results[item], dtype=h5py.string_dtype(encoding='utf-8'))
                else:
                    res_group.create_dataset(item,data=self.results[item])
        try:
            group.create_dataset("p",data=self.p)
        except AttributeError:
            pass
        # for key in self.__dict__:
        #     if key != "models":
        #         print(key)
        #         if self.__dict__[key]:
                    
        #             group.create_dataset(key, data=self.__dict__[key])
        return group
    
    def load_hdf(cls,group):

        return cls(p=group["p"])


class ContainerModel(Model):
    """
    ContainerModel class that can contain submodels each with individual parameters.

    Attributes
    ----------
    models : `list`
       List of jabble.Model objects
    parameters_per_model : `np.ndarray`
        Array with shape len(models), ith value of array is the number of parameters being fit in the ith model.
    """

    def __init__(self, models, *args, **kwargs):
        super(ContainerModel, self).__init__()
        self.models = models
        self._parameters_per_model = jnp.empty((len(models)), dtype=int)
        self.size = len(models)
        for i in range(self.size):
            self._parameters_per_model = self._parameters_per_model.at[i].set(
                len(self.models[i].get_parameters())
            )
        self.create_param_bool()

    def append(self, model):
        """
        Directing append self.models list with another model and update parameters_per_model attr

        Attributes
        ----------
        model : `jabble.Model`
            Model to be added to the list of models
        """
        self.models.append(model)
        self._parameters_per_model = jnp.concatenate(
            (self._parameters_per_model, len(model.get_parameters()))
        )
        self.create_param_bool()

    def __call__(self, p, *args):

        if len(p) == 0:
            return self.call(jnp.array([]), *args)
        else:
            return self.call(p, *args)

    def __getitem__(self, i):
        """
        Indexing, i, ContainerModel returns ith model in models list.
        """
        return self.models[i]

    def _unpack(self, p):
        """
        After optimization, fit parameters are handed back to the static Model attr.
        """
        for k, model in enumerate(self.models):
            indices = self.get_indices(k)
            model._unpack(p[indices])

    def get_parameters(self):
        """
        Return all parameters in fit mode from all submodels.

        Returns
        -------
        p : `jnp.ndarray`
            1-D jax array concatenated all submodels paraemeters that are in fit mode
        """
        x = jnp.array([])
        for i in range(self.size):
            params = self.models[i].get_parameters()
            self._parameters_per_model = self._parameters_per_model.at[i].set(
                params.shape[0]
            )
            x = jnp.concatenate((x, params))
        self.create_param_bool()
        return x

    def create_param_bool(self):
        self._param_bool = np.zeros((self.size, int(np.sum(self._parameters_per_model))))
        for i in range(self.size):
            self._param_bool[
                i,
                int(jnp.sum(self._parameters_per_model[:i])) : int(
                    jnp.sum(self._parameters_per_model[: i + 1])
                ),
            ] = jnp.ones(
                (
                    int(jnp.sum(self._parameters_per_model[: i + 1]))
                    - int(jnp.sum(self._parameters_per_model[:i]))
                ),
                dtype=bool,
            )
        self._param_bool = jnp.array(self._param_bool, dtype=bool)

    def fit(self, i=None, *args):
        """
        Set a submodel in fit mode, or all submodels to fit mode if no i is given.

        Parameters
        ----------
        i : `int`
            index of the submodel in models list to fit, if left as None all submodels will be set to fitting mode.
        """
        if i is None:
            for j in range(self.size):
                self.models[j].fit()
                self._parameters_per_model = self._parameters_per_model.at[j].set(
                    self.models[j].get_parameters().shape[0]
                )
        else:
            self[i].fit(*args)
            self._parameters_per_model = self._parameters_per_model.at[i].set(
                self[i].get_parameters().shape[0]
            )
        self.create_param_bool()

    def fix(self, i=None, *args):
        """
        Set a submodel in fixed mode, or all submodels to fixed mode if no i is given.

        Parameters
        ----------
        i : `int`
            index of the submodel in models list to fix, if left as None all submodels will be set to fixed mode.
        """
        if i is None:
            for j, model in enumerate(self.models):
                model.fix()
                self._parameters_per_model = self._parameters_per_model.at[j].set(0)
        else:
            self[i].fix(*args)
            self._parameters_per_model = self._parameters_per_model.at[i].set(0)
        self.create_param_bool()

    def to_device(self, device):
        """
        Move all parameters of all submodels to given device
        """
        for model in self.models:
            model.to_device(device)

    def display(self, string=""):
        self.get_parameters()
        super(ContainerModel, self).display(string)
        for i, model in enumerate(self.models):
            # print(model)
            tab = "  {}".format(i)
            string += tab
            model.display(string)
            string = string[: -len(tab)]

    def split_p(self, p):
        p_list = [p[self.get_indices(k)] for k in range(len(self._parameters_per_model))]

        return p_list

    def copy(self):
        return self.__class__(models=copy.deepcopy(self.models))

    def save_history(self, p):
        for i, model in enumerate(self.models):
            model.save_history(p[self.get_indices(i)])

    def get_indices(self, i):
        """
        Get array of ints for the ith submodel, in models list using parameters_per_model
        Returns
        -------
        indices : 'np.ndarray(int)`
            Array of indices for the parameters in the ith model that is in fitting mode
        """
        return self._param_bool[i]

    def save_hdf(self,file,index=[]):
        
        # group = file.create_group(self.__class__.__name__)
        group = super().save_hdf(file,index)
        for i,model in enumerate(self.models):
            model.save_hdf(group,index=index+[i])
        return group
    
    def load_hdf(cls,group):
        model_list = []
        for key in group.keys():
            cls_sub = eval(key.split(']')[-1])
            print(group[key].keys())
            model_list.append(cls_sub.load_hdf(cls_sub,group[key]))
        return cls(model_list)


class CompositeModel(ContainerModel):
    """
    ContainerModel sequentially applies models to input.
    .. math::
        f  = g_n(g_{n-1}(...g_1(g_0(x))))
    """

    def call(self, p, x, i, *args):
        for k, model in enumerate(self.models):
            indices = self.get_indices(k)
            x = model(p[indices], x, i, *args)
        return x

    def composite(self, x):
        if isinstance(x, CompositeModel):
            return CompositeModel(models=[*self.models, *x.models])
        else:
            return CompositeModel(models=[*self.models, x])


class AdditiveModel(ContainerModel):
    """
    ContainerModel applies all models to input then adds results.
    .. math::
        f  = g_n(x) + g_{n-1}(x) + ...g_1(x) + g_0(x)
    """

    def call(self, p, x, i, *args):
        output = 0.0
        # PARALLELIZABLE
        for k, model in enumerate(self.models):
            indices = self.get_indices(k)
            output += model(p[indices], x, i, *args)
        return output

    def __add__(self, x):
        if isinstance(x, AdditiveModel):
            return AdditiveModel(models=[*self.models, *x.models])
        else:
            return AdditiveModel(models=[*self.models, x])

    def __radd__(self, x):
        if isinstance(x, AdditiveModel):
            return AdditiveModel(models=[*self.models, *x.models])
        else:
            return AdditiveModel(models=[*self.models, x])


class EnvelopModel(Model):
    """
    EnvelopModel similar to ContainerModel but only containers one submodel.
    """

    def __init__(self, model):
        super(EnvelopModel, self).__init__()
        self.model = model

    def __call__(self, p, *args):
        # if there are no parameters coming in, then use the stored parameters
        if len(p) == 0:
            return self.call(np.array([]), *args)
        else:
            return self.call(p, *args)

    def __getitem__(self, i):
        return self.model[i]

    def fit(self, *args):
        self.model.fit(*args)

    def fix(self, *args):
        self.model.fix(*args)

    def to_device(self, device):
        """
        Move all parameters to given device
        """
        self.p = jax.device_put(self.model.p, device)

    def _unpack(self, p):
        self.model._unpack(p)

    def get_parameters(self):
        return self.model.get_parameters()

    def display(self, string=""):
        super(EnvelopModel, self).display(string)
        tab = "  "
        string += tab
        self.model.display(string)
        string = string[: -len(tab)]

    def split_p(self, p):
        return self.model.split_p(p)


class JaxEnvLinearModel(EnvelopModel):
    """
    Applies jax interpolation to the incoming parameters, then hands them to the submodel.
    .. math::
        f = g(j(p),x)
    """

    def __init__(self, xs, model, p=None):
        super(JaxEnvLinearModel, self).__init__(model)
        self.xs = xs
        if p is not None:
            if p.shape == self.xs.shape:
                self.p = p
            else:
                logging.error(
                    "p {} must be the same shape as x_grid {}".format(p.shape, xs.shape)
                )
        else:
            self.p = jnp.zeros(xs.shape)

    def call(self, p, x, i, *args):
        ys = self.model(p, self.xs, i, *args)
        y = jax.numpy.interp(x, self.xs, ys)
        return y


class ConvolutionalModel(Model):
    """
    Model that convolves the input, x, with parameters, p, using jax function.
    """

    def __init__(self, p=None, *args, **kwargs):
        super(ConvolutionalModel, self).__init__()
        if p is None:
            self.p = jnp.array([0, 1, 0])
        else:
            self.p = p

    def call(self, p, x, *args):
        y = jnp.convolve(x, p, mode="same")
        return y


class EpochSpecificModel(Model):
    """
    Subset of Models that have only have one parameter associated with each epoch, i.

    Parameters
    ----------
    epoches : `int`
        Number of epochs
    """

    def __init__(self, n, *args, **kwargs):
        super(EpochSpecificModel, self).__init__()
        self.n = n
        self._epoches = slice(0, n)

    def __call__(self, p, *args):
        # if there are no parameters coming in, then use the stored parameters
        if len(p) == 0:
            return self.call(self.p, *args)
        else:
            return self.call(p, *args)

    def grid_search(self, grid, loss, model, data, epoches=None):
        """
        Function that will individually grid search each parameter.

        Parameters
        ----------
        grid : `np.ndarray`
            (M,N) M searches at each epoch, N epoch array
        loss : `jabble.Loss`
            Objective being optimized.
        model : `jabble.Model`
            The full model.
        data : `jabble.Dataset`
            The dataset being optimized with respect to.

        Returns
        -------
        loss_arr : `jnp.array`
            (M,N) loss evaluated on full model at all grid values at their respective epochs.
        """

        if epoches is None:
            epoches = slice(0, self.n)

        model.fix()
        self.fit(epoches=epoches)
        if isinstance(model, ContainerModel):
            model.get_parameters()

        def _internal(grid, j):

            return jnp.array(
                [jnp.sum(loss(grid, data, i, model)) for i in range(self.n)]
            )

        loss_arr = jax.vmap(_internal, in_axes=(1, 0), out_axes=1)(
            grid, np.arange(0, grid.shape[1])
        )
        return loss_arr

    def parabola_fit(self, array1d, loss, model, data):
        """
        Finds parabolic minima of the grid search of parameters. Sets parameters to optimized result.

        Parameters
        ----------
        array1d : `np.ndarray`
            1-D array (M,) about each parameter to search.
        loss : `jabble.Loss`
            Objective being optimized.
        model : `jabble.Model`
            The full model.
        data : `jabble.Dataset`
            Dataset to optimize with respect to.

        """

        # First use grid search function to get loss grid.
        grid = np.array(self.p[:, None] + array1d[None, :])
        loss = np.array(self.grid_search(grid, loss, model, data))

        # Loop lowest value on loss grid and its 2 neighbors.
        # Fit a parabola, take derivative, then find root.
        def _internal(g, l):

            poly = jnp.polyfit(g, l, deg=2)
            lmin = jnp.roots(jnp.polyder(poly), strip_zeros=False).real
            return lmin, jnp.polyval(poly, lmin)

        minima = np.argmin(loss, axis=1).astype(int)
        inds_i = np.arange(0, loss.shape[0], dtype=int).repeat(3).reshape(-1, 3)
        inds_j = (
            (minima[:, None] + np.array([-1, 0, 1])[None, :]).flatten().reshape(-1, 3)
        )
        l_min, g_min = jax.vmap(_internal, in_axes=(0, 0), out_axes=0)(
            grid[inds_i, inds_j], loss[inds_i, inds_j]
        )
        self.p = jnp.array(jnp.squeeze(l_min))

    def add_component(self, value=0.0, n=1):
        """
        Add more epochs to model
        """
        self.p = np.concatenate((self.p, value * np.ones(n)))

    def fix(self, epoches=None):

        self._fit = False
        if epoches is not None:
            self._epoches = epoches

    def fit(self, epoches=None):

        self._fit = True
        if epoches is not None:
            self._epoches = epoches

    def get_parameters(self):
        if self._fit:
            return jnp.array(self.p)
        else:
            return jnp.array([])

    def f_info(self, model, data, device):
        """
        Get fischer information on parameters of the model.
        Since each parameter is independent of all other epochs, fischer information matrix is diagonal,
        thus returns this diagonal.

        Parameters
        ----------
        model : `jabble.Model`
            The full model to evaluate.
        data : `jabble.Dataset`
            Data to be evaluate.

        Returns
        -------
        f_info : 'jnp.array`
            (N,) array of diagonal of fischer information matrix.
        """
        model.fix()
        self.fit()
        model.display()

        datablock, metablock = data.blockify(device)
        # def _internal(self, p, datarow, metarow, model, *args)
        #     return loss(self, p, datarow, metarow, model, *args)
        duddx = jax.jacfwd(model, argnums=0)

        def get_dict(datablock, index):
            return {key: datablock[key][index] for key in datablock.keys()}

        # SUPPOSE TO BE FASTER, BUT KILLS KERNEL
        # LIKELY DUE TO MEM ALLOC
        # def _internal(datarow):
        #     return ((duddx(model.get_parameters(),datarow['xs'],datarow)**2) * datarow['yivar'][:,None]).sum(axis=0)
        # f_info = jax.vmap(_internal,(0),0)(datablock)

        f_info = np.zeros((len(data), len(model.get_parameters())))
        for i in range(len(data)):
            datarow = get_dict(datablock, i)
            metarow = get_dict(metablock, i)
            # sum over pixels
            # assumes diagonal variance in pixel, and time
            f_info[i, :] += (
                (duddx(model.get_parameters(), datarow["xs"], metarow) ** 2)
                * datarow["yivar"][:, None]
            ).sum(axis=0)

            # f_info[i,:] += ((duddx(model.get_parameters(),datarow,metarow,model)**2) * datarow['yivar'][:,None]).sum(axis=0)
        # sum over datarows
        return f_info.sum(axis=0)


class EpochShiftingModel(EpochSpecificModel):
    """
    Model that adds different value to input at each epoch.
    .. math::
        f(p,x,i) = x + p[i]

    Parameters
    ----------
    p : `np.ndarray`
        N epoch length array of initial values of p.
    """

    def __init__(self, p=None, epoches=0, *args, **kwargs):
        if p is None:
            self.p = jnp.zeros(epoches)
        else:
            self.p = jnp.array(p)
            epoches = len(p)
        super(EpochShiftingModel, self).__init__(epoches)

    def call(self, p, x, meta, *arg):

        return x - p[meta["times"]]


class ShiftingModel(EpochSpecificModel):
    """
    Model that adds different value to input at each epoch.
    .. math::
        f(p,x,i) = x + p[i]

    Parameters
    ----------
    p : `np.ndarray`
        N epoch length array of initial values of p.
    """

    def __init__(self, p=None, epoches=0, *args, **kwargs):
        if p is None:
            self.p = jnp.zeros(epoches)
        else:
            self.p = jnp.array(p)
            epoches = len(p)
        super(ShiftingModel, self).__init__(epoches)

    def call(self, p, x, meta, *arg):

        return x - p[meta["index"]]


class StretchingModel(EpochSpecificModel):
    """
    Model that multiplies different value to input at each epoch.
    .. math::
        f(p,x,i) = x * p[i]

    Parameters
    ----------
    p : `np.ndarray`
        N epoch length array of initial values of p.
    """

    def __init__(self, p=None, epoches=0, *args, **kwargs):
        if p is None:
            self.p = jnp.ones((epoches))
        else:
            self.p = p
            epoches = len(p)
        super(StretchingModel, self).__init__(epoches)

    def call(self, p, x, meta, *args):

        return p[meta["index"]] * x
    

import jabble.cardinalspline

@partial(jit, static_argnums=[3, 4, 5])
def cardinal_vmap_model(x, xp, ap, basis, a):
    """
    Evaluates cardinal basis using vmap design matrix.

    Parameters
    ----------
    x : `np.ndarray`
        array (N,) of x values to evaluate.
    xp : `np.ndarray`
        evenly spaced array (M,) of centerpoints of the basis functions
    ap : `np.ndarray`
        coefficient for basis functions
    basis : anything callable
        Basis function. Should probably make sense as a real basis function.
    a : `float`
        basis function must go to zero outside this value.

    Returns
    -------
    out : 'np.ndarray`
        y array (N,) of evaluated vmap cardinal basis
    """
    dx = xp[1] - xp[0]
    ap = jnp.array(ap)
    # assert np.all(dx == xp[1:] - xp[:-1])
    arange = jnp.floor(jnp.arange(-a - 1, a + 2, step=1.0)).astype(int)
    # get distance between each element and the closest cardinal basis to its left
    inputs = ((x - xp[0]) / dx) % 1
    # get index of the cardinal basis spline to datapoint's left
    index = ((x - xp[0]) // dx).astype(int)

    def _internal(inputs, index):

        return jnp.dot(ap[index - arange], basis(inputs + arange))

    out = jax.vmap(_internal, in_axes=(0, 0), out_axes=0)(inputs, index)

    return out

class CardinalSplineMixture(Model):
    """
    Model that evaluates input using full Irwin-Hall cardinal basis design matrix.

    Parameters
    ----------
    xs : `np.ndarray`
        centerpoints of cardinal (evenly spaced) basis functions
    p_val : `int`
        order of the Irwin-Hall basis functions
    p : `np.ndarray`
        the initial control points. If None, then initialized at zero.

    """

    def __init__(self, xs, p_val=2, p=None, *args, **kwargs):
        super(CardinalSplineMixture, self).__init__()
        # when defining ones own model, need to include inputs as xs, outputs as ys
        # and __call__ function that gets ya ther, and params (1d ndarray MUST BE BY SCIPY) to be fit
        # also assumes epoches of data that is shifted between
        self.spline = jabble.cardinalspline.CardinalSplineKernel(p_val)
        self.p_val = p_val
        self.xs = jnp.array(xs)
        if p is not None:
            if p.shape == self.xs.shape:
                self.p = p
            else:
                logging.error(
                    "p {} must be the same shape as x_grid {}".format(p.shape, xs.shape)
                )
        else:
            self.p = jnp.zeros(xs.shape)

    def call(self, p, x, *args):

        a = (self.p_val + 1) / 2
        y = cardinal_vmap_model(x, self.xs, p, self.spline, a)
        return y
    
    def save_hdf(self,file,index):
        group = super(CardinalSplineMixture,self).save_hdf(file,index)
        group.create_dataset("xs",data = self.xs)
        group.create_dataset("alphas", data = self.spline.alphas)
        group.create_dataset("p_val",data= self.p_val)
        return group
    
    def load_hdf(cls,group):

        return cls(xs=np.array(group["xs"]), p_val=int(np.array(group["p_val"])), p=np.array(group["p"]))
    

class NormalizationModel(Model):
    def __init__(self, p, model, size, *args, **kwargs):
        super(NormalizationModel, self).__init__()
        self.p = p#jnp.tile(model.p, size)
        self.model = model

        self.model_p_size = len(model.p)
        self.size = size

    def call(self, p, x, meta, *args):

        x = self.model.call(
            (p.reshape(self.size, self.model_p_size)[meta["index"]]).flatten(),
            x,
            meta,
            *args,
        )
        return x

    def save_hdf(self,file,index=[]):
        group = file.create_group(self.model.__class__.__name__)
        group = super(type(self.model),self.model).save_hdf(group,index)
        
        file = super(NormalizationModel, self).save_hdf(file,index)
        file.create_dataset("size",data = self.size)
        return file
    
    def load_hdf(cls,group):
        for key in group.keys():
            if key != "size":
                if key != "p":
                    cls_sub = eval(key.split(']')[-1])
                    temp_model = cls_sub.load_hdf(cls_sub,group[key])
        return cls(p=group["p"], model=temp_model, size=group["size"])