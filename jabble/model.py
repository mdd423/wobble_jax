from tokenize import group

import numpy as np
import math
import jax
import jax.numpy as jnp
from datetime import datetime
import jax.experimental.sparse

# import jaxopt

import jax.scipy.optimize
import scipy.constants
from functools import partial
import h5py

import copy
import logging

import pickle  # 5 as pickle
import jabble.physics

def get_submodel_indices(self, i, j=None, *args):
    # this recurses through submodels when given a set of indices to that submodel
    # then returns of a bool array of the length of the total number of parameters
    # of whole model
    # with 1's at the parameters of the specific submodel, 0's elsewhere
    s_temp = self.get_indices(i)
    if j is not None:
        temp = get_submodel_indices(self[i],j,*args)
        s_inds = s_temp[temp]
        return s_inds
    return s_temp

def load(filename):
    with h5py.File(filename, 'r') as hf:
        for key in hf.keys():
            if key == 'metadata':
                metadata = {}
                metagroup = hf['metadata']
                for key in metagroup.keys():
                    # print(h5py.check_string_dtype(metagroup[key].dtype))
                    if h5py.check_string_dtype(metagroup[key].dtype) is not None:
                        metadata[key] = np.array(metagroup[key].asstr()[:], dtype=str)
                    else:   
                        metadata[key] = np.array(metagroup[key][:])
                    
            obj_name = key.split('/')[-1]
            obj_name = obj_name.split('_')[0]
            # print(obj_name,dir('__file__'))
            if obj_name in dir(jabble.model):
                model = eval(obj_name).load(hf[key])
        model.metadata = metadata
    return model

def save(filename,model,header={}):
    with h5py.File(filename, 'w') as hf:
        obj_name = model.__class__.__name__
        group = hf.create_group(obj_name)
        model.save(group)
        
        hf.attrs['date_created'] = datetime.now().isoformat()
        for key in header.keys():
            hf.attrs[key] = header[key]
            
        metagroup = hf.create_group('metadata')
        for key in model.metadata.keys():
            if model.metadata[key].dtype.kind in {'U', 'S'}:
                metagroup.create_dataset(key,data=np.char.encode(model.metadata[key], 'utf-8'))
            else:
                metagroup.create_dataset(key,data=model.metadata[key])

def save_rvs(filename,rv_inds,model,dataset,loss,time,device):
    with h5py.File(filename, 'w') as hf:
        rvs_model = model
        for ind in rv_inds:
            rvs_model = rvs_model[ind]
        hf.create_dataset('rvs', data=jabble.physics.velocities(rvs_model.p))
        hf.create_dataset('time', data=time)
        hf.create_dataset('rv_error_fisher', data=jabble.quickplay.get_RV_error_fisher(model, dataset, device, rv_inds))
        hf.create_dataset('rv_error_2d_est', data=jabble.quickplay.get_RV_error_2d_est(model, dataset, loss, rv_inds, device))
        hf.create_dataset('rv_error_2d_full', data=jabble.quickplay.get_RV_error_2d(model, dataset, loss, rv_inds, device))


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
    size = int(math.ceil((x_max - x_min + 2 * x_padding) / step))
    model_grid = np.linspace(x_min - x_padding, x_max + x_padding, size)
    return model_grid


class Model:
    """
    Base Model class with scipy optimizer using jax.

    Attributes
    ----------
    p : `jnp.ndarray`
        jax array of parameters of the model
    _fit : `bool`
        If True, model is in fitting mode, all parameters will be varied during optimization.
    results : `np.ndarray`
        Structured array of results from optimization calls.
    metadata : `dict`
        Dictionary to hold any metadata for the model.
    """

    def __init__(self, *args, **kwargs):
        self._fit = False

        self.results = np.empty(shape=(0),dtype=[('task', 'U64'), ('funcalls', int),\
                                                            ('nit',int), ('warnflag',int),('f',np.double),('loss','U64')])

        self.metadata = {}

    @partial(jax.jit, static_argnums=(0, 4))
    def __call__(self, p, x, meta, margs=()):
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
            return self.call(self.p, x, meta, margs)
        else:
            assert self._fit == True
            return self.call(p, x, meta, margs)

    def gaussnewton(self, data, *args):
        """
        Deprecated!
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
        Choosen optimizer for jabble is jax.scipy.optimize.minimize 'bfgs'.
        optimizes all parameters in fit mode with respect to the loss function using jabble.Dataset

        Parameters
        ----------
        loss : `jabble.Loss`
            jabble.Loss function to be minimized during optimization
        data : `jabble.Dataset`
            jabble.Dataset, that is handed to the Loss function during optimization
        device_store : jax.Device
            Device to store data on
        device_op : jax.Device
            Device to perform operations on
        batch_size : int
            Number of data epochs to use in each optimization step
        options : dict
            Options to pass to the optimizer
        Returns
        ----------
        d : `dict`
            Results from scipy.fmin_l_bgs_b call
        """

        # blockify dataset
        datablock,metablock = data.blockify(device_store)

        ##########################################################
        loss.ready_indices(self)
        
        x,f,d = scipy.optimize.fmin_l_bfgs_b(
            func=loss.loss_all,
            fprime=jax.grad(loss.loss_all),
            x0=self.get_parameters(),
            args=(datablock, metablock, self, device_op, batch_size),
            **options,
        )
        # print(result)
        self.results = np.append(self.results, np.array([(d['task'],d['funcalls'],d['nit'],d['warnflag'],f,repr(loss)),],\
                                                        dtype=[('task', 'U64'), ('funcalls', int),\
                                                            ('nit',int), ('warnflag',int),('f',np.double),('loss','U64')]), axis=0) 
        self._unpack(jax.device_put(jnp.array(x), device_op))
        return x,f,d
    
    def __add__(self, x):
        if isinstance(x, AdditiveModel):
            return AdditiveModel(models=[self, *x.models])
        return AdditiveModel(models=[self, x])

    def __radd__(self, model):
        '''
        Combine two models additively, where the output of self and model are summed
        y = self(x) + model(x)
        Parameters
        ----------
        model : `jabble.Model`
            Model to add to self

        Returns
        -------
        additive_model : `jabble.Model`
            AdditiveModel object with self and model summed
        '''
        return AdditiveModel(models=[self, model])

    def composite(self, model):
        '''
        Combine two models in series, where the output of self is the input to model
        y = model(self(x))
         Parameters
         ----------
         model : `jabble.Model`
             Model to apply after self

         Returns
         -------
         composite_model : `jabble.Model`
             CompositeModel object with self and model in series
         '''
        return CompositeModel(models=[self, model])

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

    def fisher_full(model, data, device):
        """
        Get full fischer information on parameters of the model.
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
        model.fit()
        model.display()

        datablock,metablock = data.blockify(device)
        # def _internal(self, p, datarow, metarow, model, *args)
        #     return loss(self, p, datarow, metarow, model, *args)
        dfdt = jax.jacfwd(model, argnums=0)

        curvature_all = np.zeros((len(data),datablock['xs'].shape[1], len(model.get_parameters())))

        for i in range(len(data)):
            datarow = datablock.ele(i).to_device(device)
            metarow = metablock.ele(i).to_device(device)
        
            curvature_all[i,:,:] = jnp.where(~datarow["mask"][:, None]*\
                                    np.ones(len(model.get_parameters()))[None, :],
                (dfdt(model.get_parameters(), datarow["xs"], metarow)),
                0.0
            )

        f_info = np.zeros((len(model.get_parameters()),len(model.get_parameters())))
        for i in range(len(data)):
            datarow = datablock.ele(i).to_device(device)
            
            f_info += np.einsum('j,jn,jm->nm',datarow['yivar'][:],curvature_all[i,:,:],\
                        curvature_all[i,:,:])

        return f_info

    def transform_variance(model,xp,xq,covar,metarowp={},metarowq={}):
        '''
        Given covariance matrix of parameters, transform to covariance of model outputs at points xp and xq
        Parameters
        ----------
        model : `jabble.Model`
            The full model to evaluate.
        xp : `jnp.array`
            First set of points to evaluate variance at
        xq : `jnp.array`
            Second set of points to evaluate variance at
        covar : `jnp.array`
            Covariance matrix of the parameters of the model
        metarowp : `dict`, optional
            Metadata dictionary for points xp, by default {}
        metarowq : `dict`, optional
            Metadata dictionary for points xq, by default {}
        Returns
        -------
        transformed_covar : `jnp.array`
            Covariance matrix of model outputs between points xp and xq
        '''
        model.fit()
        dydt = jax.jacfwd(model, argnums=0)
        dypdt = dydt(model.get_parameters(),xp,metarowp)
        dyqdt = dydt(model.get_parameters(),xq,metarowq)
        model.fix()
        return dypdt @ covar @ dyqdt.transpose()

    def save(self,hf):
        pass

    def load(hf):
        obj_name = hf.name.split('/')[-1].split('_')[0]
        model = eval(obj_name)()
        return model

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
        self.create_param_indices()

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
        self.create_param_indices()

    @partial(jax.jit, static_argnums=(0, 4))
    def __call__(self, p, x, meta, margs=()):

        if len(p) == 0:
            return self.call(jnp.array([]), x, meta, margs)
        else:
            return self.call(p, x, meta, margs)

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
        self.create_param_indices()
        return x

    def create_param_indices(self):
        self._param_indices = []
        for i in range(len(self.models)):
            start = int(jnp.sum(self._parameters_per_model[:i]))
            stop = int(jnp.sum(self._parameters_per_model[:i+1]))
            self._param_indices.append(jnp.arange(start, stop))

    def get_indices(self, i):
        return self._param_indices[i]

    def split_p(self, p):
        p_list = [p[self.get_indices(k)] for k in range(len(self._parameters_per_model))]

        return p_list

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
        self.create_param_indices()

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
        self.create_param_indices()

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

    def copy(self):
        return self.__class__(models=copy.deepcopy(self.models))

    def save_history(self, p):
        for i, model in enumerate(self.models):
            model.save_history(p[self.get_indices(i)])

    def save(self,hf):
        
        iteration = 0
        for model in self.models:
            while model.__class__.__name__ + f'_{iteration}' in hf.keys():
                iteration += 1
            subgroup = hf.create_group(model.__class__.__name__ + f'_{iteration}',track_order=True)
            model.save(subgroup)

    def load(hf):
        models = []
        for group_key in hf.keys():
            obj_name = hf[group_key].name.split('/')[-1]
            obj_name = obj_name.split('_')[0]
            models.append(eval(obj_name).load(hf[group_key]))
            
        obj_name = hf.name.split('/')[-1].split('_')[0]
        return eval(obj_name)(models)

    def tree_sum(model,p_list,reduce_index):
        '''where A and N are the length of the parameters, theta of submodels. If a submodel is container of containers 
        like fs(g(x|theta_a)|theta_n) + ft(g(x|theta_b)|theta_m)
        then split_p only drops one layer down, (A+N,B+M)
        thus split (A+N) -> A,N '''
        sum_list = []
        for i,(ele,submodel) in enumerate(zip(p_list,model.models)):
            if isinstance(submodel, jabble.model.ContainerModel) and i == reduce_index[0]:
                new_list, mark_temp = submodel.tree_sum(submodel.split_p(ele),reduce_index[1:])
                mark_ele = mark_temp + i
                sum_list += new_list
            else:
                if i == reduce_index[0]:
                    mark_ele = i
                sum_list.append(len(ele))
        return sum_list, mark_ele
    
    def reduce_fisher(model,f_info,reduce_index):
        '''
        Given full fischer information matrix of model, reduce to fischer information of submodel at reduce_index
        Parameters
        ----------
        model : `jabble.Model`
            The full model to evaluate.
        f_info : `jnp.array`
            Full fischer information matrix of the model.
        reduce_index : `list`
            List of indices to reduce down to the submodel of interest.
        Returns
        -------
        reduced_f_info : 'jnp.array`
            Reduced fischer information matrix of the submodel at reduce_index.
        '''
        # NOW REDUCE
        p_list = model.split_p(model.get_parameters())
        sum_list, mark_ele = model.tree_sum(p_list,reduce_index)

        print(sum_list,mark_ele)
        print('a',0,int(np.sum(sum_list[:mark_ele])),int(np.sum(sum_list[:mark_ele+1])),0,\
            '\nn',int(np.sum(sum_list[:mark_ele])),int(np.sum(sum_list[:mark_ele+1])))

        faa_info_uu = f_info[0:int(np.sum(sum_list[:mark_ele])),0:int(np.sum(sum_list[:mark_ele]))]
        faa_info_ub = f_info[0:int(np.sum(sum_list[:mark_ele])),int(np.sum(sum_list[:mark_ele+1])):-1]
        faa_info_bb = f_info[int(np.sum(sum_list[:mark_ele+1])):-1,int(np.sum(sum_list[:mark_ele+1])):-1]
        faa_info_bu = f_info[int(np.sum(sum_list[:mark_ele+1])):-1,0:int(np.sum(sum_list[:mark_ele]))]

        faa_info = np.block([[faa_info_uu,faa_info_ub],[faa_info_bu,faa_info_bb]])

        fnn_info = f_info[int(np.sum(sum_list[:mark_ele])):int(np.sum(sum_list[:mark_ele+1])),\
                        int(np.sum(sum_list[:mark_ele])):int(np.sum(sum_list[:mark_ele+1]))]

        fan_info_l = f_info[int(np.sum(sum_list[:mark_ele])):int(np.sum(sum_list[:mark_ele+1])),\
                            0:int(np.sum(sum_list[:mark_ele]))]
        fan_info_r = f_info[int(np.sum(sum_list[:mark_ele])):int(np.sum(sum_list[:mark_ele+1])),\
                            int(np.sum(sum_list[:mark_ele+1])):-1]
        
        fan_info = np.concatenate([fan_info_l,fan_info_r],axis=1)

        fna_info_t = f_info[0:int(np.sum(sum_list[:mark_ele])),\
                            int(np.sum(sum_list[:mark_ele])):int(np.sum(sum_list[:mark_ele+1]))]
        fna_info_b = f_info[int(np.sum(sum_list[:mark_ele+1])):-1,\
                            int(np.sum(sum_list[:mark_ele])):int(np.sum(sum_list[:mark_ele+1]))]
        
        fna_info = np.concatenate([fna_info_t,fna_info_b],axis=0)

        print(fna_info.shape,fan_info.shape,faa_info.shape,fnn_info.shape)
        return fnn_info - (fan_info @ np.linalg.inv(faa_info) @ fna_info)
    

class CompositeModel(ContainerModel):
    """
    ContainerModel sequentially applies models to input.
    .. math::
        f  = g_n(g_{n-1}(...g_1(g_0(x))))
    """
    @partial(jax.jit, static_argnums=(0, 4))
    def call(self, p, x, meta, margs=()):
        for k, model in enumerate(self.models):
            indices = self.get_indices(k)
            x = model(p[indices], x, meta, margs)
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
    @partial(jax.jit, static_argnums=(0, 4))
    def call(self, p, x, meta, margs=()):
        output = 0.0
        # PARALLELIZABLE
        for k, model in enumerate(self.models):
            indices = self.get_indices(k)
            output += model(p[indices], x, meta, margs)
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


class EpochSpecificModel(Model):
    """
    Subset of Models that have only have one parameter associated with each epoch, i.

    Parameters
    ----------
    n : `int`
        Number of epochs
    """

    def __init__(self, p, which_key='index', *args, **kwargs):
        super(EpochSpecificModel, self).__init__()
        self.n = len(p)
        self._epoches = slice(0, self.n)
        self.p = jnp.array(p)
        
        self.which_key = which_key

    @partial(jax.jit, static_argnums=(0, 4))
    def __call__(self, p, x, meta, margs=()):
        # if there are no parameters coming in, then use the stored parameters
        if len(p) == 0:
            return self.call(self.p, x, meta, margs)
        else:
            return self.call(p, x, meta, margs)

    def grid_search(self, grid, loss, model, data, device, epoches=None):
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
        if isinstance(model, jabble.model.ContainerModel):
            model.get_parameters()

        datablock,metablock = data.blockify(device)
        def _internal(grid):
            Q = jnp.zeros((len(data)))

            for iii in range(len(Q)):
                datarow = datablock.ele(iii).to_device(device)
                metarow = metablock.ele(iii).to_device(device)
                
                Q = Q.at[iii].set(loss(grid, datarow, metarow, model).sum().astype(np.double))

            uniques = np.unique(metablock[self.which_key])

            out = jnp.zeros(self.p.shape)
            for iii,unq in enumerate(uniques):
                out = out.at[unq].set(Q[np.where(metablock[self.which_key] == unq)].sum())
            return out

        loss_arr = jax.vmap(_internal, in_axes=(1), out_axes=1)(
            grid
        )
        return loss_arr

    def parabola_fit(self, array1d, loss, model, data, device_1, device_2):
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
        loss = np.array(self.grid_search(grid, loss, model, data, device_1))

        # Loop lowest value on loss grid and its 2 neighbors.
        # Fit a parabola, take derivative, then find root.
        def _internal(g, l):

            poly = jnp.polyfit(g, l, deg=2)
            g_min = jnp.roots(jnp.polyder(poly), strip_zeros=False).real
            return g_min, jnp.polyval(poly, g_min)

        minima = np.argmin(loss, axis=1).astype(int)
        mask_ends = ((minima == 0) + (minima == loss.shape[1]-1)).astype(bool)
        inds_i = np.arange(0, np.sum(~mask_ends), dtype=int).repeat(3).reshape(-1, 3)
        
        inds_j = (
            (minima[~mask_ends, None] + np.array([-1, 0, 1])[None, :]).flatten().reshape(-1, 3)
        )
        loss = jax.device_put(loss,device_2)
        grid = jax.device_put(grid,device_2)
        g_min, l_min = jax.vmap(_internal, in_axes=(0, 0), out_axes=0)(
            grid[inds_i, inds_j], loss[inds_i, inds_j]
        )
        g_min = jax.device_put(g_min,device_1)
        self.p = jax.device_put(self.p.at[~mask_ends].set(jnp.array(jnp.squeeze(g_min))),device_1)

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
        device : `jax.Device`
            Device to perform operations on.

        Returns
        -------
        f_info : 'jnp.array`
            (N,) array of diagonal of fischer information matrix.
        """
        model.fix()
        self.fit()
        model.display()

        datablock,metablock = data.blockify(device)
        # def _internal(self, p, datarow, metarow, model, *args)
        #     return loss(self, p, datarow, metarow, model, *args)
        duddx = jax.jacfwd(model, argnums=0)


        f_info = np.zeros((len(data), len(model.get_parameters())))
        for i in range(len(data)):
            datarow = datablock.ele(i).to_device(device)
            metarow = metablock.ele(i).to_device(device)
            # sum over pixels
            # assumes diagonal variance in pixel, and time
            f_info[i, :] += jnp.where(~datarow["mask"][:, None]*np.ones(len(model.get_parameters()))[None, :],
                (duddx(model.get_parameters(), datarow["xs"], metarow) ** 2)
                * datarow["yivar"][:, None],
                0.0
            ).sum(axis=0)

            # f_info[i,:] += ((duddx(model.get_parameters(),datarow,metarow,model)**2) * datarow['yivar'][:,None]).sum(axis=0)
        # sum over datarows
        return f_info.sum(axis=0)

    def save(self, group):
        group.create_dataset('p', data=self.p)
        group.create_dataset('which_key', data=self.which_key, dtype=h5py.string_dtype(encoding='utf-8'))

    def load(hf):
        obj_name = hf.name.split('/')[-1].split('_')[0]
        return eval(obj_name)(p=jnp.array(hf['p']), which_key=str(hf['which_key'][()].decode('utf-8'))) 

class ShiftingModel(EpochSpecificModel):
    """
    Model that adds different value to input at each epoch.
    .. math::
        f(p,x,i) = x + p[i]

    Parameters
    ----------
    p : `np.ndarray`
        N epoch length array of initial values of p.
    which_key : `str`
        Key in the metadata dictionary to use for epoch indexing.
    """
    @partial(jax.jit, static_argnums=(0, 4))
    def call(self, p, x, meta, margs=()):

        return x - p[meta[self.which_key]]


class StretchingModel(EpochSpecificModel):
    """
    Model that multiplies different value to input at each epoch.
    .. math::
        f(p,x,i) = x * p[i]

    Parameters
    ----------
    p : `np.ndarray`
        N epoch length array of initial values of p.
    which_key : `str`
        Key in the metadata dictionary to use for epoch indexing.
    """
    @partial(jax.jit, static_argnums=(0, 4))
    def call(self, p, x, meta, margs=()):

        return p[meta[self.which_key]] * x
    

import jabble.cardinalspline

@partial(jax.jit, static_argnums=[3, 4])
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
    @partial(jax.jit, static_argnums=(0, 4))
    def call(self, p, x, meta, margs=()):

        a = (self.p_val + 1) / 2
        y = cardinal_vmap_model(x, self.xs, p, self.spline, a)
        return y
    
    def save(self,group):
        group.create_dataset('p',data=self.p)
        group.create_dataset('xs',data=self.xs)
        group.create_dataset('p_val',data=self.p_val)
        group.create_dataset('alpha',data=self.spline.alphas)

    def load(group):
        return CardinalSplineMixture(p=jnp.array(group['p']),p_val=int(group['p_val'][()]),xs=jnp.array(group['xs']))
    

def cardinal_vmap_matrix(x, xp, basis, a):
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
    # get distance between each element and the closest cardinal basis to its left
    inputs = ((xp[:,None] - x[None,:]) / dx)
    # get index of the cardinal basis spline to datapoint's left
    out = basis(inputs)
    return out


class FullCardinalSplineMixture(CardinalSplineMixture):
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

    def __init__(self, xs, p_val=2, p=None):
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
    @partial(jax.jit, static_argnums=(0, 4))
    def call(self, p, x, meta, margs=()):

        a = (self.p_val + 1) / 2
        y = p @ cardinal_vmap_matrix(x, self.xs, self.spline, a)
        return y


class NormalizationModel(Model):
    '''
    Model that applies model to each input, x, with different parameters based on epoch index.
    .. math::
        f(p,x,i) = g(p[i],x)
    Parameters
    ----------
    p : `np.ndarray`
        N* M length array of initial values of p, where N is number of epochs, M is number of parameters in submodel.
    model : `jabble.Model`
        Submodel to be applied with different parameters at each epoch.
    size : `int`
        Number of epochs.
    which_key : `str`
        Key in the metadata dictionary to use for epoch indexing.
    '''
    def __init__(self, p, model, size, which_key='index'):
        super(NormalizationModel, self).__init__()
        self.p = p#jnp.tile(model.p, size)
        self.model = model

        self.which_key = which_key
        self.model_p_size = len(model.p)
        self.size = size
    @partial(jax.jit, static_argnums=(0, 4))
    def call(self, p, x, meta, margs=()):

        x = self.model.call(
            (p.reshape(self.size, self.model_p_size)[meta[self.which_key]]).flatten(),
            x,
            meta,
            margs,
        )
        return x

    def save(self,group):
            modelgroup = group.create_group('model_' + self.model.__class__.__name__)
            self.model.save(modelgroup)
            group.create_dataset('p',data=self.p)
            group.create_dataset('which_key',data=self.which_key,dtype=h5py.string_dtype('utf-8'))
            group.create_dataset('size',data=self.size)

    def load(group):
        for key in group.keys():
            if key.startswith('model_'):
                obj_name = key.split('_')[-1]
                model = eval(obj_name).load(group[key])
        return NormalizationModel(p=jnp.array(group['p'][:]),model=model,which_key=str(group['which_key'][()].decode('utf-8')),size=int(group['size'][()]))