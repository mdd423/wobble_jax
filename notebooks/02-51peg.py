#!/usr/bin/env python
import sys

sys.path.append(".")
sys.path.append("..")
import jabble.model
import jabble.dataset
import jabble.loss
import jabble.physics 
import astropy.units as u

import h5py
import matplotlib.pyplot as plt
import scipy.optimize

from jaxopt import GaussNewton
import jax.numpy as jnp
import jax
import numpy as np
from mpl_axes_aligner import align

import os
import jabble.physics

import jax.config

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_disable_jit", True)


import os
import datetime
import functools
from multiprocessing import Pool

def _internal(dataset,shifts,airmass,order_num,star_name,out_dir):
    # file = h5py.File(file_name, "r")
    loss = MyChiSquare()

    resolution = 115_000
    p_val = 2
    vel_padding = 100 * u.km / u.s

    pts_per_wavelength = 1/20
    norm_p_val = 2

    # for order_num in orders:
    model_name = os.path.join('models',star_name+'_o{}.mdl'.format(order_num))

    # if not os.path.exists(model_name):
    # dataset, shifts, airmass = get_dataset(file,[order_num])

    print(dataset[0].xs.devices(),shifts.devices())
    model = get_model(dataset,resolution,p_val,vel_padding,shifts,airmass,pts_per_wavelength,norm_p_val)
    model = pre_train_cycle(model, dataset, loss)
    jabble.model.save(model_name,model)
    # else:
    #     print(model_name, ' already exists skipping.')
    #     continue

def _profilingrun(star_name, out_dir, dataset_list, shifts_list, airmass_list, orders):
    
    for dataset,shifts,airmass,order in zip(dataset_list, shifts_list, airmass_list, orders):
        _internal(star_name,out_dir,dataset,shifts,airmass,order)

def prerun_orders(file_name,star_name,out_dir):
    
    num_devices = jax.device_count()
    file = h5py.File(file_name, "r")
    orders = np.arange(file['data'].shape[0])
    dataset_list, shifts_list, airmass_list = get_dataset(file)
    if num_devices > 1:
        with Pool(num_devices) as p:
            p.starmap(functools.partial(_internal,star_name=star_name,out_dir=out_dir),zip(dataset_list, shifts_list, airmass_list, orders))
    else:
        import cProfile
        
        cProfile.run('_profilingrun(star_name,out_dir,dataset_list, shifts_list, airmass_list, orders)','02-norm.stats')
        
def get_dataset(file):

    init_shifts_list = []
    airmass_list = []
    dataset_list = []
    
    for iii in range(file["data"].shape[0]):
        ys = []
        xs = []
        yivar = []
        mask = []

        init_shifts_list.append([])
        airmass_list.append([])
        for jjj in range(file["data"].shape[1]):
            ys.append(jnp.array(file["data"][iii,jjj,:]))
            xs.append(jnp.array(file["xs"][iii,jjj,:]))
            yivar.append(jnp.array(file["ivars"][iii,jjj,:]))
            mask.append(jnp.zeros(file["data"][iii,jjj,:].shape,dtype=bool))

            init_shifts_list[-1].append(jabble.physics.shifts(file["bervs"][jjj]))
            airmass_list[-1].append(file["airms"][jjj])
        
        dataset_list.append(jabble.dataset.Data.from_lists(xs,ys,yivar,mask))
    
        init_shifts_list[-1] = jnp.array(init_shifts_list[-1])
        airmass_list[-1] = jnp.array(airmass_list[-1])
                         
    
    # dataset.to_device(device)
    # init_shifts = jax.device_put(init_shifts,device)
    # airmass = jax.device_put(airmass,device)

    return dataset_list, init_shifts_list, airmass_list


def gpu_optimize(
        self, loss, data, device_store, device_op, batch_size, options={}
    ):
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
        # mask extra points added to block
        xs, ys, yivar, mask = data.blockify(device_store)

        ##########################################################
    
        x, f, d = scipy.optimize.fmin_l_bfgs_b(
            val_gradient_function, self.get_parameters(), None, (xs,ys,yivar,mask,self,device_op,batch_size), **options
        )
        self.results.append(d)
        self._unpack(jax.device_put(jnp.array(x),device_op))
        return d



class MyChiSquare(jabble.loss.ChiSquare):
    def __call__(self, p, xs, ys, yivar, mask, i, model, *args):
        return jnp.where(~mask,yivar * (((ys - model(p,xs,i,*args))**2)),0.0)
    
    def loss_all(self,p,xs,ys,yivar,mask,model,device_op,batch_size,*args):
        
        #blockify parameters
        #what if normalization model has different number of parameters per model
        #anything that is going to take the epoch index needs to blockified and be the only parameter
        #this is an issue with the normalization model because its epoch specific but the parameters vary by epoch
        # just putting in the zero below will assume the same number of parameters as the first one
        # not the one specified, whats the better way to do multiple epoch fitting without indices
        
        def _internal(xs_row,ys_row,yivar_row,mask_row,index):
            return self(p,xs_row,ys_row,yivar_row,mask_row,index,model,*args).sum()

        indices = jnp.arange(0,xs.shape[0],dtype=int)

        rounds = int(np.ceil(xs.shape[0]/batch_size))
        out = 0.0
        for iii in range(rounds):
            top = np.min([(iii+1)*batch_size,xs.shape[0]])
            # print(device_op)
            temp = jax.vmap(_internal, in_axes=(0, 0, 0, 0, 0), out_axes=0)(jax.device_put(xs[(iii*batch_size):top],device_op), \
                                                                            jax.device_put(ys[(iii*batch_size):top],device_op), \
                                                                            jax.device_put(yivar[(iii*batch_size):top],device_op), \
                                                                            jax.device_put(mask[(iii*batch_size):top],device_op), \
                                                                            jax.device_put(indices[(iii*batch_size):top],device_op))
            out += temp.sum()
        return out



class NewNormalizationModel(jabble.model.Model):
    def __init__(self, model, size):
        super(NewNormalizationModel, self).__init__()
        self.p     = jnp.tile(model.p,size)
        self.model = model

        self.model_p_size = len(model.p)
        self.size  = size

    def call(self, p, x, i, *args):
        # indices = self.get_indices(i)
        # parameters = 
        x = self.model.call(p.reshape(self.size,self.model_p_size)[i], x, i, *args)
        return x


def get_normalization_model(dataset,norm_p_val,pts_per_wavelength):
    len_xs = np.max([np.max(dataframe.xs) - np.min(dataframe.xs) for dataframe in dataset])
    min_xs = np.min([np.min(dataframe.xs) for dataframe in dataset])
    max_xs = np.max([np.max(dataframe.xs) for dataframe in dataset])

    shifts = jnp.array([dataframe.xs.min() - min_xs for dataframe in dataset])

    x_num = int((np.exp(max_xs) - np.exp(min_xs)) * pts_per_wavelength)
    x_spacing = len_xs/x_num
    x_grid = jnp.linspace(-x_spacing,len_xs+x_spacing,x_num+2) + min_xs
    
    model = jabble.model.IrwinHallModel_vmap(x_grid, norm_p_val)
    size  = len(dataset)

    print(size,len(model.p))
    norm_model = NewNormalizationModel(model,size)
    return jabble.model.ShiftingModel(shifts).composite(norm_model)



def get_model(dataset,resolution,p_val,vel_padding,init_shifts,airmass,pts_per_wavelength,norm_p_val):
    
    dx = jabble.physics.delta_x(2 * resolution)
    x_grid = jnp.arange(np.min(np.concatenate(dataset.xs)), np.max(np.concatenate(dataset.xs)), step=dx, dtype="float64")
    
    model_grid = jabble.model.create_x_grid(
        x_grid, vel_padding.to(u.m/u.s).value, 2 * resolution
    )  
    print(len(model_grid))
    model = jabble.model.CompositeModel(
        [
            jabble.model.ShiftingModel(init_shifts),
            jabble.model.IrwinHallModel_vmap(model_grid, p_val),
        ]
    ) + jabble.model.CompositeModel(
        [
            jabble.model.IrwinHallModel_vmap(model_grid, p_val),
            jabble.model.StretchingModel(airmass),
        ]
    ) + get_normalization_model(dataset,norm_p_val,pts_per_wavelength)

    return model


def pre_train_cycle(model, dataset, loss):
    device = jax.devices(backend='cpu')[0]
    # print(device,flush=True)
    # Fit Normalization
    model.fix()
    model.fit(2,1)
    model.display()
    res1 = gpu_optimize(model, loss, dataset, device, device, 2000)
    # print(res1)

    return model

def train_cycle(model, dataset, loss, device_store, device_op):
    
    # Fit Stellar & Telluric Template
    model.fix()
    model.fit(0, 1)
    model.fit(1, 0)
    res1 = gpu_optimize(model,loss,dataset, device_store, device_op,2000)#model.optimize(loss, dataset)
    print(res1)
    
    # Fit RV
    model.fix()
    model.fit(0, 0)
    res1 = gpu_optimize(model,loss,dataset, device_store, device_op,2000)# model.optimize(loss, dataset)
    print(res1)

    # RV Parabola Fit
    # model.fix()
    # shift_search = jabble.physics.shifts(np.linspace(-10, 10, 100))
    # model[0][0].parabola_fit(shift_search, loss, model, dataset)
    # print(type(model_p[0][0].p))

    # Fit Everything
    model.fix()
    model.fit(0, 0)
    model.fit(0, 1)
    model.fit(1, 0)
    model.fit(2)
    res1 = gpu_optimize(model,loss,dataset, device_store, device_op,2000)#model.optimize(loss, dataset)
    print(res1)

    return model

def run_orders(file,star_name, device_store, device_op):
    
    for order_num in range(file['data'].shape[0]):
        model_name = os.path.join('..','models',star_name+'_o{}.mdl'.format(order_num))
        final_name = os.path.join('..','models',star_name+'_o{}_fit.mdl'.format(order_num))
        if not os.path.exists(final_name):
            dataset, shifts, airmass = get_dataset(file,[order_num],device_op)
            
            model = jabble.model.load(model_name)

            model.fix()
            model.display()
            test_inds = model[2][1].get_indices(2)
            print(len(test_inds), np.sum(test_inds))
            model.to_device(device_op)
            model = train_cycle(model, dataset, loss, device_store, device_op)
            jabble.model.save(final_name,model)
        else:
            continue


if __name__ == "__main__":
    today = datetime.date.today()
    out_dir = os.path.join('..','out',today.strftime("%y-%m-%d"))

    file_name_b = "data/barnards_e2ds.hdf5"
    file_name_p = "data/51peg_e2ds.hdf5"   

    cpus = jax.devices("cpu")
    # print(cpus,flush=True)
    star_name_b, star_name_p = 'barnards','peg51'

    for star_name,file in zip([star_name_b, star_name_p],[file_name_b,file_name_p]):
        prerun_orders(file,star_name,out_dir)
        # run_orders(file   ,star_name, cpus[0],gpus[0])