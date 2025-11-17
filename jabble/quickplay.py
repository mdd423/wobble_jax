import sys

sys.path.append("..")
import jax.numpy as jnp
import jax
import numpy as np
import jabble.model
import jabble.physics
import h5py
import glob
import os

import astropy.units as u

def combine_rvs(rv_list,err_list,time_list,max_info=1e30,min_info=0.0):
    out_time = np.unique(np.concatenate(time_list))
    out_rv = np.zeros(out_time.shape)
    out_err = np.zeros(out_time.shape)

    all_rvs = np.concatenate(rv_list)
    all_info = 1/np.concatenate(err_list)**2
    all_times = np.concatenate(time_list)
    for ii,this_time in enumerate(out_time):
        # Filter RVs by information minimum and maximum
        rv_temp   = all_rvs[all_times==this_time]
        info_temp = all_info[all_times==this_time]
        
        rv_temp   = rv_temp[info_temp < max_info]
        info_temp = info_temp[info_temp < max_info]

        rv_temp   = rv_temp[info_temp > min_info]
        info_temp = info_temp[info_temp > min_info]

        rv_temp   = rv_temp[~np.isnan(info_temp)]
        info_temp = info_temp[~np.isnan(info_temp)]

        rv_temp   = rv_temp[~np.isinf(info_temp)]
        info_temp = info_temp[~np.isinf(info_temp)]
        # Combine RVs
        out_rv[ii]  = np.average(rv_temp,weights=info_temp)
        # Combined Errors
        out_err[ii] = np.sqrt(np.average(rv_temp**2,weights=info_temp) - out_rv[ii]**2)

    return out_rv, out_err, out_time

def create_summary_hdf(filename,rv_array,dir_files,dir):
    
    with h5py.File(filename,'w') as file:
        group = file.create_group("RVs")
        group.create_dataset("RV_comb",data=rv_array["RV_comb"])
        group.create_dataset("RV_err_comb",data=rv_array["RV_err_comb"])
        group.create_dataset("Time_comb",data=rv_array["Time_comb"])

        link_group = file.create_group("links")
        for dir_file in dir_files:
            
            head, tail = os.path.split(dir_file)
            link_group[tail] = h5py.ExternalLink(os.path.join(dir,dir_file),'/')

def load_summary_hdf(filename):

    with h5py.File(filename,'r') as file:
        rv_array = np.array([*zip(file["RVs"]["RV_comb"],file["RVs"]["RV_err_comb"],file["RVs"]["Time_comb"])],\
                            dtype=[("RV_comb",np.double),("RV_err_comb",np.double),("Time_comb",np.double)])
    return rv_array

def load_model_dir(path,dir_files,device,force_run=False,max_info=1e30,min_info=0.0):
   
    all_models = []
    all_data   = []
    with h5py.File(dir_files[0],'r') as file:
        time_length = len(np.array(file["Times"]))
    loss_array = np.zeros((len(dir_files),time_length))

    rv_list = []
    err_list = []
    time_list = []
    
    for iii,filename in enumerate(dir_files):
        with h5py.File(filename,'r') as file:
            # Extract velocities, errors, and times
            rv_list.append(np.array(file["RVs"]))
            err_list.append(np.array(file["RV_err"]))
            time_list.append(np.array(file["Times"]))
            
            model = jabble.model.load(file.attrs["model"])
            dataset = jabble.model.load(file.attrs["dataset"])

            all_models.append(model)
            all_data.append(dataset)

            loss_temp = np.array(file['Loss']).mean(axis=1)
            datablock, metablock = dataset.blockify(device)
            
            for jjj,time_unq in enumerate(np.unique(metablock["times"])):
                loss_array[iii,jjj] = loss_temp[metablock["times"] == time_unq].mean()

    # Combine RVs and create HDF summary in directory
    if not os.path.isfile(os.path.join(path,'RV_Summary.hdf')) or force_run:
        
        comb_rv, comb_err, comb_time = combine_rvs(rv_list,err_list,time_list,max_info=max_info,min_info=min_info)
        
        rv_array = np.array([*zip(comb_rv,comb_err,comb_time)],\
                            dtype=[("RV_comb",np.double),("RV_err_comb",np.double),("Time_comb",np.double)])
        create_summary_hdf(os.path.join(path,'RV_Summary.hdf'),rv_array,dir_files,path)
    else:
        rv_array = load_summary_hdf(os.path.join(path,'RV_Summary.hdf'))

    # min_order_of_chunk = np.array([np.unique(model.metadata["orders"]) for model in all_models]).min(axis=1)
    min_wavelength_of_chunk = np.array([np.min(np.concatenate(dataset.xs)) for dataset in all_data])
    # Reorder RV all array by min order and times
    if not os.path.isfile(os.path.join(path,'RV_all_Summary.npy')) or force_run:
        rv_difference = np.zeros((len(rv_list),rv_array["RV_comb"].shape[0]))
        for iii,sub_lists in enumerate(zip(rv_list,time_list)):
            rv_list_sub,time_list_sub = sub_lists
            time_list_indi = np.argsort(np.argsort(time_list_sub))
            rv_difference[iii,:] = rv_list_sub - rv_array["RV_comb"][np.argsort(rv_array["Time_comb"])][time_list_indi]
        
        all_rv_array = np.array([*zip(np.stack(rv_list),np.stack(err_list),
                                      np.stack(time_list),loss_array,rv_difference,min_wavelength_of_chunk)],\
                                dtype=[("RV_all",np.double,(loss_array.shape[1])),("RV_err_all",np.double,(loss_array.shape[1])),\
                                       ("Time_all",np.double,(loss_array.shape[1])),("Loss_Avg",np.double,(loss_array.shape[1])),
                                       ("RV_difference",np.double,(loss_array.shape[1])),("min_order",int)])
        np.save(os.path.join(path,'RV_all_Summary.npy'),all_rv_array)
    else:
        all_rv_array = np.load(os.path.join(path,'RV_all_Summary.npy'))

    order_by_orders = np.argsort(min_wavelength_of_chunk)
    return rv_array, [all_models[iii] for iii in order_by_orders], all_rv_array[order_by_orders], [all_data[iii] for iii in order_by_orders]

def get_stellar_model(init_rvs, model_grid, p_val, which_key='index'):
    return jabble.model.CompositeModel([jabble.model.ShiftingModel(jabble.physics.shifts(init_rvs), which_key=which_key), \
                                        jabble.model.CardinalSplineMixture(model_grid, p_val)])

def get_tellurics_model(init_airmass, model_grid, p_val, rest_vels=None, which_key='index'):
    if rest_vels is None:
        rest_vels = np.zeros(init_airmass.shape) * u.m/u.s
    return jabble.model.CompositeModel([jabble.model.ShiftingModel(jabble.physics.shifts(rest_vels), which_key=which_key), \
                                        jabble.model.CardinalSplineMixture(model_grid, p_val), \
                                        jabble.model.StretchingModel(init_airmass)])

def get_wobble_model(init_rvs, init_airmass, model_grid, p_val,rest_vels=None, which_key='index'):
    '''
    Create a Wobble Model with Stellar and Telluric Components
    Parameters
    ----------
    init_rvs : array-like
        Initial radial velocities for each observation
    init_airmass : array-like
        Initial airmass values for each observation
    model_grid : array-like
        Wavelength grid for the model components
    p_val : int
        Number of parameters for the Cardinal Spline Mixture models
    rest_vels : array-like, optional
        Rest frame velocities for the telluric component, by default None
    which_key : str, optional
        Key to use for shifting model, by default 'index'
    Returns
    -------
    wobble_model : `jabble.model.CompositeModel`
        Composite model containing both stellar and telluric components
    '''

    return get_stellar_model(init_rvs, model_grid, p_val, which_key=which_key) + \
        get_tellurics_model(init_airmass, model_grid, p_val, rest_vels, which_key=which_key)

def get_normalization_model(dataset, norm_p_val, norm_pts):
    len_xs = np.max(
        [np.max(dataframe.xs) - np.min(dataframe.xs) for dataframe in dataset]
    )
    min_xs = np.min([np.min(dataframe.xs) for dataframe in dataset])
    max_xs = np.max([np.max(dataframe.xs) for dataframe in dataset])

    shifts = jnp.array([dataframe.xs.min() - min_xs for dataframe in dataset])
    x_spacing = len_xs / norm_pts
    x_grid = jnp.linspace(-x_spacing*((norm_p_val + 1)//2), len_xs + (x_spacing*((norm_p_val + 1)//2)), norm_pts + norm_p_val + 1) + min_xs
    model = jabble.model.CardinalSplineMixture(x_grid, norm_p_val)
    size = len(dataset)

    p = jnp.tile(model.p, size)
    norm_model = jabble.model.NormalizationModel(p, model, size)
    return jabble.model.ShiftingModel(shifts).composite(norm_model)

def train_norm(model, dataset, loss, device_store, device_op, batch_size,\
               nsigma = [0.5,2], maxiter=3,options={"maxiter": 64,"factr": 1e4},norm_model_index=[2,1]):
    # Fit Normalization Template
    for iii in range(maxiter):
        model.fix()
        model.fit(*norm_model_index)
        
        res1 = model.optimize(loss, dataset, device_store, device_op, batch_size, options=options)#model.optimize(loss, dataset)
        print(res1)
        model.fix()
        datablock = dataset.blockify(device_op)
        for data_epoch in range(len(dataset)):

            mask    = datablock.ele(data_epoch,device_op)["mask"]
            datarow = datablock[data_epoch]
            resid = dataset[data_epoch].ys - model([],dataset[data_epoch].xs,datarow['meta'])
            sigma = np.sqrt(np.nanmedian(resid**2))
            m_new = (resid < -nsigma[0]*sigma) | (resid > nsigma[1]*sigma)
            dataset[data_epoch].mask = mask | m_new[:len(mask)]

    return model

def train_cycle(model, dataset, loss, device_store, device_op, \
                batch_size, options = {"maxiter": 100_000,"factr": 1.0e-1},parabola_fit=False):
    '''
    Full Training Cycle for Wobble Model
    1) Fit Stellar and Telluric Templates
    2) Fit RVs
    3) (Optional) Fit RV Parabola
    4) Fit Everything
    Parameters
    ----------
    model : `jabble.Model`
        Full model of data. Assumed to have RV component at model[0][0], 
        stellar and telluric templates at model[0][1] and model[1][1]
    dataset : `jabble.Dataset`
        Data to be evaluated against
    loss : `jabble.Loss`
        Loss function to be optimized
    device_store : jax.Device
        Device to store parameters on
    device_op : jax.Device
        Device to perform operations on
    batch_size : int
        Number of data points to use in each optimization step
    options : dict
        Options to pass to the optimizer
    parabola_fit : bool
        Whether to perform a parabola fit to the RVs
    Returns
    -------
    model : `jabble.Model`
        Trained model
    '''
    # Fit Stellar & Telluric Template
    model.fix()
    model.fit(0,1)
    model.fit(1,1)
    model.display()
    
    res1 = model.optimize(loss, dataset, device_store, device_op, batch_size, options=options)#model.optimize(loss, dataset)
    print(res1)
    
    # Fit RV
    model.fix()
    model.fit(0,0)
    res1 = model.optimize(loss, dataset, device_store, device_op, batch_size, options=options)
    print(res1)

    # RV Parabola Fit
    if parabola_fit:
        model.fix()
        search_space = np.linspace(-100, 100, 500)
        shift_search = jabble.physics.shifts(search_space)

        
        model[0][0].parabola_fit(shift_search, loss, model, dataset, device_op, device_store)
    # model.to_device(device_op)

    # Fit Everything
    model.fix()
    model.fit(0,0)
    model.fit(0,1)
    model.fit(1,1)
    # model.fit(2,1)

    res1 = model.optimize(loss, dataset, device_store, device_op, batch_size, options=options)#model.optimize(loss, dataset)
    print(res1)

    return model

def get_RV_sigmas(model, dataset, device=None, rv_ind = [0,0]):
        """
        Return errorbar on radial velocities using fischer information

        Parameters
        ----------
        dataset : `jabble.Dataset`
            Data to be evaluated against
        model : `jabble.Model`
            Full model of data. If None, then just the stellar model(self) is used.
        """
        if device is None:
            device = dataset[0].xs.device()
        
        rv_model = model
        for xx in rv_ind:
            rv_model = rv_model[xx] 
        f_info = rv_model.f_info(model, dataset, device)
        dvddx = jnp.array(
            [jax.grad(jabble.physics.velocities)(x) for x in rv_model.p]
        )
        return np.sqrt(1 / f_info) * dvddx

def get_loss_array(model,datablock,metablock,loss,device):
    loss.ready_indices(model)
    loss_array = np.zeros((datablock['xs'].shape))
    for jjj in range(datablock['xs'].shape[0]):
        datarow = jabble.loss.dict_ele(datablock,jjj,device)
        metarow = jabble.loss.dict_ele(metablock,jjj,device)
        loss_array[jjj,:] = loss(model.get_parameters(),datarow,metarow,model)
    return loss_array
    
def save(self, filename: str, dataname: str, data, shifts, loss, device, rv_ind) -> None:
        '''
            mode: 0, just RVs
            mode: 1, RVs and template
            mode: 2, RVs, template, and residuals
            mode: 3, RVs and template in pickle
            mode: 4, RVs, template, residuals in pickle
        '''
        
        jabble.model.save(filename,self)
        jabble.model.save(dataname,data)
        
        with h5py.File(filename + "_RVS.hdf",'w') as file:
            datablock, metablock, meta_keys = data.blockify(device,return_keys=True)
            file.create_dataset("RVs",data=jabble.physics.velocities(shifts))
            file.create_dataset("RV_err",data=get_RV_sigmas(self, data, device=device, model=self,rv_ind=rv_ind))
            file.create_dataset("Times",data=meta_keys['times'])

            head, tail = os.path.split(filename)
            file.attrs['model'] = filename

            loss_array = get_loss_array(self,datablock,metablock,loss,device)
            file.create_dataset("Loss",data=loss_array)

            head,tail = os.path.split(dataname)
            file.attrs['dataset'] = dataname
        
        pass

def load(filename,mode:str):
    def load_results(model,file):
        results_row = []
        str_dtype = '<U100'
        results_dtype = []
        for key,ele in file.attrs.items():
            
            arr = np.array(file.attrs[key]).astype(str_dtype)
            results_dtype.append((key,str_dtype))
            results_row.append(arr)
            
        for key in file.keys():
            results_dtype.append((key,file[key].dtype))
            results_row.append(np.array(file[key]))

        # put named columns in the same order as in model.results
        temp = list(map(tuple, zip(*results_dtype)))
        index_1 = np.argsort(temp[0])
        index_2 = np.argsort(model.results.dtype.names)
        invindex_2 = np.argsort(index_2)

        data = list(map(tuple, zip(*np.array(results_row)[index_1][invindex_2])))
        thing = list(map(tuple,np.array(results_dtype)[index_1][invindex_2]))
        # create results array and append to existing results named array
        results_arr = np.array(data,dtype=thing)
        model.results = np.append(model.results,results_arr, axis=0) 

        return model
    
    def load_metadata(model,file):
        # I HATE MY CODE THIS IS SO FUCKING STUPID 
        # I SHOULD JUST MAKE A GENERIC
        for key in file.keys():
            model.metadata[key] = np.array(file[key])
        return model

    if mode == "hdf":
        with h5py.File(filename,'r') as file:
            # print(model_name + '.hdf',file.keys())
            # print(file.keys())
            for key in file.keys():
                if key in dir(jabble.model):
                    # print(key)
                    cls = eval('jabble.model.' + key)
                    model = cls.load_hdf(cls,file[key])
                    model = load_results(model,file['results'])
                    model = load_metadata(model,file['metadata'])

    elif mode == "pkl":
        model = jabble.model.load(filename)
    return model

def gaussian_smooth(x,sigma):

    return np.exp(-x**2/(2*sigma**2))

def convolve_load(file,sigma_l,xl,eval_a):
    # This reads in line by line a phoenix model
    # storing flux and x values to the cache around the grid point
    # then convolves to instrument resolution
    
    xl_ii   = 0
    xl_curr = xl[xl_ii]
    fl = np.zeros(xl.shape)

    x_cache = np.array([])
    f_cache = np.array([])

    # xh = []
    # fh = []
    with open(file,'r') as file_stream:
        cnt = 0
        for line in file_stream:
            cnt += 1
            if cnt > 8:
                words = line.split('    ')
                for word in words:
                    try:
                        x_val = np.log(np.double(word))
                        break
                    except ValueError:
                        pass
                        
                if x_val > (xl_curr - eval_a):
                    x_cache = np.append(x_cache,x_val)
                    # xh.append(x_val)
                    for word in words[::-1]:  
                        try:
                            f_cache = np.append(f_cache,np.double(word))
                            # fh.append(np.double(word))
                            break
                        except ValueError:
                            pass
                while x_val > (xl_curr + eval_a):
                    kern = gaussian_smooth(x_cache - xl_curr,sigma_l)
                    # print(len(x_cache))
                    # plt.plot(x_cache,f_cache,'.k')
                    # plt.vlines([xl_curr,xl_curr+eval_a,xl_curr-eval_a],np.min(f_cache),np.max(f_cache))
                    # plt.show()
                    
                    fl[xl_ii] = np.dot(f_cache,kern)/np.sum(kern)
                    plt.plot(xl,fl,'.r')
                    if xl_ii+1 >= len(xl):
                        break
                    xl_ii += 1
                    xl_curr = xl[xl_ii]
                    # print(x_cache.shape,f_c
                    f_cache = f_cache[x_cache > xl_curr - eval_a]
                    x_cache = x_cache[x_cache > xl_curr - eval_a]
                if xl_ii+1 >= len(xl):
                    break

    return fl
                    
