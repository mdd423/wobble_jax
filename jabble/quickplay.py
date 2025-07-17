import sys

sys.path.append("..")
import jax.numpy as jnp
import jax
import numpy as np
import jabble.model
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
        # print(file.keys())
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

def get_stellar_model(init_rvs, model_grid, p_val):
    return jabble.model.CompositeModel([jabble.model.EpochShiftingModel(jabble.physics.shifts(init_rvs)), jabble.model.CardinalSplineMixture(model_grid, p_val)])

def get_tellurics_model(init_airmass, model_grid, p_val, rest_vels=None):
    if rest_vels is None:
        rest_vels = np.zeros(init_airmass.shape) * u.m/u.s
    return jabble.model.CompositeModel([jabble.model.ShiftingModel(jabble.physics.shifts(rest_vels)), jabble.model.CardinalSplineMixture(model_grid, p_val), \
                                        jabble.model.StretchingModel(init_airmass)])

def get_wobble_model(init_rvs, init_airmass, model_grid, p_val,rest_vels=None):

    return get_stellar_model(init_rvs, model_grid, p_val) + get_tellurics_model(init_airmass, model_grid, p_val, rest_vels)

def get_normalization_model(dataset, norm_p_val, pts_per_wavelength):
    len_xs = np.max(
        [np.max(dataframe.xs) - np.min(dataframe.xs) for dataframe in dataset]
    )
    min_xs = np.min([np.min(dataframe.xs) for dataframe in dataset])
    max_xs = np.max([np.max(dataframe.xs) for dataframe in dataset])

    shifts = jnp.array([dataframe.xs.min() - min_xs for dataframe in dataset])

    x_num = int((np.exp(max_xs) - np.exp(min_xs)) * pts_per_wavelength)
    x_spacing = len_xs / x_num
    x_grid = jnp.linspace(-x_spacing, len_xs + x_spacing, x_num + 2) + min_xs

    model = jabble.model.CardinalSplineMixture(x_grid, norm_p_val)
    size = len(dataset)

    p = jnp.tile(model.p, size)
    norm_model = jabble.model.NormalizationModel(p, model, size)
    return jabble.model.ShiftingModel(shifts).composite(norm_model)

def get_pseudo_norm_model(init_rvs, init_airmass, model_grid, p_val,dataset,norm_p_val, pts_per_wavelength,rest_vels=None,):

    return get_stellar_model(init_rvs, model_grid, p_val) + get_tellurics_model(init_airmass, model_grid, p_val, rest_vels) + \
        get_normalization_model(dataset,norm_p_val, pts_per_wavelength)

def get_RV_sigmas(self, dataset, model=None, device=None, rv_ind = [0,0]):
        """
        Return errorbar on radial velocities using fischer information

        Parameters
        ----------
        dataset : `jabble.Dataset`
            Data to be evaluated against
        model : `jabble.Model`
            Full model of data. If None, then just the stellar model(self) is used.
        """
        if model is None:
            model = self
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
        
        # if mode == 1 or mode == 2:
        jabble.model.save(filename,self)
        jabble.model.save(dataname,data)
        # if mode == 3 or mode == 4:
            # self.save(filename,mode="pkl")
            

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
        # if mode == 2 or mode == 4:
        #     res_group = file.create_group("residuals")
        #     res_group.create_dataset("residuals",data=data)
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
                    
# x_cache = np.concatenate((x_cache,[np.double(word)]))
# low_resolution = 115_000
# x_min = np.log(4000)
# x_max = np.log(4200)
# sigma_l = jabble.physics.delta_x(low_resolution)

# xl = np.arange(x_min,x_min+(10000*sigma_l/2),sigma_l/2)
# print(xl)
# fl = convolve_load('/scratch/mdd423/wobble_jax/data/HD4307_harps/models_1750205771/bt-nextgen-gns93/lte035-5.0-0.5.BT-NextGen.7.dat.txt',\
#                        sigma_l,xl,10*sigma_l)

# def _getitem__(self, key: str | int):
#     if type(key) == int:
#         return super(type(self), self).__getitem__(key)
#     elif type(key) == str:
#         return super(type(self), self).__getitem__(np.argwhere(self.keys == key)[0][0])


# class StellarModel(jabble.model.CompositeModel):
#     """
#     StellarModel for quick use on fitting radial velocities with a stellar template.
#     CompositeModel of shifting model, and IrwinHall sparse model for the stellar template.

#     Parameters
#     ----------
#     init_shifts : `np.ndarray`
#         initial radial velocity shifts in log wavelength space
#     model_grid : `np.ndarray`
#         evenly spaced x control points for IrwinHall model
#     p_val : `int`
#         order of IrwinHall model

#     Attributes
#     ----------
#     keys : `list`
#         list of the strings that must be used to index the submodels

#     """

#     def __init__(self, init_shifts, model_grid, p_val):
#         super(StellarModel, self).__init__(
#             [
#                 jabble.model.EpochShiftingModel(init_shifts),
#                 jabble.model.CardinalSplineMixture(model_grid, p_val),
#             ]
#         )
#         self.keys = np.array(["RV", "Template"])

#     def get_RV(self):
#         """
#         Converts shifting parameters to velocity in m/s

#         Returns
#         -------
#         vels : `np.ndarray`
#             radial velocity in m/s
#         """
#         return jabble.physics.velocities(self["RV"].p)

#     def get_RV_sigmas(self, dataset, model=None, device=None):
#         """
#         Return errorbar on radial velocities using fischer information

#         Parameters
#         ----------
#         dataset : `jabble.Dataset`
#             Data to be evaluated against
#         model : `jabble.Model`
#             Full model of data. If None, then just the stellar model(self) is used.
#         """
#         if model is None:
#             model = self
#         if device is None:
#             device = dataset[0].xs.device()
#         f_info = self["RV"].f_info(model, dataset, device)
#         dvddx = jnp.array(
#             [jax.grad(jabble.physics.velocities)(x) for x in self["RV"].p]
#         )
#         return np.sqrt(1 / f_info) * dvddx

#     def __getitem__(self, key: str | int):

#         return _getitem__(self, key)
    

# class TelluricsModel(jabble.model.CompositeModel):
#     """
#     TelluricsModel for quick use on fitting airmass stretching with a telluric template.
#     CompositeModel of IrwinHall sparse model for the stellar template, and stretching model.

#     Parameters
#     ----------
#     init_airmass : `np.ndarray`
#         initial stretching airmass value in log flux space
#     model_grid : `np.ndarray`
#         evenly spaced x control points for IrwinHall model
#     p_val : `int`
#         order of IrwinHall model

#     Attributes
#     ----------
#     keys : `list`
#         list of the strings that must be used to index the submodels

#     """

#     def __init__(self, init_airmass, model_grid, p_val, rest_shifts=None):
#         if rest_shifts is None:
#             rest_shifts = np.zeros(init_airmass.shape)
#         super(TelluricsModel, self).__init__(
#             [
#                 jabble.model.ShiftingModel(rest_shifts),
#                 jabble.model.CardinalSplineMixture(model_grid, p_val),
#                 jabble.model.StretchingModel(init_airmass),
#             ]
#         )
#         self.keys = np.array(["RestShifts", "Template", "Airmass"])

#     def __getitem__(self, key: str | int):

#         return _getitem__(self, key)


# class WobbleModel(jabble.model.AdditiveModel):
#     """
#     WobbleMdoel for quick use includes StellarModel and TelluricsModel.
#     AdditiveModel of StellarModel and TelluricsModel.

#     Parameters
#     ----------
#     init_shifts : `np.ndarray`
#         initial radial velocity shifts in log wavelength space
#     init_airmass : `np.ndarray`
#         initial stretching airmass value in log flux space
#     model_grid : `np.ndarray`
#         evenly spaced x control points for IrwinHall model
#     p_val : `int`
#         order of IrwinHall model

#     Attributes
#     ----------
#     keys : `list`
#         list of the strings that must be used to index the submodels

#     """

#     def __init__(self, init_shifts, airmass, model_grid, p_val):
#         super(WobbleModel, self).__init__(
#             [
#                 StellarModel(init_shifts, model_grid, p_val),
#                 TelluricsModel(airmass, model_grid, p_val),
#             ]
#         )
#         self.keys = np.array(["Stellar", "Tellurics"])

#     def get_RV(self):
#         return self["Stellar"].get_RV()

#     def get_RV_sigmas(self, *args, **kwargs):

#         return self["Stellar"].get_RV_sigmas(*args, **kwargs)

#     def __getitem__(self, key: str | int):

#         return _getitem__(self, key)
    
    
#     def load_hdf(cls,group):
#         model = cls(np.array(group["StellarModel"]["EpochShiftingModel"]["p"]),np.array(group["TelluricsModel"]["StretchingModel"]["p"]),\
#                    np.array(group["TelluricsModel"]["CardinalSplineMixture"]["xs"]),np.array(group["TelluricsModel"]["CardinalSplineMixture"]["p_val"]))
#         model["Stellar"]["Template"].p = np.array(group["StellarModel"]["CardinalSplineMixture"]["p"])
#         model["Tellurics"]["Template"].p = np.array(group["TelluricsModel"]["CardinalSplineMixture"]["p"])
#         return model


# class PseudoNormalModel(jabble.model.AdditiveModel):
#     def __init__(self, init_shifts ,airmass, model_grid, p_val, normal_model):
#         super(WobbleModel, self).__init__(
#             [
#                 StellarModel(init_shifts, model_grid, p_val),
#                 TelluricsModel(airmass, model_grid, p_val),
#                 normal_model
#             ]
#         )
#         self.keys = np.array(["Stellar", "Tellurics", "Normal"])

#     def __getitem__(self, key: str | int):
        
#         return _getitem__(self, key)
    
#     def load_hdf(cls,group):
#         normal_model = group[-1].load_hdf(cls)
#         model = cls(np.array(group["StellarModel"]["EpochShiftingModel"]["p"]),np.array(group["TelluricsModel"]["StretchingModel"]["p"]),\
#                    np.array(group["TelluricsModel"]["CardinalSplineMixture"]["xs"]),np.array(group["TelluricsModel"]["CardinalSplineMixture"]["p_val"]))
#         model["Stellar"]["Template"].p = np.array(group["StellarModel"]["CardinalSplineMixture"]["p"],normal_model)
#         model["Tellurics"]["Template"].p = np.array(group["TelluricsModel"]["CardinalSplineMixture"]["p"])

       
#         return model
