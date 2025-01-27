import sys

sys.path.append("..")
import jax.numpy as jnp
import jax
import numpy as np
import jabble.model
import h5py
import os

import astropy.units as u

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


def get_RV_sigmas(self, dataset, model=None, device=None):
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
        f_info = self[0][0].f_info(model, dataset, device)
        dvddx = jnp.array(
            [jax.grad(jabble.physics.velocities)(x) for x in self[0][0].p]
        )
        return np.sqrt(1 / f_info) * dvddx

def save(self,filename: str,mode: str, data, device) -> None:
        '''
            mode: 0, just RVs
            mode: 1, RVs and template
            mode: 2, RVs, template, and residuals
            mode: 3, RVs and template in pickle
            mode: 4, RVs, template, residuals in pickle
        '''
        
        # if mode == 1 or mode == 2:
        self.save(filename,mode=mode)
        # if mode == 3 or mode == 4:
            # self.save(filename,mode="pkl")
            

        with h5py.File(filename + "_RVS.hdf",'w') as file:
            _, _, meta_keys = data.blockify(return_keys=True)
            file.create_dataset("RVs",data=jabble.physics.velocities(self[0][0].p))
            file.create_dataset("RV_err",data=get_RV_sigmas(self, data, device=device, model=self))
            file.create_dataset("Times",data=meta_keys['times'])

            head, tail = os.path.split(filename + "." + mode)
            if mode == 'hdf':
                file["model"] = h5py.SoftLink(filename + "." + mode)
            elif mode == 'pkl':
                file.attrs['model'] = filename + "." + mode
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

    if mode == "hdf":
        with h5py.File(filename,'r') as file:
            # print(model_name + '.hdf',file.keys())
            for key in file.keys():
                if key in dir(jabble.model):
                    cls = eval('jabble.model.' + key)
                    model = cls.load_hdf(cls,file)
                    load_results(model,file['results'])

    elif mode == "pkl":
        model = jabble.model.load(filename)
    return model

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
