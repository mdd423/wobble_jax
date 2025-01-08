import sys

sys.path.append("..")
import jax.numpy as jnp
import jax
import numpy as np
import jabble.model
import h5py


def _getitem__(self, key: str | int):
    if type(key) == int:
        return super(type(self), self).__getitem__(key)
    elif type(key) == str:
        return super(type(self), self).__getitem__(np.argwhere(self.keys == key)[0][0])


class StellarModel(jabble.model.CompositeModel):
    """
    StellarModel for quick use on fitting radial velocities with a stellar template.
    CompositeModel of shifting model, and IrwinHall sparse model for the stellar template.

    Parameters
    ----------
    init_shifts : `np.ndarray`
        initial radial velocity shifts in log wavelength space
    model_grid : `np.ndarray`
        evenly spaced x control points for IrwinHall model
    p_val : `int`
        order of IrwinHall model

    Attributes
    ----------
    keys : `list`
        list of the strings that must be used to index the submodels

    """

    def __init__(self, init_shifts, model_grid, p_val):
        super(StellarModel, self).__init__(
            [
                jabble.model.EpochShiftingModel(init_shifts),
                jabble.model.CardinalSplineMixture(model_grid, p_val),
            ]
        )
        self.keys = np.array(["RV", "Template"])

    def get_RV(self):
        """
        Converts shifting parameters to velocity in m/s

        Returns
        -------
        vels : `np.ndarray`
            radial velocity in m/s
        """
        return jabble.physics.velocities(self["RV"].p)

    def get_RV_sigmas(self, dataset, model=None):
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
        f_info = self["RV"].f_info(model, dataset)
        dvddx = jnp.array(
            [jax.grad(jabble.physics.velocities)(x) for x in self["RV"].p]
        )
        return np.sqrt(1 / f_info) * dvddx

    def __getitem__(self, key: str | int):

        return _getitem__(self, key)
    


class TelluricsModel(jabble.model.CompositeModel):
    """
    TelluricsModel for quick use on fitting airmass stretching with a telluric template.
    CompositeModel of IrwinHall sparse model for the stellar template, and stretching model.

    Parameters
    ----------
    init_airmass : `np.ndarray`
        initial stretching airmass value in log flux space
    model_grid : `np.ndarray`
        evenly spaced x control points for IrwinHall model
    p_val : `int`
        order of IrwinHall model

    Attributes
    ----------
    keys : `list`
        list of the strings that must be used to index the submodels

    """

    def __init__(self, init_airmass, model_grid, p_val, rest_shifts=None):
        if rest_shifts is None:
            rest_shifts = np.zeros(init_airmass.shape)
        super(TelluricsModel, self).__init__(
            [
                jabble.model.ShiftingModel(rest_shifts),
                jabble.model.CardinalSplineMixture(model_grid, p_val),
                jabble.model.StretchingModel(init_airmass),
            ]
        )
        self.keys = np.array(["RestShifts", "Template","Airmass"])

    def __getitem__(self, key: str | int):

        return _getitem__(self, key)


class WobbleModel(jabble.model.AdditiveModel):
    """
    WobbleMdoel for quick use includes StellarModel and TelluricsModel.
    AdditiveModel of StellarModel and TelluricsModel.

    Parameters
    ----------
    init_shifts : `np.ndarray`
        initial radial velocity shifts in log wavelength space
    init_airmass : `np.ndarray`
        initial stretching airmass value in log flux space
    model_grid : `np.ndarray`
        evenly spaced x control points for IrwinHall model
    p_val : `int`
        order of IrwinHall model

    Attributes
    ----------
    keys : `list`
        list of the strings that must be used to index the submodels

    """

    def __init__(self, init_shifts, airmass, model_grid, p_val):
        super(WobbleModel, self).__init__(
            [
                StellarModel(init_shifts, model_grid, p_val),
                TelluricsModel(airmass, model_grid, p_val),
            ]
        )
        self.keys = np.array(["Stellar", "Tellurics"])

    def get_RV(self):
        return self["Stellar"].get_RV()

    def get_RV_sigmas(self, dataset):

        return self["Stellar"].get_RV_sigmas(dataset, self)

    def __getitem__(self, key: str | int):

        return _getitem__(self, key)
    
    def save(self,filename: str,mode: str, data) -> None:
        '''
            mode: 0, just RVs
            mode: 1, RVs and template
            mode: 2, RVs, template, and residuals
        '''
        if mode == 1 or mode == 2:
            super().save(filename)
        with h5py.File(filename,'w') as file:

            datablock, metablock, meta_keys = data.blockify(return_keys=True)

            group = file.create_group("RVs")
            group.create_dataset("RVs",data=self.get_RV())
            group.create_dataset("RV_err",data=self.get_RV_sigmas(data))
            group.create_dataset("Times",data=meta_keys['times'])
            if mode == 2:
                res_group = file.create_group("residuals")
                res_group.create_dataset("residuals",data=data)
            pass


class PseudoNormalModel(jabble.model.AdditiveModel):
    def __init__(self, init_shifts ,airmass, model_grid, p_val, dataset, norm_p_val, pts_per_wavelength):
        normal_model = jabble.model.get_normalization_model(dataset, norm_p_val, pts_per_wavelength)
        super(WobbleModel, self).__init__(
            [
                StellarModel(init_shifts, model_grid, p_val),
                TelluricsModel(airmass, model_grid, p_val),
                normal_model
            ]
        )
        self.keys = np.array(["Stellar", "Tellurics", "Normal"])

    def __getitem__(self, key: str | int):

        return _getitem__(self, key)
