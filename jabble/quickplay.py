import sys

sys.path.append("..")
import jax.numpy as jnp
import jax
import numpy as np
import jabble.model


def _getitem__(self, key):

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
                jabble.model.ShiftingModel(init_shifts),
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

    def __getitem__(self, *args):
        return _getitem__(self, args)


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

    def __init__(self, init_airmass, model_grid, p_val):
        super(TelluricsModel, self).__init__(
            [
                jabble.model.StretchingModel(init_airmass),
                jabble.model.CardinalSplineMixture(model_grid, p_val),
            ]
        )
        self.keys = np.array(["Airmass", "Template"])

    def __getitem__(self, *args):

        return _getitem__(self, args)


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

    def __init__(self, init_shifts, init_airmass, model_grid, p_val):
        super(WobbleModel, self).__init__(
            [
                StellarModel(init_shifts, model_grid, p_val),
                TelluricsModel(init_airmass, model_grid, p_val),
            ]
        )
        self.keys = np.array(["Stellar", "Tellurics"])

    def get_RV(self):
        return self["Stellar"].get_RV()

    def get_RV_sigmas(self, dataset):

        return self["Stellar"].get_RV_sigmas(dataset, self)

    def __getitem__(self, *args):

        return _getitem__(self, args)
