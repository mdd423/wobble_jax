from dataclasses import dataclass

# Third Party
import numpy as np
import numpy.polynomial as polynomial

import scipy.ndimage

import jax.numpy as jnp
import jax


# important for grid search
def get_parabolic_min(loss_array, grid, return_all=False):

    epoches = loss_array.shape[0]
    grid_min = np.empty(epoches)

    xss = np.empty((epoches, 3))
    yss = np.empty((epoches, 3))
    polys = []

    for n in range(epoches):
        idx = loss_array[n, :].argmin()
        print("epch {}: min {}".format(n, idx))
        if idx == 0:
            print("minimum likely out of range")
            idx = 1
        if idx == grid.shape[1] - 1:
            print("minimum likely out of range")
            idx -= 1
        # else:

        xs = grid[n, idx - 1 : idx + 2]
        xss[n, :] = xs
        ys = loss_array[n, idx - 1 : idx + 2]
        yss[n, :] = ys

        poly = np.polyfit(xs, ys, deg=2)
        polys.append(poly)
        deriv = np.polyder(poly)

        x_min = np.roots(deriv)
        x_min = x_min[x_min.imag == 0].real
        y_min = np.polyval(poly, x_min)

        grid_min[n] = x_min

    if return_all:
        return grid_min, xss, yss, polys
    else:
        return grid_min


@dataclass
class Data:
    """Temporary Data Type"""

    def __init__(self, frames, *args):
        self.dataframes = frames
        self.metadata = {}
        self.metakeys = {}

    def __getitem__(self, i):
        return self.dataframes[i]

    @property
    def yerr(self):
        return 1 / np.sqrt(self.yivar)

    @property
    def xs(self):
        return [dataframe.xs for dataframe in self.dataframes]

    @property
    def ys(self):
        return [dataframe.ys for dataframe in self.dataframes]

    @property
    def yivar(self):
        return [dataframe.yivar for dataframe in self.dataframes]

    @property
    def mask(self):
        return [dataframe.mask for dataframe in self.dataframes]

    def __len__(self):
        return len(self.dataframes)

    def from_lists(xs, ys, yivar, ma):
        frames = []
        for iii in range(len(xs)):
            frames.append(DataFrame(xs[iii], ys[iii], yivar[iii], ma[iii]))
        return Data(frames)

    def to_device(self, device):
        for dataframe in self.dataframes:
            dataframe.to_device(device)

    def blockify(data, device=None, return_keys=False):
        if device is None:
            device = data[0].xs.device()
        max_ind = np.max([len(dataframe.xs) for dataframe in data])
        xs = np.zeros((len(data), max_ind))
        ys = np.zeros((len(data), max_ind))
        yivar = np.zeros((len(data), max_ind))
        mask = np.ones((len(data), max_ind))

        for i, dataframe in enumerate(data):
            frame_size = len(dataframe.xs)
            xs[i, :frame_size] = dataframe.xs
            ys[i, :frame_size] = dataframe.ys
            yivar[i, :frame_size] = dataframe.yivar
            mask[i, :frame_size] = dataframe.mask

        xs = jax.device_put(jnp.array(xs), device)
        ys = jax.device_put(jnp.array(ys), device)
        yivar = jax.device_put(jnp.array(yivar), device)
        mask = jax.device_put(jnp.array(mask, dtype=bool), device)

        datablock = {}
        datablock["xs"] = xs
        datablock["ys"] = ys
        datablock["yivar"] = yivar
        datablock["mask"] = mask

        ###########################################################

        metablock = {}
        meta_keys = {}
        metablock["index"] = jnp.arange(0, len(data), dtype=int)
        for key in data.metadata:
            if key in data.metakeys:
                epoch_indices = np.array(metablock["index"])
                for i, ele in enumerate(data.metakeys[key]):
                    epoch_indices[data.metadata[key] == ele] = i
                    epoch_uniques = data.metakeys[key]
            else:
                epoch_uniques, epoch_indices = np.unique(
                    data.metadata[key], return_inverse=True
                )
            metablock[key] = jax.device_put(jnp.array(epoch_indices), device)
            meta_keys[key] = epoch_uniques

        if return_keys:
            return datablock, metablock, meta_keys
        return datablock, metablock


class DataFrame:
    def __init__(self, xs: jnp.array, ys: jnp.array, yivar: jnp.array, mask: jnp.array):
        self.xs = jnp.array(xs)
        self.ys = jnp.array(ys)
        self.yivar = jnp.array(yivar)
        self.mask = jnp.array(mask)

    def to_device(self, device):
        self.xs = jax.device_put(jnp.array(self.xs), device)
        self.ys = jax.device_put(jnp.array(self.ys), device)
        self.yivar = jax.device_put(jnp.array(self.yivar), device)
        self.mask = jax.device_put(jnp.array(self.mask), device)
