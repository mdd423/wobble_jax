from dataclasses import dataclass

# Third Party
import numpy as np
import numpy.polynomial as polynomial

import scipy.ndimage

import jax.numpy as jnp
import jax

import jabble.loss
import jabble.model

def fit_continuum_jabble(x, y, ivars, device_store, device_op, norm_p_val, norm_res, nsigma=[0.8,3.0], maxniter=50,options={}):
    """Fit the continuum using sigma clipping

    Args:
        x: The wavelengths
        y: The log-fluxes
        order: The polynomial order to use
        nsigma: The sigma clipping threshold: tuple (low, high)
        maxniter: The maximum number of iterations to do

    Returns:
        The value of the continuum at the wavelengths in x
        Author: Matt Daunt

    """
    # A = np.vander(x - np.nanmean(x), order+1)
    m = np.zeros(len(x), dtype=bool)

    # x_num = int((np.exp(x.max()) - np.exp(x.min())) * pts_per_wavelength)
    x_spacing = jabble.physics.delta_x(norm_res)

    x_grid = np.arange(x.min()-(norm_p_val*x_spacing),x.max()+(norm_p_val*x_spacing),x_spacing)

    print(len(x_grid))
    loss = jabble.loss.ChiSquare()
    model = jabble.model.CardinalSplineMixture(x_grid, norm_p_val)

    for i in range(maxniter):
        m[ivars == 0] = 1

        # print(np.sum(~m))
        dataset = jabble.dataset.Data.from_lists([x],[y],[ivars],[m])
        model.fit()
        # model.display()
        # options = {'pgtol': 1e-10}
        res = model.optimize(loss, dataset, device_store, device_op, batch_size=1,options=options)
        model.fix()
        
        resid = y - model([],x)
        sigma = np.sqrt(np.nanmedian(resid**2))
        m_new = ~np.array((resid > (-nsigma[0]*sigma)) & (resid < (nsigma[1]*sigma)))
        m_new[ivars == 0] = 1

        # plt.errorbar(x,y,yerr=1/np.sqrt(ivars),fmt='.k',zorder=1,alpha=0.1,ms=5)
        # plt.plot(x,model([],x),'-r',zorder=2,alpha=0.4,ms=5)
        # plt.plot(x_grid,model([],x_grid),'.r',zorder=3,alpha=0.5,ms=5)
        # plt.plot(x,model([],x)-(nsigma[0]*sigma),'-b',zorder=2,alpha=0.4,ms=5)
        # plt.plot(x,model([],x)+(nsigma[1]*sigma),'-b',zorder=2,alpha=0.4,ms=5)
        # plt.plot(x[~m_new],y[~m_new],'og',zorder=2,alpha=0.2,ms=5)
        # plt.ylim(-5,1)
        # plt.show()
        #m_new = np.abs(resid) < nsigma*sigma

        print(m.sum(),m_new.sum())
        if m.sum() == m_new.sum():
            print('break')
            m = m_new
            break
        m = m_new
    
    return model([],x)

def fit_continuum(x, y, ivars, order=6, nsigma=[0.3,3.0], maxniter=50):
    """Fit the continuum using sigma clipping

    Args:
        x: The wavelengths
        y: The log-fluxes
        order: The polynomial order to use
        nsigma: The sigma clipping threshold: tuple (low, high)
        maxniter: The maximum number of iterations to do

    Returns:
        The value of the continuum at the wavelengths in x

    """
    A = np.vander(x - np.nanmean(x), order+1)
    m = np.ones(len(x), dtype=bool)
    for i in range(maxniter):
        m[ivars == 0] = 0  # mask out the bad pixels
        w = np.linalg.solve(np.dot(A[m].T, A[m]), np.dot(A[m].T, y[m]))
        mu = np.dot(A, w)
        resid = y - mu
        sigma = np.sqrt(np.nanmedian(resid**2))
        #m_new = np.abs(resid) < nsigma*sigma
        m_new = (resid > -nsigma[0]*sigma) & (resid < nsigma[1]*sigma)
        if m.sum() == m_new.sum():
            m = m_new
            break
        m = m_new
    return mu

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

        # datablock = np.array([*zip(xs,ys,yivar,mask)],\
        #                      dtype=[("xs",np.double,(xs.shape[1])),("ys",np.double,(xs.shape[1])),\
        #                             ("yivar",np.double,(xs.shape[1])),("mask",np.double,(xs.shape[1]))])
        # rv_array = np.array([*zip(comb_rv,comb_err,comb_time)],dtype=[("RV_comb",np.double),("RV_err_comb",np.double),("Time_comb",np.double)])
    
        ###########################################################

        meta_keys = {}

        # meta_dtype = [("index",int)]
        index_span = np.arange(0, len(data), dtype=int)
        metablock = {"index": index_span}
        for key in data.metadata:
            if key in data.metakeys:
                epoch_indices = np.zeros(index_span.shape)
                for i, ele in enumerate(data.metakeys[key]):
                    epoch_indices[data.metadata[key] == ele] = i
                epoch_uniques = data.metakeys[key]
            else:
                epoch_uniques, epoch_indices = np.unique(
                    data.metadata[key], return_inverse=True
                )
            # meta_dtype.append((key,epoch_indices.dtype))
            # mdata.append(jnp.array(epoch_indices))
            # metablock[key] = jax.device_put(jnp.array(epoch_indices), device)
            meta_keys[key] = epoch_uniques
            metablock[key] = epoch_indices.astype(int)

        # metablock = np.array([*zip(*mdata)],dtype=meta_dtype)

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
