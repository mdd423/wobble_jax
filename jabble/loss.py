import jabble.model
import numpy as np
import jax.numpy as jnp
import jax

from functools import partial

class LossFunc:
    """
    Loss or Objective function class for fitting jabble.models
    """

    def __init__(self, coefficient=1.0):
        self.coefficient = coefficient

    def __add__(self, x):

        return LossSequential(loss_funcs=[self, x])

    def __mul__(self, x):
        self.coefficient *= x
        return self

    def __rmul__(self, x):
        self.coefficient *= x
        return self

    @partial(jax.jit,static_argnums=(0,2,3,4,5,6,7))
    def loss_all(self, p, datablock, metablock, model, device_op, batch_size, margs=()):
        """
        Loops through all epochs in dataset. And adds each value to objective.

        Parameters
        ----------
        p : `jnp.array`
            Parameters of the model being fit.
        data : `jabble.Dataset`

        model : `jabble.Model`
            Model being fit.
        """
        def _internal(datarow,metarow):
            return self(p, datarow, metarow, model, margs).sum()

        rounds = int(np.ceil(len(metablock) / batch_size))
        out = 0.0

        for iii in range(rounds):
            top = np.min([(iii + 1) * batch_size, len(datablock)])

            temp = jax.vmap(_internal, in_axes=(0,0), out_axes=0)(
                    datablock.slice((iii * batch_size),top).to_device(device_op),\
                    metablock.slice((iii * batch_size),top).to_device(device_op)
                    )
            out += temp.sum()
        return out
      
    def ready_indices(self, model):

        pass

    def __repr__(self) -> str:

        return (
            "{:.2e}".format(self.coefficient) + " {obj.__class__.__name__}".format(obj=self) + "()"
        )


class LossSequential(LossFunc):
    def __init__(self, loss_funcs):
        # super().__init__(self)
        self.loss_funcs = loss_funcs

    @partial(jax.jit,static_argnums=(0,3,4,5))
    def __call__(self, p, datarow, metarow, model, margs=()):
        output = 0.0
        for loss in self.loss_funcs:
            output += loss(p, datarow, metarow, model, margs).sum()
        return output

    def __add__(self, x: LossFunc):
        if isinstance(x, LossSequential):
            out = LossSequential(loss_funcs=[*self.loss_funcs, *x])
        else:
            out = LossSequential(loss_funcs=[*self.loss_funcs, x])
        return out

    def __radd__(self, x: LossFunc):
        if isinstance(x, LossSequential):
            out = LossSequential(loss_funcs=[*self.loss_funcs, *x])
        else:
            out = LossSequential(loss_funcs=[*self.loss_funcs, x])
        return out

    def __mul__(self, x):
        for loss in self.loss_funcs:
            loss.coefficient *= x
        return self

    def __rmul__(self, x):
        for loss in self.loss_funcs:
            loss.coefficient *= x
        return self

    def ready_indices(self, model):

        for loss in self.loss_funcs:
            loss.ready_indices(model)
            # loss.indices = get_submodel_indices(model,*loss.submodel_inds)

    def __repr__(self) -> str:
        out = self.loss_funcs[0].__repr__()
        for loss in self.loss_funcs[1:]:
            out += " + " + loss.__repr__()
        return out


class ChiSquare(LossFunc):
    @partial(jax.jit,static_argnums=(0,3,4,5))
    def __call__(self, p, datarow, metarow, model, margs=()):

        return self.coefficient * jnp.where(
            ~datarow.mask,
            datarow.yivar
            * (((datarow.ys - model(p, datarow.xs, metarow, margs)) ** 2)),
            0.0,
        )

class L2Loss(LossFunc):
    @partial(jax.jit,static_argnums=(0,3,4,5))
    def __call__(self, p, datarow, metarow, model, margs=()):

        return self.coefficient * jnp.where(
            ~datarow.mask,
            (((datarow.ys - model(p, datarow.xs, metarow, margs)) ** 2)),
            0.0,
        )


class L2Reg(LossFunc):
    def __init__(self, submodel_inds=True, coefficient=1.0, constant=0.0):
        super(L2Reg, self).__init__(coefficient)
        self.constant = constant
        self.submodel_inds = submodel_inds
        # self.indices       = get_submodel_indices(model,*self.submodel_inds)

    def ready_indices(self, model):
        self.indices = jabble.model.get_submodel_indices(model, *self.submodel_inds)
    @partial(jax.jit,static_argnums=(0,3,4,5))
    def __call__(self, p, datarow, metarow, model, margs=()):
        err = self.coefficient * 0.5 * ((p[self.indices] - self.constant) ** 2)
        return err

    def __repr__(self) -> str:

        return (
            str(self.coefficient)
            + " {obj.__class__.__name__}".format(obj=self)
            + "({obj.submodel_inds})".format(obj=self)
        )
    
class L1Reg(L2Reg):
    @partial(jax.jit,static_argnums=(0,3,4,5))
    def __call__(self, p, datarow, metarow, model, margs=()):
        err = self.coefficient * jnp.abs(p[self.indices] - self.constant)
        return err