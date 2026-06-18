import jabble.model
import numpy as np
import jax.numpy as jnp
import jax
import h5py
import datetime

from functools import partial

def load(filename):
    with h5py.File(filename, 'r') as hf:
        loss = []
        for key in hf.keys():

            obj_name = key.split('_')[0]
            if obj_name in dir(jabble.loss):
                loss.append(eval(obj_name).load(hf[key]))
    return loss

def save(filename,loss):
    
    with h5py.File(filename, 'w') as hf:
        group = hf.create_group(loss.__class__.__name__ )
        loss.save(group)
  

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

    def save(self,group):
        group.create_dataset('coeff',data=self.coefficient)

    def load(group):
        obj_name = group.name.split('/')[-1].split('_')[0]
        return eval(obj_name + '()') * group['coeff'][()]


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
            out = LossSequential(loss_funcs=[*self.loss_funcs, *x.loss_funcs])
        else:
            out = LossSequential(loss_funcs=[*self.loss_funcs, x])
        return out

    def __radd__(self, x: LossFunc):
        if isinstance(x, LossSequential):
            out = LossSequential(loss_funcs=[*self.loss_funcs, *x.loss_funcs])
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

    def save(self,hf):
        
        iteration = 0
        for model in self.loss_funcs:
            while model.__class__.__name__ + f'_{iteration}' in hf.keys():
                iteration += 1
            subgroup = hf.create_group(model.__class__.__name__ + f'_{iteration}')
            model.save(subgroup)

    def load(hf):
        loss_funcs = []
        for group_key in hf.keys():
            obj_name = hf[group_key].name.split('/')[-1].split('_')[0]
            loss_funcs.append(eval(obj_name).load(hf[group_key]))
            
        obj_name = hf.name.split('/')[-1]
        print(obj_name.split('_'))
        obj_name = obj_name.split('_')[0]
        return eval(obj_name)(loss_funcs)


class ChiSquare(LossFunc):
    @partial(jax.jit,static_argnums=(0,4,5))
    def __call__(self, p, datarow, metarow, model, margs=()):

        return self.coefficient * jnp.where(
            ~datarow.mask,
            0.5 * datarow.yivar
            * (((datarow.ys - model(p, datarow.xs, metarow, margs)) ** 2)),
            0.0,
        )

class L2Loss(LossFunc):
    @partial(jax.jit,static_argnums=(0,4,5))
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
    
    @partial(jax.jit,static_argnums=(0,4,5))
    def __call__(self, p, datarow, metarow, model, margs=()):
        err = self.coefficient * 0.5 * ((p[self.indices] - self.constant) ** 2)
        return err
    
    def save(self, group):
        group.create_dataset('coeff', data=self.coefficient)
        group.create_dataset('const', data=self.constant)
        group.create_dataset('submodel_inds', data=self.submodel_inds)

    def load(group):
        obj_name = group.name.split('/')[-1].split('_')[0]
        return eval(obj_name)(constant=group['const'][()], submodel_inds=group['submodel_inds'][()]) * group['coeff'][()]
    
    def __repr__(self) -> str:

        return (
            str(self.coefficient)
            + " {obj.__class__.__name__}".format(obj=self)
            + "({obj.submodel_inds})".format(obj=self)
        )
    
class L1Reg(L2Reg):
    @partial(jax.jit,static_argnums=(0,4,5))
    def __call__(self, p, datarow, metarow, model, margs=()):
        err = self.coefficient * jnp.abs(p[self.indices] - self.constant)
        return err