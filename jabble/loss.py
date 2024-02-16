# import jabble.model as wobble_model
import numpy as np
import jax.numpy as jnp
import jax

class LossFunc: 
    """
    Loss or Objective function class for fitting jabble.models
    """
    def __init__(self,coefficient=1.0):
        self.coefficient = coefficient

    def __add__(self,x):

        return LossSequential(loss_funcs=[self,x])

    def __mul__(self,x):
        self.coefficient *= x
        return self

    def __rmul__(self,x):
        self.coefficient *= x
        return self

    def loss_all(self,p,data,model,*args):
        """
        Loops through all epochs in dataset. And adds each value to objective.

        Parameters
        ----------
        p : `jnp.array`
            Parameters of the model being fit.
        data : `jabble.Dataset`
            Dataset that model is being fit to.
        model : `jabble.Model`
            Model being fit.
        """

        output = jnp.zeros(len(data.ys))
        def _internal(ind):
            return self(p,data,ind,model,*args)
        output = jax.vmap(_internal,in_axes=(0,),out_axes=0)(jnp.arange(len(data.ys)))
        return jnp.sum(output)
        # output = 0.0
        # for ind in range(data.ys.shape[0]):

        #     output += self(p,data,ind,model,*args).sum()
        # return output
    

class LossSequential(LossFunc):
    def __init__(self,loss_funcs):
        # super().__init__(self)
        self.loss_funcs = loss_funcs

    def __call__(self,p,data,i,model,*args):
        output = 0.0
        for loss in self.loss_funcs:
            output += loss(p,data,i,model,*args).sum()
        return output

    def __add__(self,x):
        if isinstance(x,LossSequential):
            out = LossSequential(loss_funcs=[*self.loss_funcs,*x])
        else:
            out = LossSequential(loss_funcs=[*self.loss_funcs,x])
        return out

    def __radd__(self,x):
        if isinstance(x,LossSequential):
            out = LossSequential(loss_funcs=[*self.loss_funcs,*x])
        else:
            out = LossSequential(loss_funcs=[*self.loss_funcs,x])
        return out

    def __mul__(self,x):
        for loss in self.loss_funcs:
            loss.coefficient *= x
        return self

    def __rmul__(self,x):
        for loss in self.loss_funcs:
            loss.coefficient *= x
        return self

# always multiply by coefficient so that when you do mutliplication with the
# object then it translates to the output
class L2Loss(LossFunc):
    def __call__(self, p, data, i, model,*args):
        err = self.coefficient * 0.5 * ((data.ys[i,~data.mask[i,:]] - model(p,data.xs[i,~data.mask[i,:]],i,*args))**2)
        # Since jax grad only takes in 1d ndarrays you need to flatten your inputs
        # thus the targets should already be flattened as well
        return err


class ChiSquare(LossFunc):
    def __call__(self, p, data, i, model, *args):
        err = self.coefficient * (((data.ys[i] - model(p,data.xs[i],i,*args))**2) * data.yivar[i])
        return err


class L2Reg(LossFunc):
    def __init__(self,coefficient=1.0,constant=0.0,indices=True):
        super(L2Reg,self).__init__(coefficient)
        self.constant = constant
        self.indices  = indices

    def __call__(self, p, data, i, model, *args):
        err = self.coefficient * 0.5 * ((p[self.indices] - self.constant)**2)
        return err


class L2Smooth(LossFunc):
    def __init__(self,coefficient=1.,constant=0.0,submodel_ind=None):
        super(L2Smooth,self).__init__(coefficient)
        self.submodel_ind = submodel_ind
        self.constant = constant

    def __call__(self,p,data,i,model,*args):
        ps = model.split_p(p)
        for ind in self.submodel_ind:
            ps = ps[ind]
        err = self.coefficient * 0.5 * ((ps[1:] - ps[:-1])**2)
        return err
