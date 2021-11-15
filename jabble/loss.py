# import jabble.model as wobble_model
import numpy as np


class LossFunc: #,loss_func,loss_parms=1.0
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
        output = 0.0
        # recall ys are packed st that 0: epoches, 1: pixel
        for i in range(data.epoches):
            output += self(p,data,i,model,*args)
        return output


class LossSequential(LossFunc):
    def __init__(self,loss_funcs):
        # super().__init__(self)
        self.loss_funcs = loss_funcs

    def __call__(self,p,data,i,model,*args):
        output = 0.0
        for loss in self.loss_funcs:
            output += loss(p,data,i,model,*args)
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
        err = self.coefficient * 0.5 * ((data.ys[i,~data.mask[i,:]] - model(p,data.xs[i,~data.mask[i,:]],i,*args))**2).sum()
        # Since jax grad only takes in 1d ndarrays you need to flatten your inputs
        # thus the targets should already be flattened as well
        return err


class ChiSquare(LossFunc):
    def __call__(self, p, data, i, model, *args):
        err = self.coefficient * (((data.ys[i,:] - model(p,data.xs[i,:],i,*args))**2) * data.yivar[i,:]).sum()
        return err


class L2Reg(LossFunc):
    def __init__(self,coefficient=1.0,constant=0.0,indices=True):
        super(L2Reg,self).__init__(coefficient)
        self.constant = constant
        self.indices  = indices

    def __call__(self, p, data, i, model, *args):
        err = self.coefficient * 0.5 * ((p[self.indices] - self.constant)**2).sum()
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
        err = self.coefficient * 0.5 * ((ps[1:] - ps[:-1])**2).sum()
        return err
