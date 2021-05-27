import model as wobble_model
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

    def train(self,p,y,yerr,x,model,*args):
        # basically wrapper function that loops through the epoches for the
        # optimizer so all loss classes only have to consider operating on a single
        # epoch
        output = 0.0
        # two modes of calling the loss:
        # 1) if the output y shape is single dimensional, then the epoch index is set to None
        # and the shifts or whatever other parameter of epoch cannot be used, ie use when you assume new data
        # or if running check against known properties of fit dataset
        # 2) if the output shape is two dimensional, then all epoches are looped through and added together for loss
        # and epoch passed down to model
        #
        # Thus the model must have some way of dealing with None type epoch index if the user
        # wishs to make prediction on new data
        # this should happen before call in all classes
        singular = False
        if (len(y.shape)) == 1:
            singular = True
            y = np.expand_dims(y,axis=0)
        # recall ys are packed st that 0: epoches, 1: pixel
        for epoch in range(y.shape[0]):
            if singular:
                EPOCH_INDEX = None
            else:
                EPOCH_INDEX = epoch
            output += self(p,y[epoch,:],yerr[epoch,:],x[epoch,:],EPOCH_INDEX,model,*args)
        return output

class LossSequential(LossFunc):
    def __init__(self,loss_funcs):
        # super().__init__(self)
        self.loss_funcs = loss_funcs

    def __call__(self,p,y,yerr,x,i,model,*args):
        output = 0.0
        for loss in self.loss_funcs:
            output += loss(p,y,x,i,model,*args)
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
    def __call__(self, p, y, yerr, x, i, model,*args):
        err = self.coefficient * 0.5 * ((y - model(p,x,i,*args))**2).sum()
        # Since jax grad only takes in 1d ndarrays you need to flatten your inputs
        # thus the targets should already be flattened as well
        return err

class ChiSquare(LossFunc):
    def __call__(self, p, y, yerr, x, i, model, *args):
        err = self.coefficient * 0.5 * (((y - model(p,x,i,*args))**2)/ yerr**2).sum()
        return err

class L2Reg(LossFunc):
    def __init__(self,coefficient=1.0,constant=0.0,indices=True):
        super(L2Reg,self).__init__(coefficient)
        self.constant = constant
        self.indices  = indices

    def __call__(self, p, y, yerr, x, i, model, *args):
        err = self.coefficient * 0.5 * ((p[self.indices] - self.constant)**2).sum()
        return err
