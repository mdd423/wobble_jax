import model as wobble_model
import numpy as np

class LossFunc():
    def __init__(self,loss_func,loss_parms=1.0):
        if   type(loss_func) == type('str'):
            self.func_list = np.array([loss_func])
        elif type(loss_func) == type(np.array([])):
            self.func_list = loss_func
        else:
            sys.exit('loss_func parameter not correct type: str or list')
        if   type(loss_parms) == type(1.0):
            self.constant = np.array([loss_parms])
        elif type(loss_parms) == type(np.array([])):
            self.constant = loss_parms
            assert len(self.constant) == len(self.func_list)
        else:
            sys.exit('loss_func parameter not correct type: float or list')

    def __add__(self,x):

        return LossFunc(loss_func=np.append(self.func_list, x.func_list ) ,loss_parms=np.append(self.constant, x.constant ))

    def __mul__(self,x):
        return LossFunc(loss_func=self.func_list,loss_parms=x * self.constant)

    def __rmul__(self,x):
        return LossFunc(loss_func=self.func_list,loss_parms=x * self.constant)

    def __call__(self,p,y,x,*args):
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
        if (len(y.shape)) == 1:
            for i,loss in enumerate(self.func_list):
                output += self.constant[i] * loss_dict[loss](p,y,x,None,*args)
        else:
            # recall ys are packed st that 0: epoches, 1: pixel 
            for epoch in range(y.shape[0]):
                for i,loss in enumerate(self.func_list):
                    output += self.constant[i] * loss_dict[loss](p,y[epoch,:],x[epoch,:],epoch,*args)
        return output

# I want to design a class for these such that constants of the loss functions
# can be initialized like for the regularization
def L2Loss(p, y, x, i, model,*args):

    err = 0.5 * ((y - model(p,x,i,*args))**2).sum()
    # Since jax grad only takes in 1d ndarrays you need to flatten your inputs
    # thus the targets should already be flattened as well
    return err

def L2Reg(p,y,x,i,model,*args):

    try:
        constant = args[0]
    except IndexError:
        constant = 0.0
    return 0.5 * ((p - constant)**2).sum()

loss_dict = {'L2Loss': L2Loss
            ,'L2Reg' : L2Reg}
