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

    def loss_all(self,p,xs,ys,yivar,mask,model,device_op,batch_size,*args):
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
        #blockify parameters
        #what if normalization model has different number of parameters per model
        #anything that is going to take the epoch index needs to blockified and be the only parameter
        #this is an issue with the normalization model because its epoch specific but the parameters vary by epoch
        # just putting in the zero below will assume the same number of parameters as the first one
        # not the one specified, whats the better way to do multiple epoch fitting without indices
        
        def _internal(xs_row,ys_row,yivar_row,mask_row,index):
            return self(p,xs_row,ys_row,yivar_row,mask_row,index,model,*args).sum()

        indices = jnp.arange(0,xs.shape[0],dtype=int)

        rounds = int(np.ceil(xs.shape[0]/batch_size))
        out = 0.0
        for iii in range(rounds):
            top = np.min([(iii+1)*batch_size,xs.shape[0]])
            # print(device_op)
            temp = jax.vmap(_internal, in_axes=(0, 0, 0, 0, 0), out_axes=0)(jax.device_put(xs[(iii*batch_size):top],device_op), \
                                                                            jax.device_put(ys[(iii*batch_size):top],device_op), \
                                                                            jax.device_put(yivar[(iii*batch_size):top],device_op), \
                                                                            jax.device_put(mask[(iii*batch_size):top],device_op), \
                                                                            jax.device_put(indices[(iii*batch_size):top],device_op))
            out += temp.sum()
        return out
    

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

class ChiSquare(LossFunc):
    def __call__(self, p, xs, ys, yivar, mask, i, model, *args):
        return self.coefficient * jnp.where(~mask,yivar * (((ys - model(p,xs,i,*args))**2)),0.0)
    

class L2Reg(LossFunc):
    def __init__(self,coefficient=1.0,constant=0.0,indices=True):
        super(L2Reg,self).__init__(coefficient)
        self.constant = constant
        self.indices  = indices

    def __call__(self, p, xs, ys, yivar, mask, i, model, *args):
        err = self.coefficient * 0.5 * ((p[self.indices] - self.constant)**2)
        return err


