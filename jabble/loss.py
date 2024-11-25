# import jabble.model as wobble_model
import numpy as np
import jax.numpy as jnp
import jax

def dict_slice(dictionary,slice_i,slice_j,device):
        out = {}
        for key in dictionary:
            out[key] = jax.device_put(dictionary[key][slice_i:slice_j],device)
        return out

def dict_ele(dictionary,slice_i,device):
        out = {}
        for key in dictionary:
            out[key] = jax.device_put(dictionary[key][slice_i],device)
        return out

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

    def loss_all(self,p,datablock,metablock,model,device_op,batch_size,*args):
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
        
        def _internal(datarow,metarow):
            return self(p,datarow,metarow,model,*args).sum()

        rounds = int(np.ceil(metablock['index'].max()/batch_size))
        out = 0.0
        
        for iii in range(rounds):
            top = np.min([(iii+1)*batch_size,datablock['xs'].shape[0]])
            
            temp = jax.vmap(_internal, in_axes=(0,0), out_axes=0)(dict_slice(datablock,(iii*batch_size),top,device_op),\
                                                                dict_slice(metablock,(iii*batch_size),top,device_op))
            out += temp.sum()
        return out
    
    def ready_indices(self,model):

        pass
    

class LossSequential(LossFunc):
    def __init__(self,loss_funcs):
        # super().__init__(self)
        self.loss_funcs = loss_funcs

    def __call__(self,p,data,meta,model,*args):
        output = 0.0
        for loss in self.loss_funcs:
            output += loss(p,data,meta,model,*args).sum()
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
    
    def ready_indices(self,model):

        for loss in self.loss_funcs:
            loss.ready_indices(model)
            # loss.indices = get_submodel_indices(model,*loss.submodel_inds)

# always multiply by coefficient so that when you do mutliplication with the
# object then it translates to the output

class ChiSquare(LossFunc):
    def __call__(self, p, datarow, metarow, model, *args):
        
        return self.coefficient * jnp.where(~datarow['mask'],\
                                            datarow['yivar'] * (((datarow['ys'] - model(p,datarow['xs'],metarow,*args))**2)),\
                                            0.0)

def get_submodel_indices(self,i,j=None,*args):
    # this recurses through submodels when given a set of indices to that submodel
    # then returns of a bool array of the length of the total number of parameters 
    # of whole model
    # with 1's at the parameters of the specific submodel, 0's elsewhere
    s_temp = self.get_indices(i)
    if j is None:
        return s_temp
    
    s_inds = jnp.zeros(self.get_parameters().shape,dtype=bool)
    temp = get_submodel_indices(self[i],j,*args)
    s_inds = s_inds.at[s_temp].set(temp)
    return s_inds

class L2Reg(LossFunc):
    def __init__(self,submodel_inds=True,coefficient=1.0,constant=0.0):
        super(L2Reg,self).__init__(coefficient)
        self.constant      = constant
        self.submodel_inds = submodel_inds
        # self.indices       = get_submodel_indices(model,*self.submodel_inds)

    def ready_indices(self,model):
        self.indices = get_submodel_indices(model,*self.submodel_inds)

    def __call__(self, p, datarow, model, *args):
        err = self.coefficient * 0.5 * ((p[self.indices] - self.constant)**2)
        return err

