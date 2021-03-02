import model as wobble_model
import numpy as np

def L2Loss(params,*args):
        model = args[0]
        # Since jax grad only takes in 1d ndarrays you need to flatten your inputs
        # thus the targets should already be flattened as well

        err = 0.5 * ((model.ys - model.forward(params,*args))**2).sum()
        # err += 0.5*(( - model.forward(params))**2).sum()
        return err

def L2Reg(params,*args):
    model = args[0]
    try:
        constant = args[1]
    except IndexError:
        constant = 0.0
    return 0.5 * ((params - constant)**2).sum()

loss_dict = {'L2Loss': L2Loss
            ,'L2Reg' : L2Reg}
