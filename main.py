import loss as wobble_loss
import simulator as wobble_sim
import model as wobble_model

import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':

    minimum = 0
    maximum = 2
    num_params = 40
    epoches = 8
    size   = 256
    noise  = 0.045
    env    = wobble_sim.FreqEnv(minimum,maximum,epoches=epoches,noise=noise,size=size,hgt_atm_scale=0.5,hgt_str_scale=0.5)
    fluxes = np.log(env.get_flux())

    loss_1 = wobble_model.LossFunc('L2Loss')
    loss_2 = wobble_model.LossFunc('L2Reg',10.0)
    loss   = loss_1 + loss_2
    model  = wobble_model.LinModel(num_params,fluxes,env.lambdas,env.epoches,env.real_vel,env.size)

    model.optimize(loss,1.0)
    model.plot(noise)
    plt.show()

    env.plot_stellar_specter(model,log_space=True)
    plt.show()

    flux, new_vel = env.generate_new_epoch()
    ccs, shifts   = model.cross_correlation(np.log(flux),env.lambdas)
    plt.plot(shifts,ccs)
    plt.axvline(x=new_vel,color='red')
    plt.show()
