import numpy as np
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import scipy.optimize
import sys

# you can pass whatever args you want to your function after the 0th which is the model
# pass these after the loss function in the optimize call

# args can also be passed to the forward pass of the model
# if you define a model to take on additional arguments in the forward pass
# idk how this will work with multiple update steps
def L2Loss(params,*args):
        model = args[0]
        # Since jax grad only takes in 1d ndarrays you need to flatten your inputs
        # thus the targets should already be flattened as well
        targets = model.ys.flatten()
        err = 0.5 * ((targets - model.forward(params))**2).sum()
        # err += 0.5*(( - model.forward(params))**2).sum()
        return err

def L2Reg(params,*args):
    model = args[0]
    if args[1] is not None:
        constant = 0.0
    return 0.5 * ((params - constant)**2).sum()

loss_dict = {'L2Loss': L2Loss
            ,'L2Reg' : L2Reg}

class LossFunc():
    def __init__(self,loss_func,loss_parms=1.0):
        if   type(loss_func) == type('str'):
            self.func_list = [loss_func]
        elif type(loss_func) == type(['list']):
            self.func_list = loss_func
        else:
            sys.exit('loss_func parameter not correct type: str or list')
        if   type(loss_parms) == type(1.0):
            self.params = [loss_parms]
        elif type(loss_parms) == type(['list']):
            self.params = loss_parms
            assert len(self.params) == len(self.func_list)
        else:
            sys.exit('loss_func parameter not correct type: float or list')

    def __add__(self,x):

        return LossFunc(loss_func=self.func_list + x.func_list,loss_parms=self.params + x.params)

    def __call__(self,params,*args):
        output = 0.0
        for i,loss in enumerate(self.func_list):
            output += self.params[i] * loss_dict[loss](params,*args)
        return output

def createFrequencyDist(mus,stds,heights,x):
    size = len(x)
    assert len(mus)  == len(stds)
    assert len(stds) == len(heights)
    freq = np.zeros(size)
    for i in range(len(mus)):
        freq += heights[i] * np.exp(-np.power(x - mus[i], 2.) / (2 * np.power(stds[i], 2.)))
    return freq

class FreqEnv():
    def __init__(self,minimum,maximum,epoches=16,num_peaks=4,num_tells=2,size=128,noise=0.015,vel_scale=1.0,std_atm_scale=1.,hgt_atm_scale=0.5,std_str_scale=1.,hgt_str_scale=0.5):
        self.vel_scale = vel_scale
        self.noise = noise

        self.epoches = epoches

        self.size = size

        self.lambdas = np.linspace(minimum,maximum,self.size)
        # spectra of sky
        self.mu_atm     = ((maximum - minimum) * np.random.rand(num_tells)) - minimum
        self.std_atm    = (2 * (maximum - minimum)/size) * np.ones(num_tells)#1/resolution * np.ones(num_tells) #std_atm_scale * np.random.rand(num_tells)
        self.height_atm = hgt_atm_scale * np.random.rand(num_tells)
        # spectra of star
        self.mus     = ((maximum - minimum) * np.random.rand(num_peaks)) - minimum
        self.stds    = (2 * (maximum - minimum)/size) * np.ones(num_peaks)#1/resolution * np.ones(num_peaks) #std_str_scale * np.random.rand(num_peaks)
        self.heights = hgt_str_scale * np.random.rand(num_peaks)
        # true shift of each epoch
        self.real_vel = np.array([0.0])
        # velocities will range from -vel_scale to +vel_scale
        self.real_vel = jnp.append(self.real_vel,self.generate_new_vel(epoches-1))

    def get_stellar_flux(self):
        # initialize flux array and fill
        fluxes = np.ones(self.size)
        fluxes -= createFrequencyDist(self.mus,self.stds,self.heights,self.lambdas)
        return fluxes

    def get_telluric_flux(self):
        # initialize flux array and fill
        fluxes = np.ones(self.size)
        fluxes -= createFrequencyDist(self.mu_atm,self.std_atm,self.height_atm,self.lambdas)
        return fluxes

    def get_flux(self):
        # initialize flux array and fill
        fluxes = np.ones([self.epoches,self.size])
        # Shifted Epoches
        for i,vel in enumerate(self.real_vel):
            # create star flux THIS PART GETS SHIFTED
            fluxes[i,:], dummy = self.generate_new_epoch(vel=vel)
        return fluxes

    def plot_stellar_specter(self,model=None,log_space=False):
        if model is not None:
            plt.plot(model.x,model.y,'.r',linestyle='solid',linewidth=.8,zorder=2,alpha=0.5,ms=6)
        if log_space:
            plt.plot(self.lambdas,np.log(self.get_stellar_flux()))
        else:
            plt.plot(self.lambdas,self.get_stellar_flux())

    def generate_new_vel(self,size):
        return self.vel_scale * (2 * np.random.random(size) - 1)

    def generate_new_epoch(self,vel=None):
        if vel is None:
            new_vel = self.generate_new_vel(1)[0]
        else:
            new_vel = vel
        flux    = np.ones(self.size)
        flux   -= createFrequencyDist(self.mus,self.stds,self.heights,self.lambdas + new_vel)
        # add atmosphere flux
        flux   -= createFrequencyDist(self.mu_atm,self.std_atm,self.height_atm,self.lambdas)
        # add noise
        if self.noise is not 0:
            flux += np.random.normal(0.0,scale=self.noise,size=self.size)
        return flux, new_vel

def getCellArray(x,xs):
    cell_array = np.zeros(len(xs),dtype=int)
    x_val = x[0]
    j = 0
    for i, xss in enumerate(xs):
        while x_val < xss:
            j += 1
            x_val = x[j]
        cell_array[i] = int(j)
    return cell_array

def getPlotSize(model):
    size_x = np.floor(np.sqrt(model.epoches))
    size_y = model.epoches//size_x
    while model.epoches % size_y != 0:
        size_y = model.epoches//size_x
        size_x -= 1
    else:
        size_x += 1
    size_x = int(size_x)
    size_y = int(size_y)
    return size_x, size_y

class LinModel():
    def __init__(self,num_params,fluxes,lambdas,epoches,vel_shifts,size):
        self.epoches = epoches
        self.xs = lambdas
        self.ys = fluxes

        self.shifted = vel_shifts

        self.padding = abs(self.shifted).max()

        minimum = self.xs.min()
        maximum = self.xs.max()
        self.x = np.linspace(minimum-self.padding,maximum+self.padding,num_params)

        # the model x's must be shifted appropriately
        # this might have to be moved to the forward section if velocity is fit bt evals of ys

        # given x cells are shifted, the cell arrays contain the information for
        # which data points are in which cells
        self.cell_array = np.zeros([self.epoches,size],dtype=int)
        for i in range(self.epoches):
            # added the shifted to the freq dist so subtract shift from model
            self.cell_array[i,:] = getCellArray(self.x - self.shifted[i],self.xs)
        self.y = np.ones(num_params)

    def optimize(self,loss,*args):
        # Train model
        res = scipy.optimize.minimize(loss, self.y, args=(self,*args), method='BFGS', jac=jax.grad(loss),
               options={'disp': True})
        self.y = res.x
        return res

    # gives all predicted data from the input data in the model given the parameters
    def forward(self,params,*args):
        y = params
        # TO DO: determine cell array with each forward pass if we are to fit velocity shift
        ys = jnp.array([])
        for i in range(self.epoches):
            # temp_x = self.x+self.shifted[i]
            # subtracted shift from model
            cell_array = self.cell_array[i,:] #getCellArray(temp_x,self.xs)
            # the x values for the model need to be shifted here but only for the intercept
            m   = (y[cell_array] - y[cell_array-1])/(self.x[cell_array] - self.x[cell_array-1])
            ys2 = y[cell_array-1] + m * (self.xs - self.x[cell_array-1] + self.shifted[i])
            ys  = jnp.append(ys,ys2)
        return ys

    def __call__(self,x):
        # can only be used once model is optimized
        cell_array = getCellArray(self.x,x)
        # the x values for the model need to be shifted here but only for the intercept
        m  = (self.y[cell_array] - self.y[cell_array-1])/(self.x[cell_array] - self.x[cell_array-1])
        ys = self.y[cell_array-1] + m * (x - self.x[cell_array-1])
        return ys

    def plot(self,noise,env=None):
        size_x, size_y = getPlotSize(self)

        fig = plt.figure(figsize=[12.8,9.6])
        # Once again we apply the shift to the xvalues of the model when we plot it
        for i in range(self.epoches):
            ax = fig.add_subplot(size_x,size_y,i+1)
            ax.set_title('epoch %i: vel %.2f' % (i, model.shifted[i]))

            plt.plot(self.x - self.shifted[i],model.y,'.r',linestyle='solid',linewidth=.8,zorder=2,alpha=0.5,ms=6)

            plt.errorbar(self.xs,self.ys[i,:],yerr=noise,fmt='.k',zorder=1,alpha=0.9,ms=6)

            plt.xlim(min(self.xs),max(self.xs))

            plt.ylim(-0.8,0.2)
            if env is not None:
                plt.plot(env.lambdas - self.shifted[i],env.get_stellar_flux(),color='red', alpha=0.4)

    def cross_correlation(self,flux,lambdas,size=1000):

        shifts = np.linspace(-self.padding+0.01,self.padding-0.01,size)
        ccs = np.zeros(size)
        for i,shift in enumerate(shifts):
            ccs[i] = np.dot(self(lambdas + shift),flux)
        return ccs, shifts

if __name__ == '__main__':

    minimum = 0
    maximum = 2
    num_params = 40
    epoches = 8
    size   = 256
    noise  = 0.045
    env    = FreqEnv(minimum,maximum,epoches=epoches,noise=noise,size=size,hgt_atm_scale=0.5,hgt_str_scale=0.5)
    fluxes = np.log(env.get_flux())

    loss_1 = LossFunc('L2Loss')
    loss_2 = LossFunc('L2Reg',10.0)
    loss   = loss_1 + loss_2
    model  = LinModel(num_params,fluxes,env.lambdas,env.epoches,env.real_vel,env.size)

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
