import matplotlib.pyplot as plt
import numpy as np
import os
import matplotlib.ticker as ticker

def rv_all_order_plot(times,rv_e,err_e,time_comb,rv_comb,err_comb,targ_time,targ_vel,targ_err,bervs):
    fig, ax = plt.subplots(
        1,
        figsize=(6, 4),
        facecolor=(1, 1, 1),
        dpi=300,
        sharey=True
    )
    temp_vel  = targ_vel - (targ_vel)
    temp_vel -= temp_vel.mean()
    ax.errorbar(targ_time,temp_vel,targ_err,fmt='.g',zorder=3,alpha=0.30,ms=2,label='HARPS RV')

    targ_ind = np.argsort(targ_time)
    comb_indi = np.argsort(np.argsort(time_comb))
    temp_vel  = rv_comb - targ_vel[targ_ind][comb_indi]
    temp_vel -= temp_vel.mean()
    ax.errorbar(time_comb,temp_vel,err_comb,fmt='.r',zorder=5,alpha=0.50,ms=2,label='Order Jabble Combined RV')

    # temp_vel = rv_e.mean(axis=0) + bervs
    # temp_vel -= temp_vel.mean()
    # ax.errorbar(times % period,temp_vel,0.0,fmt='.b',zorder=2,alpha=0.60,ms=2,label='Avg RV')

    for i in range(rv_e.shape[0]):
        targ_ind = np.argsort(targ_time)
        comb_indi = np.argsort(np.argsort(times[i,:]))
        temp_vel  = rv_e[i,:] - targ_vel[targ_ind][comb_indi]
        temp_vel -= temp_vel.mean()
        ax.errorbar(times[i,:],temp_vel,yerr=err_e[i,:],fmt='.k',zorder=2,alpha=0.05,ms=2,label='Order Jabble RV')

    # fig.legend()
    ax.set_ylim(-500, 500)
    # fig.legend()
    ax.set_title('Barnard\'s Star Relative Radial Velocities')
    ax.set_ylabel("RV [$m/s$]")
    ax.set_xlabel( "MJD")
    plt.savefig(os.path.join(out_dir, "barn_rvs_time.png"))

    # ax.set_xlim(2.4564e6,2.45644e6)
    # ax.set_ylim(-20e3,-10e3)
    # plt.savefig(os.path.join(out_dir, "02-barns_all_order_nobervs_epoch.png"))
    plt.show()

def plot_rv_error(times,rv_e,err_e,times_comb,rv_comb,err_comb,targ_time,targ_vel,\
                  targ_err,bervs,loss_array,rv_difference_array,star_name,\
                  out_dir):
    
    epoches_span = np.arange(0,len(times_comb),dtype=int)

    # RV Err Comparison
    
    fig, ax = plt.subplots(
        1,
        figsize=(10, 4),
        facecolor=(1, 1, 1),
        dpi=300,
        sharey=True
    )
    targ_ind = np.argsort(targ_time)
    comb_indi = np.argsort(np.argsort(times_comb))

    bervs_temp = -bervs[targ_ind][comb_indi] + bervs.mean()
    norm_vel   = bervs_temp #(targ_vel[targ_ind][comb_indi] - targ_vel[targ_ind][comb_indi].mean())
    # print(np.sum(times_comb == targ_time[targ_ind][comb_indi]))

    # targ_norm  = (targ_vel[targ_ind][comb_indi] - targ_vel[targ_ind][comb_indi].mean()) - norm_vel
    targ_line = ax.plot(epoches_span,targ_err[targ_ind][comb_indi],'.g',zorder=3,alpha=0.5,ms=2,label='HARPS RV')

    
    # comb_norm = (rv_comb - rv_comb.mean()) - norm_vel
    comb_line = ax.plot(epoches_span,err_comb,'.r',zorder=1,alpha=0.3,ms=2,label='Order Jabble Combined RV')

    # ax.set_title('Barnard\'s Star Relative Radial Velocity Error')
    ax.set_ylabel("RV Error [$m/s$]")
    ax.set_xlabel( "Epochs")
    # ax.set_xlim(2.4564e6,2.45644e6)
    # ax.set_ylim(-1e2,1e2)
    plt.savefig(os.path.join(out_dir, "{}_rvs_err_epoch.png".format(star_name)),bbox_inches='tight')
    plt.show()
    
def plt_rv_comparison(times, rv_e, err_e, times_comb, rv_comb, err_comb, targ_time, targ_vel, \
                      targ_err, bervs, loss_array, rv_difference_array, star_name, out_dir):
    epoches_span = np.arange(0, len(times_comb), dtype=int)

    # RV Comparison

    fig, ax = plt.subplots(
        1,
        figsize=(10, 4),
        facecolor=(1, 1, 1),
        dpi=300,
        sharey=True
    )

    bervs_temp = -bervs[targ_ind][comb_indi] + bervs.mean()
    norm_vel   = bervs_temp

    targ_ind = np.argsort(targ_time)
    comb_indi = np.argsort(np.argsort(times_comb))

    bervs_temp = -bervs[targ_ind][comb_indi] + bervs.mean()
    norm_vel   = bervs_temp #(targ_vel[targ_ind][comb_indi] - targ_vel[targ_ind][comb_indi].mean())
    # print(np.sum(times_comb == targ_time[targ_ind][comb_indi]))

    targ_norm  = (targ_vel[targ_ind][comb_indi] - targ_vel[targ_ind][comb_indi].mean()) - norm_vel
    targ_line = ax.errorbar(epoches_span,targ_norm,targ_err[targ_ind][comb_indi],fmt='.g',zorder=3,alpha=0.5,ms=2,label='HARPS RV')

    
    comb_norm = (rv_comb - rv_comb.mean()) - norm_vel
    comb_line = ax.errorbar(epoches_span,comb_norm,err_comb,fmt='.r',zorder=2,alpha=0.3,ms=2,label='Order Jabble Combined RV')

    # temp_vel = (rv_e.mean(axis=0) - rv_e.mean()) #+ bervs_temp[targ_ind][comb_indi]
    # avg_line = ax.errorbar(epoches_span,temp_vel,0.0,fmt='.b',zorder=2,alpha=0.0,ms=2,label='Avg RV')

    for i in range(rv_e.shape[0]):
        e_ind = np.argsort(times[i,:])
        indiv_norm = (rv_e[i,:][e_ind][comb_indi] - rv_e[i,:].mean()) - norm_vel
        # times[i,:][e_ind][comb_indi]
        err_line = ax.errorbar(epoches_span,indiv_norm,yerr=err_e[i,:][e_ind][comb_indi],fmt='.k',zorder=1,alpha=0.03,ms=2,label='Order Jabble RV')

    # ax.set_ylim(-100, 100)
    ax.legend(handles=[targ_line,comb_line,err_line],loc="upper right")

    # ax.set_title('Barnard\'s Star Relative Radial Velocities')
    ax.set_ylabel("RV [$m/s$]")
    ax.set_xlabel( "Epochs")
    # ax.set_xlim(2.456e6 + 451,2.456e6+452)
    ax.set_ylim(-50,50)
    plt.savefig(os.path.join(out_dir, "{}_rvs_epoch.png".format(star_name)),bbox_inches='tight')
    plt.show()
    
    # FIGURE 2
    # fig, ax = plt.subplots(
    #     1,
    #     figsize=(10, 4),
    #     facecolor=(1, 1, 1),
    #     dpi=300,
    #     sharey=True
    # )

    # for i in range(rv_e.shape[0]):
    #     ax.errorbar(times[i,:],rv_e[i,:],yerr=err_e[i,:],fmt='.k',zorder=1,alpha=0.05,ms=2,label='Order Jabble RV')
    # ax.errorbar(times_comb,rv_comb,err_comb,fmt='.r',zorder=2,alpha=0.3,ms=2,label='Order Jabble Combined RV')

    # ax.set_ylim(-100, 100)
    # ax.set_title('Combination Check')
    # ax.set_ylabel("RV [$m/s$]")
    # ax.set_xlabel( "MJD")
    # ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('%5.1f'))
    # plt.savefig(os.path.join(out_dir, "{}_rv_comb_check.png".format(star_name)),bbox_inches='tight')
    # plt.show()

def plot_rv_difference(times, rv_e, err_e, times_comb, rv_comb, err_comb, targ_time, targ_vel, \
                       targ_err, bervs, loss_array, rv_difference_array, star_name, out_dir,top_spacing=1.2,sigma_max=5,difference_max=5,chi_max=3):
    # RV DIFFERENCE PLOT
    epoches_span = np.arange(0, len(times_comb), dtype=int)

    fig, axes = plt.subplots(
        3,
        figsize=(10, 4),
        facecolor=(1, 1, 1),
        dpi=300,
        sharey=True
    )
    im1 = axes[0].imshow(rv_difference_array,interpolation ='nearest',vmin=-difference_max,vmax=difference_max,cmap=plt.get_cmap('cmr.fusion'))
    fig.colorbar(im1,ax=axes[0])
    axes[0].set_title('$\Delta$ RV')
    
    im2 = axes[1].imshow(err_e,interpolation ='nearest',vmin=0,vmax=sigma_max,cmap=plt.get_cmap('cmr.sunburst'))
    fig.colorbar(im2,ax=axes[1])
    axes[1].set_title('$\sigma_{RV}$')
    
    im3 = axes[2].imshow(rv_difference_array/err_e,interpolation ='nearest',vmin=-chi_max,vmax=chi_max,cmap=plt.get_cmap('cmr.fusion'))
    print(np.sqrt(np.mean((rv_difference_array/err_e)**2)))
    print(np.median(np.abs((rv_difference_array/err_e))))
    fig.colorbar(im3,ax=axes[2])
    axes[2].set_title('$\Delta$ RV /$\sigma_{RV}$')
    
    axes[0].set_xticks([])
    axes[1].set_xticks([])
    plt.xlabel('Epochs')
    axes[1].set_ylabel('Chunks')
    plt.subplots_adjust(top=top_spacing)

    plt.savefig(os.path.join(out_dir,'{}_drv.png'.format(star_name)),bbox_inches='tight')
    plt.show()

    plt.hist((rv_difference_array/err_e).flatten(),bins=30)
    plt.title('$\Delta$ RV /$\sigma_{RV}$')
    plt.show()
    
def plot_loss(times, rv_e, err_e, times_comb, rv_comb, err_comb, targ_time, targ_vel, \
                       targ_err, bervs, loss_array, rv_difference_array, star_name, out_dir):
    # LOSS ARRAY PLOT
    plt.figure(figsize=(12,6),facecolor=(1, 1, 1),dpi=300)
    im = plt.imshow(loss_array,interpolation ='nearest',vmin=0,vmax=10000)

    xs,ys = np.where(np.isnan(loss_array))
    plt.scatter(ys,xs,s=10,facecolors='none', edgecolors='b')

    xs,ys = np.where(np.isinf(loss_array))
    plt.scatter(ys,xs,s=10,facecolors='none', edgecolors='r')
    plt.xlabel('Epochs')
    plt.ylabel('Chunks')
    plt.title('$\chi^2$')
    plt.colorbar(im,shrink=0.3)  
    plt.savefig(os.path.join(out_dir,'{}_loss.png'.format(star_name)),bbox_inches='tight')
    plt.show()

    # RV difference divided by RV error
    
    plt.show()


    # loss = np.array([model.results[-1][''] for model in all_models
    # loss_mean = np.mean(loss,axis=3).mean(axis=1)
    # print(loss_mean.shape)

    # fig, ax = plt.subfigures((2,2),figsize=(12,6),facecolor=(1, 1, 1),dpi=300,height_ratios=[1,4],width_ratios=[4,1],sharex='col',sharey='row')
    # im = ax[1,0].imshow(loss_mean,interpolation ='nearest',vmin=0,vmax=10)
    
    
    # ax[0,1].axis('off')
    # plt.xlabel('epoches')
    # plt.ylabel('chunks')
    # plt.title('$\chi^2$')
    # plt.colorbar(im,shrink=0.3)
    # plt.savefig(os.path.join(out_dir,'barn_obj.png'))
    # plt.show()

import jabble.loss

import jabble.model
import numpy as np
def make_subplot(axes,model,dataset,plt_epoch,device,lrange):

    model.fix()
    data, meta, keys = dataset.blockify(device,return_keys=True)
    
    datarow = jabble.loss.dict_ele(data,plt_epoch,device)
    metarow = jabble.loss.dict_ele(meta,plt_epoch,device)
    # fig.suptitle("Order {}".format(keys["orders"][meta["orders"][plt_epoch]]))
    # axes[0,ii].title.set_text('Date: {}'.format(keys["times"][meta["times"][plt_epoch]]))

    # print(metarow['index'],plt_epoch)
    
    xplot = np.linspace(np.log(lrange.min()),np.log(lrange.max()),\
                        dataset.xs[plt_epoch].shape[0]*10)

    yplot_norm_tot  = model([],xplot,metarow)
    yplot_norm_stel = model[0]([],xplot,metarow)
    yplot_norm_tell = model[1]([],xplot,metarow)
    # yplot_norm      = model[2]([],xplot,metarow)
    # for epoch in np.where(indices):
    yhat = model([],dataset.xs[plt_epoch][:],metarow)
    axes[0].set_xlim(xplot.min(),xplot.max())

    velocity = jabble.physics.velocities(model[0][0].p[plt_epoch])
    
    # Data
    # print(datarow)
    axes[0].errorbar(datarow["xs"][:],datarow["ys"][:],yerr=1/np.sqrt(datarow["yivar"][:]),fmt='.k',zorder=2,alpha=0.1,ms=5)

    # Stellar Model        
    axes[0].plot(xplot,yplot_norm_stel,'-r',linewidth=1.2,zorder=10,alpha=0.7,ms=6)
    # Telluric Model
    axes[0].plot(xplot,yplot_norm_tell,'-b',linewidth=1.2,zorder=10,alpha=0.7,ms=6)
    # Total
    # axes[0].plot(xplot,yplot_norm_tot,'-m',linewidth=1.2,zorder=10,alpha=0.7,ms=6)
    # Norm
    # axes[0,ii].plot(xplot,yplot_norm,'-g',linewidth=1.2,zorder=10,alpha=0.7,ms=6)
    # Theory Model
    # theory_ax = axes[0,ii].twinx()
    # theory_ax.plot(dataset_theory.xs[0][:],dataset_theory.ys[0][:],'-y',linewidth=1.2,zorder=10,alpha=0.7,ms=6)
    # theory_ax.set_ylim(-5,5)
    # Line List
 
    # Residuals
    axes[1].step(dataset.xs[plt_epoch],dataset.ys[plt_epoch] - yhat,\
                             'k',where='mid',zorder=1,alpha=0.3,ms=3)

    axes[0].text(np.log(lrange.min()+0.1),0.3,"Order: ${}$".format(model.metadata["orders"][plt_epoch]))
    # axes[0].text(np.log(lrange.max()-0.1),0.3,"MJD: ${}$".format(model.metadata["times"][plt_epoch]))

    axes[0].set_ylim(-2.5,0.5)
    axes[1].set_ylim(-1,1)
    
    # axes[0].set_xticks([])
    axes[0].set_xticks(np.log(lrange))
    axes[1].set_xticks(np.log(lrange))

    axes[0].set_xticklabels(['' for x in lrange])
    axes[1].set_xticklabels(['{:0.1f}'.format(x) for x in lrange])
    
    axes[0].set_xlim(np.log(lrange.min()),np.log(lrange.max()))
    axes[1].set_xlim(np.log(lrange.min()),np.log(lrange.max()))

    return axes

def make_grid_plots(datasets,models,size_n,size_m,plt_epochs,device,plt_name,out_dir,line_list=None):
    l_width = np.arange(-4,4,2)
    
    fig, axes = plt.subplots(2*size_n,size_m,figsize=(5*size_m,5*size_n),sharey='row',\
                             facecolor=(1, 1, 1),height_ratios=[4,1]*size_n,dpi=200)
    # plt.tight_layout()

    for ii, nn in enumerate(range(size_n)):
        for jj, mm in enumerate(range(size_m)):
            this_index = size_m*ii+jj
            
            lrange = np.floor(np.exp(datasets[this_index].xs[plt_epochs[this_index]]).mean()) + l_width
            # print(lrange)
            axes[2*ii:((2*ii)+2),jj] = make_subplot(axes[2*ii:((2*ii)+2),jj],models[this_index],datasets[this_index],0,device,lrange)
            if line_list is not None:
                plot_line_list(axes[2*ii:((2*ii)+2),jj],line_list,lrange)

    
    plt.subplots_adjust(wspace=0.15)
    # fig.text(0.5, 0.00, 'Wavelength $[\AA]$', ha='center')
    # fig.text(0.08, 0.5, 'Normalized Flux', va='center', rotation='vertical')
    if plt_name is not None:
        plt.savefig(os.path.join(out_dir, plt_name),bbox_inches='tight')
    plt.show()


def make_order_plot(dataset,model,lrange,plt_epoches,device,out_dir,plt_name=None,line_list=None):
    # model = jabble.model.load(model_name)
    data_orders = np.unique(model.metadata["orders"])
    model.fix()
        
    fig, axes = plt.subplots(2,len(plt_epoches),figsize=(4*len(plt_epoches),4),sharex='col',sharey='row',\
                             facecolor=(1, 1, 1),height_ratios=[4,1],dpi=200)

    for ii, plt_epoch in enumerate(plt_epoches):
        axes[:,ii] = make_subplot(axes[:,ii],model,dataset,plt_epoch,device,lrange[ii,:])#(axes,model,dataset,plt_epoch,device,lrange)
        if line_list is not None:
            plot_line_list(axes[:,ii],model,line_list,lrange[ii,:],plt_epoch)
    
    # plt.x
    # plt.text(1, 1, 'Wavelength ($\AA$)', ha='center')
    # plt_name = os.path.join(out_dir, "02-spectra_{}_{}-{}.png".format(os.path.split(model_name)[-1],lmin,lmax))
    # plt.savefig(plt_name,dpi=200,bbox_inches='tight')
    # fig.suptitle('Barnards Star')
    # fig.text(0.5, 0.00, 'Wavelength $[\AA]$', ha='center')
    # fig.text(0.08, 0.5, 'Normalized Flux', va='center', rotation='vertical')
    plt.subplots_adjust(wspace=0.3)
    if plt_name is not None:
        plt.savefig(os.path.join(out_dir, plt_name),bbox_inches='tight')
    plt.show()

def plot_line_list(axes,model,line_list,lrange,plt_epoch):
    for line in line_list[1].data[(line_list[1].data["Wave"] > lrange.min()) * (line_list[1].data["Wave"] < lrange.max())]:
        print(line["Species"])
        axes[0].axvline(np.log(line["Wave"]) + model[0][0].p[plt_epoch],-5,5,c='k',linestyle='dashed',alpha=0.4)