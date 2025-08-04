import numpy as np
import matplotlib.pyplot as plt
import corner

def singular_values_plots(S, labelsize = 25, figsize = (6,5), dpi = 200, linewidth = 3, s = 90, fontsize = 25, \
                          xlabel = r'Components $j$', ylabel = r'Singular Value $s_j/s_1$'):
    """  
    Self monologing: Needed variables to plot this graph: 
    + S matrix from singular value decomposition (probably has been saved in the class attributes).
    + labelsize of graph.
    + figsize of the figure.
    + dpi of the figure.
    + linewidth of the plot.
    + s of the plot.
    + fontsize
    + label of the x axis
    + label of the y axis
    """
    plt.rc("xtick", labelsize= labelsize)
    plt.rc("ytick", labelsize= labelsize)


    fig, ax = plt.subplots(figsize= figsize, dpi= dpi)
    fig.patch.set_facecolor('white')

    #We plot only up to n-1 singular values if we centered the data because the nth value is numerical noise ~10**(-15)
    ax.scatter(np.arange(1,S.size), S[0:-1]/S[0],color='purple',s= s)
    ax.plot(np.arange(1,S.size), S[0:-1]/S[0],color='purple',linewidth= linewidth)
     






    # if centering_data:
    #     ax.scatter(np.arange(1,S.size), S[0:-1],color='purple',s=90)
    #     ax.plot(np.arange(1,S.size), S[0:-1],color='purple',linewidth=3)
    # else:
    #     ax.scatter(np.arange(1,S.size+1), S,color='purple',s=90)
    #     ax.plot(np.arange(1,S.size+1), S,color='purple',linewidth=3)    

    ax.set_yscale('log')
    ax.set_xlabel(f"{xlabel}",fontsize = fontsize)
    ax.set_ylabel(f"{ylabel}",fontsize = fontsize)
    # plt.ylim(2*10**(-3),1.5)
    # ax.set_xticks([0,5,10,15])
    # plt.title("Singular values decay",fontsize=30)
    plt.show()

def Plotter2D_single(xvals, yvals, zvals, quantity_of_interest, model, labelsize = 25, figsize = (12,8), dpi = 200, \
                     fontsize = 25, xlabel = 'Neutrons', ylabel = 'Protons'):
    # xvals,yvals,zvals = data
    #A plotter to see principal components in 2D
    plt.rc("xtick", labelsize= labelsize)
    plt.rc("ytick", labelsize= labelsize)
    
    fig, ax = plt.subplots(figsize= figsize, dpi= dpi)

    # Create scatter plot
    sc = ax.scatter(xvals, yvals, c=zvals, s=25, cmap='plasma', marker='s')
    # plt.colorbar(sc, label='Z-Value')
    cbar = plt.colorbar(sc, ax=ax)
    cbar.set_label(f'{quantity_of_interest}', fontsize= fontsize)

    cbar.ax.tick_params(labelsize= labelsize) 
    plt.xlabel(f'{xlabel}',fontsize= fontsize)
    plt.ylabel(f'{ylabel}',fontsize= fontsize)

    ax.grid(True)
    # ax.axis('equal')
    plt.title(f'{model}', fontsize = 25)
    # plt.show()

def Plotter3D_single(xvals, yvals, zvals, title, labelsize = 15, fontsize = 20, figsize = (8,6), dpi = 200, dxy = 1.5, \
                     xlabel = 'Neutrons', ylabel = 'Protons', zlabel = 'Z Value', elev=30, azim=-60):
    # xvals,yvals,zvals = data
    #A plotter to see principal components in 3D
    plt.rc("xtick", labelsize= labelsize)
    plt.rc("ytick", labelsize= labelsize)

    z_min = zvals.min()
    z_max = zvals.max()
    z_normalized = zvals

    # Create figure for 3D plot
    fig = plt.figure(figsize= figsize, dpi= dpi)
    ax = fig.add_subplot(111, projection='3d')



    # Define the size of the bars
    dx = dy = dxy  # Width of the bars in the x and y direction
    dz = z_normalized        # Height of the bars (z values)

    # Create 3D bar plot
    ax.bar3d(xvals, yvals, np.zeros_like(zvals), dx, dy, dz, color=plt.cm.plasma((zvals - zvals.min()) / (zvals.max() - zvals.min())))
    ax.set_zlim(np.minimum(0, zvals.min()), zvals.max())
    ax.view_init(elev=elev, azim=azim) 

    # Setting labels and title
    ax.set_xlabel(f'{xlabel}')
    ax.set_ylabel(f'{ylabel}')
    ax.set_zlabel(f'{zlabel}')
    ax.set_title(title)

    mappable = plt.cm.ScalarMappable(cmap='plasma', norm=plt.Normalize(vmin=zvals.min(), vmax=zvals.max()))
    mappable.set_array(zvals)

    # Add the colorbar to the plot
    cbar = plt.colorbar(mappable, ax=ax, shrink=0.5, aspect=10, location='right', pad=0.2)
    cbar.set_label(title, fontsize= fontsize)
    cbar.ax.tick_params(labelsize= labelsize)
    
    # plt.show()



def Plotter2D(ax, xvals, yvals, zvals, validation_set, quantity_of_interest, model, xlabel = 'Neutrons', ylabel = 'Protons'):
    #A plotter to see principal components in 2D, part of the big plotter


    # Create scatter plot
    sc = ax.scatter(xvals, yvals, c=zvals, s=25, cmap='plasma',marker='s')
    # plt.colorbar(sc, label='Z-Value')
    cbar = plt.colorbar(sc, ax=ax)
    ax.scatter(validation_set.T[0], validation_set.T[1], color = 'black', alpha = 1, marker = 'o', s = 35)
    cbar.set_label(f'{quantity_of_interest}', fontsize= 25) 
    cbar.ax.tick_params(labelsize= 20) 




    plt.xlabel(f'{xlabel}',fontsize= 25)
    plt.ylabel(f'{ylabel}',fontsize= 25)

    plt.title(f'{model}', fontsize = 25)
    ax.grid(True)
    ax.axis('equal')

    # plt.show()

def Plotter3D(ax,xvals,yvals,zvals, xlabel = 'Neutrons', ylabel = 'Protons', zlabel = 'Scaled Z', elev=30,azim=-60):
    #A plotter to see principal components in 3D, part of the big plotter
    z_min = zvals.min()
    z_max = zvals.max()
    z_normalized = zvals

    # Create figure for 3D plot
    # fig = plt.figure(figsize=(8, 6), dpi=200)
    # ax = fig.add_subplot(111, projection='3d')



    # Define the size of the bars
    dx = dy = 1.5  # Width of the bars in the x and y direction
    dz = z_normalized        # Height of the bars (z values)

    # Create 3D bar plot
    ax.bar3d(xvals, yvals, np.zeros_like(zvals), dx, dy, dz, color=plt.cm.plasma((zvals - zvals.min()) / (zvals.max() - zvals.min())))
    ax.set_zlim(np.minimum(0, zvals.min()), zvals.max())
    ax.view_init(elev=elev, azim=azim) 

    # Setting labels and title
    ax.set_xlabel(f'{xlabel}')
    ax.set_ylabel(f'{ylabel}')
    ax.set_zlabel(f'{zlabel}')


    # plt.show()    



def PlotMultiple(xvals, yvals, zvals, validation_set, quantity_of_interest, model, xlabel = 'Neutrons',\
                  ylabel = 'Protons', zlabel = 'Scaled Z',angles_sheet=None):
    # Determine the number of rows needed for the plots
    # n_rows = len(data_sets)
    # Each data_sets element should look like [x,y,z]

    # Create a figure with subplots
    fig = plt.figure(figsize=(20, 8), dpi=200)


    # 2D plot
    ax_2d = fig.add_subplot(1, 2, 1)
    Plotter2D(ax_2d, xvals, yvals, zvals, validation_set, quantity_of_interest, model, xlabel = 'Neutrons', ylabel = 'Protons')

    # 3D plot
    ax_3d = fig.add_subplot(1, 2, 2, projection='3d')
    if angles_sheet == None:
        Plotter3D(ax_3d, xvals, yvals, zvals, xlabel = 'Neutrons', ylabel = 'Protons', zlabel = 'Scaled Z')
    else:
        Plotter3D(ax_3d, xvals, yvals, zvals,elev=angles_sheet[0],azim=angles_sheet[1])

    plt.tight_layout()
    # plt.show()

    
def models_projections():
    """ 
    Plot of the pricipal components into the space of the models.
    """

    
def PCs_arrows(PC_index1, PC_index2, Vt_hat_normalized, models, colors, labelsize = 35, fontsize = 35, \
               figsize = (10,10), dpi = 100, headwidth = 0.02, head_length = 0.05):
    """
    Plot of the models into the space of the principal components. Non0default arguments will include indiced of the corresponding principal
    components that we are graphing. 
    + PC_index: integers denoting indices of interested principal components
    """
    plt.rcParams['text.usetex'] = False

    plt.rc("xtick", labelsize= labelsize)
    plt.rc("ytick", labelsize= labelsize)

    # fig, ax = plt.subplots(figsize=(10,8), dpi=100)
    fig, ax = plt.subplots(figsize= figsize, dpi= dpi)



    visited_model=0
    for model in models:
        
        ax.arrow(0, 0, Vt_hat_normalized.T[visited_model][PC_index1],Vt_hat_normalized.T[visited_model][PC_index2], 
                head_width= headwidth, head_length= head_length, fc=colors[visited_model], ec=colors[visited_model], label=model)
        # plt.text(Vt_hat_normalized.T[visited_model][2], Vt_hat_normalized.T[visited_model][1], model,fontsize=20,ha='center', va='center') 

        visited_model=visited_model+1




    # Determine plot limits based on the maximum absolute values of the vectors
    # max_val = max(max(abs(coordinate) for vector in list1 + list2 for coordinate in vector))
    # plt.xlim(-max_val-1, max_val+1)
    # plt.ylim(-max_val-1, max_val+1)

    plt.xlabel(fr'Projection on $\phi_{{{PC_index1}}}$',fontsize= fontsize)
    plt.ylabel(fr'Projection on $\phi_{{{PC_index2}}}$',fontsize= fontsize)

    # plt.legend(fontsize=15)

    
    plt.xticks(fontsize= fontsize)
    plt.yticks(fontsize= fontsize)

    
    
    plt.axis('equal')

    # plt.show()

def samples_visualizations(samples, labelsize, bins, dpi, linewidth, fontsize):
    """ 
    Corner plot of the principal components weights and uncertainties from sampling algorithm. Non-default arguments include:
    + samples: shape (10000, components_kept+1)
    """
    plt.rcParams['text.usetex'] = False
    # plt.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'

    label_global = [f'$b_{i}$' for i in range(1, samples.shape[0])]
    label_global.append(r'$\sigma$')

    plt.rc("xtick", labelsize= labelsize)
    plt.rc("ytick", labelsize= labelsize)

    figure = corner.corner(samples, bins= bins, dpi= dpi, hist_kwargs={"linewidth": linewidth},
                        labels=label_global, 
                        #    truths=np.append(beta,np.sqrt(sigma_squared)),
                        #    truth_color="r",
                        label_kwargs={"fontsize": fontsize}, labelpad= 0.1)
    
    # plt.show()


def models_weights(samples, Vt_hat, models, colors, labelsize, figsize, dpi, fontsize, ticksize, capsize):
    """ 
    Plots of all the models' weights:
    + We can transform the samples' weights into the models weights. For their uncertainties, we should calculate 
    the standard error of the weights
    """
    plt.rcParams['text.usetex'] = False
    # plt.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'

    # This is the principal components' weights
    betas = samples[:, :-1]
    # Compute model weights: shape (10000, num_models)
    default_weights = np.full(Vt_hat.shape[1], 1 / Vt_hat.shape[1])
    model_weights_random = betas @ Vt_hat + default_weights  # broadcasting default_weights

    plt.rc("xtick", labelsize= labelsize)
    plt.rc("ytick", labelsize= labelsize)
    plt.figure(figsize= figsize, dpi= dpi)
    # y_max=0.6

    i=0
    for model in models:
            
            # ax.plot(filtered_models_output["N"], filtered_models_output[key_list[model_index]], 
            #         label = n_Labels[class_index], color=colors[class_index],alpha = alpha_models,linewidth=5)
            # plt.bar(model , np.mean(samples_naive_space_sigma.T[i]), color=colors[i])
            plt.bar(model , np.mean(model_weights_random.T[i]),yerr=2*np.std(np.std(model_weights_random.T[i])), color=colors[i], capsize= capsize)
            i=i+1

    plt.minorticks_off()


    # plt.xlabel('Models',fontsize=15)
    plt.ylabel(r'$\omega_k$', fontsize= fontsize)
    # plt.ylim(-0.2,0.5)
    # plt.yticks([0, 0.2, 0.4])
    plt.xticks(fontsize= ticksize,rotation='vertical')
    plt.yticks(fontsize= ticksize)
    plt.grid(axis='y')
    plt.tight_layout()
    # plt.show()


def coverage_graph(percentiles, coverage_train, coverage_val, coverage_test, colors, linewidth, fontsize):

    plt.rc("xtick", labelsize=22)
    plt.rc("ytick", labelsize=22)
    # Plotting the results
    plt.figure(figsize=(6, 6),dpi=200)

    width_line=2

    plt.plot(percentiles, coverage_train, label='Train',color = colors[0], linewidth = linewidth)
    plt.plot(percentiles, coverage_val, label='Validation',color= colors[1], linewidth = linewidth)
    plt.plot(percentiles, coverage_test, label='Test',color = colors[2], linewidth = linewidth)


    # plt.plot(percentiles, coverage_train_simplex,linestyle='dashed', label='Training Simplex',color=color_trainig,linewidth=width_line)
    # plt.plot(percentiles, coverage_validation_simplex, linestyle='dashed',label='Validation Simplex',color=color_validation,linewidth=width_line)
    # plt.plot(percentiles, coverage_test_simplex,linestyle='dashed', label='Test Simplex',color=color_testing,linewidth=width_line)



    plt.plot(percentiles, percentiles,label='reference',color='k',linewidth = linewidth)
    plt.xlabel(r'Credible Interval (\%)',fontsize = fontsize)
    plt.ylabel(r'Coverage (\%)',fontsize = fontsize)

    plt.xticks(ticks=[0, 20,40,60,80,100])

    # plt.title('Coverage of Observed Data Points by Credible Intervals')
    # plt.grid(True)
    plt.legend(fontsize = fontsize)

    # plt.savefig('Plots/Radii Data/Coverage graph/Coverag graph of global radii unconstrained case.png')
    plt.show()