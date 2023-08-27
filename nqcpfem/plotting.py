import numpy as np
import pyvista
import dolfinx



def plot_eigenvector(eigenvector,function_space,color=None,scaling =True,length_scale=1,drop_abs=False,keep_sign=False,return_plot=False,**kwargs):
        """
        Plot the eigenvector of a band-model
        :param np.ndarray eigenvector: the eigenvector solution
        :param np.ndarray mesh_geometry: The coordinates of the mesh points. array of shape (N_points,3)
        :param int|None color: how tol color the plot #todo
        :param str|None arrow: what arrows in the plot represent. Default is spin
        :return:
        """
        from .fenics import FEniCsModel
        
        eigenvector = eigenvector.copy() # prevent overwriting the array
        
        if isinstance(function_space,FEniCsModel):
            function_space = function_space.function_space()
        normalization = np.linalg.norm(eigenvector.reshape(-1,eigenvector.shape[-1]), axis=0) # flatten the spinor dims to one dim and normalize
        n_tol = 1e-7
        eigenvector[...,normalization < n_tol] = np.nan # drop
        if scaling: 
            factor = np.max(normalization)
            scaled_vector = eigenvector/factor
        else:
            scaled_vector = eigenvector
        
        p = pyvista.Plotter()
        topology, cell_types, x = dolfinx.plot.create_vtk_mesh(function_space)
        x = x*length_scale # scale x axis
        grid = pyvista.UnstructuredGrid(topology, cell_types, x)
        
        if keep_sign:
            if len(scaled_vector.shape) >3 or (len(scaled_vector.shape) == 2 and scaled_vector.shape[1]>1):
                raise SyntaxError(f'wrong shape of keep_sign=True')
            grid["u"] = scaled_vector
        else:
            grid["u"] = np.linalg.norm(scaled_vector.reshape((-1,eigenvector.shape[-1])), axis=0)  # positional wave-function shape
            grid['c'] = np.linalg.norm(scaled_vector.reshape((-1,eigenvector.shape[-1])),axis=0) # for color scale
        #grid['hh_lh'] = hh_lh_expec
        warped = grid.warp_by_scalar("u")
        grid['u'] = grid['u']

        

        #p.add_mesh(grid, scalars='hh_lh', nan_color='black', clim=[-1, 1])
        p.add_mesh(warped,scalars='u',color='c',**kwargs)
        p.show_axes()
        if return_plot:
            return p

        p.show()
  
        """
          if spin:
            xy_eigvec = np.linalg.norm(eigenvector.reshape((eigenvector.shape[0],eigenvector.shape[1],-1)),axis=-1)
            z_expec = 3 / 2 * np.abs(xy_eigvec[:, 0]) ** 2 + 1 / 2 * np.abs(xy_eigvec[:, 1]) ** 2 - 1 / 2 * np.abs(
                xy_eigvec[:, 2]) ** 2 - 3 / 2 * np.abs(xy_eigvec[:, 3]) ** 2
            spin_vec = np.zeros( (len(warped.points[:, 0]), 3))

            spin_vec[::10, 2] = z_expec[::10]
            warped['arrows'] = spin_vec * 3
            arrows = warped.glyph(scale='arrows', orient='arrows')
            p.add_mesh(arrows,lighting=False, color='black',opacity=0.5)
        """


def plot_function(func,function_space,rescale=True,length_scale=1,show_xy_plane=False,return_plotter=False):
    """
    Plots a function on a function space grid.
    :param func:
    :param function_space:
    :param bool rescale: whether to rescale the function output so that max is 1
    :param float length_scale: Evaluates the function at points multiplied by this scale
    :return:
    """
    p = pyvista.Plotter()
    from .fenics import FEniCsModel
    if isinstance(function_space,FEniCsModel):
        length_scale = function_space.length_scale()
        function_space=function_space.function_space() #create function_space
    topology, cell_types, x = dolfinx.plot.create_vtk_mesh(function_space)
    grid = pyvista.UnstructuredGrid(topology, cell_types, x)
    vals = func(length_scale*x.T)
    if rescale:
        vals = vals/np.max(np.abs(vals))
    grid["u"] = vals
    warped = grid.warp_by_scalar("u")
    p.add_mesh(warped, scalars='u')
    p.show_axes()

    if show_xy_plane:
        mesh = pyvista.Plane()
        mesh.point_data.clear()
        p.add_mesh(mesh,show_edges=True)
    if return_plotter:
        return p

    p.show()

def HH_LH_components(eigenvector):
    """
    Compute the HH and LH components of the eigenvector
    :param np.ndarray eigenvector: the eigenvector to compute HH_LH splitting from
    :return:
    """
    HH_proj = np.zeros((4,4),dtype='complex')
    HH_proj[0,0] = 1
    HH_proj[3,3] = 1

    LH_proj = np.zeros((4,4),dtype='complex')
    LH_proj[1,1] = 1
    LH_proj[2,2] = 1

    HH_comp = expec_val(eigenvector,HH_proj,index=1)
    LH_comp = expec_val(eigenvector,LH_proj,index=1)

    return HH_comp,LH_comp


def compute_angular_momentum(eigenvector):
    from . import ANGULAR_MOMENTUM as AM
    J = AM['3/2']
    return expec_val(eigenvector,J)


def expec_val(eigenvector, operator, index=1):
    """
    Compute the expectationvalue of the specified operator
    :param np.ndarray eigenvector: quantum state to measure expectation value wrt.
    :param np.ndarray operator: operator which to compute expectation value of. The dimensions of the operator are paired (n,n,m,m,l,l,...)
    if the number of dimensions os odd, the first dimension is interpreted as indexing different operators
    :param int|tuple(int) index: which index or indices the operator contracts over.
    :return:
    """
    index = (index,) if isinstance(index, int) else index # convert to tuple

    start_from = 0
    only_diagonal = False
    evec_dims = len(eigenvector.shape)
    if eigenvector.shape[0] < 100:
        # reserve index 0 and 1 to index eigenstates
        evec_dims += -1
        start_from +=1
    if len(operator.shape)%2 == 1:
        if len(operator.shape) == 1:
            only_diagonal =True
        else:
            start_from +=1

    out_index = [_ for _ in range(start_from)]

    l_index = [_ + start_from for _ in range(evec_dims)]

    l_contract_indices = [l_index[i] for i in index]  # these inds of left eig are contracted
    if not only_diagonal:
        r_contract_indices = [_ + 1 + l_index[-1] for _ in
                            range(len(l_contract_indices))]  # these inds of left eig are contracted


        lr_inds = [l_contract_indices, r_contract_indices]
        contract_indices = [l[i] for i in range(len(index)) for l in lr_inds]  # zip contraction indices together
    else:
        contract_indices = l_contract_indices
        r_contract_indices = contract_indices

    r_index = l_index.copy()
    for i, contrac in zip(index, r_contract_indices):
        r_index[i] = contrac  # replace with contracting index

    if len(operator.shape)%2 == 1 and len(operator.shape)>1:
        contract_indices = [out_index[0]]+contract_indices

    if start_from>0:  # add 0 and 1 inices to the lists
        l_index = [out_index[start_from-1]] + l_index
        r_index = [out_index[start_from-1]] + r_index


    return np.einsum(np.conj(eigenvector), l_index, operator, contract_indices, eigenvector, r_index, out_index)



def density_plot_3d(eigenvector,x,**kwargs):
    from matplotlib import pyplot as plt
    import matplotlib as mpl
    from matplotlib import colors
    # last axis is positional, so we take the norm over everything else:

    eigenvector = eigenvector.reshape((-1,eigenvector.shape[-1]))
    density = np.linalg.norm(eigenvector,axis=0)

    cmap= kwargs.get('cmap',mpl.colormaps['viridis'])
    c = cmap(density/np.max(density))
    alpha_shift = kwargs.get('alpha_shift',0.01)
    alpha_factor = np.clip(density/np.max(density)+alpha_shift,0,1)

    drop_under = kwargs.get('c_drop_under',0)
    drop_over = kwargs.get('c_drop_over',1)
    norm=colors.Normalize(np.max(density)*drop_under,np.max(density)*drop_over,clip=True)
    alpha_under = kwargs.get('alpha_drop_under',0.01)
    alpha_over = kwargs.get('alpha_drop_over',20)
    alpha_norm = colors.Normalize(alpha_under*np.max(density),alpha_over*np.max(density),clip=True)

    c = cmap(norm(density))
    c[:,-1] = alpha_norm(density)

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.scatter3D(x[:,0],x[:,1],x[:,2],c=c)#,scale_factor =0.7)

    return fig,ax