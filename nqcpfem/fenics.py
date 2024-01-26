from re import A
from . import band_model,envelope_function
from .updatable_object import ComputedAttribute,AttributeSnapshot,auto_update
from .envelope_function import Domain
import numpy as np
import sympy
from typing import List,Tuple
import ufl
from mpi4py import MPI
import dolfinx
from petsc4py import PETSc

if PETSc.ScalarType != np.complex128:
    raise ImportError(f'usage of FEniCS requires complex build of PETSc, but the current PETSc build is for real numbers')


from . import band_model as bm
from . import symbolic as symb
from .functions import SymbolicFunction
import logging
LOGGER = logging.getLogger(__name__)
LOGGER.addHandler(logging.NullHandler())




class FEniCsModel(envelope_function.EnvelopeFunctionModel):
    def __init__(self, band_model, domain,boundary_condition=None,function_class=('CG',1),fix_energy_scale=False):

        """
        EnvelopeFunctionModel which evaluates the k operator using finite element methods, built on FEniCSX. 	
        Given a band_model and a Domain, it can assemble a Hermitian matrix (and S-matrix) for solving the generalized eigenvalue problem
        :param	BandModel band_model: The BandModel describing the Hamiltonian of the system.
        :param	Domain domain: The Domain which the model must be solved on
        :param	float|None boundary_condition: Constant value for the boundary condition. By default, the boundary condition is 0.
        :param	Tuple[str,int] function_class: Class of functions to use in the finite elements. Defaults to ('CG',1).
        :param bool fix_energy_scale: Whether to fix the energy scale to the first time it is computed or not. This will speed up consecutive model constructions, but assumes that the energies between the two models are similar (which they of often are)
        """
        # check that domain has the required attributes:
        must_have = {}
        if isinstance(domain,envelope_function.RectangleDomain):
            must_have['resolution'] =  List
        elif isinstance(domain,envelope_function.CircleDomain):
            raise NotImplementedError

        for attr,attr_type in must_have.items():
                if hasattr(domain,attr):
                    if not isinstance(getattr(domain,attr),attr_type):
                        raise TypeError(f'domain specified had attribute "{attr}" of wrong type.'
                                        f'expected {attr_type}, but recieved {type(getattr(domain,attr))} ')
                else:
                    raise AttributeError(f'domain was missing required attribute: {attr}')

        # check that FEM ordering of ks is performed      
        k_ordering = band_model.independent_vars['model_defining_params'].get('k_signature_type',None)
        if k_ordering is None:
            LOGGER.warn(f'the k-ordering of the band model must be FEM, but it was not set. setting it to FEM wit left reduction direction')
            band_model.fix_k_arrangement('FEM')
        elif k_ordering != 'FEM':
            raise ValueError(f'k-ordering of band model must be "FEM" but got {k_ordering}')
        
        
        
        
        independent_vars = {'band_model':band_model,
                            'domain':domain,
                            'boundary_condition':boundary_condition,
                            'function_class':function_class,
                            'fix_energy_scale':fix_energy_scale}
        
        super().__init__(**independent_vars)
    
    @property
    def k_signature(self):
        return 'FEM'
        
    @auto_update
    def mesh(self):
        """Constructs the mesh of the system depending on number of spinor components, the domain, and the specified function class
        """
        return self.__make_mesh__(self.independent_vars['domain'], self.independent_vars['band_model'].spatial_dim, 1 / self.length_scale())

    @staticmethod
    def __make_mesh__(domain:Domain,spatial_dim:int,scale_factor:float=1):
        """
        Constructs a mesof the specified domain
        :param  domain:
        :param scale_factor: Scales the dimensions of the model by a factor ( multiples by this)
        :return:
        """
        mesh_comm = MPI.COMM_WORLD
        if domain.mesh is not None:
            return domain.mesh
            # use existing mesh
            from dolfinx.io import gmshio

            gmsh_model_rank = 0
            domain, cell_markers, facet_markers = gmshio.model_to_mesh(domain.mesh, mesh_comm, gmsh_model_rank,
                                                                    gdim=spatial_dim)
            return domain

        else:
            if not hasattr(domain,'resolution'):
                raise ValueError(f'specified domain did not have a resolution defined.')
            from .envelope_function import RectangleDomain, CircleDomain
            if isinstance(domain, RectangleDomain):

                lower_left = [l * scale_factor for l in domain.lower_left_corner[:spatial_dim]]
                upper_right = [l * scale_factor for l in domain.upper_right_corner[:spatial_dim]]
                if spatial_dim == 2:
                    """
                    Lx = domain.Lx *scale_factor
                    Ly = domain.Ly * scale_factor
                    lower_left = [-Lx / 2, -Ly / 2]
                    upper_right = [Lx / 2, Ly / 2]
                    """

                    return dolfinx.mesh.create_rectangle(mesh_comm, [lower_left, upper_right],
                                                                  domain.resolution[:spatial_dim],
                                                                      cell_type=dolfinx.mesh.CellType.triangle)#dolfinx.mesh.CellType.quadrilateral)

                elif spatial_dim == 3:
                    """
                    Lx = domain.Lx * scale_factor
                    Ly = domain.Ly * scale_factor
                    Lz = domain.Lz * scale_factor
                    lower_left = [-Lx / 2, -Ly / 2, -Lz / 2]
                    upper_right = [Lx / 2, Ly / 2, Lz / 2]
                    """
                    return dolfinx.mesh.create_box(mesh_comm, [lower_left, upper_right],
                                                            domain.resolution[:spatial_dim])
                else:
                    raise NotImplementedError(f'yet to write mesh construction for 1D')
            else:
                raise NotImplementedError(f'yet to implement circle domain construction')

    @auto_update
    def function_space(self):

        function_shape = self.independent_vars['band_model'].tensor_shape[::2]
        return self.__make_function_space__(self.mesh(),self.independent_vars['function_class'],function_shape)
    @staticmethod
    def __make_function_space__(mesh,function_class,function_shape):
        """
        Constructs the
        :param dolfinx.mesh.Mesh mesh: The mesh to define the function space over
        :param tuple[str,int] function_class: the function class to build the function space over
        :param tuple[int] function_shape: the shape of the function
        :return:
        """
        if isinstance(function_shape,int):
            function_shape = (function_shape,)
        if len(function_shape) == 1:
            if function_shape[0] > 1:
                # vector case (spinors)
                return dolfinx.fem.VectorFunctionSpace(mesh, function_class,
                                                                        dim=function_shape[0])
            else:
                # scalar case (scalar field)
                return dolfinx.fem.FunctionSpace(mesh, function_class)
        else:
            # tensor case (e.g. spinor + additional band componentes )
            return dolfinx.fem.TensorFunctionSpace(mesh, function_class,shape=function_shape)

    @auto_update
    def dolfin_bc(self):
        if self.independent_vars['boundary_condition'] is not None:
            if self.independent_vars['band_model'].tensor_shape == (1,1): # scalar case
                boundary_state = PETSc.ScalarType(self.independent_vars['boundary_condition'])  # scalara boundary condition
            else:
                boundary_state = np.full(self.independent_vars['band_model'].tensor_shape[::2],self.independent_vars['boundary_condition'],
                                        dtype=PETSc.ScalarType) # vector or tensor boundary condition
            if isinstance(self.independent_vars['domain'], envelope_function.RectangleDomain):
                fdim = self.mesh().topology.dim - 1

                boundary_facets = dolfinx.mesh.locate_entities_boundary(
                    self.mesh(), fdim, lambda x: np.full(x.shape[1], True, dtype=bool))

                boundary_facets = dolfinx.mesh.exterior_facet_indices(self.mesh().topology)
                boundary_condition = [dolfinx.fem.dirichletbc(boundary_state,
                                                    dolfinx.fem.locate_dofs_topological(self.function_space(), fdim,
                                                                                        boundary_facets),
                                                    self.function_space())]

            elif isinstance(self.independent_vars['domain'], envelope_function.CircleDomain):
                fdim = self.mesh().topology.dim - 1

                def on_boundary(x):
                    return np.isclose(np.sqrt(x[0] ** 2 + x[1] ** 2), self.independent_vars['domain'].R)

                boundary_dofs = dolfinx.fem.locate_dofs_geometrical(self.function_space(), on_boundary)
                boundary_condition = [dolfinx.fem.dirichletbc(boundary_state,
                                                    dolfinx.fem.locate_dofs_topological(self.function_space(), fdim,
                                                                                        boundary_dofs),
                                                    self.function_space())]
            else:
                raise NotImplementedError()
            return boundary_condition
        else:

                return []

    @auto_update
    def length_scale(self):
        if self.independent_vars['domain'].__mesh_scale__ is not None:
            return float(self.independent_vars['domain'].__mesh_scale__)

        if isinstance(self.independent_vars['domain'],envelope_function.RectangleDomain):
            length_scale = np.max([self.independent_vars['domain'].Lx,self.independent_vars['domain'].Ly,self.independent_vars['domain'].Lz])
        elif isinstance(self.independent_vars['domain'],envelope_function.CircleDomain):
            length_scale = self.independent_vars['domain'].R
        else:
            raise NotImplementedError
        return float(length_scale) # this number should never be complex
    
    
    
    def __make_energy_scale__(self):
        """
        determines the energy scale used in the assembled array. This is because numerics are unstable with
        very small eigenvalues.
        :return:
        """

        # divide out length scale to get dimensions equal for all terms.
        #max_vals =np.array([np.max(np.abs(np.array(sympy.Array(T).subs({d:0 for d in self.independent_vars['band_model'].position_symbols}))))/self.length_scale()**R for R,T in self.independent_vars['band_model'].numerical_tensor_repr().items()]).astype(np.float64)
        #return np.real(np.max(max_vals))#.astype(np.float64) # this number should never be complex
        max_vals = []
        
        
        # determien the energy scale based off of the 
        
        #TODO: construct the numerical array by constructing the Valuemap using band_model. adding kx,ky,kz and x,y,z to it (eval to -1j/L and 0 respectivel)
        # and add the functions to it as well. 
        # given dummy map and lambda_template from band_model, construct subsituted array directly, 
        
        # compute it from the numerical array instead
        LOGGER.debug('constructing array')
        arr = sympy.Array(self.band_model.numerical_array())
        LOGGER.debug('substituting')
        arr=arr.subs({k:-1j/self.length_scale() for k in self.band_model.momentum_symbols}).subs({x:0 for x in self.band_model.position_symbols})
        LOGGER.debug('casting back')
        arr = np.array(arr).astype('complex')
        
        LOGGER.debug('maximizing')
        e= float(np.max(np.real(arr)))
        return e #this is fast enough
        
        for R,T in self.band_model.numerical_tensor_repr().items():
            try:
                arr = [np.abs(np.array(sympy.Array(T).subs({d:0 for d in self.independent_vars['band_model'].position_symbols})))]
                max_vals.append(np.max(arr)/self.length_scale()**R)
            except TypeError as err:
                pass
            
#        max_vals =np.array([ for R,T in self.independent_vars['band_model'].numerical_tensor_repr().items()]).astype(np.float64)
        if len(max_vals):
            ee= float(np.real(np.max(max_vals)))#.astype(np.float64) # this number should never be complex
            print(ee)
            return(ee)
        else:
            return 1

    @auto_update
    def energy_scale(self):
        
        fixed_E = self.independent_vars['fix_energy_scale']
        
        if fixed_E:
            try:
                return self._saved_energy_scale.value # return the precomputed energy scale type:igonre
            except Exception:
                pass
        
        LOGGER.debug('computing energy_scale')
        return self.__make_energy_scale__()
    @auto_update
    def converted_functions(self):

        converted_funcs = {}
        symbolic_funcs = {}
        scalar_function_space = self.__make_function_space__(self.mesh(),self.independent_vars['function_class'],1) # interpolate the scalar functions as the same class as the trial and test functions
        fem_x = ufl.SpatialCoordinate(self.mesh())
        variables =  self.dolfinx_constants().copy()
        pos_syms = self.band_model.position_symbols
        variables.update({pos_syms[i]:fem_x[i]*self.length_scale() for i in range(self.band_model.spatial_dim)}) # type: ignore
        for sym,func in self.band_model.post_processed_functions().items():
            if isinstance(func,SymbolicFunction):
                if isinstance(func.expression,sympy.Piecewise):
                    # we have to traverse through any sympy piecewise functions and convert them, since UFL stuff cannot handle regular comparison operators. 
                    # NB! There should'nt be any free symbols other than those contained in parameter_dict and constants as wells as x,y, and z in the piecewise expression.
                    # this is done becuase UFL types and sympy types DO NOT MIX and UFL types use special relation operators. 
                    converted_piecewise = convert_piecewise(func.expression,__ufl_relations__,variables =variables)
                    converted_funcs[sym] = converted_piecewise
                else:
                    symbolic_funcs[sym] = func.expression
            else:
                
                # other functions are interpolated in the usual way
                f = dolfinx.fem.Function(scalar_function_space)
                f.name = sym.name
                
                #we need to scale the input
                scaled_func=lambda x:func(x*self.length_scale())
                f.interpolate(scaled_func)
                converted_funcs[sym] = f 
        return symbolic_funcs,converted_funcs   
    
    @auto_update
    def dolfinx_constants(self):
        """
        Constructs a dictionary mapping the free sympy symbols of the post-processed array to dolfinx constanps
        """

        LOGGER.debug('creating constants dict')
        # We avoid accessing the parameter_dict (and hence raising the changed flag) by directly looking at the post-processed functions and arrays for free symbols 
        band_model = self.independent_vars['band_model']
        free_syms = tuple( s for s in band_model.post_processed_array().free_symbols if (s not in band_model.momentum_symbols + band_model.position_symbols) and (s.name[-3:] != '(x)')) # functions and special variables are excluded as they do not need casting as dolfinx constants
        func_syms = set() # add free symbols containted
        for f in band_model.post_processed_functions().values():
            if isinstance(f,SymbolicFunction):
                func_syms.update(f.expression.free_symbols)
        
        func_syms.update(free_syms)
        free_syms = tuple(func_syms)
        pos_syms= sympy.symbols('x,y,z')
        
        # parameter dict with all the symbols to assing and the corresponding 
        new_param_dict =  {s:dolfinx.fem.Constant(self.mesh(), np.complex128(1)) for s in free_syms} #type: ignore
        

        scalar_function_space = self.__make_function_space__(self.mesh(),self.independent_vars['function_class'],1) # interpolate the scalar functions as the same class as the trial and test functions
        for sym,func in new_param_dict.items():
            if sym.name[-3:] == '(x)':
                if isinstance(func,sympy.Piecewise):
                    # we have to traverse through any sympy piecewise functions and convert them, since UFL stuff cannot handle regular comparison operators. 
                    # NB! There should'nt be any free symbols other than those contained in parameter_dict and constants as wells as x,y, and z in the piecewise expression.
                    # this is done becuase UFL types and sympy types DO NOT MIX and UFL types use special relation operators. 

                    converted_piecewise = convert_piecewise(func,__ufl_relations__,variables = new_param_dict)
                    new_param_dict[sym] = converted_piecewise
                
                else:
                    # other functions are interpolated in the usual way
                    f = dolfinx.fem.Function(scalar_function_space)
                    f.name = sym.name
                    
                    f.interpolate(lambda x: func(x*self.length_scale()))
                    new_param_dict[sym] = f 
        
        
        new_param_dict['__energy_scale__'] = dolfinx.fem.Constant(self.mesh(),np.complex128(1)) # add the energy scale as a constant as well since it depends on the parameter_dict and we want the ufl_form to be independent of the parameter_dict
        if sympy.symbols(r'\hbar') in new_param_dict.keys():
            new_param_dict[sympy.symbols(r'hbar')] = new_param_dict[sympy.symbols(r'\hbar')]
            
        fem_x = ufl.SpatialCoordinate(self.mesh())
        new_param_dict.update({pos_syms[i]:fem_x[i]*self.length_scale() for i in range(self.band_model.spatial_dim)}) 
        return new_param_dict
    @auto_update
    def ufl_form(self):
        LOGGER.debug('computing ufl_form')
        is_scalar_problem = self.independent_vars['band_model'].tensor_shape == (1,1)
        
        import sympy
        
        
        new_param_dict = self.dolfinx_constants().copy()
        symbolic_funcs,numerical_funcs = self.converted_functions()
        new_param_dict.update(numerical_funcs)
        
        ufl_tensors = {}
        for rank,tensor in self.band_model.symbolic_tensor_repr().items():
            
            # we first substitute everything into, then scale by the relevant parameters.
            
            # replace symbolic functions with their symbolic expressions:
            tensor = sympy.Array(tensor).subs(symbolic_funcs)

            
            
            numerical_tensor = sympy.lambdify(new_param_dict.keys(),tensor,dummify=True,modules=[{'conjugate':fenics_conjugate},'numpy'])(*new_param_dict.values())
            scaling = new_param_dict['__energy_scale__']*np.complex128(self.length_scale()**rank)
            normalized_tensor = np.asarray(np.asarray(numerical_tensor)/scaling)#.astype(np.complex128)
            
            
            # (IF shape requires it) rollaxis (two places )of the tensor so that the first index corresponds to the right-most derivative coefficient!
            
            if all(i == 0 for i in normalized_tensor.ravel()):
                continue # drop all zero tensors as they will throw an error.
            
            
            #else:
            #   scaling = self.energy_scale*self.length_scale**tensor.spatial_rank
            #  scaled_tensor = band_modeling.PositionalTensorOperator(tensor.tensor/float(scaling),tensor.spatial_rank)
            # normalized_tensor= scaled_tensor(ufl.real(x)*self.length_scale,relation_operators=__ufl_relations__)
            #else:
            #   normalized_tensor = tensor.astype(PETSc.ScalarType)/(self.energy_scale*self.length_scale**tensor.spatial_rank)#/(self.energy_scale*self.length_scale**tensor.spatial_rank)# assure complex tensor and normalize

            if is_scalar_problem:
                ufl_version = ufl.as_tensor(normalized_tensor[0,0]) if tensor.spatial_rank else normalized_tensor[0,0] # drop axes of length 1
            else:
                try:
                    ufl_version = ufl.as_tensor(normalized_tensor)
                except ufl.UFLException as err:
                    raise ufl.UFLException(f'trying to cast a tensor (type {type(normalized_tensor)} as ufl failed:\n') from err
            ufl_tensors[rank] = ufl_version
        # L(u,v) = <u,Hv>
        L = self.__ufl_tensor_dict_to_bilinear_form__(ufl_tensors)
        return L
    def __ufl_tensor_dict_to_bilinear_form__(self,ufl_tensors,u=None,v=None):
        """
        The function `__ufl_tensor_dict_to_form__` takes a dictionary of UFL tensors and converts them
        into a UFL form.
        
        :param ufl_tensors: The `ufl_tensors` parameter is a dictionary where the keys are integers
        representing the rank of the tensor, and the values are UFL tensors. UFL tensors are symbolic
        expressions used in the Unified Form Language (UFL) for defining finite element forms
        :return: the form `L`, which is a UFL expression representing a linear form.
        """
        import ufl
        if u is None and v is None:
            u = ufl.TrialFunction(self.function_space())
            v = ufl.TestFunction(self.function_space())
        elif (u is not None) and  (v is None):
            raise SyntaxError('it is not allowed to only pass one of function u and v')
        L = None
        is_scalar_problem = self.independent_vars['band_model'].tensor_shape == (1,1)
        for rank,ufl_tensor in ufl_tensors.items():
            # assume grouped ordering of the indices so [aL,aR,bL,bR,...,x1,x2,x3,...] where the last x axes are the spatial ones

            I_ten = tuple() if is_scalar_problem else ufl.indices(len(self.band_model.tensor_shape))
            I_ten_left = tuple([I_ten[i] for i in range(0,len(I_ten),2)])
            I_ten_right = tuple([I_ten[i] for i in range(1, len(I_ten), 2)])
            I_space = ufl.indices(rank) # returns empty tuple if not rank
            del_v = ufl.conj(v) # conjugate the right input in agreement with mathematician's definitonv of complex function inner product
            for _ in range(rank-1):
                del_v = 1j*ufl.grad(del_v) # test this. No minus because of the integration by parts step involved in moving the derivative to the testfunction
            del_u = -1j*ufl.grad(u) if rank>0 else u # only take derivative if term involves k.

            if I_space != tuple():
                term = del_u[tuple(I_ten_left+(I_space[0],))] * ufl_tensor[I_ten+I_space] * del_v[I_ten_right+I_space[1:]]*ufl.dx

            else:
                if is_scalar_problem:
                    term = ufl_tensor*del_u*del_v*ufl.dx
                else:
                    term = del_u[I_ten_left] * ufl_tensor[I_ten] * del_v[I_ten_right]*ufl.dx
            if L is None:
                L = term
            else:
                L+= term

        return L
    
    def __ufl_tensor_dict_to_linear_form__(self,ufl_tensors,u):
        import ufl
        L = None
        is_scalar_problem = self.independent_vars['band_model'].tensor_shape == (1,1)
        for rank,ufl_tensor in ufl_tensors.items():
            # assume grouped ordering of the indices so [aL,aR,bL,bR,...,x1,x2,x3,...] where the last x axes are the spatial ones

            
            
            I_ten = tuple() if is_scalar_problem else ufl.indices(len(self.band_model.tensor_shape))
            I_ten_left = tuple([I_ten[i] for i in range(0,len(I_ten),2)])
            I_ten_right = tuple([I_ten[i] for i in range(1, len(I_ten), 2)])
            I_space = ufl.indices(rank) # returns empty tuple if not rank
            del_u = u
            for _ in range(rank-1):
                del_u = -1j*ufl.grad(del_u)

            if I_space != tuple():
                # TODO: check that this index  slicing works. Alternatively. Do the loop explicitly and assemble a nested list and cast as tensor /vector
                term = del_u[tuple(I_ten_left+(I_space[0],))] * ufl_tensor[I_ten_left+(Ellipsis,)+I_space]

            else:
                if is_scalar_problem:
                    term = ufl_tensor*del_u
                else:
                    term = del_u[I_ten_left] * ufl_tensor[I_ten_left+(Ellipsis,)]
            if L is None:
                L = term
            else:
                L+= term

        return L
    @auto_update
    def __gather_constants__(self):
        LOGGER.debug('gathering constants')
        constants_dict = self.dolfinx_constants()
        
        # update the value of the constants based on the values stored in the band_model parameter_dict and constants dict:
        numbers_dict = self.independent_vars['band_model'].independent_vars['parameter_dict']
        const_dict = self.independent_vars['band_model'].independent_vars['constants']
        
        for k,const in constants_dict.items():
            if k == '__energy_scale__':
                const.value = np.complex128(self.energy_scale())
            elif k not in self.independent_vars['band_model'].position_symbols:
                try:
                    const.value = np.complex128(numbers_dict[k])
                    
                except KeyError as err:
                    try:
                        const.value = np.complex128(const_dict[k])
                    except KeyError as e:
                        raise KeyError(f'key {k} in dolfinx_constants() was not found in neither the parameter_dict nor the constants dict of the band_model.') from e
        #we can now assemble the form  
        
        return constants_dict
    @auto_update
    def bilinear_form(self,overwrite=False):
        """
        Assemble the bilinear form for the band model by fixing the values of the constants to the values specified ind the models parameter dict
        :param bool overwrite: whether to overwrite any previously computed bilinear form
        :return:
        """
        LOGGER.debug('gathering bilinear_form')

        assembled_form = dolfinx.fem.form(self.ufl_form(),jit_options={'timeout':20})
        return assembled_form
        """
        # assemble L (this workaround has to be used since the program doesn't handle all zero terms...)
        #i, j, k, l, i_, j_, k_ = ufl.indices(7)
        I_d = ufl.indices(6*len(self.spinor_dim)) # 6 indices for each spinor dimension (2 A, and B and C each)
        I_x = ufl.indices(3) # 3 indices for coordinate indices

        L = None
        if not is_none[0]:
            I_A_tuple = I_d[:2*len(self.spinor_dim)]
            row_spinor_I = tuple([i for n,i in enumerate(I_A_tuple) if n%2==0]) # the row indices of each tensor axis
            collum_spinor_I = tuple([i for n,i in enumerate(I_A_tuple) if n%2==1])   # the column indices of each tensor axis
            L = ufl.grad(u[collum_spinor_I])[I_x[0]] * A_ten[(I_x[0],I_x[1])+I_A_tuple] * ufl.grad(ufl.conj(v)[row_spinor_I])[I_x[1]] * ufl.dx
        if not is_none[1]:
            I_B_tuple = I_d[2*len(self.spinor_dim):4*len(self.spinor_dim)] #
            row_spinor_I = tuple([i for n,i in enumerate(I_B_tuple) if n%2==0]) # the row indices of each tensor axis
            collum_spinor_I = tuple([i for n,i in enumerate(I_B_tuple) if n%2==1])   # the column indices of each tensor axis
            add_B = ufl.Dx(u, I_x[2])[collum_spinor_I] * B_ten[(I_x[2],)+I_B_tuple] * ufl.conj(v)[row_spinor_I] * ufl.dx
            if L is None:
                L = add_B
            else:
                L += add_B
        if not is_none[2]:
            I_C_tuple = I_d[4 * len(self.spinor_dim):]  #
            row_spinor_I = tuple(
                [i for n, i in enumerate(I_C_tuple) if n % 2 == 0])  # the row indices of each tensor axis
            collum_spinor_I = tuple(
                [i for n, i in enumerate(I_C_tuple) if n % 2 == 1])  # the column indices of each tensor axis
            add_C = u[collum_spinor_I]* C_ten[I_C_tuple] * ufl.conj(v)[row_spinor_I]*ufl.dx
            if L is None:
                L = add_C
            else:
                L += add_C


        ##
        #if not is_none[0]:
        #	L = ufl.Dx(u, i)[k] * A_ten[i, j, k, l] * ufl.Dx(ufl.conj(v), j)[l] * ufl.dx
        #if not is_none[1]:
        #	if L is None:
        #		L = ufl.Dx(u, i_)[j_] * B_ten[i_, j_, k_] * ufl.conj(v)[k_] * ufl.dx
        #	else:
    #			L += ufl.Dx(u, i_)[j_] * B_ten[i_, j_, k_] * ufl.conj(v)[k_] * ufl.dx
    #	if not is_none[2]:
    #		L = ufl.inner(u, C_ten * v) * ufl.dx
    #	else:
    #		L += ufl.inner(u, C_ten * v) * ufl.dx
    
        self._bilinear_form = dolfinx.fem.form(L)
    """
    
    #@auto_update
    def infinite_boundary_vec(self):
        """Constructs a diagonal array which takes an `infinite` (i.e. very large ) value on the points that are on the boundary of the domain
        """
        boundary_value= 1234e11
            #boundary_value = 0
        self.mesh().topology.create_connectivity(self.mesh().topology.dim - 1, self.mesh().topology.dim)
        boundary_facets = dolfinx.mesh.exterior_facet_indices(self.mesh().topology)
        boundary_dofs = dolfinx.fem.locate_dofs_topological(self.function_space(), self.mesh().topology.dim - 1, boundary_facets)
        boundary_state = np.complex128(boundary_value) if self.independent_vars['band_model'].tensor_shape == (1,1) else np.full(self.independent_vars['band_model'].tensor_shape[::2],boundary_value, dtype='complex')
        infinity_boundary = dolfinx.fem.dirichletbc(boundary_state, boundary_dofs, self.function_space())
        u_boundary = dolfinx.fem.Function(self.function_space())
        dolfinx.fem.petsc.set_bc(u_boundary.vector, [infinity_boundary])
        u_boundary.x.scatter_forward()
        return u_boundary.vector # retun the petsc vector
        from scipy.sparse import diags
        return diags(u_boundary.vector.getArray(),offsets=0,format='csr')
    
    #@auto_update
    def __make_unassembled_array__(self):
        LOGGER.debug('creating PETSc Matrix')
        try:
            old_a = self._saved_assemble_array.value #type:ignore
            old_a.destroy() # delete old object if it is out of date
        except Exception:
            pass    

        A = dolfinx.fem.petsc.create_matrix(self.bilinear_form())#diagonal=1234e10)
        return A
    
    def assemble_array(self,petsc_array=None):
        """ Assembles the array. This always returns a new instance because petsc arrays are altred"""

        
        petsc_A = self.__assemble_petsc_array__()
        if petsc_array is None:
            from scipy.sparse import csr_matrix
            return csr_matrix(petsc_A.getValuesCSR()[::-1])
        
        elif isinstance(petsc_array,PETSc.Mat):
            return petsc_A.copy(petsc_array) # copy the array into the passed matrix 
        elif isinstance(petsc_array,bool) and petsc_array:
            return petsc_A.copy() # create a new copy
        

    #@auto_update
    def __assemble_petsc_array__(self, sparse=True): # only use old solution of user knows what he is doing!
        """ This metho actull cnstructs the assembled array, but since petsc matrices can be altered, we have to protect te output of this in order t guarantee tha it can be reused without having been altered"""
        
        
        A = self.__make_unassembled_array__() #this reuses existing elements
        #A.zeroEntries() # clear the entries of any previous run
        constants = self.__gather_constants__() # value is not used, but this function also updates the DolfinxConstants 
        
        LOGGER.debug(f'assembling array') #with parameters: {self.independent_vars["band_model"].independent_vars["parameter_dict"]}')
        A = dolfinx.fem.petsc.assemble_matrix(A,self.bilinear_form(),bcs=self.dolfin_bc())#diagonal=1234e10)
        A.assemble()

        A.setDiagonal(self.infinite_boundary_vec().copy(),PETSc.InsertMode.ADD_VALUES)
        A.assemble()
        return A
        from scipy.sparse import  csr_matrix
        array = csr_matrix(A.getValuesCSR()[::-1]) # return as scipy sparse matrix
        # set boundary diagonals to zero
            
        array = array + self.infinite_boundary_vec()
        
        if sparse:
            return array
        else:
            return array.todense()
    
    def positional_rep(self, vector, x_vals=None):
        if x_vals is not None:
            raise SyntaxError(f'passing x_vals to FEniCs model not possible')

        x_vals = self.mesh().geometry.x*self.length_scale()
        if vector.shape != self.band_model.tensor_shape[::2]+(x_vals.shape[0],):
            print(f'vector had shape {vector.shape} but expected {self.band_model.tensor_shape[::2] + (x_vals.shape[0],)}')
            try:
                vector = self.eigensolutions_to_eigentensors(vector)
            except ValueError as err:
                raise ValueError(f'eigensolution passed to could not be broadcast to a correct shape:',err)

        return vector,x_vals

    def eigensolutions_to_eigentensors(self, eigensolutions):
        if len(eigensolutions.shape) == 1:
            drop_first_index = True
            eigensolutions = eigensolutions[:,np.newaxis] # ad axis if we only have one eigenvector
        else:
            drop_first_index = False
        eigensolutions = eigensolutions.T # transpose to have rows indexing the vector
        # infer psotional shape from the tensor shape.
        from functools import reduce
        pos_shape = int(eigensolutions.shape[1] / reduce(lambda x,y: x*y, self.band_model.tensor_shape[::2]))
        eigentensors = eigensolutions.reshape((eigensolutions.shape[0],pos_shape)+self.band_model.tensor_shape[::2]) # first index indexes which eigenstate we are looking at. second index indexes the positional coordiante and the remaining are the tensor components
        # transpose to get positional index to be the last one, by cyclically permuting all indeices except the first
        transposition = (0,)+tuple(np.roll(np.arange(1,len(eigentensors.shape)),-1))
        return_tensors = eigentensors.transpose(transposition)
        if drop_first_index:
            return_tensors = return_tensors[0,...]
        return return_tensors

    def flatten_eigentensors(self,eigentensors):
        if eigentensors.shape != self.band_model.tensor_shape[::2]+ (self.mesh().geometry.x.shape[0],):
            # left index is assigned to be indexing the tensors
            start = 1
        else:
            start = 0
        
        transpose_shape= [0]*start + list(range(1+start,len(eigentensors.shape))) + [start]
        transposition = eigentensors.transpose(transpose_shape)
        if start:
            return transposition.reshape(eigentensors.shape[0],-1)
        else:
            return transposition.flatten()
        

    @staticmethod
    def __make_S_array__(old_S_array:ComputedAttribute,function_space,band_model,dolfin_bc):
        u = ufl.TrialFunction(function_space.value)
        v = ufl.TestFunction(function_space.value)

        if band_model.value.tensor_shape == (1,1): #scalar case
            L = u*ufl.conj(v)*ufl.dx

        else:
            vector_rank = int(len(band_model.value.tensor_shape)/2)
            I= ufl.indices(vector_rank)
            L = u[I]*ufl.conj(v[I])*ufl.dx

        form = dolfinx.fem.form(L)
        A = dolfinx.fem.petsc.assemble_matrix(form,dolfin_bc.value)
        A.assemble()

        return ComputedAttribute(A,time.time(),old_S_array.constructor,(function_space,band_model,dolfin_bc),old_S_array.attr_name)
    
    
    
    def make_S_array(self,petsc_array=None):
        S = self.__make_S_petsc_mat__()
        S.assemble()
        
        if petsc_array is None:
            from scipy.sparse import  csr_matrix
            return csr_matrix(S.getValuesCSR()[::-1]) # return as scipy sparse matrix
        
        elif isinstance(petsc_array,PETSc.Mat):
            return S.copy(petsc_array) # copy the array into the passed matrix 
        elif isinstance(petsc_array,bool) and petsc_array:
            return S.copy() # create a new copy
        
    
    @auto_update
    def __make_S_petsc_mat__(self):
        u = ufl.TrialFunction(self.function_space())
        v = ufl.TestFunction(self.function_space())

        if self.independent_vars['band_model'].tensor_shape == (1,1): #scalar case
            L = u*ufl.conj(v)*ufl.dx

        else:
            vector_rank = int(len(self.independent_vars['band_model'].tensor_shape)/2)
            I= ufl.indices(vector_rank)
            L = u[I]*ufl.conj(v[I])*ufl.dx

        form = dolfinx.fem.form(L)
        A = dolfinx.fem.petsc.assemble_matrix(form,self.dolfin_bc())
        return A




    def project_operator(self,operator):
        # NB! the array produced here is wrt a basis that is NOT orthonormal! proceed with caution.
        symbols = (symb.X,symb.Y,symb.Z,symb.Kx,symb.Ky,symb.Kz)
        
        
        if not all(s in symbols for s in operator.free_symbols):
            raise ValueError(f'only alloed symbols are {symbols} but operator had symbols: {operator.free_symbols}')
        
        # cast as FEM form (in this form we directly know that the Ks only act on the trial and test function)
        aranged_O = symb.arange_ks_array(operator,self.k_signature)
        
        tensor_repr = bm.BandModel.__make_tensor_dict__(aranged_O,self.band_model.spatial_dim)
        
        # bring the tensors to the correct shape.
        
        ufl_tensors = {}
        for order,arr in tensor_repr.items():
            
            # replace the x,y,z with ufl.SpatialCoordinate. This is done by treating expressions in arr as polynomials of x,y,z and replacing it according to polynomial order
            arr = sympy.Array(arr)
            x_order_dict = symb.array_sort_out_polynomials(symb.enforce_commutativity(arr),sympy.symbols('x,y,z'))
            big_tensor = np.zeros(arr.shape) 
            for x_orders,coeff in x_order_dict.items():
                tensor = np.array(coeff).astype(np.complex128)
                X = ufl.SpatialCoordinate(self.mesh())
                for i,xo in enumerate(x_orders):
                #for each array_constant add the position expression according to the order
                    if xo:
                        tensor = tensor* X[i]**xo # allows for posibility of changing datatype
                big_tensor = big_tensor + tensor # allows for posibility of changing datatype
            
            if (big_tensor != 0).any():
                ufl_tensors[order] = ufl.as_tensor(big_tensor)
            
        
        form = self.__ufl_tensor_dict_to_bilinear_form__(ufl_tensors)
        assembled_array = dolfinx.fem.form(form)
        
        A = dolfinx.fem.petsc.assemble_matrix(assembled_array,self.dolfin_bc())#diagonal=1234e10)
        A.assemble()

        from scipy.sparse import  csr_matrix
        array = csr_matrix(A.getValuesCSR()[::-1]) # return as scipy sparse matrix
        # set boundary diagonals to zero
        
        #evaluate the form
        
        return array   
                
    
    def construct_observable(self,operator):
        if isinstance(operator,np.ndarray):
            operator = sympy.Array(operator)
        return FEniCsObservable(operator,self)

    
    def solution_shape(self):
        band_model_shape = self.band_model.tensor_shape
        x_shape = (self.mesh().geometry.x.shape[0],)
        return band_model_shape[::2] + x_shape
        
    
def make_function_space(domain,spatial_dim,function_class=('CG',1),shape=(1,),scale_factor=1):
    """
    Makes a function space over a domain.
    :param int spatial_dim: The spatial dimension of the domain.
    :param Domain domain: The domain of definition
    :param tuple[str,int] function_class: The function class to build the function space over
    :param tuple[int] shape: the shape of the functions. Defaults to scalar
    :return:
    """

    mesh = FEniCsModel.__make_mesh__(domain,spatial_dim,scale_factor)

    return FEniCsModel.__make_function_space__(mesh,function_class,shape)


# Observable
from .observables import AbstractObservable
class FEniCsObservable(AbstractObservable):

    def __init__(self, operator:sympy.Array|np.ndarray, envelope_model:envelope_function.EnvelopeFunctionModel):
        super().__init__(operator, envelope_model)

        # we can already assemble ufl form and just substitute the functions when mel or apply are called
        self.left_v = dolfinx.fem.Function(self.envelope_model.function_space(),name='vl')
        self.right_v =dolfinx.fem.Function(self.envelope_model.function_space(),name='vr')
        from . import symbolic as symb
        symbols = (symb.X,symb.Y,symb.Z,symb.Kx,symb.Ky,symb.Kz)
        
        if not all(s in symbols for s in operator.free_symbols):
            raise ValueError(f'only alloed symbols are {symbols} but operator had symbols: {operator.free_symbols}')
        
        # cast as FEM form (in this form we directly know that the Ks only act on the trial and test function)
        aranged_O = symb.arange_ks_array(operator,self.envelope_model.k_signature)
        
        tensor_repr = bm.BandModel.__make_tensor_dict__(aranged_O,self.envelope_model.band_model.spatial_dim)
        
        # bring the tensors to the correct shape.
        self.ufl_tensors = {}
        for order,arr in tensor_repr.items():
            
            # replace the x,y,z with ufl.SpatialCoordinate. This is done by treating expressions in arr as polynomials of x,y,z and replacing it according to polynomial order
            arr = sympy.Array(arr)
            x_order_dict = symb.array_sort_out_polynomials(symb.enforce_commutativity(arr),sympy.symbols('x,y,z'))
            big_tensor = np.zeros(arr.shape,dtype='O') 
            for x_orders,coeff in x_order_dict.items():
                # replace complex numbers by dolfinx.fem.Constant!
                tensor = np.array([dolfinx.fem.Constant(self.envelope_model.mesh(),np.complex128(c)) for c in np.array(coeff).ravel()]).reshape(coeff.shape)

                X = ufl.SpatialCoordinate(self.envelope_model.mesh())
                for i,xo in enumerate(x_orders):
                #for each array_constant add the position expression according to the order
                    if xo:
                        tensor = tensor* X[i]**xo*self.envelope_model.length_scale()**xo # allows for posibility of changing datatype
                big_tensor +=tensor # allows for posibility of changing datatype
            
            if (big_tensor != 0).any():
                # extend te shape of tensor.
                required_shape = self.envelope_model.band_model.tensor_shape
                if big_tensor.shape != required_shape:
                    save_shape = big_tensor.shape
                    missing =required_shape[len(big_tensor.shape)::2]# we can only suffix
                    for m in missing:
                        big_tensor = np.tensordot(big_tensor,np.eye(m),axes=0)
                
                if big_tensor.shape != required_shape:
                    raise ValueError(f'unable to extend tensor of shape {save_shape} to the required shape: {required_shape}')
                
                
                self.ufl_tensors[order] = ufl.as_tensor(big_tensor/(self.envelope_model.length_scale()**order))
            
        
        self.ufl_bilinear_form = self.envelope_model.__ufl_tensor_dict_to_bilinear_form__(self.ufl_tensors,u=self.right_v,v=self.left_v)
        self.bilinear_form = dolfinx.fem.form(self.ufl_bilinear_form)
        #THISDOES NOT WORK
        self.ufl_linear_form = self.envelope_model.__ufl_tensor_dict_to_linear_form__(self.ufl_tensors,self.right_v)
        #self.linear_form = dolfinx.fem.Expression(self.ufl_linear_form,self.envelope_model.function_space().element.interpolation_points())
    
    def mel(self,vector,other_vector =None):

        if all(ss == 0 for ss in np.array(self.abstract_operator).ravel()):
            return 0 # trivial case, which breaks FEniCs
        
        
        
        #handle cases where multiple vectors are passed 
        if  vector.shape != self.envelope_model.solution_shape():
            return np.array([self.mel(v,other_vector) for v in vector]) # loop over the passed vectors. This automatically also catches case where vec and other vec are both lists of vectors and returns an array of numbers
        if other_vector is not None and other_vector.shape != self.envelope_model.solution_shape():
            return np.array([self.mel(vector,vr) for vr in other_vector])
        
        vector = self.envelope_model.flatten_eigentensors(vector)
        if other_vector is None:
            other_vector = vector
        else:
            other_vector =self.envelope_model.flatten_eigentensors(other_vector) 
        
        
        # sign the numerical value of the functions in the array    
        self.left_v.x.array[:] = vector.flatten()
        self.right_v.x.array[:] = other_vector.flatten()

        # assemble the array and build 
        res = dolfinx.fem.assemble_scalar(self.bilinear_form)
        
        if other_vector is vector:
            return np.real(res)
        else:
            return res
                
        
    def apply(self,vector):
        
        self.right_v.x.array[:] =  self.envelope_model.flatten_eigentensors(vector)
        
        # assemble expression and interpolate outcome
        result = dolfinx.fem.Function(self.envelope_model.function_space())
        result.interpolate(lambda x: self.linear_form(x*self.envelope_model.length_scale()))
        
        res_array = np.array(result.x.array)
        
        res = self.envelope_model.eigensolutions_to_eigentensors(res_array)
        
        return res



# region ufl relation and piecewise functions and Constnat subclassing

def fenics_conjugate(expr):
    return ufl.conj(expr)


__ufl_relations__ = {'==':ufl.eq, '>':ufl.gt, '>=':ufl.ge,
                    '<': ufl.lt, '<=':ufl.le, '!=':ufl.ne,'conditional':ufl.conditional,'real':ufl.real,'and':ufl.And,'or':ufl.Or,'not':ufl.Not}


def is_piecewise(expression):
    if isinstance(expression,sympy.MutableDenseNDimArray):
        expression = sympy.Array(expression)
    if isinstance(expression,sympy.Piecewise):
        return True
    elif not len(expression.args):
        return True
    else:
        return any(is_piecewise(arg) for arg in expression.args)





def convert_piecewise(expression,relation_operators,variables):
    """Converts a sympy piecewise expression to the specified conditional. Usefull when converting to UFL for using with fenicsx

    :param expression: THe expression to convert
    :type expression: sympy.Expr
    :param relation_operators: Dictionary containing the relation operators to use, as well as the conditional function and functions casting coplex numbers ot real.
    :type relation_operators: dict[str,typing.Any]
    :param variables: dict containing sympy.Symbols as keys and their corresponding substituted values as values.
    :type variables: dict[sympy.Symbol,typing.Any]
    :raises NotImplementedError: _description_
    :return: _description_
    :rtype: _type_
    """
    
    
    condition_func = relation_operators['conditional']
    try:
        real_part = relation_operators['real']
    except KeyError as err:
        real_part = lambda x: x
    # dictionary to convert negations
    negations = {'==':'!=','!=':'==',
             '>':'<=','<=': '>',
             '<': '>=','>=': '<',
             'or': 'and', 'and': 'or'}

    def recursive_conditionals(conditions,final_value=0.0):
        if len(conditions) == 1:
            next_iter = final_value
        else:
            next_iter = recursive_conditionals(conditions[1:],final_value=final_value)

        this_condition = conditions[0]
        condition_type = determine_relation(this_condition[1])
        if isinstance(condition_type,bool):
            if condition_type:
                # return the function because this is always true
                return sympy.lambdify(variables.keys(),this_condition[0],"numpy",dummify=True)(*variables.values())
            else:
                # just go down the chain because this condition will never be true
                return next_iter


        # convert the condition 
        converted_cond = convert_condition(this_condition[1],relation_operators)

        # evaluate function and conditionals
        function = expression_traversal_lambdify(this_condition[0])
        return condition_func(converted_cond,function,next_iter)


    def convert_condition(cond,conversion_dict):
        condition_type = determine_relation(cond)
        
        if condition_type in ('and','or','not'):
            # we also convert all the parts of ands and ors 
            converted_parts = [convert_condition(c,conversion_dict) for c in cond.args]
            
            # combine them using the condition:
            if condition_type == 'not':
                assert len(converted_parts) == 1
                return conversion_dict[condition_type](converted_parts[0])
            
            converted = conversion_dict[condition_type](converted_parts[0],converted_parts[1])
            for c in converted_parts[2:]:
                converted = conversion_dict[condition_type](converted,c)
            return converted
        # lambdify left and right conditions 
        left_conditional = expression_traversal_lambdify(cond.args[0])
        right_conditional = expression_traversal_lambdify(cond.args[1])
        return conversion_dict[condition_type](real_part(left_conditional),real_part(right_conditional))
        
        
            
    def expression_traversal_lambdify(expr):
        dummy_expr = expr.func
        if dummy_expr == sympy.Piecewise:
            return recursive_conditionals(expr.args,final_value=0.0)
        dummy_args = sympy.symbols('a0:%d'%len(expr.args))

        if isinstance(expr,sympy.core.relational.Relational):
            return convert_condition(expr,conversion_dict=relation_operators)
        
        converted_args = []
        for arg in expr.args:
            if arg in variables.keys():
                converted_args.append(variables[arg])
            else:
                converted_args.append(expression_traversal_lambdify(arg))

        if len(converted_args) == 0:
            converted_args=tuple(range(len(dummy_args)))
        if not len(expr.args):

            lambdified = sympy.lambdify(variables.keys(),expr,"numpy")(*variables.values())
        else:
            lambdified = sympy.lambdify(dummy_args,dummy_expr(*dummy_args),"numpy")(*converted_args)
        return lambdified

    result = expression_traversal_lambdify(expression)
    return result

def determine_relation(condition):

    if isinstance(condition, sympy.core.relational.Equality):
        return '=='
    elif isinstance(condition, sympy.core.relational.LessThan):
        return '<='
    elif isinstance(condition, sympy.core.relational.StrictLessThan):
        return '<'
    elif isinstance(condition, sympy.core.relational.GreaterThan):
        return '>='
    elif isinstance(condition, sympy.core.relational.StrictGreaterThan):
        return '>'
    elif isinstance(condition, sympy.core.relational.Unequality):
        return '!='
    elif isinstance(condition, sympy.Or):
        return 'or'
    elif isinstance(condition, sympy.And):
        return 'and'
    elif isinstance(condition,sympy.Not):
        return 'not'
    elif isinstance(condition,sympy.logic.boolalg.BooleanFalse):
        return False
    elif isinstance(condition,sympy.logic.boolalg.BooleanTrue):
        return True
    else:
        raise NotImplementedError(f'non recognized relation: {type(condition)}')








# endregion