"""
This module describes all things related to Hamiltonians, which isn't specific to any set of basis functions of the Hilbert space. The Hamiltonians are all expressed in position space,
where positionally dependent terms are diagonal (so they can be expressed as scalars). The terms in the Hamiltonian are expressed as tensors:
h_{i_1,...,i_n, \\kappa_1,\\kappa_2,...,\\kappa_m} where the indices i_j label spinor components or similar degrees of freedom, while the
\\kappa indices (\\kappa = 0,1,2) describe which k operators h is related to:
h_{i_1,...,i_n,\\kappa_1,\\kappa_2,...,\\kappa_m}
is contracted with the operator k_{\\kappa_1}k_{\\kappa_2}...k_{\\kappa_m}

CONVENTIONS:
## Magnetic field terms:
In order for magnetic field terms to work with the G-matrix formalism as well as time-reversal, the magnetic field terms MUST
be added as constructed tensors, with the functions mapping the magnetic field (along with model specific parameters
such as g-factor) to the Hamiltonian term expresses as STATIC methods and with the argument representing the magnetic
field named `Bvec`.
"""

from sre_parse import SubPattern
import numpy as np
import sympy
import time
import logging
from . import PETSc
from typing import Any, Iterable,Union,Sequence,Annotated
import typing
LOGGER = logging.getLogger(__name__)
LOGGER.addHandler(logging.NullHandler())

# specify global names for important variables 
__MOMENTUM_NAMES__ = (r'k_{x}',r'k_{y}',r'k_{z}')
__POSITION_NAMES__ = (r'x',r'y',r'z')
__MAGNETIC_FIELD_NAMES__ = (r'B_{x}(x)',r'B_{y}(x)',r'B_{z}(x)') # These are always treated as functions!
__VECTOR_FIELD_NAMES__ =  (r'A_{x}(x)',r'A_{y}(x)',r'A_{z}(x)') # these are always treated as functions!

from .updatable_object import UpdatableObject, auto_update,ComputedAttribute
    

class BandModel(UpdatableObject):
    __post_processor_ordering__ = ("OTHER","A-field","z-confinement","BdG")
    __time_reversal_change_of_basis__ = None # this attr stores the change of basis that is involved in time-reversal of the model.
    def __init__(self,array:sympy.ImmutableDenseNDimArray,spatial_dim:int=3,unit_convention='SI') -> None:
        """This class describes the Hamiltonian of the system. In the array, the position variables must be symbols which are named ´x,y,z´ and the momentum operators must be symbols named ´k_x,k_y,k_z´.
        Args:
            expression (sympy.ImmutableDenseNDimArray): The array which represents the Hamiltonian 
            spatial_dim (int, optional): Spatial dimension of the system. Defaults to 3.
        
        """
        self.spatial_dim = spatial_dim

        # Dict containing all the attributes that could be changed from outside the instance.
        independent_vars = {} 
        
        """ Hierarchy of expressions: 
        Computing the Final shape of the Hamiltonian is done in sequential steps and after each step the state is saved to avoid recomputing it. 
        Changing parameters of the model will affect the computation at different steps and therefore changes some (but not necessarily all) saved computations.
        the steps are as follows:
            - preprocessed array: This is the underlying array describing a system in the specified dimension (without superconductivity or confinement or A-field)
                - CHANGES:
                    * This array is only changed if the `.preprocessed_array` property is set, which occurs whenever a new term is added to the model.
            
            - post-processed Array: A-fields, z-confinement and BdG along with other post-processing steps are performed on the preprocessed array to get this.
                - CHANGES:
                    * post-processing function: The function that takes the preprocessed array and returns the post-processed array. Changing the function changes the postprocessing array.
                    
            - Numerical Array: The post-processed array is turned into an array where all constants (\\hbar _mu_B etc.) and parameters (m,g_factor etc.) are specified, but the spatial (x,y,z) coordinates and momentum values (kx,ky,kz) are still there).
                - CHANGES:
                    * parameter_dict: If the values of the parameter dict changes so does this
                    * unit_convention: If the unit convention is changed this affects the numerical values in this array
                    * constants: changing the constants also changes the this array.
                    
            - Tensor_representation: The Numerical array is cast as a dict of higher dimensional tensors according to the terms of the numerical array and its order in the momentum parameters.
                - CHANGES:
                    * 
        
        In order to keep track of this, the versions are saved along with the time at which they were modified so that one can check whether things are all up to date.
        """
                
        # globally defined position and momentum symbols, since these are important, are often referenced in the methods, and the momentum symbols should not be commutative (as we may or may not add an A-field later)
        position_symbols = {}
        momentum_symbols = {} 
        for symbol in array.free_symbols:
            if not isinstance(symbol,sympy.Symbol):
                raise TypeError(f'symbol was not of type sympy.Symbol. Got: {symbol} ({type(symbol)})')
            
            if symbol.name in __MOMENTUM_NAMES__[:]:
                momentum_symbols[symbol.name] = symbol
            if symbol.name in __POSITION_NAMES__[:]:
                position_symbols[symbol.name] = symbol
        
        self.position_symbols = tuple(position_symbols.get(n,sympy.symbols(n)) for n in __POSITION_NAMES__) 
        self.momentum_symbols = tuple(momentum_symbols.get(n,sympy.symbols(n)) for n in __MOMENTUM_NAMES__)
        non_commuting_ks = sympy.symbols(__MOMENTUM_NAMES__,commutative=False)
        # make sure that all k symbols are mate non-commuting:
        # this flags allow invalidation of saves down the line of computing if it turns out that an earlier computation has been changed.
        #        self.__sync_status__ = {'post_processed_array':False,'numerical_array':False,'tensor_repr':False}
        
        independent_vars['preprocessed_array'] = array.subs(zip(self.momentum_symbols,non_commuting_ks))
        self.momentum_symbols = tuple(non_commuting_ks)
        
        
        # post processor constructor is constructed from a dict and alters the array
        
        independent_vars['postprocessing_function_specification'] = {}
        
        # dictionary of all parameters defining the model. 
        # The .constants dict is meant for physical constants which should only change if one changes the unit system that this model is represented in
        # the .parameter_dict is meant for all other parameters. The keys are the symbols and the values must be floats (or float like) 
        # NB If the symbols name ends with "(x)", e.g. V(x) or f(x), then the corresponding value must be a function which as many arguments as spatial_dim (so that it can be evaluated at a given position). or a Sympy expression involving x,y,z and NOT kx,ky,kz. these functions will be stored in the functions dict
        from . import constants,values
        constant_names = tuple(c.name for c in constants.values())

        independent_vars['parameter_dict'] =   {s:None for s in independent_vars['preprocessed_array'].free_symbols if (s.name not in __MOMENTUM_NAMES__+__POSITION_NAMES__+constant_names) and (s.name[-3:] != '(x)')} #type: ignore
        
        independent_vars['function_dict']  = {s:None for s in independent_vars['preprocessed_array'].free_symbols if  s.name[-3:] == '(x)'} # type: ignore
        # Superconducting order parameter
        independent_vars['SC_Delta'] = None
        
        # unit_convention_stuff
        if unit_convention == 'SI':
            self.unit_convention = {} # todo: Allows for changing the unit convention to something different from SI base units
            
            independent_vars['unit_convention'] = {}
            
            independent_vars['constants']= {k:v for k,v in values.items() if isinstance(k,sympy.Symbol)} # import SI values from init.py

            #self.constants = SyncDict({})  # dict mapping symbols representing constants such as hbar, pi etc to their numerical values (expressed in SI units). Transformation to different unit systems is done dynamically.

        else:
            raise NotImplementedError('yet to implement saving numerical values in different unit scales')

        # init the parent to assure that computable items can be computed and saved.
        super().__init__(**independent_vars)
        
        return      

    # region postprocessing and building  / casting    
    
    #region array_shape properties
    @property
    def tensor_shape(self):
        
        return self.post_processed_array().shape
    
    @property
    def n_bands(self):
        return np.prod(self.tensor_shape[::2])
    #endregion
    def __fix_parameters_of_function__(self,func,parameter_dict:None|dict[sympy.Symbol,Any] = None):
        """
        The function __fix_parameters_of_function__ takes a function and a dictionary of parameter
        values, and returns a new function with the parameter values substituted.
        
        :param func: The `func` parameter is a symbolic expression or function that you want to modify
        or evaluate. It can be an expression involving one or more symbolic variables
        :param parameter_dict: The `parameter_dict` parameter is a dictionary that maps `sympy.Symbol`
        objects to any value. It is used to substitute the symbols in the `func` expression with the
        corresponding values before returning the new function. If `parameter_dict` is not provided, the
        function will use the parameter_dict of the model.
        :type parameter_dict: None|dict[sympy.Symbol,Any]
        :return: The function `__fix_parameters_of_function__` returns a new function `new_func` after
        substituting the values of the parameters specified in the `parameter_dict` into the original
        function `func`.
        """
        new_func = func.subs(self.independent_vars['parameter_dict'] if parameter_dict is None else parameter_dict)
        return new_func 
    
    def __check_functions__(self,raise_error=True):
        """Verifies that the functions are correct
        """
        import inspect
        from typing import Callable
        avoid = self.position_symbols[3-self.spatial_dim:]
        for sym,func in self.independent_vars['function_dict'].items():
            if isinstance(func,Callable) and len(inspect.signature(func).parameters) != self.spatial_dim:
                if raise_error:
                    raise Exception(f'function {sym} did not have the correct number of ars insignature. expected {self.spatial_dim} but got {inspect.signature(func)}')
                else:
                    return False
            elif hasattr(func,'free_symbols') and any(s in func.free_symbols for s in avoid):
                if raise_error: 
                    raise Exception('function {sym} had wrong spatil coordinates. Allowed coordinates: {self.position_symbols[:self.spatial_dim]}.\n function: {func}')
                else:
                    return False
        
        return True
        
    
    def __postprocessing_constructor__(self,post_processing_spec,skip_bdg=False):
        """Function for building the post-processing function, either with or without a bdg step. This allows us to also use it  for post-processing the SC order parameter.

        :param skip_bdg: Whether to not do the BdG extension part if it is supplied, defaults to False
        :type skip_bdg: bool, optional
        :return: Function which takes an array and a BandModel and returns a post-processed array
        :rtype: Sympy.Array
        """
        
        _ = post_processing_spec    # this just calls the dependency so that dependencies work even if the pp_func_spec is an empty dict 
        # create list of functions in the correct ordering
        post_processing_func_list = []
        for func_name in self.__class__.__post_processor_ordering__:
            if func_name == 'OTHER':
                # run all keys that are not in ordering
                post_processing_func_list.extend(f for name,f in post_processing_spec.items() if name not in self.__class__.__post_processor_ordering__)
            elif skip_bdg and func_name == "BdG":
                continue # do not add bdg extension function. 
            elif func_name in post_processing_spec.keys():
                
                post_processing_func_list.append(post_processing_spec[func_name])
        if not len(post_processing_func_list):
            post_processing_func = lambda x,model:x
        else:
            from functools import reduce
            def chain(funcs):
                """
                The function "chain" takes in multiple functions as arguments and returns a new function
                that applies each of the input functions in sequence to its input.
                """
                def chained_call(arg,model):
                    result = arg
                    for func in funcs:
                        result=func(result,model=model)
                    return result
                return chained_call
            post_processing_func = chain(post_processing_func_list)
        
        return post_processing_func
    
    @auto_update
    def array_post_processor(self):
        """Returns a function F(array,model) which converts the preprocessed array to the post-processed one based on parameters of the passed model 

        :return: _description_
        :rtype: _type_
        """
        return self.__postprocessing_constructor__(self.independent_vars['postprocessing_function_specification'],skip_bdg=False)

    
    @auto_update
    def post_processed_array(self):
        """
        Makes the post-processed array (array representing the Final Hamiltonian without any bound values). Returns the
        precomputed version of the array if available, otherwise it computes it. 

        :param force_recompute: force_recompute is a boolean parameter that determines whether the
        function should recompute the post-processed expression or return the precomputed version. If
        force_recompute is True, the function will recompute the post-processed expression even if it
        has already been computed before. If force_recompute is False (, defaults to False
        :type force_recompute: bool (optional)
        :return: the post-processed expression array. If the post-processed expression has already been
        computed and the force_recompute flag is False, then it returns the precomputed version.
        """
        return self.array_post_processor()(self.independent_vars['preprocessed_array'],self)
        
    
    @auto_update
    def numerical_array(self):
        """

        creates a numpy array where the only free symbols left are x,y,z and k_x,k_y,k_z. THe ks are replaced with new symbols with the same name but commutative.

        Args:
            float_format (_type_, optional): The format for the floats *that are not in the parameter_dict*. Defaults to PETSc.ScalarType.
        """
        
        
        var_dict = self.independent_vars['parameter_dict'].copy() # make copy for adding constants and x,y,z and k_x,k_y,k_z. These will just be evaluated as their symbols
        # specify B field from A field if it is present:
        
        unit_transformed_constants = self.independent_vars['constants'].copy() # todo  make transformation of unit scales 
        
        var_dict.update(unit_transformed_constants)
        var_dict.update({k:sympy.symbols(k.name) for k in self.momentum_symbols})  #casts momentum symbols as commutative from now on!

        unspecified = [k for k,v in var_dict.items() if v is None]
        if unspecified:
            raise ValueError(f'the following values have not been specified so the model cannot be cast as a numerical array:\n {unspecified}')
        # evaluate all functions:
        from typing import Callable
        position_set = set(self.position_symbols)
        for sym,func in self.independent_vars['function_dict'].items():
            if func is None:
                if sym in sympy.symbols(__MAGNETIC_FIELD_NAMES__):
                    var_dict[sym] = None # this may be overwritten later if A fields are specified. Otherwise and exception will be raised later
                    continue
                raise ValueError(f'function with symbol {sym} was not assigned a value')
            if isinstance(func,Callable):
                var_dict[sym] = func(*self.position_symbols[:self.spatial_dim])
            else:
                # substitue all parameters
                if any((s,s) in var_dict.items() for s in func.free_symbols):
                    raise ValueError(f'Infinite recursion due to substituion of symbol with itself: {[s for s in func.free_syms if (s,s) in var_dict.items()]}')
                func = func.subs(var_dict)
                if position_set.union(func.free_symbols) != position_set:
                    raise ValueError(f'Passed function had symbols not contained in the parameter-dict. Only x,y,z should be left but we have: {func.free_symbols}')
                var_dict[sym] = func
    
                
        
        A=sympy.symbols(__VECTOR_FIELD_NAMES__) 
        B=sympy.symbols(__MAGNETIC_FIELD_NAMES__)
        
        # Check if A and B are in the model, and if so that A is specified but B is not. In that case write B as the curl of A 
        if (A[0] in var_dict.keys()) and (B[0] in var_dict.keys()):
            if all(var_dict.get(a,None) is not None for a in A) and all(var_dict[b] is None for b in B):
                for i in range(3):
                    j = (i+1)%3
                    k = (i+2)%3
                    try:
                        var_dict[B[i]]=sympy.diff(var_dict[A[k]],self.position_symbols[j])-sympy.diff(var_dict[A[j]],self.position_symbols[k])
                    except Exception as err:
                        raise Exception(f'unable to differentiate function {var_dict[A[k]]} wrt {self.position_symbols[j]} ') from err
            elif any(var_dict.get(a,None) is not None for a in A):
                raise TypeError('A fiel and/or B-field not correctly specified. Exactly one of them have to be completely specified')
            else:
                raise TypeError()
                LOGGER.debug('weird A-B case happened')
        
        
        var_dict.update({k:sympy.symbols(k.name) for k in self.position_symbols})  #add position symbols as themselves (do it after function spec to avoid infinite recursion)
        
        unspecified = [k for k,v in var_dict.items() if v is None]
        if unspecified:
            raise ValueError(f'the following values have not been specified so the model cannot be cast as a numerical array:\n {unspecified}')
        
        LOGGER.debug(f'building numerical array with var_dict: \n{var_dict}')
        numpy_version = np.array(sympy.lambdify(var_dict.keys(),self.post_processed_array(),dummify=True)(*var_dict.values())) # set Dummify to True to make sure that symbol names do not break the code
        
        
        return numpy_version

    @auto_update
    def symbolic_tensor_repr(self):
        """
        COnstructs a tensor representation of the model, where the Ks are replaced with indices in a tensor. The symbols are kept.
        """
        
        commuting_momentum_symbols = sympy.symbols(__MOMENTUM_NAMES__) # these are the commuting momentum symbols which are in the numerical version of the Hamiltonian
        array =   self.post_processed_array().subs({k:ck for k,ck in zip(self.momentum_symbols,commuting_momentum_symbols)}).subs({sympy.symbols(r'\hbar'):sympy.symbols('hbar')}) # \hbar breaks lambdify so we replace with a working name :( 
        
        disassemble_dict = self.__make_tensor_dict__(array)
        

        return disassemble_dict
    
    @auto_update
    def numerical_tensor_repr(self):
        """
        Constructs a tensor representation of the model where the Ks are replaced with indices in a tensor. The resulting dict contains numpy arrays of complex numbers.
        """
        disassemble_dict = self.__make_tensor_dict__(self.numerical_array())
        return disassemble_dict
    
    
    def __make_tensor_dict__(self,array,force_recompute:bool=False,):
        """Constructs a dictionary where the keys are are integers and elements are arrays of shape `self.tensor_shape`. The keys indicate how many factor of the momentum operators are in front of the specified tensor.

        :params force_recompute: Whether to force the recomputation of the array or not. Defaults to False.
        :params use_numerical: True by default. Uses the numerical array generated from make_numerical() rather than the array returned by make_array().

        Returns: dict[int,np.ndarray]: dictionary with the Hamiltonian split up into coefficients in front of k operators
        """
        
        
        
        
        # Fix this bit using symbolic, and make it so that the array is populated as if the k-s did not commute i.e. it matters whether to pick T[...,0,1]  or T[...,1,0] (kx*ky vs ky*kx). This is needed to assure that ordering is preserved for FEM
        from sympy.polys.polytools import degree_list
        disassemble_dict = {} # dict containing tensors corresponding to the respective order

        commuting_momentum_symbols = sympy.symbols(__MOMENTUM_NAMES__) # these are the commuting momentum symbols which are in the numerical version of the Hamiltonian
        
        
        for i,val in enumerate(np.array(array).ravel()):
            # split up value into terms 
            val = sympy.sympify(val).expand()
            terms = val.args if isinstance(val,sympy.core.add.Add) else (val,)
            for term in terms: 
                try:
                    orders = degree_list(term,commuting_momentum_symbols)
                except TypeError as e:
                    continue 
                arr = disassemble_dict.get(sum(orders),np.zeros(self.tensor_shape+(self.spatial_dim,)*(sum(orders)),'O'))
                # k-s commute, and by convention we will always order them as k_x k_x ... k_x k_y k_y ... k_y k_z k_z ... k_z
                arr_i = np.unravel_index(i,self.tensor_shape)+(0,)*orders[0]+(1,)*orders[1]+(2,)*orders[2]
                addition =   sympy.lambdify(commuting_momentum_symbols,term)(1,1,1) if any(orders) else term # this could give an error if ks and functions V(x) are mixed!
                arr[arr_i] += addition

                # add coefficient to array
                disassemble_dict[sum(orders)] = arr

            
        return disassemble_dict

    #endregion

    # region spectrum analysis
            
    def eig(self,kvec:typing.Sequence[float],position:Sequence=(0,0,0),drop_eigenvectors:bool=False,symbolic_computation=False):
        import numpy as np
        import numpy.typing as npt
        kv:npt.NDArray = np.array(kvec) if not isinstance(kvec,np.ndarray) else kvec
        
        if len(kv.shape) == 1:
            return_array = False
            kv = kv[np.newaxis,:] # type: ignore
        else:
            return_array = True 
            
        # make function which plugs k-vectors into the array
        fixed_array:sympy.tensor.ImmutableDenseNDimArray = sympy.Array(self.numerical_array()).subs({pos_sym:pos for pos_sym,pos in zip(self.position_symbols,position)}) # type: ignore
        commuting_momentum_symbols = sympy.symbols(__MOMENTUM_NAMES__)
        array_func = sympy.lambdify(commuting_momentum_symbols,fixed_array,dummify=True)
        
        # populate as an iterable which should be stacked rather than this mess of indexing!
        mat_shape = np.prod(fixed_array.shape[::2]) # shape of the matrix to diagonalize
        def array_generator(k_vecs):
            for k_vector in k_vecs:
                kvec_list = list(k_vector)
                while len(kvec_list)<3:
                    kvec_list.append(0) # all other k moments should be zero
                
                transposition = tuple(range(0,len(fixed_array.shape),2))+ tuple(range(1,len(fixed_array.shape),2)) # reordering og axes so that left axes are ket axes and right are bra axes
                bound_array = np.array(array_func(*kvec_list)).astype(np.complex128).transpose(transposition).reshape((mat_shape,mat_shape))
                yield bound_array
        
        
        
        
        array_stack = np.stack(tuple(array_generator(kv))) # type: ignore # shape must be `N_kvecs` x mat_shape x mat_shape
        
        eigvals,eigvecs = np.linalg.eigh(array_stack)
        if return_array:
            if drop_eigenvectors:
                return eigvals
            return eigvals,eigvecs
        else:
            if drop_eigenvectors:
                return eigvals[0,:]
            else:
                return eigvals[0,:],eigvecs[0,:]
            
        

    def spectrum(self,k_range:typing.Sequence[float],k_direction:typing.Sequence[float],n_points:int,band_sorting:bool=False,drop_eigenvectors:bool=True,position=(0,0,0)):
        
        # create array of k-values and assemble:
        k_unit_vector = np.array(k_direction)
        k_unit_vector = k_unit_vector / np.linalg.norm(k_unit_vector)
        k_vals = np.linspace(k_range[0],k_range[1],n_points)
        
        k_vectors = np.einsum('i,j->ij',k_vals,k_unit_vector)
        # reuse sorting of old code 
        LOGGER.debug(f'finding eigenvalues')
        if band_sorting:
            results = self.eig(np.stack(k_vectors,axis=0),position=position)
            # reshape the output from Eval,Evec to (Eval,Evec)
            res = [ (results[0][i],results[1][i]) for i in range(len(k_vectors))]
            results = res
            #results = [self.eig(k_vec, drop_eigenvectors=False,skip_build=True) for k_vec in k_vectors]
            sorting = np.arange(len(results[0][0]))  # sorting order is fixed by first res
            eigenvals = [results[0][0]]
            eigenvects = [results[0][1]]

            LOGGER.debug(f'sorting results')
            for i in range(1, len(results)):
                # compare eigenvectors of each and determine which eigenvalue belongs to which band
                overlaps = np.abs(np.einsum('ij,ik->jk', results[i][1].conj(), results[i - 1][1]))
                maximums = np.argmax(overlaps, axis=0)
                sorting = sorting[maximums]
                eigenvals.append(results[i][0][sorting])
                eigenvects.append(results[i][1][:, sorting])

            eigenvals = np.stack(eigenvals, axis=0)
            eigenvects = np.stack(eigenvects, axis=0).T


        else:
            eigenvals,eigenvects = self.eig(k_vectors,position=position)#np.stack([self.eig(k_vec)[0] for k_vec in k_vectors],
                                # axis=0).T  # columns indicate k_value, so each row is a plot
            
        if drop_eigenvectors:
            return k_vals, eigenvals
        else:
            return k_vals, eigenvals, eigenvects
        
        
    #endregion
    
    # region model construction
    def material_spec(self,material_name:str):
        """Specifies the parameters according to Winkler for the specified material

        Args:
            material_name (str): Name of the material.
        """
        from . import _m_e
        if material_name == 'Ge':
            parameter_dict = {r'\Delta_{0}': 0.296,
                            r'\gamma_{1}': 13.38,
                            r'\gamma_{2}': 4.24,
                            r'\gamma_{3}': 5.69,
                            r'\kappa': 3.41,
                            r'q': 0.06,
                            r'D_{u}': 3.3195,
                            r"D'_{u}": 5.7158,
                            r'c_{11}': 12.40,
                            r'c_{12}': 4.13,
                            r'c_{44}': 6.83,
                            r'a': 5.65791,
                            r'\epsilon': 16.5,
                            r'm': _m_e}
        else:
            raise ValueError(f'material name {material_name} not in database')
        
        
        # reuse existing symbols if they exist. Else, create new symbols 
        existing_names = {k.name:k for k in self.independent_vars['parameter_dict'].keys()}
        addendum_keys = (existing_names.get(name,sympy.symbols(name)) for name in parameter_dict.keys())
        self.independent_vars['parameter_dict'].update({k:parameter_dict[k.name] for k in addendum_keys}) 
        
        return self

    def add_diagonal_terms(self,diagonal,axis:int|None=None):
        pass
    def add_potential(self,potential,parameters=None,constants=None):
        """Adds a potential (e.g. electrostatic confinement) to the model. The potential is expressed as a sympy expression with the symbols ´x,y´ and ´z´ denoting the positional coordinates. Any other symbols are added to the parameter dict to be specified, unless they are keys in the constants dict passed., in which case they are treated as constants.

        :param potential: The (scalar) potential which affects all bands equally
        :type potential: sympy.Expression
        :param parameters: dict mapping symbol names (strings) to floats. Allows specifying parameters in the potential expression immediately.
        :type parameters: dict[str,float|None]|None
        :param constants: dict mapping symbols to floats specifying if there are any symbols that should be treated as constants.
        :type constants: dict[sympy.Symbols,float]|None
        """
        
        if (parameters is not None and constants is not None) and any(((k in parameters.keys()) for k in constants.keys())):
            k = next(k for k in constants.keys() if k in parameters.keys())
            raise ValueError(f'A symbol appeared in both the specified parameter dict and the constants dict:\n {k}')
    
        
        
        diagonal_array = potential* sympy.eye(self.independent_vars['preprocessed_array'].shape[0])
        
        self.independent_vars['preprocessed_array'] = self.independent_vars['preprocessed_array']+ sympy.Array(diagonal_array)
        
        constants = {} if constants is None else constants
        
        # make dict containing all parameters
        new_params = {s:None for s in potential.free_symbols if (s not in constants.keys() and s not in self.position_symbols+self.momentum_symbols)}
        new_params.update( {} if parameters is None else parameters)
        
        self.independent_vars['parameter_dict'].update(new_params)
        
        self.independent_vars['constants'].update(constants)
        
        return self
        
    def add_vector_potential(self,vector_potential=None,charge=None):
        """
        Add a static electromagnetic vector potential A to the model. In the model, the vector potential is included via a minimal coupling: k-> k-Ae/hbar
        :param vector_potential: vector potential function. Must have length 3 and contain expressions which can be constant or depend on x,y,z (but not kx,ky,kz)
        :type vector_potential: list
        """
        pass
        self.independent_vars['function_dict'][sympy.symbols(__VECTOR_FIELD_NAMES__[0])] = None if vector_potential is None else vector_potential[0]
        self.independent_vars['function_dict'][sympy.symbols(__VECTOR_FIELD_NAMES__[1])] = None if vector_potential is None else vector_potential[1]
        self.independent_vars['function_dict'][sympy.symbols(__VECTOR_FIELD_NAMES__[2])] = None if vector_potential is None else vector_potential[2]
        
        self.independent_vars['parameter_dict'][sympy.symbols('q_{c}')] = charge
        
        def vector_potential_adder(array,model):
            """Adds a vector potential to the model by replacing k_i with k_i +eA/hbar where e is the charge defined in the model
            :param array: array to replace k symbols with
            :type array: sympy.Array
            :param model: model whose parameters to use
            :type model: BandModel
            """
            K = model.momentum_symbols
            from . import constants
            A= list(sympy.symbols(__VECTOR_FIELD_NAMES__))
            substituted_array = array.subs({K[i]:(K[i]+sympy.symbols('q_{c}')*A[i]/constants['hbar']) for i in range(3)})
            return substituted_array

        self.independent_vars['postprocessing_function_specification']['A-field'] = vector_potential_adder
        
        return self

        
        
    # region BdG methods
    def BdG_extension(self):
        """Performs the BdG extension to also include time-reversed states in the Hamiltonian. This allows modeling of superconductivity.
        """
        # add a postprocessing function to the post-processor
        
        def BdG_extension(array:sympy.tensor.ImmutableDenseNDimArray,model:BandModel):
            """given an array, H, returns H \\oplus H^* 
            where H^* is the time-reversed state. This is computed by replacing:
                B -> -B (magnetic field)
                A -> -A (vector potential)
                k -> -k (momentum)
                all constants -> their complex conjugate
            as well as performing a unitary transformation which depends on the model in question. (this is what ultimately describes the flipping of spin under time-reversal).
            The time-reversal unitary transform is given as the attribute ´time_reversal_change_of_basis´
            :param array: The array to extend
            :type array: sympy.Array
            :param model: The model from where the array it originates from. The parameter specification of this model will be used.
            :type model: BandModel
            """
            
            # complex conjugate everything and then convert the above named operators as k^* -> -k
            
            # the shape of the returned array. respects tensor structure.
            return_shape = (array.shape[0]*2,)*2+array.shape[2:] 
                
            #change of basis array     
            U = model.__time_reversal_change_of_basis__
            if U is None:
                raise NotImplementedError(f'time-reversal change of basis array has not been specified for the model: {self.__class__}')
            
            if len(return_shape) >2:
                
                
                #we have to extend the change of basis matrix (which should only act on the first tensor components!) to act trivially on wrt the other indices.
                if not U.shape == array.shape[:2]:
                    raise ValueError(f'Time-reversal change of basis array has unexpected shape. expected: {array.shape[:2]} but got: {U.shape} ')
                
                U = sympy.tensorproduct(U,*(sympy.Array(sympy.eye(d)) for d in array.shape[2::2]))  # U can now be reshaped just like the array!
                
                
                
                # cast as matrix
                matrix_size = np.product(array.shape[::2])
                
                # permute the dims such that the left-most indices are the bra indices and the right-most are the ket indices
                dim_permutation=tuple(range(0,len(array.shape),2))+tuple(range(1,len(array.shape),2))
                array = sympy.tensor.array.permutedims(array,dim_permutation) # type: ignore
                array = array.reshape(matrix_size,matrix_size)
                
                U = sympy.tensor.array.permutedims(U,dim_permutation)
                U = U.reshape(matrix_size,matrix_size) #type: ignore
                
                
            else:
                dim_permutation = [] # not needed for matrix shaped stuff i.e. in case there has been no z-confinement
                
                
            
            time_reversed_array = array.copy().conjugate().subs({k:-1*sympy.conjugate(k) for k in model.momentum_symbols+sympy.symbols(__MAGNETIC_FIELD_NAMES__+__VECTOR_FIELD_NAMES__)}) #type: ignore # momentum symbols separate because they are non-commuting
            time_reversed_array = sympy.tensorcontraction(sympy.tensorproduct(U,time_reversed_array,U.conjugate().transpose()),(1,2),(3,4)) #NB! this one may need another minus sign! H -> -\Theta H \Theta^{-1} = - U H^* U^\dagger 
            
            bdg_extended = sympy.Array(sympy.Matrix(sympy.matrices.expressions.blockmatrix.BlockDiagMatrix(sympy.Matrix(array),sympy.Matrix(time_reversed_array)))) # cast first as matrix to get a dense version in stead of a sparse one before casting to array
            
            if bdg_extended.shape != return_shape:
                
                
                # reorder the axes again and reshape back to desired shape!
                permuted_return_shape = [return_shape[p] for p in dim_permutation]
                
                
                bdg_extended = sympy.tensor.array.permutedims(bdg_extended.reshape(*permuted_return_shape),dim_permutation)
                
            # Add SC order parameter
            if model.independent_vars['SC_Delta'] is not None:
                # us the same post processing func dict as the post processor for the model. This ensures consistency
                post_processed_delta = model.__postprocessing_constructor__(self.independent_vars['postprocessing_function_specification'],skip_bdg=True)(model.independent_vars['SC_Delta'],model)
                np_delta = np.array(post_processed_delta)
                np_delta_adj = np.swapaxes(np.array(post_processed_delta.conjugate()),0,1)
                np_version = np.array(bdg_extended)
                upper_right_quadrant = (slice(0,U.shape[0]),slice(U.shape[0],2*U.shape[0]))
                lower_left_quadrant = (slice(U.shape[0],2*U.shape[0]),slice(0,U.shape[0]))

                np_version[upper_right_quadrant] = np_delta
                np_version[lower_left_quadrant] = np_delta_adj
                bdg_extended = sympy.Array(np_version)
                
            return bdg_extended
        
        self.independent_vars['postprocessing_function_specification']['BdG'] = BdG_extension
        return self
    

        
    #endregion
    #endregion 
    
    # region z-confinement
    def add_z_confinement(self,nz_modes,z_confinement_type,lz=None):
        """Evaluates the z-direction on a finite set of basis vectors depending on the shape of the z-confinement.


        Args:
            lz (float): size of the system in the z-direction
            nz_modes (int): number of z-modes
            z_confinement_type (str): the type of confinement potential. Currently, either "well" or "box" are supported.
        """
        self.independent_vars['parameter_dict'][sympy.symbols('l_z')] = lz
        
        self.independent_vars['constants'][sympy.symbols(r'n_{z-modes}')] = nz_modes  #  should probably not be changed...
        self.independent_vars['constants'][sympy.symbols(r'z_{confinement_type}')] = z_confinement_type # should probably also not be changed 
        
        
        from sympy.polys.polytools import degree
        
        def z_confinement_func(array,model):
            """
            Performs z_confinement remapping of the passed array based on the parameter dict in the specified model
            """
            
            
            z_confinement_type = model.independent_vars['constants'][sympy.symbols(r'z_{confinement_type}')]
            if  z_confinement_type== 'box':
                def k_z_func(order):
                    def __k_mat__(order, L, n_modes):
                        """
                        compute matrix version of k operator for box modes
                        :param order:
                        :param L:
                        :param n_modes:
                        :return:
                        """
                        if order > 2:
                            raise NotImplementedError(f'fix this. K operator for order > 2 is not well defined (not hermitian). Could be fixed by just projecting k onto the (finite Hilbert space) ')  # todo

                        def k_mel(n, m):
                            # matrix elements of the momentum operator p^order = (-i\\hbar \\del)^order
                            # indexing with 0 so we add one here to
                            n += 1
                            m += 1
                            if order % 2:
                                # odd case
                                with np.errstate(divide='ignore', invalid='ignore'):  # suppress divide by zero warnings
                                    re = n * ((-1) ** (n + m) - 1) / (sympy.pi * (m ** 2 - n ** 2)) * (-1j * m * sympy.pi / L) ** order
                                return np.where(n != m, re, 0)

                            else:
                                # even case
                                return np.where(n == m, (n * sympy.pi / L) ** order, 0)

                        def p_mat(order):  # assemble momentum matrix
                            if order == 0:
                                return np.eye(n_modes, dtype='complex')
                            P = np.fromfunction(k_mel, (n_modes, n_modes))

                            return sympy.Array(P)

                        return p_mat(order)
                            
                    k_mat = __k_mat__(order,sympy.symbols('l_z'),model.independent_vars['constants'][sympy.symbols('n_{z-modes}')])
                    print(k_mat)
                    return k_mat #__k_mat__(order,model.parameter_dict[sympy.symbols('l_z')],model.constants[sympy.symbols('n_{z-modes}')])
            
                def z_func(order):
                    if order==0:
                        return np.eye(model.independent_vars['constants'][sympy.symbols('n_{z-modes}')]) # just return identity
                    else:
                        raise NotImplementedError(f'{order}')
            
            elif z_confinement_type == 'well':
                raise NotImplementedError()
                def k_z_func(order):
                    raise NotImplementedError()
            else:
                raise ValueError(f'confinement type {z_confinement_type} not recognized')
        
        
            disassemble_dict = {} # dict containing tensors corresponding to the respective order in kz
            
            kz = model.momentum_symbols[-1]
            z = model.position_symbols[-1] 
            
            # construct disassemble_dict (divide Hamiltonian into terms according to their dependency on kz and z)
            for i,val in enumerate(np.array(array).ravel()):
                
                val = val.expand(mul=True) # expand any parentheses
                terms = val.args if isinstance(val,sympy.core.add.Add) else (val,)
                for term in terms: 
                    try:
                        
                        # replace all momentum_symbols with a new symbols that commute and determine the order (because it for some reason doesn't work when there are non-commuting numbers).
                        D = sympy.Dummy(commutative=True) # dummy symbol for all other Ks
                        Dz = sympy.Dummy(commutative=True) # dummy symbol for kz
                        dummify={k:D for k in model.momentum_symbols[:-1]}
                        dummify[kz] = Dz
                        commuting_term = term.subs(dummify)
                        
                        k_order = degree(commuting_term,Dz) #type:ignore
                        z_order = degree(commuting_term,z) # type:ignore
                        
                        if isinstance(k_order,sympy.core.numbers.NegativeInfinity):
                            k_order = 0
                        if isinstance(z_order,sympy.core.numbers.NegativeInfinity):
                            z_order = 0
                        total_order=(z_order,k_order)
                    except TypeError as e:
                        continue 
                    
                    if total_order in disassemble_dict.keys():
                        arr = disassemble_dict[total_order]
                    else:    
                        arr = np.zeros(array.shape,dtype='O') 
                    # k-s commute, and by convention we will always order them as k_x k_x ... k_x k_y k_y ... k_y k_z k_z ... k_z

                    arr_i = np.unravel_index(i,array.shape)
                    arr[arr_i] += term.subs({kz:1,z:1}) # replace kz and z by 1 
                    disassemble_dict[total_order] = arr

            # tensor_dot the tensors by their kz_operator and z_operator expressed as a matrix
            def combiner(o,x):
                if nz_modes ==1:
                    return x*k_z_func(order=o[1])[0,0]*z_func(order=o[0])[0,0]    
                else:  # NB! we assume that the operators are ordered as z**N @ kz**M ! and we assume that we can compute the matrix element of this operator as the matrix multiplication of the matrix representations of z**N and kz**M
                    z_kz_matrix = z_func(order=o[0]) @ k_z_func(order=o[1])  
                    return np.tensordot(x,z_kz_matrix,axes=0)
                    
                    
                #combiner = lambda o,x: x*k_z_func(o)[0,0]
            
                #combiner = lambda o,x: np.tensordot(x,k_z_func(o),axes=0)# TODO: why [0,0],axes=0) # probably from the `drop excited modes case?`
                
            new_expression = sympy.Array(sum(combiner(o,mat) for o,mat in disassemble_dict.items())) # type: ignore
            
            return new_expression

            
        
        self.independent_vars['postprocessing_function_specification']['z-confinement'] = z_confinement_func
        return self
        

    # endregion
    
    # region saving, printing and comparing
    def __repr__(self) -> str:
        return super(BandModel,self).__repr__()
        #return self.independent_vars['preprocessed_array'].subs({r'\hbar':'hbar'}).__repr__()

    def __eq__(self,other)-> bool:
        if not isinstance(other,self.__class__):
            return False
        
        # if the independent vars are equal the models are equal:
        return self.independent_vars.as_dict() == other.independent_vars.as_dict()
        
        
    def __getstate__(self):
        return self.__dict__
    def __setstate__(self,state):
        self.__dict__.update(state)
    # endregion

class FreeBoson(BandModel):
    __time_reversal_change_of_basis__ = sympy.Array([[1]]) # no spin so no change of basis!
    
    def __init__(self, mass: float|None = None, spatial_dim: int = 3,unit_convention='SI') -> None:
        """ A model for a free boson (single parabolic band)
        :param float mass: numerical value for the mass of the boson. Defaults to None.
        :param int spatial_dim: Spatial dimension of the system.  Defaults to 3.
        """
        m = sympy.symbols('m')
        
        from . import constants,values
        hbar = constants['hbar']
        
        array = sympy.ImmutableDenseNDimArray([[hbar**2/(2*m)*sum((k**2 for k in sympy.symbols(__MOMENTUM_NAMES__[:spatial_dim])))]])
        super().__init__(array, spatial_dim)
        self.independent_vars['parameter_dict'][m] = mass
        self.independent_vars['constants'][hbar] = values[hbar]
        
class FreeFermion(BandModel):
    __time_reversal_change_of_basis__ = sympy.Array([[0,1],[-1,0]])
    def __init__(self,mass:float|None= None,spatial_dim: int =3, unit_convention='SI') -> None:
        m= sympy.symbols('m')
        
        from . import constants,values
        hbar = constants['hbar']
        
        diagonal = hbar**2/(2*m)*sum((k**2 for k in sympy.symbols(__MOMENTUM_NAMES__[:spatial_dim])))
        array = sympy.ImmutableDenseNDimArray([[diagonal,0],[0,diagonal]])
        super().__init__(array, spatial_dim)
        self.independent_vars['parameter_dict'][m] = mass
        self.independent_vars['constants'][hbar] = values[hbar]
        
    #region model construction
    def add_zeeman_term(self,g_tensor=None,Bvec=None,tensorize=False):
        """
        Adds a Zeeman term splitting the spins. 
        :param float|np.ndarray|None| g_tensor: The g-tensor. Can be passed as a scalar (isotropic) or as a tensor. The form of the added term depends on the shape of the passed g-tensor. If nothing is passed, the shape of the form is determined by `tensorize`.
        :param Sequence[float]|None Bvec: The magnetic field. This values cannot be passed together with an A-field.
        :param bool tensorize: whether the term should be expressed in terms of an anisotropic g-factor or be isotropic. The choice of this defined how many symbols are needed to express the g-factor. 
        """
        from . import ANGULAR_MOMENTUM,constants,values#todo set as sympy
        sigma = sympy.Array(ANGULAR_MOMENTUM['1/2'])
        mu_B = constants['mu_B']
        
        # cast sigma as tensor
        
        if any(s.name == __VECTOR_FIELD_NAMES__[0] for s in self.independent_vars['parameter_dict'].keys()):
            
            if Bvec is not None:
                raise ValueError('Bvector was specified but A-field already appears in model. This is inconsistent')
            else:
                pass
            """
            dd = lambda f,x : sympy.diff(f,x,evaluate=False)
            Ax,Ay,Az = sympy.symbols(__VECTOR_FIELD_NAMES__)
            x,y,z = self.position_symbols
            B = sympy.Array([dd(Ay,z)-dd(Az,x),dd(Az,x)-dd(Ax,z),dd(Ax,y)-dd(Ay,x)])
            """
        else:
            Bx,By,Bz = sympy.symbols(__MAGNETIC_FIELD_NAMES__)
            B = sympy.Array([Bx,By,Bz])
            
            Bvec = (0,0,0) if Bvec is None else Bvec
            self.independent_vars['function_dict'].update({B[i]:Bvec[i] for i in range(3)})
        
        
        
        
        if tensorize or isinstance(g_tensor,np.ndarray):
            g_components =  sympy.symbols(r'g_{xx},g_{yy},g_{zz},g_{yz},g_{xz},g_{xy}')
            gxx,gyy,gzz,gyz,gxz,gxy = g_components
            
            if isinstance(g_tensor,np.ndarray):
                values = (g_tensor[i,j] for (i,j) in ((0,0),(1,1),(2,2),(1,2),(0,2),(0,1)))
            elif isinstance(g_tensor,float):
                values = (g_tensor,g_tensor,g_tensor,0,0,0)
            else:
                values = (None,None,None,None,None,None)
            
            self.independent_vars['parameter_dict'].update({g_comp:v for g_comp,v in zip(g_components,values)})
            g = sympy.Array([[gxx,gxy,gxz],[gxy,gyy,gyz],[gxz,gyz,gzz]])
            term = mu_B*sympy.tensorcontraction(sympy.tensorproduct(sigma,g,B),(0,3),(4,5)) # contract sigma with g_tensor row_index and g_tensor column index with B_vec
        else:
            g = sympy.symbols('g')
            self.independent_vars['parameter_dict'][g] = g_tensor
            term = mu_B*g*sympy.tensorcontraction(sympy.tensorproduct(sigma,B),(0,3)) #contract first and last index of tensor sigma \\otimes B
            

        self.independent_vars['preprocessed_array'] = self.independent_vars['preprocessed_array'] + term
        from . import values
        self.independent_vars['constants'][mu_B] = values[mu_B]
        return self
    #endregion

class LuttingerKohnHamiltonian(BandModel):
    __time_reversal_change_of_basis__ = sympy.Array([[0,0,0,-1],[0,0,1,0],[0,-1,0,0],[1,0,0,0]])
    
    
    @staticmethod
    def __make_LK_Hamiltonian__(theta:float|sympy.Symbol|None=None,phi:float|sympy.Symbol|None=None):
        """Constructs the canonical Luttinger Kohn Hamiltonian matrix (with coordinate system pointing along the X,Y and Z directions)
            Follows the definition given in Winkler
        :return: _description_
        :rtype: _type_
        """

        sq = sympy.sqrt(3)
        Jx = sympy.Matrix([[0, sq, 0, 0], [sq, 0, 2, 0], [0, 2, 0, sq], [0, 0, sq, 0]]) * 1 / sympy.Integer(2)
        Jy = sympy.Matrix([[0, -sq, 0, 0], [sq, 0, -2, 0], [0, 2, 0, -sq], [0, 0, sq, 0]]) * sympy.I / sympy.Integer(2)
        Jz = sympy.Matrix([[3, 0, 0, 0], [0, 1, 0, 0], [0, 0, -1, 0], [0, 0, 0, -3]]) / sympy.Integer(2)
        Id = sympy.Matrix([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]])
        
        J = [Jx,Jy,Jz]
        J2 = [j @ j for j in J]
        Jsq = J2[0]+J2[1]+J2[2]
        K = sympy.symbols(__MOMENTUM_NAMES__,commutative=False) # Non-commutative to allow for adding A-fields later! 
        
        
        # Rotate J and K if angles are given
        if theta is not None or phi is not None:
            if theta is None:
                theta = 0
            elif phi is None:
                phi = 0
            
            rotation_mat = sympy.Array(
                [[sympy.cos(theta) * sympy.cos(phi), sympy.cos(theta) * sympy.sin(phi),
                -sympy.sin(theta)],
                [-sympy.sin(phi), sympy.cos(phi), 0],
                 [sympy.sin(theta) * sympy.cos(phi), sympy.sin(theta) * sympy.sin(phi),
                sympy.cos(theta)]])
            # rotate the AM matrices and the K operators
            Kvec = sympy.Matrix(K)
            Krot = rotation_mat @ Kvec
            
            K = list(Krot)
            
            # we rotate the Js in the same way by simply substituting Ks for Js
            J = [k.subs({K[i]:J[i] for i in range(3)}) for k in K]
            
            
        ss = lambda n:sympy.symbols(n)
        H_LK = ss(r'\gamma_{1}')* (K[0]**2+ K[1]**2+ K[2]**2)*Id
        for i in range(3):
            j = (i+1)%3
            k = (i+2)%3
            H_LK = H_LK - ss(r'\gamma_{3}')*((J[j] @ J[k]+J[k]@J[j])*(K[j]*K[k]+K[k]*K[j])) - 2*ss(r'\gamma_{2}')*(J2[i]-Jsq/3)*K[i]**2
        
        from . import constants
        hbar =  constants['hbar']
        m = sympy.symbols('m')
        
        H_LK = hbar**2/(2*m)*sympy.Array( H_LK )
        return H_LK
    
    def __init__(self,mass:float|None = None, spatial_dim: int =3, unit_convention='SI',
                gamma1:float|None=None,
                gamma2:float|None=None,
                gamma3:float|None=None):
        
        
        H_LK = self.__make_LK_Hamiltonian__() #  not rotated
        
        
        super().__init__(H_LK,spatial_dim,unit_convention)
        
        ss = lambda n:sympy.symbols(n)
        self.independent_vars['parameter_dict'].update({ss(r'\gamma_{1}'):gamma1, ss(r'\gamma_{2}'):gamma2,ss(r'\gamma_{3}'):gamma3,ss('m'):mass})

    def rotate_crystallographic_direction(self,theta=None,phi=None):
        """Performs a the postprocessing step of rotating the coordinate system, such that the coordinates are not aligned with the crystallographic axis. 
        Under the hood, the regular kinteic part of the Luttinger Kohn Hamiltonian is subtracted and another version, with rotated Ks and  Js is added.
        We have chosen to do so to make the expressions simpler if the crystal is not rotated (which is the most common scenario)

        :param theta: The angle between the crystallographic z-direction and the coordinate z axis
        :type theta: float|sympy.Symbol
        :param phi: The angle between the crystallographic y-direction and the coordinate y axis
        :type phi: float|sympy.Symbol
        """
        theta_val = theta
        phi_val = phi
        theta,phi = sympy.symbols(r'\theta,\phi') # redefine theta and phi as symbols 
        self.independent_vars['parameter_dict'][theta] =theta_val
        self.independent_vars['parameter_dict'][phi] =phi_val
        
        
        
        def rotate_crystal(array,model):
            """
            Rotates the crystal
            :param array: Array to transform.
            :type array: sympy.Array
            :param model: uses the parameters and constants of this model
            :type model: BandModel
            """
            
            rotated_H_LK = model.__make_LK_Hamiltonian__(theta,phi)
            regular_H_LK = model.__make_LK_Hamiltonian__()
            array = array - regular_H_LK + rotated_H_LK
            return array
            
        self.independent_vars['postprocessing_function_specification']['crystal rotation'] = rotate_crystal
        return self
    def add_zeeman_term(self,kappa:float|None=None,q:float|None=None,B:list|None=None):
        """Adds a Zeeman term to the Model. The B-field can either be passed here, or it can be inferred from an added vector field.

        :param kappa: Luttinger parameter for the strength of the term proportional to J , defaults to None
        :type kappa: float, optional
        :param q: Luttinger parameter for the strength of the term proportional to J^3 , defaults to None
        :type q: float, optional
        :param B: magnetic field. Can be defined explicitly or, if a vector potential is specified, inferred from the A-field when computing the numerical version, defaults to None
        :type B: list[float], optional
        """
        from . import SP_ANGULAR_MOMENTUM,constants,values
        J = SP_ANGULAR_MOMENTUM['3/2']
        mu_B = constants['mu_B']
        if not mu_B in self.independent_vars['constants']:
            self.independent_vars['constants'][mu_B] = values[mu_B]
            
            
        # add magnetic field symbols to parameter dict 
        Bsym = sympy.symbols(__MAGNETIC_FIELD_NAMES__)
        Bx = None if B is None else B[0]
        By = None if B is None else B[1]
        Bz = None if B is None else B[2]
        kappa_sym,q_sym = sympy.symbols(r'\kappa,q')
        self.independent_vars['function_dict'].update({Bsym[i]:b for i,b in enumerate((Bx,By,Bz))})
        self.independent_vars['parameter_dict'][kappa_sym] = kappa
        self.independent_vars['parameter_dict'][q_sym] = q
        
        Zeeman_term = sympy.Matrix([[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0]])
        for i in range(3):
            Zeeman_term = Zeeman_term - 2*mu_B*(kappa_sym*J[i]+q_sym*(J[i]@J[i]@J[i]))*Bsym[i]
        
        Zeeman_term = sympy.Array(Zeeman_term)
        self.independent_vars['preprocessed_array'] = self.independent_vars['preprocessed_array']+ Zeeman_term 
        
        return self
        
# region vector-field functions:
def make_vector_field(B_field,position_symbols,gauge='symmetric'):
    """
    Given a B-field, constructs a vector potential A according the the gauge specification

    :param B: list containing the components of B
    :type B: list
    :param position_symbols: the symbols which represent x,y,z.
    :type position_symbols: list[sympy.Symbols]
    :param gauge: which gauge to chose, defaults to 'symmetric'
    :type gauge: str, optional
    :return: Returns a list of the a filed components
    """
    if gauge == 'symmetric':
        A = []
        for i in range(3):
            j = (i+1)%3
            k = (i+2)%3
            A.append((B_field[j]*position_symbols[k]-B_field[k]*position_symbols[j])/2)
    elif gauge == 'other':
        BX=lambda i,j:B_field[i]*position_symbols[j]
        A =[-BX(2,1)/2,
                        BX(2,0)/2,
                        (BX(0,1)-BX(1,0))]
    else:
        raise NotImplementedError(f'gauge not implemented: {gauge}')


    return A 

#endregion