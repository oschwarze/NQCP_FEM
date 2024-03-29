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

import numpy as np
import sympy
from .functions import SymbolicFunction,NumericalFunction,Function,PlaceHolderFunction
from . import functions as funcs
import logging
from . import PETSc
from typing import Any, Iterable,Union,Sequence,Annotated,Callable
import typing
LOGGER = logging.getLogger(__name__)
LOGGER.addHandler(logging.NullHandler())

# specify global names for important variables 
from .symbolic import __MOMENTUM_NAMES__,__POSITION_NAMES__,dummify_non_commutative,expand_term
__MAGNETIC_FIELD_NAMES__ = (r'B_{x}(x)',r'B_{y}(x)',r'B_{z}(x)') # These are always treated as functions!
__VECTOR_FIELD_NAMES__ =  (r'A_{x}(x)',r'A_{y}(x)',r'A_{z}(x)') # these are always treated as functions!

__STRAIN_NAMES__ = tuple(r'\varepsilon_{'+('x','y','z')[i]+('x','y','z')[j]+'}(x)' for i in range(3) for j in range(i,3))

from .updatable_object import UpdatableObject, auto_update,ComputedAttribute
from .functions import NumericalFunction,SymbolicFunction

class BandModel(UpdatableObject):


    __post_processor_ordering__ = ("OTHER","A-field","z-confinement","BdG",'K-ordering')
    __time_reversal_change_of_basis__ = None # this attr stores the change of basis that is involved in time-reversal of the model.
    def __init__(self,array:sympy.ImmutableDenseNDimArray,spatial_dim:int=3,unit_convention='SI') -> None:
        """This class describes the Hamiltonian of the system. In the array, the position variables must be symbols which are named ´x,y,z´ and the momentum operators must be symbols named ´k_x,k_y,k_z´.
        Args:
            expression (sympy.ImmutableDenseNDimArray): The array which represents the Hamiltonian 
            spatial_dim (int, optional): Spatial dimension of the system. Defaults to 3.
        
        """
        
        

        # Dict containing all the attributes that could be changed from outside the instance.
        independent_vars = {'spatial_dim':spatial_dim} 
        
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
        independent_vars['Delta_SC'] = None
        
        # unit_convention_stuff
        if unit_convention == 'SI' or unit_convention == {}:
            self.unit_convention = {} # todo: Allows for changing the unit convention to something different from SI base units
            
            independent_vars['unit_convention'] = {}
            
            independent_vars['constants']= {k:v for k,v in values.items() if isinstance(k,sympy.Symbol)} # import SI values from init.py

            #self.constants = SyncDict({})  # dict mapping symbols representing constants such as hbar, pi etc to their numerical values (expressed in SI units). Transformation to different unit systems is done dynamically.

        else:
            raise NotImplementedError('yet to implement saving numerical values in different unit scales')

        independent_vars['model_defining_params'] = {} # paams like number of z modes and k_ordering etc
        
        # init the parent to assure that computable items can be computed and saved.
        super().__init__(use_sympy_dict=True,**independent_vars)
        
        return      

    
    #region properties for acessing independent_vars
    @property
    def parameter_dict(self):
        return self.independent_vars['parameter_dict']
    @parameter_dict.setter
    def parameter_dict(self,value):
        self.independent_vars['parameter_dict'] = value

    @property
    def constants(self):
        return self.independent_vars['constants']
    @constants.setter
    def constants(self,value):
        self.independent_vars['constants'] = value
    
    @property
    def function_dict(self):
        return self.independent_vars['function_dict']
    
    @function_dict.setter
    def function_dict(self,value):
        self.independent_vars['function_dict'] = value
    
    
    @property
    def Delta_SC(self):
        return self.independent_vars['Delta_SC']
    
    @Delta_SC.setter
    def Delta_SC(self,value):
        self.independent_vars['Delta_SC'] = value
    
    #endregion
    @property
    def spatial_dim(self):
        return self.independent_vars['spatial_dim']
    
    # region postprocessing and building  / casting    
    
    def validate_model(self):
        
        if any(f is not None and not isinstance(f,Function) for f in self.function_dict.values()):
            wrong_funcs = tuple(v for v,f in self.function_dict.values() if ((f is not None) and not isinstance(f,Function)))
            raise TypeError(f' the following functions in the function dict were not of type nqcpfem.functions.Function: {wrong_funcs}')
    
    def validate_numerical_model(self):
        
        if any(p is None for p in self.parameter_dict.values()):
            wrong_params = tuple(v for v,f in self.parameter_dict.values() if f is None )
            raise ValueError(f'the following parameters were unspecified: {wrong_params}')
        if any(p is None for p in self.function_dict.values()):

            wrong_params = tuple(v for v,f in self.function_dict.values() if f is None )
            raise ValueError(f'the following functions were unspecified: {wrong_params}')
        
        if any(p is None for p in self.constants.values()):
            wrong_params = tuple(v for v,f in self.constants.values() if f is None )
            raise ValueError(f'the following constants were unspecified: {wrong_params}')
        
        
    #region array_shape properties
    @property
    def tensor_shape(self):
        
        return tuple(int(s) for s in self.post_processed_array().shape)
    
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
        
    
    def __postprocessing_constructor__(self,array_pp_spec,skip_bdg=False):
        """Function for building the post-processing function, either with or without a bdg step. This allows us to also use it  for post-processing the SC order parameter.

        :param skip_bdg: Whether to not do the BdG extension part if it is supplied, defaults to False
        :type skip_bdg: bool, optional
        :return: Function which takes an array and a BandModel and returns a post-processed array
        :rtype: Sympy.Array
        """
        
        _ = array_pp_spec    # this just calls the dependency so that dependencies work even if the pp_func_spec is an empty dict 
        # create list of functions in the correct ordering
        array_pp_sorted = []
        for func_name in self.__class__.__post_processor_ordering__:
            if func_name == 'OTHER':
                # run all keys that are not in ordering
                array_pp_sorted.extend(f for name,f in array_pp_spec.items() if name not in self.__class__.__post_processor_ordering__)

            elif skip_bdg and func_name == "BdG":
                continue # do not add bdg extension function. 
            else:
                if func_name in array_pp_spec.keys():
                    array_pp_sorted.append(array_pp_spec[func_name])
        
        # we need a function that post-processed the functions and the array simultaneously
        
        
        
        if not len(array_pp_sorted):
            post_processing_func = lambda arr,funcs,model:(arr,funcs)
        else:
            def chain(funcs):
                """
                The function "chain" takes in multiple functions as arguments and returns a new function
                that applies each of the input functions in sequence to its input.
                """
                def chained_call(arr,func_dict,model):
                    result=(arr,func_dict)
                    for func in funcs:
                        result=func(*result,model=model)
                    return result
                return chained_call
            post_processing_func = chain(array_pp_sorted)
        
        return post_processing_func
    
    @auto_update
    def post_processor(self):
        """Returns a function F(array,model) which converts the preprocessed array to the post-processed one based on parameters of the passed model 

        :return: _description_
        :rtype: _type_
        """
        return self.__postprocessing_constructor__(self.independent_vars['postprocessing_function_specification'],skip_bdg=False)

    @auto_update
    def __post_process__(self):
        return self.post_processor()(self.independent_vars['preprocessed_array'],self.independent_vars['function_dict'],self)
    
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
        return self.__post_process__()[0]
    
    @auto_update
    def post_processed_functions(self):
        return self.__post_process__()[1]
    
    
    @auto_update
    def numerical_lambda(self,replace_symbolic_functions=True):
        """

        creates a numpy array where the only free symbols left are x,y,z and k_x,k_y,k_z. The ks remain as commutative symbols.

        Args:
            float_format (_type_, optional): The format for the floats *that are not in the parameter_dict*. Defaults to PETSc.ScalarType.
        """
        
        #TODO: extract construction of the var_dict and make it update only the relevant parts instead of remaking everything whenever a value is changed (just update the value in the dict as well)
        #TODO: extract the function substitution part o the construction and pass it a specific dictionary so that it can be used by fenics energy_scale to subs x,y,z with 0 fast.
        
        var_dict = self.independent_vars['parameter_dict'].copy() # make copy for adding constants and x,y,z and k_x,k_y,k_z. These will just be evaluated as their symbols
        # specify B field from A field if it is present:
        
        unit_transformed_constants = self.independent_vars['constants'].copy() # todo  make transformation of unit scales 
        
        var_dict.update(unit_transformed_constants)

        unspecified = [k for k,v in var_dict.items() if v is None]
        if unspecified:
            raise ValueError(f'the following values have not been specified so the model cannot be cast as a numerical array:\n {unspecified}')
        # evaluate all functions:
        from typing import Callable
        position_set = set(self.position_symbols)
        for sym,func in self.post_processed_functions().items():
            # if the function is symbolic, we replace it with its symbolic representation, else, we keep it as a symbol
            if isinstance(func,SymbolicFunction):
                if replace_symbolic_functions:
                    # substitue all parameters
                    func = func.expression
                    if any((s,s) in var_dict.items() for s in func.free_symbols):
                        raise ValueError(f'Infinite recursion due to substituion of symbol with itself: {[s for s in func.free_syms if (s,s) in var_dict.items()]}')
                    func = func.subs(var_dict)
                    if position_set.union(func.free_symbols) != position_set:
                        raise ValueError(f'Passed function had symbols not contained in the parameter-dict. Only x,y,z should be left but we have: {func.free_symbols}')
                    var_dict[sym] = func
                else:
                    var_dict[sym] = sym


        
        var_dict.update({k:k for k in self.momentum_symbols})  #casts momentum symbols as themseelves so that we can dummify commutative symbols
        var_dict.update({k:k for k in self.position_symbols})  #add position symbols as themselves (do it after function spec to avoid infinite recursion)
        # add function numerical functions to dict so that we can use lambdify
        var_dict.update({s:s for s,f in self.post_processed_functions().items() if isinstance(f,NumericalFunction)})
        
        if any(True for f in self.independent_vars['function_dict'].values() if not isinstance(f,(NumericalFunction,SymbolicFunction,type(None),PlaceHolderFunction) )):
            wrong_dict =  {s:f for s,f in self.independent_vars['function_dict'].items() if not isinstance(f,(NumericalFunction,SymbolicFunction))}
            raise TypeError(f'functions must be passed as either numerical function or symbolic functions. Recieved: {wrong_dict}')


        unspecified = [k for k,v in var_dict.items() if v is None]
        if unspecified:
            raise ValueError(f'the following values have not been specified so the model cannot be cast as a numerical array:\n {unspecified}')
        
        
        # we have to dummify ourselves, because dummies created in lambdify are commutative which we don't want
        #dummy_map = dummify_non_commutative(self.post_processed_array())
        #dummified = self.post_processed_array().subs(dummy_map)

        #value_map = {dummy_map.get(s,s):val for s,val in var_dict.items()} # cast dummy symbols to values
        
        dummy_map,lambdified_arr = self.lambdify_non_commutative()
        
        value_map = {v:var_dict[k] for k,v in dummy_map.items()}
        return value_map,lambdified_arr

    @auto_update
    def numerical_array(self,replace_symbolic_functions=True):
        LOGGER.debug(f'building numerical array')
        value_map,lambdified_arr = self.numerical_lambda(replace_symbolic_functions=True)

        numpy_version = np.array(lambdified_arr(*value_map.values())) # set Dummify to True to make sure that symbol names do not break the code
        return numpy_version
    
    #@auto_update
    def lambdify_non_commutative(self):
        LOGGER.debug('constructing lambda map')
        dummy_map = dummify_non_commutative(self.post_processed_array())
        dummified = self.post_processed_array().subs(dummy_map)
        lambdified = sympy.lambdify(dummy_map.values(),dummified,dummify=False,docstring_limit=0)
        return dummy_map,lambdified
    
    @auto_update
    def symbolic_tensor_repr(self):
        """
        COnstructs a tensor representation of the model, where the Ks are replaced with indices in a tensor. The symbols are kept.
        """
        
        commuting_momentum_symbols = sympy.symbols(__MOMENTUM_NAMES__) # these are the commuting momentum symbols which are in the numerical version of the Hamiltonian
        #array =   self.post_processed_array().subs({k:ck for k,ck in zip(self.momentum_symbols,commuting_momentum_symbols)}).subs({sympy.symbols(r'\hbar'):sympy.symbols('hbar')}) # \hbar breaks lambdify so we replace with a working name :( 
        
        disassemble_dict = self.__make_tensor_dict__(self.post_processed_array(),self.spatial_dim) # we do not need commuting Ks anymore
        

        return disassemble_dict
    
    @auto_update
    def numerical_tensor_repr(self):
        """
        Constructs a tensor representation of the model where the Ks are replaced with indices in a tensor. The resulting dict contains numpy arrays of complex numbers.
        """
        disassemble_dict = self.__make_tensor_dict__(self.numerical_array(),self.spatial_dim)
        return disassemble_dict
    
    @staticmethod
    def __make_tensor_dict__(array,spatial_dim):
        """Constructs a dictionary where the keys are are integers and elements are arrays of shape `self.tensor_shape`. The keys indicate how many factor of the momentum operators are in front of the specified tensor.

        :params force_recompute: Whether to force the recomputation of the array or not. Defaults to False.
        :params use_numerical: True by default. Uses the numerical array generated from make_numerical() rather than the array returned by make_array().

        Returns: dict[int,np.ndarray]: dictionary with the Hamiltonian split up into coefficients in front of k operators
        """
        from . import symbolic as symb 
        momentum_symbols = (symb.Kx,symb.Ky,symb.Kz)
        
        
        disassemble_dict = {} # dict containing tensors corresponding to the respective order
        from .symbolic import k_component_signature,expand_term
        
        for i,val in enumerate(np.array(array).ravel()):
            # split up value into terms 
            val = sympy.sympify(val)

            
            val = val.expand()
            terms = val.args if isinstance(val,sympy.core.add.Add) else (val,)
            for term in terms: 
                # compute the k-component_signature
                if all(k not in term.free_symbols for k in momentum_symbols ):
                    # skip the simple case. This also avoids problematic piecewise terms 
                
                    arr = disassemble_dict.get(0,np.zeros(array.shape,'O'))
                    arr_i = np.unravel_index(i,array.shape)
                    term = term.subs({sympy.symbols(r'\hbar'):sympy.symbols('hbar')})
                    arr[arr_i] += term
                    disassemble_dict[0] = arr

                    continue
                
                if isinstance(term,sympy.Piecewise):
                    # check if the term is a valid piecewise. If not, raise Notimplemented (determine what you want to do)
                    from .symbolic import extract_valid_bipartition_part
                    piecewise_parts,remains = extract_valid_bipartition_part(term)
                    if isinstance(remains,sympy.Piecewise):
                        raise NotImplementedError('TODO')
                    term = remains
                else:
                    piecewise_parts = []
                
                
                k_signature = k_component_signature(expand_term(term))
                arr = disassemble_dict.get(len(k_signature),np.zeros(array.shape+(spatial_dim,)*(len(k_signature)),'O'))
                # k-s commute, and by convention we will always order them as k_x k_x ... k_x k_y k_y ... k_y k_z k_z ... k_z, by sorting the signature
                arr_i = np.unravel_index(i,array.shape)+tuple(sorted(k_signature))
                term = term.subs({sympy.symbols(r'\hbar'):sympy.symbols('hbar')})

                addition =   term.subs({k:1 for k in momentum_symbols}) if len(k_signature) else term 
                arr[arr_i] += addition*sympy.Mul(*piecewise_parts) # add the piecewise parts back on again

                # add coefficient to array
                disassemble_dict[len(k_signature)] = arr

            
        return disassemble_dict
    
    
    def fix_position(self,array,x,y=None,z=None):
        def subst_position(array,pos):
            # replace all functions as well:
            subs_dict = {xx:pp for xx,pp in zip(self.position_symbols,pos)}
            subs_dict.update({s:f(*pos) for s,f in self.post_processed_functions().items()})           
            return sympy.Array(array).subs(subs_dict)
            
            
        
        if isinstance(array,dict):
            #case when we use this to fix postion of tensor_repr
            new_dict = {}
            for k,arr in array.items():
                new_arr = subst_position(arr,(x,y,z))
                if isinstance(arr,np.ndarray):
                    new_arr = np.array(new_arr)
                new_dict[k] = new_arr
            return new_dict
            
        else:
            #case when we use this to fix position of arrays
            new_arr = subst_position(array,(x,y,z))
            if isinstance(array,np.ndarray):
                new_arr = np.array(new_arr)
            return new_arr
    
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
        #fixed_array:sympy.tensor.ImmutableDenseNDimArray = sympy.Array(self.numerical_array()).subs({pos_sym:pos for pos_sym,pos in zip(self.position_symbols,position)}) # type: ignore
        fixed_array = sympy.Array(self.fix_position(self.numerical_array(),*position))
        commuting_momentum_symbols = sympy.symbols(__MOMENTUM_NAMES__)
        array_func = sympy.lambdify(self.momentum_symbols,fixed_array,dummify=True) # dummify replaces ks with commuting symbols, but we substitue with numbers anyways 
        
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

        from . import _m_e,_e
        from . import UNIT_CONVENTION
        eV = 1/UNIT_CONVENTION['J to eV']
        if material_name == 'Ge':
            parameter_dict = {r'\Delta_{0}': 0.296,
                            r'\gamma_{1}': 13.38,
                            r'\gamma_{2}': 4.24,
                            r'\gamma_{3}': 5.69,
                            r'kappa': 3.41,
                            r'q': 0.06,
                            r'C_{1}':0, # set to zero since it is of no relevance in the LK hamiltonian as it descirbes the cobnduction band shift caused bu strin.
                            r'D_{d}':-2*eV, # not irellevant since strain is spatially dependent and therefore the overall shift of the valence band matters! 
                            r'C_{4}':0, # set to zero since Ge is inversion symmetric ( see https://link.aps.org/doi/10.1103/PhysRevB.20.686 )
                            r"C'_{5}":0,# set to zero since Ge is inversion symmetric ( see https://link.aps.org/doi/10.1103/PhysRevB.20.686 )
                            r'D_{u}': 3.3195*eV,
                            r"D'_{u}": 5.7158*eV,
                            r'c_{11}': 12.40,
                            r'c_{12}': 4.13,
                            r'c_{44}': 6.83,
                            r'a': 5.65791*1e-10,
                            r'\epsilon': 16.5,
                            r'm': _m_e,
                            r'q_{c}': _e} #electric charge
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

        if isinstance(potential,Function):
            self.function_dict[potential.symbol]= potential
            potential = potential.symbol
        
        diagonal_array = potential* sympy.eye(self.independent_vars['preprocessed_array'].shape[0])
        
        self.independent_vars['preprocessed_array'] = self.independent_vars['preprocessed_array']+ sympy.Array(diagonal_array)
        
        constants = {} if constants is None else constants
        
        # make dict containing all parameters
        new_params = {s:None for s in potential.free_symbols if (s not in constants.keys() and s not in self.position_symbols+self.momentum_symbols and s.name[-3:] !='(x)')}
        new_params.update( {} if parameters is None else parameters)
        
        new_functions = {s:PlaceHolderFunction(s) for s in potential.free_symbols if s.name[-3:] =='(x)'}
        if parameters is not None:
            new_functions.update({p:v for p,v in parameters.items() if p in new_functions})
        
        self.function_dict.update({p:v for p,v in new_functions.items() if p not in self.function_dict})
        
        self.independent_vars['parameter_dict'].update({k:v for k,v in new_params.items() if k not in self.independent_vars['parameter_dict']})
        
        self.independent_vars['constants'].update({k:v for k,v in constants.items() if k not in self.independent_vars['constants']})
        
        return self
        
    def add_vector_potential(self,vector_potential=None,charge=None):
        """
        Add a static electromagnetic vector potential A to the model. In the model, the vector potential is included via a minimal coupling: k-> k-Ae/hbar
        :param vector_potential: vector potential function. Must have length 3 and contain expressions which can be constant or depend on x,y,z (but not kx,ky,kz)
        :type vector_potential: list
        """
        
        A=sympy.symbols(__VECTOR_FIELD_NAMES__,commutative=False) 
        
        
        self.independent_vars['function_dict'][A[0]] = None if vector_potential is None else vector_potential[0]
        self.independent_vars['function_dict'][A[1]] = None if vector_potential is None else vector_potential[1]
        self.independent_vars['function_dict'][A[2]] = None if vector_potential is None else vector_potential[2]
        
        #add parameter if needed
        if (charge is not None) or (sympy.symbols('q_{c}') not in self.independent_vars['parameter_dict']):
                self.independent_vars['parameter_dict'][sympy.symbols('q_{c}')] = charge
        
        #cast as Functions
        for a in A: 
            if not isinstance(self.independent_vars['function_dict'][a],funcs.Function):
                f = self.independent_vars['function_dict'][a] 
                if f is None:
                    self.independent_vars['function_dict'][a] = PlaceHolderFunction(a)
                else:
                    self.independent_vars['function_dict'][a] = NumericalFunction(f,a,[0,1,2]) if isinstance(f,Callable) else SymbolicFunction(sympy.sympify(f),a)
                
        # only alter post-processing specification if it is necessary. This avoids raising any 'changed' flags
        if 'A-field' not in self.independent_vars['postprocessing_function_specification']:
            self.independent_vars['postprocessing_function_specification']['A-field'] = self.vector_potential_adder
        
        return self

    def fix_k_arrangement(self,signature_type:str,signature_reduction_direction:str='left',allow_placeholder_functions=False):

        # given an array arrange them and update the new functions that come out of this.
        # determine the updated functions 
        consts = self.independent_vars['model_defining_params']
        consts['k_signature_type'] = signature_type
        consts['k_signature_reduction'] = signature_reduction_direction
        consts['allow_placeholder_functions']= allow_placeholder_functions
        

        self.independent_vars['postprocessing_function_specification']['K-ordering'] = self.rearrange_ks
        return self
        
    def BdG_extension(self):
        """Performs the BdG extension to also include time-reversed states in the Hamiltonian. This allows modeling of superconductivity.
        """
        # add a postprocessing function to the post-processor

        self.independent_vars['postprocessing_function_specification']['BdG'] = self.BdG_extender

        return self
    
    #endregion 
    
    # region z-confinement
    def add_z_confinement(self,nz_modes,z_confinement_type,lz=None,omega=None):
        """Evaluates the z-direction on a finite set of basis vectors depending on the shape of the z-confinement.
        Args:
            lz (float): size of the system in the z-direction
            nz_modes (int): number of z-modes
            z_confinement_type (str): the type of confinement potential. Currently, either "well" or "box" are supported.
        """
        
        if z_confinement_type == 'box':
            self.independent_vars['parameter_dict'][sympy.symbols('l_z')] = lz
            eval_kwargs = {'L':sympy.symbols('l_z')}
        elif z_confinement_type == 'well':
            self.independent_vars['parameter_dict'][sympy.symbols(r'\omega')] = omega
            eval_kwargs = {'omega':sympy.symbols(r'\omega'),'m':sympy.symbols('m')}
        else:
            raise NotImplementedError(f'{z_confinement_type} is not implemneted')
        
        self.independent_vars['model_defining_params'][sympy.symbols(r'n_{z-modes}')] = nz_modes  #  should probably not be changed...
        self.independent_vars['model_defining_params'][sympy.symbols(r'z_{confinement_type}')] = z_confinement_type # should probably also not be changed 
        
        
        
        self.independent_vars['model_defining_params']['z_confinement_eval_kwargs'] = eval_kwargs
        self.independent_vars['postprocessing_function_specification']['z-confinement'] = self.z_confinement_func
        # For the functions that depend on z, we need to 
        
        self.independent_vars['spatial_dim'] = 2 
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
    
    
    def __setstate__(self,state):
        self.__init__(state['preprocessed_array'],state['spatial_dim'],state['unit_convention'])

        from .updatable_object import ConstantsDict
        
        self._independent_vars_ = ConstantsDict(state)
        return self
    # endregion

    
    
    # region post-processing functions
    @staticmethod
    def z_confinement_func(array,func_dict,model):
        """
        Performs z_confinement remapping of the passed array based on the parameter dict in the specified model
        """
        
        
        z_confinement_type = model.independent_vars['model_defining_params'][sympy.symbols(r'z_{confinement_type}')]
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
                        
                k_mat = __k_mat__(order,sympy.symbols('l_z'),model.independent_vars['model_defining_params'][sympy.symbols('n_{z-modes}')])
                return k_mat #__k_mat__(order,model.parameter_dict[sympy.symbols('l_z')],model.constants[sympy.symbols('n_{z-modes}')])

            def z_func(order):
                from .functions import box_polynomial_matrix
                
                L = model.independent_vars['parameter_dict'][sympy.symbols('l_z')]
                nz = model.independen_vars['model_defining_params'][sympy.symbols(r'n_{z-modes}')]
                return box_polynomial_matrix(nz,L,order)
        
        elif z_confinement_type == 'well':
            
            
            raise NotImplementedError()
            def k_z_func(order):
                raise NotImplementedError()
        else:
            raise ValueError(f'confinement type {z_confinement_type} not recognized')
    
    
        
        kz = model.momentum_symbols[-1]
        z = model.position_symbols[-1] 
        
        
        eval_kwargs = model.independent_vars['model_defining_params']['z_confinement_eval_kwargs']
        
        # construct disassemble_dict (divide Hamiltonian into terms according to their dependency on kz and z)
        nz_modes = model.independent_vars['model_defining_params'][sympy.symbols(r'n_{z-modes}')]
        new_arr = sympy.tensor.Array.zeros(*array.shape,nz_modes,nz_modes).as_mutable() # check this works
        
        new_funcs = func_dict.copy()
                
        for i,val in enumerate(np.array(array).ravel()):
            
            val = val.expand(mul=True) # expand any parentheses
            terms = val.args if isinstance(val,sympy.core.add.Add) else (val,)
            for term in terms: 
                search_symbols = tuple(f for f in term.free_symbols if f.name[-3:] == '(x)')
                term = expand_term(term,split_pow=False) # expand into factors 

                coeff = 1
                matrix = sympy.Matrix.eye(nz_modes) # check this works )
                for t in term:
                    if not t.is_constant(kz):
                        # compute the matrix_expression of kz and z and Vs
                        matrix = matrix @ sympy.Matrix(k_z_func(t.args[1] if isinstance(t,sympy.Pow) else 1))
                    elif not t.is_constant(z):
                        matrix = matrix @ sympy.Matrix(z_func(t.args[1] if isinstance(t,sympy.Pow) else 1))
                    elif len(search_symbols) and not t.is_constant(*search_symbols):
                        # split this term up if it is a Power:
                        if isinstance(t,sympy.Pow):
                            func = new_funcs[t.args[0]].pow(t.args[1])
                            new_funcs[func.symbol] = func
                        else:
                            func = new_funcs[t]
                        
                        projected,added_funcs = func.project_to_basis(2,nz_modes,type=z_confinement_type,**eval_kwargs)
                        new_funcs.update(added_funcs)
                        matrix = matrix @sympy.Matrix(projected)
                    else:
                        coeff = coeff*t
                # with the signature, we can replace kz and z in the correct order.

                index = np.unravel_index(i,array.shape)+(slice(0,matrix.shape[0]),slice(0,matrix.shape[1]))
                
                new_arr[index] += sympy.Array(coeff*matrix)
        
        
        
        return sympy.Array(new_arr),new_funcs

    @staticmethod
    def BdG_extender(array:sympy.tensor.ImmutableDenseNDimArray,func_dict,model):
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
        time_reversed_array = -1*sympy.tensorcontraction(sympy.tensorproduct(U,time_reversed_array,U.conjugate().transpose()),(1,2),(3,4)) #NB! this one may need another minus sign! H -> -\Theta H \Theta^{-1} = - U H^* U^\dagger 
        
        bdg_extended = sympy.Array(sympy.Matrix(sympy.matrices.expressions.blockmatrix.BlockDiagMatrix(sympy.Matrix(array),sympy.Matrix(time_reversed_array)))) # cast first as matrix to get a dense version in stead of a sparse one before casting to array
        
        if bdg_extended.shape != return_shape:
            
            
            # reorder the axes again and reshape back to desired shape!
            permuted_return_shape = [return_shape[p] for p in dim_permutation]
            
            
            bdg_extended = sympy.tensor.array.permutedims(bdg_extended.reshape(*permuted_return_shape),dim_permutation)
            
        # Add SC order parameter
        if model.independent_vars['Delta_SC'] is not None:
            # us the same post processing func dict as the post processor for the model. This ensures consistency
            post_processed_delta = model.__postprocessing_constructor__(model.independent_vars['postprocessing_function_specification'],skip_bdg=True)(model.independent_vars['Delta_SC'],model.independent_vars['function_dict'],model)[0]
            np_delta = np.array(post_processed_delta)
            np_delta_adj = np.swapaxes(np.array(post_processed_delta.conjugate()),0,1)
            np_version = np.array(bdg_extended)
            upper_right_quadrant = (slice(0,U.shape[0]),slice(U.shape[0],2*U.shape[0]))
            lower_left_quadrant = (slice(U.shape[0],2*U.shape[0]),slice(0,U.shape[0]))

            np_version[upper_right_quadrant] = np_delta
            np_version[lower_left_quadrant] = np_delta_adj
            bdg_extended = sympy.Array(np_version)
            
        return bdg_extended,func_dict
    
    @staticmethod
    def vector_potential_adder(array,func_dict,model):
        """Adds a vector potential to the model by replacing k_i with k_i +eA/hbar where e is the charge defined in the model
        :param array: array to replace k symbols with
        :type array: sympy.Array
        :param model: model whose parameters to use
        :type model: BandModel
        """
        K = model.momentum_symbols
        from . import constants
        A= list(sympy.symbols(__VECTOR_FIELD_NAMES__,commutative=False))
        substituted_array = array.subs({K[i]:(K[i]+sympy.symbols('q_{c}')*A[i]/constants['hbar']) for i in range(3)})

    
        # region add B-field from A 
        A=sympy.symbols(__VECTOR_FIELD_NAMES__,commutative=False) 
        B=sympy.symbols(__MAGNETIC_FIELD_NAMES__,commutative=False)
        new_dict = func_dict.copy()
        
        
        for b in B:
            # check that B field wsa not supplied
            if func_dict.get(b,None) is not None and not isinstance(func_dict.get(b,None),PlaceHolderFunction):
                if isinstance(func_dict[b],SymbolicFunction) and func_dict[b].expression != 0: # we overwrite if B is specified as zero.# we overwrite if B is specified as zero.
                    raise ValueError(f'Bot magnetic field and vector potential were specified for the model. When vector potentials are included in the model, the magnetic field is computed automatically')

        # substitue B for curl of A  
        for i in range(3):
            j = (i+1)%3
            k = (i+2)%3
            
            def numerical_B_factory(ai,aj):
                def B_func(x,y=None,z=None):
                    return ai(x,y,z)-aj(x,y,z)
                return B_func   
            
            akj = func_dict[A[k]].derivative(j)
            ajk =func_dict[A[j]].derivative(k)
            
            if any(isinstance(a,PlaceHolderFunction) for a in (akj,ajk)):
                continue
            elif any(isinstance(a,NumericalFunction) for a in (akj,ajk)):
                spatial_deps = [i for i in range(3) if i in akj.spatial_dependencies+ajk.spatial_dependencies]
                b_function = NumericalFunction(numerical_B_factory(akj,ajk),B[i],spatial_deps)
            else: 
                b_function = SymbolicFunction(akj.expression-ajk.expression,B[i])
                
                
            new_dict[B[i]]= b_function #func_dict[A[k]].derivative(j) -func_dict[A[j]].derivative(k)

        # endregion
        
        return substituted_array,new_dict
    
    @staticmethod 
    def rearrange_ks(array,func_dict,model):
        """
        Rearrance the Momentum operators by commuting them with other operators in the array. 
        This in general will involve taking derivatives of specified functions.
        Default behaviour for handling this is to throw an error if the functions are numerical functions.
        If ´accept_placeholders´ is True, then derivatives of nummerical functions will be placeholder functions.
        Handling these placeholder functions can then be
        """
        consts = model.independent_vars['model_defining_params']
        from .symbolic import arange_ks_array
        new_array = arange_ks_array(array,consts['k_signature_type'],consts['k_signature_reduction'])#,consts.get('allow_placeholder_functions',False))
        

        # region functions computing
        from .symbolic import arange_ks_array
        from .functions import decompose_func_name
        

        allow_placeholder_functions = consts.get('allow_placeholder_functions',False)

        for s in new_array.free_symbols:
            if s.name[-3:]=='(x)' and s not in func_dict:
                # try to make s. we know that the symbols have been projected the same way so only derivatives can differ
                s_decomp = decompose_func_name(s.name)
                for f,v in func_dict.items():
                    f_decomp = decompose_func_name(f.name)
                    if s_decomp[0] == f_decomp[0] and s_decomp[2] == f_decomp[2]:
                        func_dict[s] = v.determine_derived_function(s.name,
                                                        accept_placeholders=allow_placeholder_functions)
                        break
                else:
                    raise ValueError(f'function {s} could not be established')
        # endregion
        return new_array,func_dict 
    #endregion
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
            Bx,By,Bz = sympy.symbols(__MAGNETIC_FIELD_NAMES__,commutative=False)
            B = sympy.Array([Bx,By,Bz])
            
            Bvec = (None,None,None) if Bvec is None else Bvec
            self.independent_vars['function_dict'].update({B[i]:Bvec[i] for i in range(3)})
        
            #cast as Functions
            for b in __MAGNETIC_FIELD_NAMES__: 
                if not isinstance(self.independent_vars['function_dict'][sympy.symbols(b,commutative=False)],funcs.Function):
                    f = self.independent_vars['function_dict'][sympy.symbols(b,commutative=False)] 
                    if f is None:
                        self.independent_vars['function_dict'][sympy.symbols(b,commutative=False)] = PlaceHolderFunction(b)
                    else:
                        self.independent_vars['function_dict'][sympy.symbols(b,commutative=False)] = NumericalFunction(f,sympy.symbols(b,commutative=False),[0,1,2]) if isinstance(f,Callable) else SymbolicFunction(sympy.sympify(f),sympy.symbols(b,commutative=False))
                    
        
        
        
        
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
    __post_processor_ordering__ = ('crystal rotation',) + BandModel.__post_processor_ordering__ # crystal rotation is the first thing that happends
    
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
        It is important that this prosprocessing step occurs beofre anything that alters the LK hamiltonian!
        
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
        
        self.independent_vars['postprocessing_function_specification']['crystal rotation'] = self.rotate_crystal
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
        Bsym = sympy.symbols(__MAGNETIC_FIELD_NAMES__,commutative=False)
        Bx = None if B is None else B[0]
        By = None if B is None else B[1]
        Bz = None if B is None else B[2]
        
        Bfuncs = []
        for i,b in enumerate((Bx,By,Bz)):
            
            if not isinstance(b,Function):
                if b is None:
                    Bfuncs.append(PlaceHolderFunction(Bsym[i]))
                elif isinstance(b,Callable):
                    Bfuncs.append(NumericalFunction(b,Bsym[i],None))
                else:
                    Bfuncs.append(SymbolicFunction(b,Bsym[i]))
            else:
                Bfuncs.append(b)
        
        
        kappa_sym,q_sym = sympy.symbols(r'kappa,q')
        self.independent_vars['function_dict'].update({Bsym[i]:b for i,b in enumerate((Bfuncs))})

        params = self.independent_vars['parameter_dict']
        if kappa_sym in params:
            if kappa is not None:
                params[kappa_sym] = kappa
        else:
                params[kappa_sym] = kappa
            
        if q_sym in params:
            if q is not None:
                params[q_sym] = q
        else:
                params[q_sym] = q
            
        
        Zeeman_term = sympy.Matrix([[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0]])
        for i in range(3):
            Zeeman_term = Zeeman_term + 2*mu_B*(kappa_sym*J[i]+q_sym*(J[i]@J[i]@J[i]))*Bsym[i] # plus sign in front because our LK hamiltonian as a global - to make dispersion look particle like 
        
        Zeeman_term = sympy.Array(Zeeman_term)
        self.independent_vars['preprocessed_array'] = self.independent_vars['preprocessed_array']+ Zeeman_term 
        
        return self
    
    def add_strain(self,epsilon,spatial_dim = 3,Dd=None,Du=None,Du_prime=None,C4=None,C5_prime=None):
        """
        Add strain to the model. The strain tensor epsilon must be passed as a list containing the strian components in the order (xx,yy,zz,xy,xz,yx)
        """
        cs = sympy.symbols(r"D_{d},D_{u} D'_{u},C_{4},C'_{5}")
        es= sympy.symbols(__STRAIN_NAMES__,commutative=False) # ordered as xx,xy,xz,yy,yz,zz -> 0,3,4,1,5,2
        
        # epsilon as a symmetric matrix
        e = sympy.Array([[es[0],es[1],es[2]],[es[1],es[3],es[4]],[es[2],es[4],es[5]]])
        
                
        if isinstance(epsilon,(tuple,list)) and len(epsilon) != 6:
            raise ValueError('strain tensor must either be passed as an array or a list containing the tensor elements in the forllowing order ´00,11,22,01,02,12´')

        from . import SP_ANGULAR_MOMENTUM

        J = SP_ANGULAR_MOMENTUM['3/2']
        Jsq =[J[0]@J[0],J[1]@J[1],J[2]@J[2]]
        JJ = Jsq[0]+Jsq[1]+Jsq[2]
        K = self.momentum_symbols
        
        strain_term = cs[0]*(e[0,0]+e[1,1]+e[2,2])*sympy.eye(4).as_mutable() # Tr(\epsilon) term
        
        
        for i in range(3):
            j = (i+1)%3
            k = (j+1)%3
            strain_term = strain_term + sympy.sympify(2)/3*cs[1]*(Jsq[i]-1/sympy.sympify(3)*JJ)*e[i,i]+2/(sympy.sympify(3))*cs[2]*((J[i]@J[j]+J[j]@J[i])*e[i,j])
            strain_term = strain_term + (cs[3]*(e[j,j]-e[k,k])*K[i]+cs[4]*(e[i,j]*K[j]-e[i,k]*K[j]))*J[i]
        
        self.independent_vars['preprocessed_array'] = self.independent_vars['preprocessed_array'] + sympy.Array(strain_term)#FIXME  # minus sign because we have a global minus sign in the LK ham from mkaing spectrum particle like

        esym_reorder = [es[i] for i in (0,3,5,1,2,4)]

        #convert epsilons expressions to symbols:
        eps_functions= []
        for esym,epsilon_expr in zip(esym_reorder,epsilon):
            if isinstance(epsilon_expr,Function):
                func = epsilon_expr
            elif isinstance(epsilon_expr,Callable):
                func = NumericalFunction(epsilon_expr,esym,spatial_dependencies=list(range(spatial_dim)))
            else:
                func = SymbolicFunction(epsilon_expr,esym)

            self.independent_vars['function_dict'][esym] = func





    
    #region post-processing_functions
    @staticmethod
    def rotate_crystal(array,funcs,model):
        """
        Rotates the crystal
        :param array: Array to transform.
        :type array: sympy.Array
        :param model: uses the parameters and constants of this model
        :type model: BandModel
        """
        
        theta,phi = sympy.symbols(r'\theta,\phi')
        rotated_H_LK = model.__make_LK_Hamiltonian__(theta,phi) # add rotation verison with symbolic angles!
        regular_H_LK = model.__make_LK_Hamiltonian__()
        array = array - regular_H_LK + rotated_H_LK
        return array,funcs
    
    #endregion       





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






#region band sorting
def covariant_sorting(results,S=None,return_sorting_index = False):
    # if the eigenvecs is an array of shape [n,m,...] then this functions returns an integer array of shape [n,m]
    # which sorts the eigenvectors such that eigenvector of all eigenvectors in row (i-1), eigenvector [i,j] has the most overlap with eigenvector [i-1,j] 
    
    
    
    eigenvals,eigenvecs = results
    n,m = eigenvecs.shape[0],eigenvecs.shape[1]
    einsum_idx = list(range(2,len(eigenvecs.shape))) # [0,1] reserved for indxing overlaps array, and we only need n_shape -2 indices to index the relevant entries 
    
    def find_overlaps(vecs_0,vecs_1):
        if S is None:
            norms_0 = np.linalg.norm(vecs_0.reshape(vecs_0.shape[0],-1),axis=1)
            norms_1 = np.linalg.norm(vecs_1.reshape(vecs_1.shape[0],-1),axis=1)
            return np.abs(np.einsum(vecs_0,[0]+einsum_idx, 1/norms_0,[0] ,vecs_1.conj(),[1]+einsum_idx, 1/norms_1,[1],[0,1]))#     lv /norm(lv) * rv /norm(rv)
        else:
            return np.abs(S.mel(vecs_0,vecs_1))
    
    I = np.zeros((n,m))
    # sort the first according to the eigenvals. shift by its index to the how much to 'move' the element instead of where to move it
    I[0] = np.argsort(eigenvals[0]) - np.arange(m)
    previous_set = eigenvecs[0]
    
    for i,evec_set in enumerate(eigenvecs[1:],1): #skip first
        # compute overlap between every evec in the set and eigenvecs in previous set:
        overlaps = find_overlaps(evec_set,previous_set) # output is m x m
        l = np.argmax(overlaps,axis=1) # find which of the previous ones is best matching 
        # shift l by index to get the relative difference between an elemes index in this set and an elements index in the previous set
        # ad to it the relative shift of the previous set to get the relative shift wrt the array in 0
        I[i] = l-np.arange(m) + I[i-1][np.argsort(l)] 
        
        previous_set = evec_set
        # convert from relative index to absolute. First we shift from relative movement to absolute movement (shift and modulus) 
        # and then we convert from the elements x=I[i,j] that tell that evec [i,j] goes to x, to element I[i,j] showing which evec[i,x] we want
        
    I=    np.argsort(np.mod(np.mod(I,3)+np.arange(m),3)).astype(int)

    
    if return_sorting_index:
        return I        
    
    sorted_evals = np.take_along_axis(eigenvals,I,axis=1)
    sorted_evecs = np.stack([a[i] for a,i in zip(eigenvecs,I)])
    
    
    
    return sorted_evals,sorted_evecs
        
#endregion

