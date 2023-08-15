from typing import Callable,Sequence,Iterable,Tuple
import sympy
from .symbolic import __POSITION_NAMES__,derivative_of_function,sort_out_polynomials,extract_valid_bipartition_part,symbolize_array
import logging
import numpy as np
from functools import lru_cache
LOGGER = logging.getLogger(__name__)
LOGGER.addHandler(logging.NullHandler())
from abc import ABC,abstractmethod


# NB! These symbols are commuting since we are not dealing wiht k-operators at all, and sympy.Piecewise does not work with non-commuting symbols
X,Y,Z = sympy.symbols(__POSITION_NAMES__)
class Function(ABC):
    """A class representing a function. This function specifies the available methods for working with functions.
    """
    def __init__(self,symbol,spatial_dependencies,is_constant=False,**kwargs):
        self.spatial_dependencies = spatial_dependencies # array of ints representing the spatial dependencies
        self.is_constant = is_constant
        self.projection_spec = kwargs.get('projection_spec',None) # spec how function was projected if it is a projected function
        if isinstance(symbol,str):
            symbol = sympy.Symbol(symbol,commutative=False)
        else:
            if symbol.is_commutative or symbol.name[-3:] != '(x)': 
                raise ValueError(f'function symbols should end with (x) and be non-commutative')
            
        self.symbol = symbol # the symbol representing the function. Must end on (x)
        self._derivatives_ = {}
        super(Function,self).__init__()
    @abstractmethod
    def __call__(self,x,y=None,z=None):
        """Functions should be callable. The call signature should take as many arguments as len(self.spatial_dependencies).
        They should return the value of the function at that position
        :param x: x-position
        :type x: Any
        :param y: y-position
        :type y: Any
        :param z: z-position
        :type z: Any
        """
        raise NotImplementedError()
    
    def derivative(self,directions):
        """Returns a new Function describing the derivative of this function

        :param dir: integer representing the direction: 0,1 or 2
        :type dir: int
        :param order: the order of the derivate operatorion
        :type order: int
        """
        if isinstance(directions,int):
            directions = [directions,]
        derivative_name = self.symbol
        for d in directions:
            sym = (X,Y,Z)[d]
            derivative_name = derivative_of_function(derivative_name,sym)
        if derivative_name not in self._derivatives_:
            self._derivatives_[derivative_name] = self.__compute_derivative__(derivative_name,directions)
        
        return self._derivatives_[derivative_name]
        
    @abstractmethod
    def __compute_derivative__(self,name,directions):
        """In order to permute K operators with the function we must be able to take derivatives of the function.
        This should return another Function with symbol being equal to `name` 

        """
        raise NotImplementedError()
    
    @staticmethod
    def __projection_arg_preprocess__(directions,n_modes):
        # sort the arguments
        if isinstance(directions,int):
            directions = [directions]
        sorted_i = np.argsort(directions)
        if isinstance(n_modes,int):
            n_modes = [n_modes]
        n_modes = [n_modes[i] for i in sorted_i]
        return directions,n_modes
    
    @abstractmethod
    def project_to_basis(self,directions,n_modes,type='box',**kwargs):
        """ # computes the matrix elements <n|F|m> for the function F (self) and some basis states |n>, |m> which are classified by `type`


        :param directions: which direction(s) to 'integrate out'
        :type directions: int|Iterable[int]
        :param n_modes: number of modes of the basis to include
        :type n_modes: int
        :param type: which kind of eigenstates to used, defaults to 'box', meaning eigenstates of a particle in a box system.
        :type type: str, optional
        
        returns a Sympy Array containing explicit parameters as well as new functions labeled as f'{self.symbol.name[:-3]}'+"_{i}(x)" for i an integer.
        The corresponding functions are also returned in the form of a dict mapping symbols to Functions.+
        """
        #This the return value should only have to be computed once so use the auto_update feature! this also ensures that we get the same Function instances every time.
        # The return value must be an array contianing floats and symbols representing new functions: `(`symbol_name`)_{nm}(x)`  as well as a dict of the corresponding functions (keys are the symbols)
        raise NotImplementedError()
    
    
    @abstractmethod
    def pow(self,order):
        """from f(x) Comute a new function g(x) = (f(x))**order as a new function. The symbol of the new function should be f^order(x) with f being the base name """
        raise NotImplementedError
    
    def determine_derived_function(self,other_function_name):
        
        
        my_name,my_derivs,my_proj = decompose_func_name(self.symbol.name)
        
        other_name,other_derivs,other_proj = decompose_func_name(other_function_name)
        if my_name != other_name:
            raise ValueError(f'unable to determine funcitonal expression of {other_function_name} from {self.symbol.name} (wrong base function)')

        # assert projections are OK
        if my_proj is not None:
            if my_proj != other_proj:
                raise ValueError(f'unable to determine funcitonal expression of {other_function_name} from {self.symbol.name} (wrong projection component)')
        elif other_proj is not None:
            # determine which projection we want
            raise NotImplemnetedError(' TODO FIx this. It should be the responsibility of the band_model to commute Ks!')
            #new_projection = other_proj

        if other_derivs is None:
                other_derivs = tuple()
        if my_derivs is not None:
            new_derivs = [other_derivs.count(i) - my_derivs.count(i) for i in range(3)]
            if any(n < 0 for n in new_derivs):
                raise ValueError(f'unable to determine funcitonal expression of {other_function_name} from {self.symbol.name} (wrong non-compatible derivatives)')
        else:
            new_derivs = other_derivs
            
        # take new derivatives of the function:
        new_func = self
        for n in new_derivs:
            new_func = new_func.derivative(n)
        
        return new_func
                    
                
    def __hash__(self):
        return self.symbol.__hash__() # one-to-one correspondence between function and symbol!    
        
        
        
    
class SymbolicFunction(Function):
    def __init__(self,expression,symbol,**kwargs):
        expression = sympy.sympify(expression)
        if isinstance(symbol,str):
            symbol = sympy.Symbol(symbol,commutative=False)
        if symbol.name[-3:] != '(x)':
            raise ValueError(f'symbol representing functio MUST end with `(x)`. Recieved {symbol} with name: {symbol.name}')
        spatial_dependencies = [i for i in range(3) if not expression.is_constant((X,Y,Z)[i])]
        Function.__init__(self,symbol,spatial_dependencies,bool(len(spatial_dependencies)),**kwargs)
        self._expression_ = expression
    
    @property
    def expression(self):
        return self._expression_
    
    def __call__(self,x,y=None,z=None):
        if isinstance(x,np.ndarray) and y is None and z is None:
            n_cords = x.shape[0]
            call_tuple = (x[0],)
            for i in range(1,n_cords):
                call_tuple = call_tuple+(x[i],)
            
            return sympy.lambdify((X,Y,Z)[:n_cords],self.expression)(*call_tuple)
        
        return self.expression.subs({X:x,Y:y,Z:z})
    
    def __compute_derivative__(self,name,directions,):
        res = self.expression
        for d in directions:
            sym = (X,Y,Z)[d]
            res = sympy.diff(res,sym)
        
        return SymbolicFunction(res,name)
    
    @lru_cache(32)
    def project_to_basis(self,directions:int|Tuple[int], n_modes:int|Tuple[int], type='box',**kwargs):
        """ 
        Structure of expressions: 
        IF the expression is a piecewise function, it will be instance of Piecewise.
        We therefore split it up into valid piecewise parts and the remainder.
        If remainder it is a sum of terms we handle each term individually.
        for the individual terms in remainder we check that it is a polynomial in the relevant direcitons. If not, we compute numerically,
        else, we compute analytically
        """
        
        
        # check that the expression is One of the analytically solved types. else we have to compute it numerically
        directions,n_modes = self.__projection_arg_preprocess__(directions,n_modes)

        if all(d not in self.spatial_dependencies for d in directions):
            # just return identity array
            return self.symbol*sympy.tensor.tensorproduct(*(sympy.Array(sympy.Matrix.eye(n)) for n in n_modes)) , {self.symbol:self}
        relevant_position_syms = [(X,Y,Z)[i] for i in sorted(directions)] # we always do X,Y,Z
        
        if isinstance(self._expression_,sympy.Piecewise):
            #extract the valid piecewise components and constants.
            piecewise_parts,remains = extract_valid_bipartition_part(self._expression_)      
            
        else:
            piecewise_parts = None
            remains = self._expression_


            polynomials = sort_out_polynomials(remains,relevant_position_syms) if remains is not None else []
        
        # needs fixing?
        if type =='box':
            if not 'L' in kwargs:
                raise SyntaxError(f'projection onto box modes requires passing the `L` kwarg ')
            L = kwargs['L']
            if not isinstance(L,Iterable):
                L = (L,)
            
            
            from .box_modes import __basis_function_factory__
            
            index_map = {X:0,Y:1,Z:2}
            # compute the piecewise matrices

            if piecewise_parts is not None:
                coordinate_wise_matrices = {p:None for p in relevant_position_syms}
                for pw in piecewise_parts:
                    p = list(pw.free_symbols)[0] # this should be either X,Y or Z
                    if p in relevant_position_syms:
                        index = directions.index(index_map[p])
                        coordinate_wise_matrices[p] = box_bipartition_matrix(N_modes=n_modes[index],L=L[index])
                #contruct the matrix to be filled. Must have shape Nmodes[0] x Nmodes [0] x ... (depending on how many modes we are talking about)
                
                for (p,N) in zip(relevant_position_syms,n_modes):
                    if coordinate_wise_matrices[p] is None:
                        # generate identity_matrix as array:
                        coordinate_wise_matrices[p] = sympy.Array(sympy.Matrix.Eye(N))
                        
                
                matrix_form = sympy.tensor.tensorproduct(*(coordinate_wise_matrices[p] for p in relevant_position_syms))
            else:
                # construct zero array of correct shape
                matrix_form = sympy.tensor.tensorproduct(*(sympy.Array(sympy.Matrix.zeros(n,n)) for n in n_modes))
            # Compute polynomials
            
            for poly in polynomials:
                coordinate_wise_matrices = {p:None for p in relevant_position_syms}
                had_numerical=False
                for i,(p,N,l) in enumerate(zip(relevant_position_syms,n_modes,L)):

                    if poly[0][i] is None:
                        had_numerical = True
                        # numerical eval necessary:
                        func_factory = __basis_function_factory__(l,use_sympy=True)
                        coordinate_wise_matrices[p]= numerically_evaluate_symbolic_elements(poly[1],p,(-l,l),func_factory,N,kwargs.get('integration_method','sympy'))
                    else:
                        coordinate_wise_matrices[p]=box_polynomial_matrix(N,l,poly[0][i])
                for (p,N) in zip(relevant_position_syms,n_modes):
                    if coordinate_wise_matrices[p] is None:
                        # generate identity_matrix as array:
                        coordinate_wise_matrices[p] = sympy.Array(sympy.Matrix.eye(N))
                
                addition = sympy.tensor.tensorproduct(*(coordinate_wise_matrices[p] for p in relevant_position_syms))

                if not had_numerical:
                    addition = addition *poly[1]
                
                
                matrix_form = matrix_form +addition
                

        elif type == 'well':
            if not 'sigma' in kwargs:
                raise SyntaxError(f'projection onto box modes requires passing the `L` kwarg ')
            sigma = kwargs['sigma']
            if not isinstance(sigma,Iterable):
                sigma = (sigma,)
            
            
            raise NotImplementedError('make SHO matrix representation')
        else:
            raise SyntaxError(f'invalid type specification: {type}')
        
        """
        # Take every element of the matrix form and it it depends on X,Y or Z, replace it with a function 
        functions = {}
        matrix_elements = []
        pos_set = {X,Y,Z}
        for i,t in enumerate(np.array(matrix_form).ravel()):
            if pos_set.intersection(t.free_symbols):
                name = '('+self.symbol[-3:]+')_{('+''.join(str(s) for s in np.unravel_index(i,matrix_form.shape))+')}'
                func = SymbolicFunction(t,name)
                functions[func.symbol] = func
                matrix_elements.append(func)
            else:
                matrix_elements.append(t)
        """
        symbolic_array,funcs = symbolize_array(sympy.Array(matrix_form),self.symbol)
        projection_spec = {'directions':directions,
                          'n_modes': n_modes,
                          'type':type,
                          '__kwargs__':kwargs}
        func = {f:SymbolicFunction(v,f,projection_spec=projection_spec) for f,v in funcs.items()}
        return symbolic_array,func

    def pow(self,order):
        new_name = pow_name(self.symbol.name,order)
        from copy import copy
        new = copy(self)
        new.symbol = sympy.Symbol(new_name,commutative=False)
        new._expression_ = self.expression**order
        
        return new
        
        
    def __getstate__(self):
        return {'expr':self.expression,'symbol': self.symbol,'projection_spec':self.projection_spec}
    
    def __setstate__(self,state):
        self.__init__(state['expr'],state['symbol'],projection_spec=state['projection_spec'])
    
    def __eq__(self, __value: object) -> bool:
        if not isinstance(__value,type(self)):
            return False
        return self.expression == __value.expression

    def __hash__(self):
        return super().__hash__()
    
# region Analytically computed Matrices

# region box system
@lru_cache(maxsize=8)
def box_bipartition_matrix(L,n_modes):
    """
    COnstructs matrix representation of V(x) = \\chi_{{x>0}} * V where x is the specified direction which lies in the range (-L/2,L/2)
    :param float V: strength of the potential
    :param int n_modes: Number of modes in the direction x
    :return:
    """
    def V_mel(m, n):
        n += 1
        m += 1
        sinpi2 = lambda i: (1 - (-1) ** i) / 2 * (-1) ** ((i - 1) / 2)
        cospi2 = lambda i: sinpi2(i + 1)

        with np.errstate(divide='ignore', invalid='ignore'):
            diagonal = 0.5
            off_diagonal = 2 / (sympy.pi * (m ** 2 - n ** 2)) * (
                    m * cospi2(m) * sinpi2(n) - n * cospi2(n) * sinpi2(m))
    
        return np.where(n == m, diagonal, off_diagonal,)

    V_mat = np.fromfunction(V_mel, (n_modes, n_modes), dtype='O')
    
    return sympy.Array(V_mat)




    raise NotImplementedError('#todo')

@lru_cache(maxsize=8)
def box_well_matrix(n_modes,omega,m,L):
    """
    The function `__make_harmonic_matrix__` calculates a harmonic matrix based on the given parameters
    and returns the result multiplied by 0.5*m*omega^2.
    
    :param omega: The parameter "omega" represents the angular frequency of the harmonic oscillator. It
    can be either a float value or a symbolic variable (e.g., sympy.Symbol)
    :type omega: float|sympy.Symbol
    :param m: mass of the system
    :type m: float|sympy.Symbol
    :param L: The parameter L represents the length of the system
    :type L: float|sympy.Symbol
    :param n_modes: The parameter `n_modes` represents the number of modes in the harmonic matrix
    :type n_modes: int
    :return: a matrix that is calculated based on the input parameters omega, m, L, and n_modes. The
    matrix is multiplied by 0.5*m*omega^2 before being returned.
    """
    
    def V_mel(n, m):
        # indexing with zero requires us to add one to convert to the mode numbers
        n += 1
        m += 1

        with np.errstate(divide='ignore', invalid='ignore'):  # supress divide by zero error
            diagonal = 1 / (12 * np.pi ** 2 * n ** 2) * (np.pi ** 2 * n ** 2 - 6)
            off_diagonal = (1 + (-1) ** (n + m)) * 2/ (np.pi ** 2 * (m ** 2 - n ** 2) ** 2) * n * m

        return np.where(n == m, diagonal, off_diagonal)

    matrix = np.fromfunction(V_mel, (n_modes, n_modes), dtype=float)

    return sympy.Array(matrix) * 0.5*m*omega**2*L**2

    
    
    
@lru_cache(maxsize=8)
def box_linear_matrix(L,n_modes):
    """
    Given a potential of the form V(x) = V*x, with x being the position coordinate ranging in (-L/2,L/2) 
    :param V: 
    :type V: float
    :return:
    """
    def V_mel(m, n):
        # recast m,n from basis function indices starting from 0 to basis function numbering
        m += 1
        n += 1
        with np.errstate(divide='ignore', invalid='ignore'):
            diagonal = 0
            off_diagonal = ((-1) ** (n + m) - 1) * 4 * m * n / (sympy.pi ** 2 * (m ** 2 - n ** 2) ** 2)

        return np.where(n == m, diagonal, off_diagonal)

    V_mat = np.fromfunction(V_mel, (n_modes, n_modes), dtype='O')
    return V_mat * L 

def box_polynomial_matrix(N_modes,L,order):
    if order == 0:
        return sympy.Array(sympy.Matrix.eye(N_modes))
    if order == 1:
        return box_linear_matrix(n_modes=N_modes,L=L)
    elif order == 2:
        return box_well_matrix(n_modes=N_modes,L=L,omega=1,m=2) # omega and m such that we only evalute x**2
    else:
        raise NotImplementedError()
#endregion

#region Well system

# TODO

#endregion

#endregion

# region Numerical evaluation of matrix expressions

def numerically_evaluate_symbolic_elements(expression_sym,expression,var_symbol,integration_range,basis_factory,N_modes,integration_method= 'sympy',**kwargs):
    """
    The function numerically_evaluate_symbolic_elements evaluates symbolic expressions numerically using
    different integration methods.
    
    :param expression: The symbolic expression that you want to numerically evaluate
    :param var_symbol: The variable symbol that represents the independent variable in the expression
    :param integration_range: The integration range is a tuple that specifies the lower and upper limits
    of integration. For example, if integration_range = (0, 1), it means that the integration will be
    performed from 0 to 1
    :param basis_factory: The `basis_factory` is a function that takes an integer `n` as input and
    returns a symbolic expression representing the `n`-th basis function
    :param N_modes: The parameter N_modes represents the number of modes or basis functions used in the
    calculation. It determines the size of the resulting array and the number of iterations in the for
    loop
    :param integration_method: The `integration_method` parameter determines the method used for
    numerical integration. Defaults to sympy (optional)
    :return: a sympy Array object that contains the results of the numerical evaluation of the symbolic
    elements. The results are reshaped into a matrix with dimensions N_modes x N_modes.
    """

    from itertools import product
    
    #TODO: parallelize
    results = []
    for n,m in product(range(N_modes),repeat=2):
        # numerical integration:
        integrand = sympy.conjugate(basis_factory(n,var_symbol))*expression*basis_factory(m,var_symbol)
        
        if integration_method == 'sympy':
            integral = sympy.integrate(integrand,(var_symbol,integration_range[0],integration_range[1]))
            results.append(integral)
        elif integration_method == 'Gauss-Legendre':
            if  integration_range[0] != -integration_range[1]:
                raise NotImplementedError('TODO: shift') 
                
            from sympy.integrals.quadrature import gauss_legendre
            points = gauss_legendre(kwargs.get('n_points',50),kwargs.get('n_digits',16))
            
            # lambdify for quick eval
            try:
                func = sympy.lambdify(var_symbol,integrand)
            except Exception as err:
                func = lambda x: integrand.subs(var_symbol,x)
            integral = sum(w*func(p*integration_range[1]) for p,w in points)
            results.append(integral)
        
    return sympy.Array(results).reshape(N_modes,N_modes)
        
    

def numerically_evaluate_numerical_elements(expression,integration_range,basis_factory,N_modes,integration_method='scipy',**kwargs):
    """
    The function numerically_evaluate_numerical_elements evaluates the numerical elements of an
    expression using integration over a given range and a specified basis.
    
    :param expression: The expression to be numerically evaluated. It can be a mathematical expression
    involving variables and functions
    :param integration_range: The integration range is a tuple that specifies the lower and upper limits
    of integration. For example, if integration_range = (0, 1), the integration will be performed from 0
    to 1
    :param basis_factory: The `basis_factory` is a function that takes an integer `n` as input and
    returns a function that represents the `n`-th basis function
    :param N_modes: The parameter N_modes represents the number of modes or basis functions used in the
    calculation. It determines the size of the resulting matrix
    :param integration_method: The `integration_method` parameter is used to specify the method to be
    used for numerical integration. The default value is 'scipy', which means that the
    `scipy.integrate.quad` function will be used for integration. (optional)
    :return: a numpy array of the results of the numerical evaluation of the expression for each
    combination of n and m in the given range of N_modes. The shape of the array is N_modes x N_modes.
    """
    import numpy as np
    from scipy import integrate
    from itertools import product
    
    #TODO: parallelize
    results = []
    for n,m in product(range(N_modes),repeat=2):
        
        function = lambda x:expression*np.conjugate(basis_factory(n)(x))*basis_factory(m)(x)
        results.append(integrate.quad(function,integration_range[0],integration_range[1])[0])
        
    return sympy.Array(results).reshape(N_modes,N_modes)


def numerically_integrate(func:Callable,integration_range:Sequence,integration_method='scipy'):
    
    from functools import partial
    def nested_integration(f,int_range):
        def integrand(*p):
            return numerically_integrate(partial(f,*p),int_range,integration_method)

        return integrand
    
    ranges = list(integration_range) 
    if isinstance(integration_range[0],Sequence):   
        # we have to integrate coordinate-wise
        this_function = func
        while len(ranges):
            this_range = ranges.pop(-1) # integrate arguments from right to left, so that we can use partial to assign the left-most args to variables
            this_function = nested_integration(this_function,this_range)
        return this_function() # the last function should not need any arguments (all args should have been integrated out)    
                
    if integration_method =='scipy':
        from scipy import integrate
        return integrate.quad(func,integration_range[0],integration_range[1])[0]
    else:
        raise NotImplementedError(f'{integration_method} not implemented.')

def numerically_compute_matrix(integrand_constructor,intergration_range,N_modes):
    
    mode_ranges = []
    for n in N_modes:
        mode_ranges.extend([np.arange(n),np.arange(n)])
    
    mode_grid = np.meshgrid(*mode_ranges)
    
    raise NotImplementedError('TODO: questionable whether this is even numerically feasable (poor scaling)')



#endregion




class NumericalFunction(Function):

    def __init__(self,func,symbol,spatial_dependencies,is_constant=False):
        self._function_ = func
        super(NumericalFunction,self).__init__(symbol,spatial_dependencies,is_constant)


    def __call__(self,x,y=None,z=None):
        args = [(x,y,z)[i] for i in self.spatial_dependencies]
        return self._function_(*args)

    def __compute_derivative__(self, name, directions):
        raise NotImplementedError(f'TODO: numerical differentiation')
        return super().__compute_derivative__(name, directions)
    
    def project_to_basis(self,directions, n_modes, type='box', **kwargs):
        directions,n_modes = self.__projection_arg_preprocess__(directions,n_modes)
        raise NotImplementedError(f'TODO:  numerial projection')
        return super().project_to_basis(n_modes, type, **kwargs)
    
    def pow(self,order):
        def power_func(func,order):
            def power(x,y=None,z=None):
                return func(x,y,z)**order
            return power
        import copy
        new = copy.copy(self)
        new._function_ = power_func(self,order)
        new.symbol = sympy.Symbol(pow_name(self.symbol.name,order),commutative=False)
        return new
        
class PlaceHolderFunction(NumericalFunction):
    
    def __init__(self,symbol,spatial_dependencies=None,is_constant=False):
        # is instance of numerical function to tell that we do not know anythong about how the functions looks
        if spatial_dependencies is None:
            spatial_dependencies = [] if is_constant else [1,2,3]

        super(PlaceHolderFunction,self).__init__(None,symbol,spatial_dependencies,is_constant)
        
    def __call__(self,x,y=None,z=None):
        raise NotImplementedError(f'funciton {self.symbol} was not specified so it is not possible to call it')
    
    def __compute_derivative__(self, name, directions):
        
        new_func = PlaceHolderFunction(name,self.spatial_dependencies,is_constant=self.is_constant)
        return new_func
    
    def project_to_basis(self,directions, n_modes, type='box', **kwargs):

        
        directions,n_modes = self.__projection_arg_preprocess__(directions,n_modes)
        # return full array of new PlaceHolderFunctions
        import re
        def new_name(old_name,i,j):
            # assuming that the name is of the form: (BASE_NMAE)_{abcdefg}(x) or BASE_NAME(x)
            name = re.search(r'\((.*)\)_\{([^\(\)]*)\}\(x\)',old_name)
            if name is None: # no existing sufix
                base_name = old_name[:-3] # drop (x)
                suffix = f'{i}{j}'
            else:
                base_name = name.group(1)
                old_suffix = name.group(2)
                suffix = old_suffix+f'{i}{j}'
            
            updated_name = '('+base_name+')_{'+suffix+'}(x)'
            return updated_name
        
        name_list = [(tuple(),self.symbol.name)]
        
        for n in n_modes:
            new_name_list = [(index+(i,j),new_name(name,i,j)) for index,name in name_list for i in range(n) for j in range(n)]
            
            name_list = new_name_list
        
        
        
        new_spatial_dep = [s for s in self.spatial_dependencies if s not in directions]
        # convert list of indices and names to functions and symbols:
        func_dict = {sympy.Symbol(name,commutative=False): PlaceHolderFunction(sympy.Symbol(name,commutative=False),new_spatial_dep,self.is_constant) for _,name in name_list }
        
        array_shape = sum(([n]*2 for n in n_modes),[])
        new_array = sympy.Array.zeros(*array_shape).as_mutable()
        
        for id,name in name_list:
            new_array[id] = sympy.Symbol(name,commutative=False)
        
        return sympy.Array(new_array),func_dict

    def pow(self,order):
        new_sym = sympy.Symbol(pow_name(self.symbol.name,order),commutative=False)
        
        return PlaceHolderFunction(new_sym,self.spatial_dependencies,self.is_constant)

def decompose_func_name(func_name):
    # decompose into base_name projection stuff and derivatives
    import re
    
    name = re.search(r'\((.*)\)_\{(.*)\}\(x\)',func_name)
    if name is not None:
        func_name = name.group(1)+'(x)'
        # projection coords is tuple of ints
        projection = tuple(int(s) for s in name.group(2))
    else:
        projection = None
    
    
    # remove derivative syntax: FUNC_{ABC(xxyyzz)}(x) -> FUNC_{ABC}(x)
    has_deriv = re.search(r'(.*)_\{(.*)\((.*)\)\}',func_name)
    if has_deriv is not None:
        func_name = has_deriv.group(1)
        if has_deriv.group(2) != '':
            func_name = func_name+'_{'+has_deriv.group(2)+'}'
        
        func_name= func_name + '(x)'  # put (x) back on
        
        # derivatives is tuple of 0,1,2 for x ,y, z
        symbol_map = {'x':0,'y':1,'z':2}
        derivatives = tuple(symbol_map[s] for s in has_deriv.group(3))
    else:
        derivatives = None
    return func_name,derivatives,projection


def assemble_func_name(func_name,derivatives=None,projection=None):
    import re
    # check if the base_func name has subscript:
    has_suffix = re.search('(.*)_\{(.*)\}\(x\)',func_name)
    if has_suffix is not None:
        suffix = has_suffix.group(2)
        func_name = has_suffix.group(1)
    else:
        suffix = ''
        func_name = func_name[:-3] # remove (x)
    
    
    symbol_map = {0:'x',1:'y',2:'z'}
    if derivatives is not None:
        suffix = suffix +'('+ ''.join(symbol_map[s] for s in derivatives)+')'

    new_name = func_name
    if suffix != '':
        new_name = new_name +'_{' + suffix + '}'
    
    if projection is not None:
        new_name = '(' + new_name + ')_{'+''.join(str(s) for s in projection)+'}'
    
    new_name = new_name +'(x)'
    return new_name
    
    
def extract_base_function_name(func_name,skip_derivatives=False):


    import re
    # remove projection syntax: (FUNC_{ABC})_(1234...)(x) -> FUNC_ABC(x)
    
    name = re.search('\((.*)\)_\{.*\}(x)',func_name)
    if name is not None:
        func_name = name.group(1)+'(x)'
    
    
    if not skip_derivatives:
        # remove derivative syntax: FUNC_{ABC(xxyyzz)}(x) -> FUNC_{ABC}(x)
        has_deriv = re.search(r'(.*)_\{(.*)\(.*\)\}',func_name)
        if has_deriv is not None:
            func_name = has_deriv.group(1)+'_{'+has_deriv.group(2)+'}'+'(x)'
    
    return func_name

def pow_name(func_name,order):
    f_base,f_deriv,f_proj = decompose_func_name(func_name)
    import re
    subscript = re.search(r'(.*)_\{(.*)\}\(x\)',f_base)
    if subscript is not None:
        new_name = subscript.group(1)+ f'^{order}' + '_{' + subscript.group(2) + '}(x)'
    else:
        new_name = f_base[:-3] + f'^{order}(x)'
    return assemble_func_name(new_name,f_deriv,f_proj)
    