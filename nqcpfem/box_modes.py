import numpy as np
import scipy as sp
import itertools

from .updatable_object import auto_update
from .envelope_function import EnvelopeFunctionModel
from typing import Union
from . import _hbar
from functools import partial
import sympy
import .functions


class BoxEFM(EnvelopeFunctionModel):
    def __init__(self, band_model, domain, nx, ny, nz):
        """
        Box Modes for the EFM
        :param band_model:
        :param domain:
        """
        self.n_modes = [nx, ny, nz]
        from envelope_function import RectangleDomain
        if not isinstance(domain,RectangleDomain):
            raise TypeError(f'unsupported domain type. Only `RectangleDomain` is supported, got: {domain} ({type(domain)})')
        super().__init__(band_model, domain,**{'nx':nx,'ny':ny,'nz':nz})

    @auto_update
    def potential_matrix(self, sparse=True):
        """
        Computes the matrix representation of the all potential functions, i.e. all the parameters of the band_model that have `(x)` in their name:
        :param bool sparse: whether the return value should be dense or not
        :return:
        """
        V_mat = np.zeros(
            (self.n_modes[0], self.n_modes[0], self.n_modes[1], self.n_modes[1], self.n_modes[2], self.n_modes[2]),
            dtype='complex')
        # add the analytical potentials:
        for sym,func in ((s,v) for s,v in self.band_model.independent_vars['parameter_dict'].items() if s.name[-3:] == '(x)'):
            # check the the potential function is a sympy expression and if so, if it is of one of the easier forms.
            try:
                symbolic = func(*self.band_model.position_symbols)
            except Exception:
                symbolic = None
    
            if symbolic is None:
                # we have to integrate numerically.
                raise NotImplementedError('todo: Numerical integration using scipy')
            else:
                # try to see if the function is analytically computable:
                # check order of the dict.
                from sympy.polytools import degree_list
                
                symbolic = sympy.sympify(symbolic).expand()
                terms = symbolic.args if isinstance(symbolic,sympy.core.add.Add) else (symbolic,)
                function_coeffients = {}
                for term in terms: 
                    try:
                        orders = degree_list(term,self.band_model.position_symbols)
                    except TypeError as e:
                        continue 
                    arr = function_coefficients.get(sum(orders),np.zeros(self.tensor_shape+(self.spatial_dim,)*(sum(orders)),'O'))
                    # k-s commute, and by convention we will always order them as k_x k_x ... k_x k_y k_y ... k_y k_z k_z ... k_z
                    arr_i = np.unravel_index(i,self.tensor_shape)+(0,)*orders[0]+(1,)*orders[1]+(2,)*orders[2]
                    addition =   sympy.lambdify(commuting_momentum_symbols,term)(1,1,1) if any(orders) else term # this could give an error if ks and functions V(x) are mixed!
                    arr[arr_i] += addition

                    # add coefficient to array
                    disassemble_dict[sum(orders)] = arr
            
        
        box_dims = [getattr(self.domain, l) for l in ['Lx', 'Ly', 'Lz']]
        var = tuple(sympy.symbols('x,y,z'))

        for P in self.potentials:
            for dir_tup, funcs in P.get_separable_terms().items():
                if len(dir_tup) == 0:
                    # constant terms go here:
                    V = np.complex128(sum((f(None) for f in funcs)))
                    for d in self.n_modes:
                        V = np.tensordot(V, np.eye(d, dtype='complex'), axes=0)
                    V_mat += V
                    continue
                # integration variables:
                # integration parameters and their ranges
                integ_bounds = tuple([(var[i], -box_dims[i] / 2, box_dims[i] / 2) for i in dir_tup])
                term_shape = [self.n_modes[i] for i in dir_tup for _ in range(2)]  # (N,N,M,M, ...)
                dirs = [d for d in dir_tup for _ in range(2)]  # (x,x,y,y,z,z)
                V_term = np.zeros(term_shape, dtype='complex')
                # loop over V to fill in terms:
                iterlist = [range(d) for d in term_shape]
                for index_tup in itertools.product(*iterlist):
                    # determine function as product of V function,
                    basis_function_term = 1
                    for dir, index in zip(dirs, index_tup):
                        # we index in the code with zero, so the first basis mode has n=1
                        basis_function_term *= self.__basis_function_factory__(index + 1, dir)(var[dir])
                    from typing import Callable
                    functions = [f(*[t[0] for t in integ_bounds]) if isinstance(f, Callable) else f for f in funcs]

                    term = sum([f * basis_function_term for f in functions])
                    if term == 0:
                        res = 0
                    else:
                        res = sympy.Integral(term, *integ_bounds).evalf(verbose=True)
                    # res = 0 if np.isclose(res,0) else res
                    V_term[index_tup] += res

                # extend V_term over missing directions:
                missing = []
                for i in range(3):
                    if i not in dir_tup:
                        missing.append(i)
                        V_term = np.tensordot(V_term, np.eye(self.n_modes[i], dtype='complex'), axes=0)
                # reorder
                ordering = list(dir_tup) + missing
                transposition = [2 * ordering.index(i) + j for i in range(3) for j in
                                 range(2)]  # j loop is because each direction has two axes
                V_mat += V_term.transpose(transposition)
        return V_mat

    def __compute_k_matrix__(self, coord_dir, order=1, sparse=True):
        if order > 2:
            raise NotImplementedError(f'fix this')  # todo
        """
        Compute
        :param bool sparse: whether the return array should be dense or not
        :return:
        """
        if not sparse:
            n_modes = self.n_modes[coord_dir]  # number of modes in the directon
            coord_name = ['Lx', 'Ly', 'Lz']
            L = getattr(self.domain, coord_name[coord_dir])  # box dimension

            return __k_mat__(order,L,n_modes)
            def k_mel(n, m):  # matrix elements of the momentum operator p^order = (-i\hbar \del)^order
                # indexing with 0 so we add one here to
                n += 1
                m += 1
                if order % 2:
                    # odd case
                    with np.errstate(divide='ignore', invalid='ignore'):  # suppress divide by zero warnings
                        re = (2 * n) * ((-1) ** (n + m) - 1) / (np.pi * (m ** 2 - n ** 2)) * (
                                    -1j * m * np.pi / L) ** order
                    return np.where(n != m, re, 0)

                else:
                    # even case
                    return np.where(n == m, (n * np.pi / L) ** order, 0)

            def p_mat(order):  # assemble momentum matrix
                if order == 0:
                    return np.eye(n_modes, dtype='complex')
                P = np.fromfunction(k_mel, (n_modes, n_modes))

                return P

            return p_mat(order)

    def positional_rep(self, vector, x_vals=None, as_vector=True):

        if list(vector.shape) != self.__eigen_tensor_shape__:
            try:
                vector = self.eigensolutions_to_eigentensors(vector)
            except ValueError as err:
                raise ValueError(f'eigensolution passed to could not be broadcast to a correct shape:', err)

        if x_vals is None:
            # just reshape solutions so that last axis is what desribes the positional part of the solution
            return vector.reshape(vector.shape[:-3]+(np.prod(vector.shape[-3:]),)),None
        elif isinstance(x_vals, int):
            n_points = [x_vals] * 3
            box_shape = [self.domain.Lx, self.domain.Ly, self.domain.Lz]
            X = [np.linspace(-L / 2, L / 2, N) for L, N in zip(box_shape, n_points)]
            x_vals = np.stack(X, axis=1)


        # construct lists of sin_(i,n)(x) for all x points and directions, i, and modes n
        basis_evals = []
        for i, n in enumerate(self.n_modes):
            mode_wise_eval = []
            for m in range(n):
                basis_func = self.__basis_function_factory__(m + 1, i,
                                                             use_sympy=False)  # indexing with 0 so frst mode has n=1
                mode_wise_eval.append(basis_func(x_vals[:, i]))
            basis_evals.append(np.stack(mode_wise_eval, axis=0))

        basis = basis_evals[0]
        for b in basis_evals[1:]:
            basis = np.tensordot(basis, b, axes=0)
        # transpose to get shape into (n,m,l,x,y,z) (we currently have (n,x,m,y,l,z))
        basis = basis.transpose([0, 2, 4, 1, 3, 5])

        vector_is = list(range(len(vector.shape)))
        basis_is = vector_is[-3:] + list(range(vector_is[-1] + 1, vector_is[-1] + 4))
        if not as_vector:
            return np.einsum(vector, vector_is, basis, basis_is), x_vals
        # flatten the array to be just a vector:
        tensor = np.einsum(vector, vector_is, basis, basis_is)
        vector = tensor.reshape(tensor.shape[:-3] + (-1,))
        # construct x to also be like this
        pos = np.stack(np.meshgrid(x_vals[:, 0], x_vals[:, 1], x_vals[:, 2]), axis=0)
        pos = pos.reshape((pos.shape[0], -1)).transpose()
        return vector, pos

    @property
    def __eigen_tensor_shape__(self):
        return list(self.band_model.tensor_shape[::2]) + list(self.n_modes)

    @auto_update
    def assemble_array(self, sparse=False):
        # replace Ks with matrices. For this we extract As a tensor_dict.
        
        decomposition_dict = self.band_model.symbolic_tensor_repr()
        # Only order 0 can have X,Y,Z dependence for this to work.

        if 0 in decomposition_dict:
            pass
        # extend V to act diagonally on the tensor space
        tensor_shape = self.band_model.tensor_shape
        tensor_id = np.eye(tensor_shape[0], dtype='complex')  # assumes grouped ordering
        for d in tensor_shape[2::2]:
            tensor_id = np.tensordot(tensor_id, np.eye(d, dtype='complex'), axes=0)
        V = np.tensordot(tensor_id, V, axes=0)

        # add momentum parts to array
        tensor_slice = tuple([slice(0, t, 1) for t in tensor_shape])
        tensors = self.band_model.tensors.copy()
        for ten in tensors:

            order = ten.spatial_rank
            for index_t in itertools.product(*([range(self.band_model.spatial_dim)] * order)):
                addition = ten[tensor_slice + index_t]  # 'factor' which is in front of the matrix
                # multiply onto the matrices:
                for d in range(self.band_model.spatial_dim):  # loop in this order to get (ten_shape,x_dim,x_dim,y_dim,y_dim,z_dim,z_dim)
                    p_order = index_t.count(d)
                    addition = np.tensordot(addition, self.__compute_k_matrix__(d, p_order, sparse=sparse), axes=0)
                V += addition
        # cast V as matrix
        V_mat = grouped_to_canonical(V)
        vector_dim = np.prod(V_mat.shape[:int(len(V_mat.shape) / 2)])
        V_mat = V_mat.reshape((vector_dim, vector_dim))

        self._array = V_mat

    def eigensolutions_to_eigentensors(self, eigensolutions):
        # transpose to get eigensolutions as listed along first axis:
        # reshape second axis

        if not eigensolutions.shape[0] == np.prod(self.__eigen_tensor_shape__):
            raise ValueError(
                f'shape of eigenvector did not match expected shape: expected {np.prod(self.__eigen_tensor_shape__)}'
                f', got {eigensolutions.shape[0]}')
        if len(eigensolutions.shape) > 1:
            eigs = eigensolutions.T
            eigs = eigs.reshape([eigs.shape[0], ] + self.__eigen_tensor_shape__)
            return eigs
        else:
            eigs = eigensolutions.reshape(self.__eigen_tensor_shape__)  # only one eigenvector passed
            return eigs

    def make_S_array(self):
        return 1  # identity as modes are orthogonal

def __k_mat__(order, L, n_modes):
    """
    compute matrix version of k operator for box modes
    :param order:
    :param L:
    :param n_modes:
    :return:
    """
    if order > 2:
        raise NotImplementedError(f'fix this')  # todo


    def k_mel(n, m):
        # matrix elements of the momentum operator p^order = (-i\hbar \del)^order
        # indexing with 0 so we add one here to
        n += 1
        m += 1
        if order % 2:
            # odd case
            with np.errstate(divide='ignore', invalid='ignore'):  # suppress divide by zero warnings
                re = n * ((-1) ** (n + m) - 1) / (np.pi * (m ** 2 - n ** 2)) * (-1j * m * np.pi / L) ** order
            return np.where(n != m, re, 0)

        else:
            # even case
            return np.where(n == m, (n * np.pi / L) ** order, 0)

    def p_mat(order):  # assemble momentum matrix
        if order == 0:
            return np.eye(n_modes, dtype='complex')
        P = np.fromfunction(k_mel, (n_modes, n_modes))

        return P
    return p_mat(order)

def __basis_function_factory__(L,use_sympy=True):
    """
    The function `__basis_function_factory__` returns a factory function that generates basis functions
    for a given value of `n`.
    
    :param L: L is the length of the interval over which the basis functions are defined. If sympy is used L can be a symbol.
    :param use_sympy: The parameter "use_sympy" is a boolean flag that determines whether the code
    should use the sympy library or the numpy library for mathematical operations. If "use_sympy" is set
    to True, the code will use sympy for mathematical operations. If it is set to False, Numpy is used. Defaults to
    True (optional)
    :return: The function `basis_func_factory` is being returned. This function takes an integer and returns the corresponding a function which respresents the basis function. 
    """
    if use_sympy:
        sin = sympy.sin
        pi = sympy.pi
        sqrt = sympy.sqrt
    else:
        sin = np.sin
        pi = np.pi
        sqrt = np.sqrt

    def basis_func_factory(n):
        def basis_func(x):
            return sqrt(2 / L) * sin(n * pi * (x / L + 1 / 2))

    return basis_func_factory

def box_mode_projection(array,directions,functions_dict,Nmodes,):
    pass
    
    # projects a band_model down to a basis of box-modes for the specified directions. Functions which are specifed in the functions dict are also altered and returned as a new dict of the projected functions
    
    
    
    """
    
        functions_dict = {}
    else:
        pass

    for_numerical_eval = {}
    bipartition_types = {}
    polynomial_types = {}
    for sym,func in functions_dict.items():
        
        # check that functions are either Bipartition (and constant on the partitions) or polynomial.
        # evaluate everything else numerically and store 
        if isinstance(func,sympy.sympy.Piecewise):
            for arg in func.args:
                for s in (s for s in arg.free_symbols if s in position_symbols):
                    if arg[1] in ((s>0),(s>=0)):
                        direction = s
                        coefficient  = arg[0]
                        break
            else:
                for_numerical_eval[sym] = func # this piecewise did not have the correct shape

            if (func/coefficient).simplify() != sympy.sympy.Piecewise((1,direction>0),(0,True)):
                for numerical_eval[sym] = func # this piecewise did not have the correct shape

            else:
                
                
    return 

    # find all functions and make the position_dependence_explicit
    # for the functions where this is not possible, do it numerically -> this will give an array.
    # replace position symbols with the corresponding matrix
    
    # split up into dict dependening on degree of x,y,z (or arbitrary position syms)
    # use (None,None,NOne) for the ones where it isnt possible
    # evalue imposible numericcaly.
    # combine array.
    
def evaluate_numerically(func,position_symbol,N_modes):
    # COmpute matrix expresion of a function numerically ()
    return 
    """