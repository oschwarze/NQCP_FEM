import numpy as np
import scipy as sp
import itertools

from .updatable_object import auto_update
from .envelope_function import EnvelopeFunctionModel,RectangleDomain
from typing import Union
from . import _hbar
from functools import partial
import sympy
from . import functions
from . import band_model as bm
from . import symbolic as symb


class BoxEFM(EnvelopeFunctionModel):
    
    """
    Evaluating a band_model in terms of eigenstates of a particle in a box. Since for this basis, the basis is separable,
    we represent the basis states by a tuple (nx,ny,nz). Therefore the assembled array will have shape
    (Tn,nx,ny,nz,Tm,mx,my,mz) and the eigenstates will be tensors of shpae (Tn,nx,ny,nz) where Tn are superindices representing the tensor shape of the band_model
    """
    
    def __init__(self, band_model, domain, nx, ny, nz):
        """
        Box Modes for the EFM
        :param band_model:
        :param domain:
        """
        self.n_modes = [nx, ny, nz]
        if not isinstance(domain,RectangleDomain):
            raise TypeError(f'unsupported domain type. Only `RectangleDomain` is supported, got: {domain} ({type(domain)})')
        super().__init__(band_model, domain,**{'nx':nx,'ny':ny,'nz':nz})
    
    @property
    def k_signature(self) -> str:
        return 'all right'
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
            box_shape = [self.independent_vars['domain'].Lx, self.independent_vars['domain'].Ly, self.independent_vars['domain'].Lz]
            X = [np.linspace(-L / 2, L / 2, N) for L, N in zip(box_shape, n_points)]
            x_vals = np.stack(X, axis=1)


        # construct lists of sin_(i,n)(x) for all x points and directions, i, and modes n
        basis_evals = []
        L = [getattr(self.independent_vars['domain'],name) for name in ('Lx','Ly','Lz')]
        for i, n in enumerate(self.n_modes):
            mode_wise_eval = []
            func_factory = __basis_function_factory__(L[i],use_sympy=False)
            for m in range(n):
                basis_func = func_factory(m+1)
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
        
        # Take post-processed Ham and cast as right-aligned
        
        # unravel array and sort the terms into analytically computable terms (involving only Ks, or involving x,x**2 or Piecewise(1,x>0, etc.) (and relevant combinations thereof))
        
        H = symb.arange_ks_array(sympy.Array(self.band_model.numerical_array()),'all right')
        # substituate symbolic functions with their corresponding expressions
        H = H.subs({f.symbol:f.expression for f in symb.present_functions(H) if isinstance(f,functions.SymbolicFunction)})

        # need these for checking polynmial order
        kx,ky,kz = sympy.symbols(symb.__MOMENTUM_NAMES__,commutative=True)
        x,y,z = sympy.symbols(symb.__POSITION_NAMES__,commutative=True)
        Vs = tuple(f for f in symb.present_functions(H) if isinstance(f,functions.NumericalFunction))
        
        # substiute Ks and xs for commuting ones now since they we have the order fixed from now on and need commuting symbols for interpreting the array as a polynomial
        subs_dict = {k:ck for k,ck in zip(self.band_model.momentum_symbols,(kx,ky,kz))}
        subs_dict.update({x:cx for x,cx in zip(self.band_model.position_symbols,(x,y,z))})
        subs_dict.update({f:sympy.Symbol(f.name,commutative='False') for f in Vs})
        
        H = H.subs(subs_dict)
        
        
        nx= self.independent_vars['nx']
        ny= self.independent_vars['ny']
        nz= self.independent_vars['nz']
        projected_H = np.zeros(H.shape+(nx,nx,ny,ny,nz,nz))
        
        

        symbol_search = (kx,x,ky,y,kz,z)+Vs # order like this to make checking for conflicts easy
        Lx = self.independent_vars['domain'].Lx
        Ly = self.independent_vars['domain'].Ly
        Lz = self.independent_vars['domain'].Lz
        L = [Lx,Ly,Lz]
        
        def combine_matrices(matrix_dict):
            
            # from the tuple keys we construct indexing orderings of the axes for the matrices where 1,2 belong to x indecies, 2,3 to y and 4,5 to z
            d_tuples = tuple(matrix_dict.keys())
            indexing_tuples = {tup:[None]*(2*len(tup)) for tup in d_tuples} 
            for d in range(self.band_model.spatial_dim):
                # find the key with d in it
                i = next(i for i in range(len(d_tuples)) if d in d_tuples[i])
                t = d_tuples[i]
                ti = 2*t.index(d)
                indexing_tuples[t][ti] = 2*d
                indexing_tuples[t][ti+1] = 2*d+1
            
            combi = []
            for d_tup in d_tuples:
                combi.extend((np.array(matrix_dict[d_tup]).astype(float),tuple(indexing_tuples[d_tup])))
            
            combi.append(tuple(range(2*self.band_model.spatial_dim))) # output shape
            return np.einsum(*combi)
        
        def numerical_integration(term,spatial_directions):
            # assuming that term is right-alligned
            # construct lambda for numerical integration
            
            if isinstance(spatial_directions,int):
                spatial_directions = [spatial_directions]
            
            
            
            numerical_funcs = {sympy.Symbol(f.name,commutative=True):f for f in Vs}
            
            
            
            
            # split up piecewise
            if isinstance(term,sympy.Piecewise):
                piecewise_parts = term.args
                terms = [p[0] for p in piecewise_parts]
            else:
                piecewise_parts = None
                terms = [term]
            
            # determine k-order of every term. This takes care of any Add-type terms as well 
            splits = [symb.sort_out_polynomials(t,(kx,ky,kz)) for t in terms]
            k_orders = [[ss[0] for ss in s] for s in splits] 
            # cast coefficients as numerical function of x,y,z as well as V(x) for all numerical functions 
            coeff_funcs = [[sympy.lambdify((x,y,z)+tuple(numerical_funcs.keys()),ss[1],dummify=True) for ss in s]for s in splits]
            
            

            
            def integrand_product(left_basis,middle,right_basis):
                
                def funcprod(x,y=None,z=None,*func_vals):
                    
                    res = left_basis(x,y,z)*middle(x,y,z,*func_vals)*right_basis(x,y,z)
                    return res
                return funcprod    
            
            def basis_func_combiner(fx,fy=None,fz=None):
                
                def product_basis(x,y=None,z=None):
                    res = fx(x)
                    if fy is not None:
                        res *=fy(y)
                    if fz is not None:
                        res *= fz(z)
                    return res

                return product_basis
            
            def function_sum(*funcs):
                
                def funcsum(x,y=None,z=None,*func_vals):
                    res = 1
                    for func in funcs:
                        res += func(x,y,z,*func_vals)
                    return res
                
                return funcsum    
            
                
            
            def integrand_maker(*args):
                """Returns the integrand for computing the matrix element specified by `args`.
                args contains 2*D ints with D being the dimension to integrate over.
                The args are ordered as (nx,mx,ny,my,nz,mz) (or fewer directions if integration is over 2D or 1D). n are the row-indices and m are the collumn indices.
                """
                # for each term, multiply together basis funcs (with correct number of derivatives) and coeff_func
                
                term_funcs = []
                for j in range(len(splits)):
                    split_funcs = []
                    for k_order,coeff_func in zip(k_orders[j],coeff_funcs[j]):
                        left_basis_funcs = []
                        right_basis_funcs = []
                        for i in range(len(args),2):
                            n = args[i]
                            m =args[i+1]
                            k = k_order[int(i/2)] # number of times to differentiate the right basis func.
                            basis_func_factory = __basis_function_factory__(L=L[int(i/2)])
                            
                            left_basis_funcs.append(basis_func_factory(n))
                            right_basis_funcs.append(basis_func_factory(m,k))
                            
                            
                        split_funcs.append(integrand_product(basis_func_combiner(*left_basis_funcs),coeff_func,basis_func_combiner(*right_basis_funcs)))
                    term_funcs.append(function_sum(*split_funcs))
                
                if piecewise_parts is not None:
                    # lambdify the piecewise by building a new piecewise with dummies as arguments and lambdifying 
                    dummies = {sympy.Dummy():t for t in terms}
                    piecewise_form = ((d,c[1]) for d,c in zip(dummies.keys(),piecewise_parts))
                    numerical_piecewise = sympy.lambdify((x,y,z),sympy.Piecewise(*piecewise_form))
                    def numerical_integrand(x,y=None,z=None):
                        # cast numerical functions to their values 
                        numerical_evals = [f(x,y,z) for f in numerical_funcs.values()]
                        
                        # check piecewise to see which term to call:
                        relevant_dummy = numerical_piecewise(x,y,z)
                        
                        return dummies[relevant_dummy](x,y,z,*numerical_evals)
                else:
                    def numerical_integrand(x,y=None,z=None):
                        numerical_evals = [f(x,y,z) for f in numerical_funcs.values()]
                        return terms[0](x,y,z,*numerical_evals) 
                    
                return numerical_integrand
            return integrand_maker    
        
        for i,expr in enumerate(np.array(H).ravel()):
            
                        
            #check for piecewise:
            valid_piecewise,term = symb.extract_valid_bipartition_part(expr)

            if isinstance(term,sympy.Piecewise):

                matrices = {}
                # we have to compute the stuff numerically
                if valid_piecewise is None:
                    relevant_directions = [0,1,2]
                    pw_part = None
                else:
                    relevant_directions = [i for i in range(3) if all(((x,y,z)[i] not in pw[1].free_symbols) for pw in valid_piecewise)]
                    for i in range(3):
                        if i not in relevant_directions:
                            matrices[(i,)] = functions.box_bipartition_matrix(None,(nx,ny,nz)[i])
                    
                ranges = [[-L[i]/2,L[i]/2] for i in relevant_directions]
                N_modes = [(nx,ny,nz)[i] for i in relevant_directions]
                matrices[tuple(relevant_directions)] = functions.numerically_compute_matrix(numerical_integration(term,relevant_directions),ranges,N_modes)

                matrix_term = combine_matrices(matrices)
                
            else:    
                # split up into polynomial in kx,ky,kz,x,y,z and the numerical Vs
                parts = symb.sort_out_polynomials(term,symbol_search)
                matrix_term = np.zeros((nx,nx,ny,ny,nz,nz))
                
                for (orders,coeff) in parts:
                    # project every part down to the box basis.
                    
                    # WE have to do nummerical eval if: 
                        # numerical functions are involved
                        # we are not dealing with polynomials in the ks and xs
                    
                    spatial_deps =[]
                    for f,o in zip(Vs,orders):
                        if o is None or o>0:
                            spatial_deps.append(set(f.spatial_dependencies))
                    
                    
                    for d in range(3):
                        if orders[2*d] is None or orders[2*d+1] is None:
                            spatial_deps.append([d])
                    # combine spatial dependencies into disjoint sets
                    while any(len(a.intersection(b)) for a in spatial_deps for b in spatial_deps if a != b) and len(spatial_deps)>1:
                        for i,a in enumerate(spatial_deps[:-1]):  # while loop assures that len>1
                            for b in spatial_deps[i+1:]: 
                                if len(a.intersection(b)):
                                    b.update(a)
                                    spatial_deps.pop(i)
                                    break 
                            else:
                                continue # get here if b-loop finishes
                            break # get here if we break out of b loop. We start looping from the begining
                    
                    # Coordinates in `spatial deps` are now grouped together according to hwo the must beevaluated. Coordinates which are not present can be computed analytically. 
                    matrices = {}
                    
                    for coordinate_grp in spatial_deps:
                        raise NotImplementedError('this needs reworking: we need to account for the coefficient exactly once, but it can have non-polynomial terms.')
                        # COmbine all directions together which have None and integrate those combined with the coeff. the others directiosn can be numerically/analyticall integrted
                        if valid_piecewise is None:
                            relevant_directions = sorted(coordinate_grp)
                        
                        else:
                            raise NotImplementedError()
                            
                        ranges = [[-L[i]/2,L[i]/2] for i in relevant_directions]
                        N_modes = [(nx,ny,nz)[i] for i in relevant_directions]
                        
                        # reconstruct relevant parts of the expression 
                        t = coeff # also contains possible non-polynomial contributions which need numerical eval
                        for i in relevant_directions:
                            if ord is not None:
                                for o,ord in zip(symbol_search,orders):
                                    pass
                            if ord is not None:
                                t *= o**ord

                        matrices[tuple(coordinate_grp)] = functions.numerically_compute_matrix(numerical_integration(t,relevant_directions),ranges,N_modes)

                    
                    for d in range(3):
                        if all(d not in grp for grp in spatial_deps):
                            # analytical computation. Each direction is separated so it's a one-variable problem.
                            kd_o = orders[2*d] # order of k operator in direction d
                            xd_o = orders[2*d+1] # order of x operator in direction d
                            
                            if kd_o:
                                if xd_o:
                                    if kd_o % 2:
                                        # If there are even powers of K, it will act as a diagonal matrix
                                        diagonal = __k_mat__(kd_o,L[d],(nx,ny,nz)[d]) # should be diagonal
                                        x_mat = functions.box_polynomial_matrix((nx,ny,nz)[d],L[d],xd_o)
                                        matrices[(d,)] = x_mat @ diagonal
                                    else:
                                        raise NotImplementedError()
                                        # we have to do different numerical eval for this type ( sin * X**n * cos)
                                else:
                                    matrices[(d,)] = __k_mat__(kd_o,L[d],(nx,ny,nz)[d])
                            else:
                                
                                x_mat = functions.box_polynomial_matrix((nx,ny,nz)[d],L[d],xd_o)

                                matrices[(d,)] = x_mat
                    
                    mat = np.array(combine_matrices(matrices)).astype(float)
                    if not len(spatial_deps):
                        mat *= float(coeff) # spatial dependence has not been accounted for
                    matrix_term += mat
            
            ix = np.unravel_index(i,H.shape)
            projected_H[ix] += matrix_term
            
            
            # we have to reorder the array and reshape to N x N matrix 
            reordering = tuple(range(0,len(projected_H.shape),2))+ tuple(range(1,len(projected_H.shape),2))
            
            matrix_shape = np.prod(projected_H.shape[::2])
            return projected_H.transpose(reordering).reshape((matrix_shape,matrix_shape))

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

    def project_operator(self, operator):
        raise NotImplementedError('todo')
        return super().project_operator(operator)
    
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
    The function `__basis_function_factory__` returns a basis function factory that generates basis
    functions for a given value of `n` and `x`.
    
    :param L: The parameter L represents the length of the interval over which the basis functions are
    defined
    :param use_sympy: The parameter `use_sympy` is a boolean flag that determines whether the code
    should use the `sympy` library for symbolic calculations or the `numpy` library for numerical
    calculations. If `use_sympy` is set to `True`, the code will use `sympy, defaults to True (optional)
    :param derivative_order: The `derivative_order` parameter determines the order of the derivative of
    the basis function that will be returned. For example, if `derivative_order = 0`, the basis function
    itself will be returned. If `derivative_order = 1`, the first derivative of the basis function will
    be returned., defaults to 0 (optional)
    :return: The function `__basis_function_factory__` returns a function `basis_func_factory`.
    """

    if use_sympy:
        sin = sympy.sin
        cos = sympy.cos
        pi = sympy.pi
        sqrt = sympy.sqrt
    else:
        sin = np.sin
        cos = np.cos
        pi = np.pi
        sqrt = np.sqrt

    def basis_func_factory(n,derivative_order = 0):
        def basis_func(x):
            if not derivative_order %2: # even number of derivatives
                return sqrt(2/L)* sin(n*pi*(x/L + 1/2))*(n*pi/L)**derivative_order *(-1)**int(derivative_order/2)
            else: # odd numer of derivatives
                return sqrt(2/L)* cos(n*pi*(x/L + 1/2))*(n*pi/L)**derivative_order *(-1)**np.floor(derivative_order/2)
        return basis_func
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