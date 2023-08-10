from mimetypes import init
import numpy as np
import logging,sys
from .band_model import BandModel,LuttingerKohnHamiltonian,FreeFermion,__MAGNETIC_FIELD_NAMES__,__VECTOR_FIELD_NAMES__
from .envelope_function import EnvelopeFunctionModel
from .updatable_object import UpdatableObject,auto_update
from .spin_gap import find_spin_gap
from petsc4py import PETSc
import copy
import sympy
from . import functions as funcs
from . import symbolic as symb
from .solvers import ModelSolver
from nqcpfem import updatable_object

LOG = logging.getLogger(__name__)
LOG.addHandler(logging.StreamHandler(sys.stderr))
LOG.setLevel(logging.DEBUG)

class GMatrix(UpdatableObject):

	
	def __init__(self,envelope_model,model_solver):
		"""
		Computes the g_matrix of a band model at some specific parameter point, as well as the partial derivatives wrt.
		each of the parameters specified.

		:param EnvelopeFunctionModel envelope_model: band model to compute g matrix of. This band model needs to have terms that depend
		on B in order for this to make sense. The specific values of the parameters of the model are the ones with which respect we compute the g matrix.
		:param ModelSolver model_solver: Solver to use when computing eigenstates.
		"""
		
		self._derivatives = {} # dict containing partial derivatives
		self._tolerances = {} # dict containing the tolerances used in computing the partial derivatives

		super(GMatrix, self).__init__(**{'envelope_model':envelope_model,'solver':model_solver})

	@property
	def envelope_model(self):
		return self.independent_vars['envelope_model']

	@envelope_model.setter
	def envelope_model(self,value):
		self.independent_vars['envelope_model'] = value

	@property
	def solver(self):
		return self.independent_vars['solver']

	@solver.setter
	def solver(self,value):
		self.independent_vars['solver'] = value

	@auto_update
	def matrix(self):
		return self.compute_g_matrix()[0]

	@auto_update
	def ground_state_pair(self):
		return self.compute_g_matrix()[1]

	@auto_update
	def __make_M_operator__(self):
		Ax,Ay,Az = sympy.symbols(__VECTOR_FIELD_NAMES__,commutative=False)
		A_syms=[symb.derivative_of_function(a,x) for a in (Ax,Ay,Az) for x in (symb.X,symb.Y,symb.Z)]
		A_field = np.array(A_syms).reshape(3,3)  # for easier indexing
	
		x_base_group = [__MAGNETIC_FIELD_NAMES__[0],A_field[2,1].name,A_field[1,2].name]
		y_base_group = [__MAGNETIC_FIELD_NAMES__[1],A_field[0,2].name,A_field[2,0].name]
		z_base_group = [__MAGNETIC_FIELD_NAMES__[2],A_field[1,0].name,A_field[0,1].name]

		coordinate_bases = {'x':x_base_group,'y':y_base_group,'z':z_base_group}
		
		H = sympy.Array(self.envelope_model.band_model.numerical_array(replace_symbolic_functions=False)) # make numerical so we don't have to bother with constants and parameters

		
		# ddd all the symbols that have anything to do with Bx,By,Bz or the derivatives of A (derivatives, projections etc) to the relevat_symbols:
		# for each function, take the base_name and the spatial derivatives and compare with the base groups
		for f in H.free_symbols:
			if f.name[-3:] == '(x)':
				
				f_base,derivs,proj = funcs.decompose_func_name(f.name)
				
				for d,name_list in coordinate_bases.items():
					for name in name_list:
						base_name ,name_derivs,_ = funcs.decompose_func_name(name)
						if f_base == base_name and derivs == name_derivs:
							coordinate_bases[d].append(f.name) 
							break # base groups are disjoint
				
		relevant_symbols = sum((sympy.symbols(group) for group in coordinate_bases.values()),[])
		# assume from now on that the position if the Ks is fixed so we can make the commutative
		comm_relevant_symbols = symb.enforce_commutativity(relevant_symbols)
		comm_H = symb.enforce_commutativity(H)
		# deconstruct H into polynomials in the relevant symbols:
		decomposed_H = symb.array_sort_out_polynomials(comm_H,comm_relevant_symbols)

		# determine the Mx,My,Mz parts 

		blank = sympy.Array.zeros(*comm_H.shape)
		M_parts = {'x':blank,'y':blank,'z':blank}
		single_one = lambda n: tuple(0 if i!=n else 1 for i in range(len(relevant_symbols)))
		for i,sym in enumerate(comm_relevant_symbols):
			for d,name_list in coordinate_bases.items():
				if sym.name in name_list:
					
					# we need minus sign in fron if symbol is diAj for ij anti-cyclic
					sign = -1 if funcs.extract_base_function_name(sym.name,skip_derivatives=True) == name_list[2] else 1
					M_parts[d] = M_parts[d] + sign*decomposed_H.get(single_one(i),blank)
					break 
			
		

		#cast each Mx,My,Mz as an abstract operator
		observable_constructor = self.envelope_model.construct_observable
		Mx = observable_constructor(M_parts['x'])
		My = observable_constructor(M_parts['y'])
		Mz = observable_constructor(M_parts['z'])
		return Mx,My,Mz

	def compute_g_matrix(self,param_dict = None,covariant_transform = False,precomputed_solution=None,**spin_gap_kwargs):
		"""
		Compute the g_matrix of the band model. Default is to use the parameters of the model, but alterntive values can be supplied
		:param dict param_dict: alternative parameter specification to use (only the ones different from the models
		parameters need to be supplied)
		:param dict eig_solve_kwargs: kwargs for the .get_eigenvectors method of envelope_model
		:param tuple(np.ndarray.np.ndarray|None precomputed_solution: If the model has already been solved
		:return:
		"""
		# currently only the .add_LK_magnetic_field magnetic field is supported. This only works for magnetic terms
		# that are independent of position (so both B field and g_tensor must be homogenous in position)
		from copy import deepcopy
		envelope_model = deepcopy(self.envelope_model) 
		if all(sympy.symbols(n,commutative=False) not in envelope_model.band_model.post_processed_functions() for n in __MAGNETIC_FIELD_NAMES__+__VECTOR_FIELD_NAMES__) :
			raise ValueError(f'band model had no magnetic term. g matrix can only be computed for band models with magnetic terms defined')

		# replace all band_model parameters with zerofunction

		envelope_model.band_model.independent_vars['function_dict'].update({sympy.symbols(n,commutative=False):funcs.SymbolicFunction(sympy.sympify(0),sympy.symbols(n,commutative=False)) for n in __MAGNETIC_FIELD_NAMES__+__VECTOR_FIELD_NAMES__})

		# Determine the M operator components:
		M= self.__make_M_operator__()
		
		# determine the ground state Kramer's pair
		if precomputed_solution is None:
			solution = self.solver.solve(envelope_model)
		else:
			solution = precomputed_solution
		gap, states, has_intermediate = find_spin_gap(solution,envelope_model, **spin_gap_kwargs)
		from .observables import spin,gram_schmidt_orthogonalization
		do_GS = False
		if do_GS:
			states = gram_schmidt_orthogonalization(states) # orthogonalize the states (in case numerics has had a hard time doing this)
			# This shouldn't be wrong in any way, as we know that the gs kramers pair is degenerate, and the two solutions
			# are linear independent, so we can convert them to an ONB of the ground state subspace.
		from . import UNIT_CONVENTION, _hbar
		from .fenics import FEniCsModel
		plot = 0
		if isinstance(self.envelope_model, FEniCsModel) and plot > 0:
			# PLOTTING
			import pyvista
			import dolfinx
			eigentensors = solution[1]
			topology, cell_types, x = dolfinx.plot.create_vtk_mesh(envelope_model)
			for i in range(len(solution[0])):
				if i>-1:
					vec,_ = envelope_model.positional_rep(eigentensors[i])
					wave_func = np.linalg.norm(vec.reshape((-1,vec.shape[-1])),axis=0)
					p = pyvista.Plotter()

					grid = pyvista.UnstructuredGrid(topology, cell_types, x)
					grid["u"] = np.abs(wave_func) / np.max(np.abs(wave_func))
					warped = grid.warp_by_scalar("u")
					p.add_mesh(warped, scalars='u')
					p.show()
		#raise ValueError(f"Unable to find ground state kramer's pair:")
		log_message = f"found ground state kramer's pair, with energy gap: {gap*UNIT_CONVENTION['J to eV']} eV."
		log_message += ' Intermediate states were present.' if has_intermediate else ' No intermediate states present.'
		LOG.info(log_message)
		from . import _mu_B
		if covariant_transform:
			# Compute the g-matrix elements wrt. the covariant basis vectors
			basis_transform = self.covariant_basis_transform(states)
			states = np.einsum('ij,j...->i...', basis_transform, states)


		# indices 0,1,2,3 are reserved for indexing the vectors and spin contraction
		solution_dims = len(envelope_model.band_model.tensor_shape) # how many indices are needed to express the tensor # how many indices are needed to express the tensor acting on the states
		# one positional index (0), 2 indices indexing which eigenvector (1,2), one index giving coordinate (x,y,z) of C (3)
		C_indices = [i for i in range (4, 4 + solution_dims)] # and 2*n_spinor_dims indices to index elements of C_x, C_y, C_z
		right_i = [i for i in range(4, 4 + solution_dims, 2)]  # even indices index rows (start from 4 since 0,1,2,3 are reserved)
		left_i = [i for i in range(5, 4 + solution_dims, 2)] #  odd indices index columns

		gs_pos, _ = envelope_model.positional_rep(states[0])
		ex_pos, _ = envelope_model.positional_rep(states[1])

		states = np.stack([gs_pos,ex_pos],axis=0)

		M_projection = np.zeros((3,2,2),dtype='complex') # n,j,k with n being indexing which M, and j,k which of the basis vectors

		for n in range(3):
			for j in range(2):
				for k in range(2):
					M_projection[n,j,k] = M[n].mel(states[j],states[k])
        #		M_mel = np.einsum(states.conj(),[1,]+left_i+[0,],M_parts,[3]+C_indices,states,[2,]+right_i+[0,],[3,1,2])

		
		# NB: factor 2 comes from the fact that we assume that the two states have spin 1/2 i.e. S operator is \hbar * paulis but this will not be the case for LK hamiltonian
		g_matrix = -2/_mu_B * np.vstack([np.real(M_projection[:,0,1]),
		                               np.imag(M_projection[:,0,1]),
		                               np.real(M_projection[:,1,1])]) #last np.real is just to make the matrix elements all completely real (it shouldn't discard anything above machine precision)
		return g_matrix,states

	def derivative(self,params,step_sizes,overwrite=False,method='regular',**diff_kwargs):
		"""
		Determine the partial derivative with respect to the given parameters. By default
		it only computes not previously determined derivatives (reagrdless of the step_size).
		:param bool overwrite: whether to overwrite derivatives if the tolerances do not match
		:param str||list[str]|list[(str,int)] params: parameters with respect to which we want to determine the derivatives
		If the parameter is a vector, it should be passed as a tuple with the name and the integer referencing wrt. which component to differentiate
		:param float|list[float] step_sizes: the step_size used to compute the derivative. Must have same shape as params.
		:return:
		"""
		if isinstance(params,str):
			params = [params]
			if isinstance(step_sizes, float):
				step_sizes = [step_sizes]

		if isinstance(step_sizes, (list, tuple, np.ndarray)) and len(step_sizes) !=len(params):
			raise ValueError(f'tolerance list was not of same length as params list: {len(params)} vs {len(step_sizes)}')

		#compute new partial derivatives
		for param,step_size in zip(params, step_sizes):
			if param not in self._derivatives or overwrite:
				left_param_dict = self.default_params.copy()
				right_param_dict = self.default_params.copy()
				if isinstance(param,tuple):
					index=param[1]
					parameter = param[0]
					left_arg = copy.copy(self.default_params[parameter])
					left_arg[index] = left_arg[index]+step_size

					right_arg = copy.copy(self.default_params[parameter])
					right_arg[index] = right_arg[index]-step_size
				else:
					parameter = param
					left_arg = self.default_params[parameter]+step_size
					right_arg = self.default_params[parameter]-step_size

				left_param_dict[parameter] = left_arg
				right_param_dict[parameter] = right_arg

				left_matrix=self.compute_g_matrix(left_param_dict,covariant_transform=True)
				right_matrix = self.compute_g_matrix(right_param_dict,covariant_transform=True)

				derivative = (left_matrix-right_matrix)/(2*np.linalg.norm(step_sizes))

				self._derivatives[param] = derivative

		return [self._derivatives[param] for param in params]
		"""
		for param,tol in zip(params, step_sizes):
			if param not in self._derivatives or overwrite:

				if isinstance(param,tuple):
					# we get here if param is a vector
					index = param[1]
					param = param[0]

					def func(x):
						param_val = np.array(self.default_params[param],dtype=float)
						param_val[index] = x
						return self.compute_g_matrix({param:param_val})
					self._derivatives[(param,index)] = derivative(func, self.default_params[param][index], tol,method=method, **diff_kwargs)
				else:
					def func(x):
						return self.compute_g_matrix({param:x})

					self._derivatives[param] = derivative(func, self.default_params[param], tol, method=method)

				self._tolerances[param] = tol

		return [self._derivatives[param] for param in params ]
		"""
	def compute_beta(self,vectors):
		"""
		Compute the beta factor from appendix B in https://arxiv.org/pdf/1807.09185.pdf
		:param np.ndarray vectors: vectors describing the ground state spin pair of the model. shape is  (2,vector_shape)
		:return:
		"""
		from .observables import inner_product
		basis = self.ground_state_pair
		inv_beta_sq = np.abs(inner_product(basis[0],vectors[0]))**2 + np.abs(inner_product(basis[0],vectors[1]))**2
		return 1/np.sqrt(inv_beta_sq)

	def covariant_basis_transform(self,vectors):
		"""
		Computes the covariant transform used in computing the derivative of the g-matrix according to appendix B in
		https://arxiv.org/pdf/1807.09185.pdf
		:param np.ndarray vectors: vectors describing the ground state spin pair of the model. shape is  (2,vector_shape)
		:return:
		"""
		from .observables import inner_product
		basis = self.ground_state_pair
		a = inner_product(vectors[0],basis[0])
		b = inner_product(vectors[1],basis[0])
		c = inner_product(vectors[0],basis[1])
		d = inner_product(vectors[1],basis[1])
		return self.compute_beta(vectors)*np.array([[a,c],[b,d]])
