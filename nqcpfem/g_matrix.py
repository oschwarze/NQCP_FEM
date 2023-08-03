import numpy as np
import logging,sys
from .band_model import BandModel,LuttingerKohnHamiltonian,FreeFermion,__MAGNETIC_FIELD_NAMES__,__VECTOR_FIELD_NAMES__
from .envelope_function import EnvelopeFunctionModel
from .spin_gap import find_spin_gap
from petsc4py import PETSc
import copy
import sympy
from . import functions as funcs
from . import symbolic as symb
from .solvers import ModelSolver

LOG = logging.getLogger(__name__)
LOG.addHandler(logging.StreamHandler(sys.stderr))
LOG.setLevel(logging.DEBUG)

class GMatrix():
	def __init__(self,envelope_model,model_solver):
		"""
		Computes the g_matrix of a band model at some specific parameter point, as well as the partial derivatives wrt.
		each of the parameters specified.

		:param EnvelopeFunctionModel envelope_model: band model to compute g matrix of. This band model needs to have terms that depend
		on B in order for this to make sense. The specific values of the parameters of the model are the ones with which respect we compute the g matrix.
		:param ModelSolver model_solver: Solver to use when computing eigenstates.
		"""
		self.envelope_model = envelope_model
		self.default_params = self.envelope_model.band_model.parameter_dict.copy()
		self.solver = model_solver
		self._derivatives = {} # dict containing partial derivatives
		self._tolerances = {} # dict containing the tolerances used in computing the partial derivatives
		self._matrix = None
		self._ground_state_pair = None

	@property
	def matrix(self):
		if self._matrix is None:
			self._matrix,ground_state_pair = self.compute_g_matrix()
			self._ground_state_pair = ground_state_pair
		return self._matrix

	@property
	def ground_state_pair(self):
		if self._ground_state_pair is None:
			self._matrix,ground_state_pair = self.compute_g_matrix()
			self._ground_state_pair = ground_state_pair
		return self._ground_state_pair

	def __make_M_operator__(self):
		Ax,Ay,Az = sympy.symbols(__VECTOR_FIELD_NAMES__,commutative=False)
		if any(A in self.envelope_model.band_model.independent_vars['function_dict'] for A in (Ax,Ay,Az)):
			# ordered as flattened version oaf M_ij = d_j(A_i) where d_j is derivative in direciont j and A_i is field in direction i
			# Indexing wrt flattend index measn that M_N = M_ij with N=3*i+j
			has_A_field = True
			relevant_symbols = [symb.derivative_of_function(a,x) for a in (Ax,Ay,Az) for x in (symb.X,symb.Y,symb.Z)]
		else:
			has_A_field=False
			relevant_symbols = []
		
		# add magnetic field to this 
		relevant_symbols.extend(sympy.symbols(__MAGNETIC_FIELD_NAMES__,commutative=False))

		H = sympy.Array(self.envelope_model.band_model.numerical_array(replace_symbolic_functions=False)) # make numerical so we don't have to bother with constants and parameters
		aranged_H = symb.arange_ks_array(H,self.envelope_model.k_signature)

		# assume from now on that the position if the Ks is fixed so we can make the commutative

		comm_relevant_symbols = symb.enforce_commutativity(relevant_symbols)
		comm_H = symb.enforce_commutativity(aranged_H)
		# deconstruct H into polynomials in the relevant symbols:
		decomposed_H = symb.array_sort_out_polynomials(comm_H,comm_relevant_symbols)

		# determine the Mx,My,Mz parts 
		blank = sympy.Array.zeros(*comm_H.shape)
		single_one = lambda n: tuple(0 if i!=n else 1 for i in range(N))
		N = len(relevant_symbols)
		Mx_parts = decomposed_H.get(single_one(N-3),blank)
		My_parts = decomposed_H.get(single_one(N-2),blank)
		Mz_parts = decomposed_H.get(single_one(N-1),blank)
		
		if has_A_field:
			Mx_parts = Mx_parts + decomposed_H.get(single_one(3*2+1),blank) - decomposed_H.get(single_one(3*1+2),blank) #dyAz-dzAy
			My_parts = My_parts + decomposed_H.get(single_one(3*0+2),blank) - decomposed_H.get(single_one(3*2+0),blank) #dzAx-dxAz
			Mz_parts = Mz_parts + decomposed_H.get(single_one(3*0+2),blank) - decomposed_H.get(single_one(3*2+0),blank) #dxAy-dyAx

		#cast each Mx,My,Mz as an abstract operator
		observable_constructor = self.envelope_model.construct_observable
		Mx = observable_constructor(Mx_parts)
		My = observable_constructor(My_parts)
		Mz = observable_constructor(Mz_parts)
		return Mx,My,Mz



	def compute_g_matrix(self,param_dict = None,return_ground_state_pair=False,covariant_transform = False,precomputed_solution=None,**spin_gap_kwargs):
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
		band_model = deepcopy(self.envelope_model.band_model) 
		if all(sympy.symbols(n,commutative=False) not in band_model.post_processed_functions() for n in __MAGNETIC_FIELD_NAMES__+__VECTOR_FIELD_NAMES__) :
			raise ValueError(f'band model had no magnetic term. g matrix can only be computed for band models with magnetic terms defined')

		# replace all band_model parameters with zerofunction
		band_model.independent_vars['function_dict'].update({sympy.symbols(n,commutative=False):funcs.SymbolicFunction(sympy.sympify(0),sympy.symbols(n,commutative=False)) for n in __MAGNETIC_FIELD_NAMES__+__VECTOR_FIELD_NAMES__})

		# Determine the M operator components:
		M= self.__make_M_operator__()
		
		# determine the ground state Kramer's pair
		if precomputed_solution is None:
			solution = self.solver.solve(self.envelope_model)
		else:
			solution = precomputed_solution
		gap, states, has_intermediate = find_spin_gap(solution, self.envelope_model, **spin_gap_kwargs)
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
			topology, cell_types, x = dolfinx.plot.create_vtk_mesh(self.envelope_model.function_space)
			for i in range(len(solution[0])):
				if i>-1:
					vec,_ = self.envelope_model.positional_rep(eigentensors[i])
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
		solution_dims = len(self.envelope_model.band_model.tensor_shape) # how many indices are needed to express the tensor # how many indices are needed to express the tensor acting on the states
		# one positional index (0), 2 indices indexing which eigenvector (1,2), one index giving coordinate (x,y,z) of C (3)
		C_indices = [i for i in range (4, 4 + solution_dims)] # and 2*n_spinor_dims indices to index elements of C_x, C_y, C_z
		right_i = [i for i in range(4, 4 + solution_dims, 2)]  # even indices index rows (start from 4 since 0,1,2,3 are reserved)
		left_i = [i for i in range(5, 4 + solution_dims, 2)] #  odd indices index columns

		gs_pos, _ = self.envelope_model.positional_rep(states[0])
		ex_pos, _ = self.envelope_model.positional_rep(states[1])

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
		if return_ground_state_pair:
			return g_matrix,states
		return g_matrix

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

