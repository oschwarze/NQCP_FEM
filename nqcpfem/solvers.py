from abc import ABC,abstractmethod
from .envelope_function import EnvelopeFunctionModel
from typing import Tuple


import logging

LOGGER = logging.getLogger(__name__)
LOGGER.addHandler(logging.NullHandler())



"""
SOLVER ARGUMENT CONVENTION:
- k: int: Number of eigenvectors to find
- sigma: float: shift and invert to find eigenstates close to this number

"""

class ModelSolver(ABC):
	def __init__(self,sparse=True,**solver_kwargs):
		self.solver_kwargs = solver_kwargs
		self.sparse = sparse
		self.return_tensor = True

	@abstractmethod
	def solve(self,model,sparse=None):
		"""

		Args:
			model: EnvelopeFunctionModel
			sparse: Bool

		Returns: Tuple[eigenvalues,eigenstates]

		"""
		raise NotImplementedError

	@staticmethod
	def scipy(k=10,sigma=0):
		return ScipySolver(**{'k':k,'sigma':sigma})

	@staticmethod
	def petsc(k=10,sigma=0):
		return PETScSolver(**{'k':k,'sigma':sigma})


	def convert_solver_kwargs(self,model):
		"""
		converts the solver kwargs to be used when solving an EF model. (like converting energy scale units )
		:param EnvelopeFunctionModel model: the model to solve
		:return:
		"""
		converted = self.solver_kwargs.copy()
		if 'sigma' in converted.keys():
			converted['sigma'] = converted['sigma'] / model.energy_scale

		return converted

class ScipySolver(ModelSolver):
	def solve(self, model,sparse=None):
		"""
		Solve and Envelope Function eigenvalue Problem and return results as
		array of eigenvalues eigenvectors and array of eigenvectors
		:param EnvelopeFunctionModel model:
		:return Tuple[eigenvalues,eigenstates]:
		"""
		sparse = self.sparse if sparse is None else sparse
		array = model.assemble_array(sparse)

		solver_kwargs = self.convert_solver_kwargs(model)

		if sparse:
			from scipy.sparse.linalg import eigsh as solver
			s_mat_dict = {'M':None}
			drop_solutions = 0
		else:
			from scipy.linalg import eigh as solver
			s_mat_dict = {'b':None}
			drop_solutions = solver_kwargs.get('k',0)
			if drop_solutions:
				del solver_kwargs['k']
		M = model.make_S_array() # add S array for generalized eigenvalue solving
		if not (isinstance(M,int) and M == 1):
			s_mat_dict[list(s_mat_dict.keys())[0]] = M


		import numpy as np
		if s_mat_dict[list(s_mat_dict.keys())[0]] is None:
			solution = solver(array, **solver_kwargs)

		else:
			solution = solver(array,**s_mat_dict,**solver_kwargs)

		if drop_solutions:
			# sort
			mask = np.argsort(solution[0])[:drop_solutions]
			solution=(solution[0][mask],solution[1][:,mask])

		# normalize the eigenvectors:
		norms = np.linalg.norm(solution[1], axis=0) # axis 0 since the columns are the eigenvectors
		if not np.all(np.isclose(norms, 1)):
			normalized_eigenvects = solution[1] /norms[np.newaxis, :]
			## NB FEniCs eigenvalues do not need recalculation (eigenvectors are likely scaled based on dimension to
			# keep the elements of the states within a reasonable scale)

			#image_evecs = array @ normalized_eigenvects
			#eigenvalues = np.einsum('jn,jn->n',normalized_eigenvects.conj(),image_evecs)
			#solution = (eigenvalues, normalized_eigenvects)
		else:
			normalized_eigenvects = solution[1]

		if self.return_tensor:
			normalized_eigenvects = model.eigensolutions_to_eigentensors(normalized_eigenvects)

		solution = (solution[0]*model.energy_scale, normalized_eigenvects)

		return solution
class PETScSolver(ModelSolver):
	from slepc4py import SLEPc
	solver_methods = {
		'krylovschur':SLEPc.EPS.Type.KRYLOVSCHUR,
		'power':SLEPc.EPS.Type.POWER,
		'subspace':SLEPc.EPS.Type.SUBSPACE,
		'arnoldi': SLEPc.EPS.Type.ARNOLDI,
		'lanczos': SLEPc.EPS.Type.LANCZOS,
		'GD':SLEPc.EPS.Type.GD,
		'JD':SLEPc.EPS.Type.JD,
		'RQCG':SLEPc.EPS.Type.RQCG,
		'LOBPCG':SLEPc.EPS.Type.LOBPCG,
		'CISS':SLEPc.EPS.Type.CISS,
		'LYAPII':SLEPc.EPS.Type.LYAPII,
		'LAPACK':SLEPc.EPS.Type.LAPACK
	}

	which_conversion = {
		'SA': SLEPc.EPS.Which.SMALLEST_REAL,
		'LA': SLEPc.EPS.Which.LARGEST_REAL,
		'SM': SLEPc.EPS.Which.SMALLEST_MAGNITUDE,
		'LM': SLEPc.EPS.Which.LARGEST_MAGNITUDE,
	}


	def solve(self,model,sparse=True):
		"""
		Solve the envelope function model
		:param EnvelopeFunctionModel model:
		:param bool sparse: whether to solve the system as a sparse system or not
		:return:
		"""
		import numpy as np
		from slepc4py import SLEPc
		from petsc4py import PETSc
		from mpi4py import MPI


		A = model.assemble_array(sparse=True)
		petsc_A = PETSc.Mat().createAIJ(size=A.shape,csr=(A.indptr,A.indices,A.data))

		S = model.make_S_array()
		if isinstance(S,int) and S == 1:
			petsc_S = None
		else:
			petsc_S = PETSc.Mat().createAIJ(size=S.shape,csr=(S.indptr,S.indices,S.data))

		petsc_S.assemble()
		petsc_A.assemble()


		eig_problem = SLEPc.EPS().create(comm=MPI.COMM_WORLD)

		if self.solver_kwargs.get('sigma',None) is not None:
			# do shift invert
			spectral_transform = SLEPc.ST().create()
			spectral_transform.setType(SLEPc.ST.Type.SINVERT)
			eig_problem.setST(spectral_transform)
			if self.solver_kwargs.get('which','LM') == 'LM':
				eig_problem.setWhichEigenpairs(SLEPc.EPS.Which.TARGET_MAGNITUDE)
			eig_problem.setTarget(self.solver_kwargs['sigma'])
		else:
			eig_problem.setWhichEigenpairs(self.which_conversion[self.solver_kwargs.get('which','LA')])

		LOGGER.debug(f'solving problem:')
		eig_problem.setDimensions(self.solver_kwargs['k']) # number of eigenvalues to find
		eig_problem.setProblemType(SLEPc.EPS.ProblemType.GHEP) # hermitian eigenvalue problem
		eig_problem.setType(self.solver_kwargs.get('method',self.solver_methods['krylovschur'])) # set method for solving
		eig_problem.setOperators(petsc_A,petsc_S) # define matrices of the problem
		eig_problem.solve()


		vr, vi = petsc_A.createVecs()

		LOGGER.debug(f' number of converged eigenvectors: {eig_problem.getConverged()}')
		if eig_problem.getConverged() < self.solver_kwargs['k']:
			raise ValueError(f'eigenvectors did not all converge')

		eigenvalues = []
		eigenvectors = []
		for i in range(eig_problem.getConverged()):
			lmbda = eig_problem.getEigenpair(i, vr, vi)
			eigenvalues.append(lmbda.real*model.energy_scale) # convert back to actual units

			eigenvector = np.array(vr.array) + 1j*np.array(vi)
			eigenvectors.append(eigenvector/np.linalg.norm(eigenvector))



		eigenvectors = model.eigensolutions_to_eigentensors(np.stack(eigenvectors,axis=1)) # cast into the eigentensor format
		eigenvalues = np.array(eigenvalues)
		return eigenvalues,eigenvectors





