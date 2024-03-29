from abc import ABC,abstractmethod
from re import A
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
        self.normalize = False

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
            converted['sigma'] = converted['sigma'] / model.energy_scale()

        return converted

class ScipySolver(ModelSolver):
    def solve(self, model,sparse=None):
        """
        Solve and Envelope Function eigenvalue Problem and return results as
        array of eigenvalues eigenvectors and array of eigenvectors
        :param EnvelopeFunctionModel model:
        :return Tuple[eigenvalues,eigenstates]:
        """
        array = model.assemble_array(sparse)
        from scipy.sparse import csr_matrix
        sparse = isinstance(array,csr_matrix) if sparse is None else sparse

        solver_kwargs = self.convert_solver_kwargs(model)

        if sparse:
            from scipy.sparse.linalg import eigsh as solver
            s_mat_dict = {'M':None}
            drop_solutions = 0
        else:
            from scipy.linalg import eigh as solver
            s_mat_dict = {'b':None}
            if solver_kwargs.get('k','is_none') != 'is_none':
                if solver_kwargs['k'] is not None:
                    solver_kwargs['subset_by_index']=[0,solver_kwargs.get('k')-1]
                del solver_kwargs['k']
                #FIXME: request the right eigenvlaues based on these kwargs

            try:
                del solver_kwargs['sigma']
            except KeyError as err:
                pass
            try:
                del solver_kwargs['which']
            except KeyError as err:
                pass

            
        M = model.make_S_array() # add S array for generalized eigenvalue solving
        if not (isinstance(M,int) and M == 1):
            s_mat_dict[list(s_mat_dict.keys())[0]] = M


        import numpy as np
        if s_mat_dict[list(s_mat_dict.keys())[0]] is None:
            solution = solver(array, **solver_kwargs)

        else:
            solution = solver(array,**s_mat_dict,**solver_kwargs)

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

        solution = (solution[0]*model.energy_scale(), normalized_eigenvects)

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
    spetrac_transform = {'shift_invert':SLEPc.ST.Type.SINVERT,
                         'q':0}

    which_conversion = {
        'SA': SLEPc.EPS.Which.SMALLEST_REAL,
        'LA': SLEPc.EPS.Which.LARGEST_REAL,
        'SM': SLEPc.EPS.Which.SMALLEST_MAGNITUDE,
        'LM': SLEPc.EPS.Which.LARGEST_MAGNITUDE,
    }

    def __init__(self, sparse=True, **solver_kwargs):
        super().__init__(sparse, **solver_kwargs)
        self.__petsc_mat_A__ = None
        self.__petsc_mat_S__ = None
    
    
    def create_initial_vector(self,vecs,array_size,model:EnvelopeFunctionModel):
        pass
        from petsc4py import PETSc
        import numpy as np
        #TODO: check if a single vector or multipl are passed by comparing length of vector (number of elements with size of array)
        if vecs.shape[0] != array_size[0]:
            initial_vec = np.sum(vecs,axis=0)
        else:
            initial_vec = vecs
        
        flattened = model.flatten_eigentensors(initial_vec)
        petsc_vec = PETSc.Vec().createWithArray(flattened)
        return petsc_vec
        

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

        p = True if self.__petsc_mat_A__ is None else self.__petsc_mat_A__
        petsc_A = model.assemble_array(petsc_array=p)
        #self.__petsc_mat_A__ = petsc_A


        p = True if self.__petsc_mat_S__ is None else self.__petsc_mat_S__
        petsc_S = model.make_S_array(petsc_array=p)
        #self.__petsc_mat_S__ = petsc_S
        if isinstance(petsc_S,int) and petsc_S == 1:
            petsc_S = None
        else:
            petsc_S.assemble()

        petsc_A.assemble()


        eig_problem = SLEPc.EPS().create(comm=MPI.COMM_WORLD)

        if self.solver_kwargs.get('sigma',None) is not None:
            # do shift invert
            spectral_transform = SLEPc.ST().create()
            if self.solver_kwargs.get('method','krylovschur') == 'GD':
                spectral_transform.setType(SLEPc.ST.Type.PRECOND)
            else:
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
        eig_problem.setType(self.solver_methods[self.solver_kwargs.get('method','krylovschur')]) # set method for solving

        if self.solver_kwargs.get('method','krylovschur') == 'GD':
            eig_problem.setGDDoubleExpansion(True)

        
        eig_problem.setOperators(petsc_A,petsc_S) # define matrices of the problem
        print(petsc_A.size)
        if 'initial_guess' in self.solver_kwargs:
            if self.solver_kwargs.get('method','krylovschur') == 'krylovschur':
                initial_vecs = [self.create_initial_vector(self.solver_kwargs['initial_guess'],petsc_A.size,model)]
                initial_vecs[0].assemble()
                eig_problem.setInitialSpace(initial_vecs[0])
            else:
                initial_vecs = [PETSc.Vec().createWithArray(vec) for vec in model.flatten_eigentensors(self.solver_kwargs['initial_guess'])]
                for vec in initial_vecs:
                    vec.assemble()
                eig_problem.setInitialSpace(initial_vecs)
        else:
            initial_vecs = []
        try:
            eig_problem.solve()
        except Exception as err:
            if err.args == (71,):
                raise Exception('This can occur when the magnitude of the array elements span many orders of magnitude. Check that all numerical values are correct') from err
            elif err.args == (95,):
                LOGGER.info('eigenproblem solvin with initial_guess failed. retrying with random initial guess')
                old_kwargs = self.solver_kwargs.copy()
                del self.solver_kwargs['initial_guess']
                return self.solve(model)
            else:
                print(err.args)
                raise err

        vr, vi = petsc_A.createVecs()

        LOGGER.debug(f' number of converged eigenvectors: {eig_problem.getConverged()}')
        if eig_problem.getConverged() < self.solver_kwargs['k']:
            raise ValueError(f'eigenvectors did not all converge')

        eigenvalues = []
        eigenvectors = []
        for i in range(eig_problem.getConverged()):
            lmbda = eig_problem.getEigenpair(i, vr, vi)
            eigenvalues.append(lmbda.real*model.energy_scale()) # convert back to actual units

            eigenvector = np.array(vr.array) + 1j*np.array(vi)
            if self.normalize:
                eigenvector = eigenvector/np.linalg.norm(eigenvector)

            eigenvectors.append(eigenvector)



        eigenvectors = model.eigensolutions_to_eigentensors(np.stack(eigenvectors,axis=1)) # cast into the eigentensor format
        eigenvalues = np.array(eigenvalues)
        
        petsc_A.destroy()
        vr.destroy()
        vi.destroy()
        eig_problem.destroy()
        if petsc_S is not None:
            petsc_S.destroy()
        if self.solver_kwargs.get('sigma',None) is not None:
            spectral_transform.destroy()

        for vec in initial_vecs:
            vec.destroy()
        
        return eigenvalues,eigenvectors

class IterativeSolver(ModelSolver):
    def __init__(self,first_solver:ModelSolver,second_solver:ModelSolver|None=None):
        # Solver that uses the results of the previous solve to solve the next problem. ca be configured such that the very first solving is
        # computed in a different way
        self.first_solver = first_solver
        self.second_solver = first_solver if second_solver is None else second_solver
        self.result_vecs = None
        self.result_evals = None
        
    def solve(self,model):
        if self.result_vecs is None:
            new_res = self.first_solver.solve(model)
        else:
            self.second_solver.solver_kwargs['initial_guess'] = self.result_vecs
            new_res = self.second_solver.solve(model)
        
        self.result_vecs = new_res[1]
        self.result_evals = new_res[0]
        return new_res



