import unittest
import numpy as np
import os,sys

src_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.append(src_path)

import nqcpfem.band_model

import logging
LOG=logging.getLogger(__name__)
LOG.setLevel(logging.DEBUG)
LOG

class TestFEniCsModel(unittest.TestCase):
    def setUp(self) -> None:
        from petsc4py import PETSc
        import os, sys
        print(os.environ)
        print(sys.executable)
        print((np.dtype(PETSc.ScalarType)))
        self.assertTrue(np.dtype(PETSc.ScalarType).kind == 'c',
                        msg=f'Python environment is configured incorrectly: PETSc hat bo be built with complex numbers: {np.dtype(PETSc.ScalarType)}')
        from nqcpfem.band_model import FreeBoson
        from nqcpfem.envelope_function import RectangleDomain
        from nqcpfem.fenics import FEniCsModel
        from nqcpfem import _m_e, _hbar
        self.mass = _m_e
        L = 1e-6
        self.Lx = L
        self.Ly = L
        self.Lz = L
        self.band_model = FreeBoson(self.mass, 2)
        from nqcpfem.band_model import FreeFermion
        self.band_model = FreeFermion(self.mass, 2)
        self.domain = RectangleDomain(self.Lx, self.Ly, self.Lz)
        self.resolution = 150
        self.domain.resolution = [self.resolution, self.resolution]
        self.function_class = ('CG', 1)
        self.model = FEniCsModel(self.band_model, self.domain, 0, self.function_class)
        import sympy
        from nqcpfem import envelope_function
        x, y = sympy.symbols('x,y')
        self.omega = 1e11
        omega = sympy.symbols('omega')
        V = 0.5 * omega ** 2 * _m_e * (x ** 2 + y ** 2)

        self.model.band_model.add_potential(V,{omega:self.omega})

        self.box_domain = RectangleDomain(self.Lx, self.Ly, self.Lz)
        self.box_domain.resolution = [75, 75]
        self.box_model = FEniCsModel(self.band_model, self.box_domain, 0, self.function_class)

    def test_assemble_array(self):
        # testing harmonic oscillator modes
        from nqcpfem import _m_e,_hbar
        import numpy as np
        #self.model.boundary_condition=12345e10
        A= self.model.assemble_array()
        print(A.shape)
        import dolfinx
        import scipy.sparse as sp

        from scipy.sparse.linalg import eigsh
        N= 25
        S = self.model.make_S_array()
        print('solving SHO problem...')
        evals,evecs = eigsh(A,M=S,k=N,sigma=-1000,which='LM')
        topology, cell_types, x = dolfinx.plot.vtk_mesh(self.model.function_space())

        N_found = evals/(_hbar*self.omega)*self.model.energy_scale()
        print('SHO_modes:\n',N_found)
        """#PLOTTING
        for i in range(N):
            if i in [1,2]:
                vec = evecs[:,i]
                p = pyvista.Plotter()
                grid = pyvista.UnstructuredGrid(topology, cell_types, x)
                grid["u"] = np.abs(vec)/np.max(np.abs(vec))
                warped = grid.warp_by_scalar("u")
                p.add_mesh(warped, scalars='u')
                p.show()
        """
        np.testing.assert_array_less(N_found,100) #check that we actually get small eigenvalues
        np.testing.assert_allclose(N_found,N_found.astype(np.int32),rtol=1e-2,atol=5.2e-1)

        facit_list = []
        next_n = iter(range(1,N+1))
        while len(facit_list)<2*N: # we have to make a lot to assure that we get the smallest ones
            new_n = next(next_n)
            facit_list.extend([new_n]*(2*new_n))
        
        facit_list = np.sort(facit_list)
        np.testing.assert_allclose(np.sort(N_found),facit_list[:N],atol=5.2e-1,rtol=1e-2)
        
        import sympy
        self.box_model.band_model.parameter_dict[sympy.symbols('omega')] = 0
        A_box = self.box_model.assemble_array()
        S_box = self.box_model.make_S_array()


        print('Solving box problem...')
        box_solution = eigsh(A_box,M=S_box,k=N,sigma=-2000)
        N_box_found = box_solution[0]/(_hbar**2*np.pi**2/(2*self.mass*self.Lx**2))*self.box_model.energy_scale()

        print('box_modes:\n',N_box_found)
        N_box_found[N_box_found<0.5] = 0
        np.testing.assert_array_less(N_found,100) #check that we actually get small eigenvalues
        np.testing.assert_allclose(N_box_found,N_box_found.astype(np.int32),rtol=1e-2,atol=1e-3)

        facit = []
        next_n = iter(range(2,N+2)) # start from 2 since that is ground state in both x and y
        while len(facit)<2*N:
            n = next(next_n)
            # find all different partitions
            partitions = set((i,n-i) for i in range(1,n)) # minimum of each element i 1
            
            facit.extend(p[0]**2+p[1]**2 for p in partitions for _ in range(2)) # add everything twice because of spind degeneracy
            
        facit = np.sort(facit)
        np.testing.assert_allclose(np.sort(N_box_found),facit[:N],atol=5.2e-1,rtol=1e-2)
        
    def test_positional_rep(self):
        import dolfinx
        from nqcpfem.band_model import FreeFermion
        band_model = FreeFermion(1,2)
        from nqcpfem.fenics import FEniCsModel
        evf_model = FEniCsModel(band_model,self.domain,0)
        du = dolfinx.fem.Function(evf_model.function_space())
        array=du.vector.getArray()

        mock_vector = np.linspace(0,1,array.size).reshape(array.shape)
        mock_vector = mock_vector.reshape((int(mock_vector.shape[-1]/2),2)).transpose([1,0])
        positional_rep,x = evf_model.positional_rep(mock_vector)

        np.testing.assert_array_equal(x,self.model.mesh().geometry.x*self.model.length_scale())
        np.testing.assert_array_equal(mock_vector,positional_rep)

    def test_eigensolutions_to_eigentensors(self):
        import dolfinx
        du = dolfinx.fem.Function(self.model.function_space())
        array=du.vector.getArray()

        mock_vector = np.linspace(0,1,array.size).reshape(array.shape)
        #mock_vector = mock_vector.reshape((mock_vector.shape[-1],1,1)).transpose([1,2,0])
        mock_set = np.stack([mock_vector,mock_vector],axis=1)
        result = self.model.eigensolutions_to_eigentensors(mock_set)

        facit_vector = mock_vector.reshape((int(mock_vector.shape[-1]/2),2)).transpose([1,0])
        facit_vectors = np.stack([facit_vector,facit_vector],axis=0)
        np.testing.assert_array_equal(result,facit_vectors)

        result = self.model.eigensolutions_to_eigentensors(mock_vector)
        np.testing.assert_array_equal(result,facit_vector)

    def test_mesh(self):
        xvals = self.model.mesh().geometry.x
        max_coords = np.max(xvals,axis=0)
        min_coords = np.min(xvals,axis=0)
        np.testing.assert_array_almost_equal(max_coords,np.array([0.5,0.5,0]))
        np.testing.assert_array_almost_equal(min_coords,np.array([-0.5,-0.5,0]))

    def test_redefining_constants(self):
        
        
        # check that after building an Array, I can change the mass in the bilinear form without having to reassemble it
        self.model.independent_vars['domain'].resolution=[10,10]
        old_A = self.model.assemble_array()
        S =self.model.make_S_array()
        import sympy
        import copy 
        # check that altering the constants dict alters the resulting array but not build another ufl_form.
        old_form = copy.copy(self.model._saved_ufl_form)
        self.model.band_model.independent_vars['constants'][sympy.symbols(r'\hbar')] *= 2
        self.assertEqual(self.model._saved_ufl_form._modified_time_,old_form._modified_time_)
        self.assertIs(self.model.ufl_form(),old_form.value)
        #del self.model._saved_assemble_array 
        new_A = self.model.assemble_array()
        import numpy as np
        np.testing.assert_allclose(old_A.todense(),4*new_A.todense()-3*np.diag(self.model.infinite_boundary_vec().getArray()))
        
        self.model.band_model.independent_vars['parameter_dict'][sympy.symbols('omega')] = 0
        old_A = self.model.assemble_array()
        self.assertIs(self.model.ufl_form(),old_form.value)
        self.assertEqual(self.model._saved_ufl_form._modified_time_,old_form._modified_time_)
        

        self.model.band_model.independent_vars['parameter_dict'][sympy.symbols('m')] *= 2 
        new_A = self.model.assemble_array()
        self.assertIs(self.model.ufl_form(),old_form.value)
        self.assertEqual(self.model._saved_ufl_form._modified_time_,old_form._modified_time_)
        
    def test_projcet_operator(self):
        # assert that identity operator matches S operator
        import sympy
        operator = sympy.Array([[1,0],[0,1]])
        O = self.model.project_operator(operator)
        S = self.model.make_S_array()
        diff = O-S
        
        max = diff.max()
        min = diff.min()
        np.testing.assert_allclose([max,min],0) 
        
        
        # Chekc that X**2 operator has the same sparsity pattern as the S matrix
        X = sympy.symbols('x',commutative=False)
        operator = operator*X**2 
        O = self.model.project_operator(operator)
        
        O_is = np.split(O.indices,O.indptr)[1:-1]
        
        S_is = np.split(S.indices,S.indptr)[1:-1]
        
        for i,(o_row,s_row) in enumerate(zip(O_is,S_is)):
            np.testing.assert_array_equal(o_row,s_row,err_msg=f'row {i} did not have the same sparsity pattern')
        
        
        # test that it works just like assemble_array
        from nqcpfem.symbolic import Kx
        from nqcpfem import _hbar
        self.setUp()
        self.model.domain.resolution = [100,100]
        operator = self.model.band_model.post_processed_array().subs({'m':self.mass,'omega':0,r'\hbar':_hbar})
        O = self.model.project_operator(operator)
        S = self.model.make_S_array()
        
        # assert that the array gives particle in a box eigenmodes:
        import scipy.sparse as sparse
        O = O +sparse.diags(self.model.infinite_boundary_vec().getArray())
        factor = (_hbar**2*np.pi**2/(2*self.mass*self.Lx**2))
        
        eigvals,eigvecs = sparse.linalg.eigsh(O,k=10,M=S,sigma=-1000)
        nsq = eigvals/factor
        np.testing.assert_allclose(nsq,np.round(nsq))
        
        
    def test_make_observable(self):
        # make poisson equatio and
        from nqcpfem.symbolic import Kx,Ky
        from nqcpfem.band_model import BandModel
        from nqcpfem import fenics
        from nqcpfem.solvers import PETScSolver
        import sympy
        poisson_spinor = sympy.Array([[Kx**2+Ky**2+np.pi**2,0],[0,Kx**2+Ky**2-np.pi**2]])
        bm = BandModel(poisson_spinor,2)
        
        # use domain from setup
        model = fenics.FEniCsModel(bm,self.domain,0,('CG',1))
        
        # solve the model and obtain eigenstates in the usual way.
        solver = PETScSolver(which='SM',sigma=0,k=10)
        
        eigvals,eigvecs = solver.solve(model)
        
        # take lowest eigenstates and compute the MEL of sigma_z operator
        
        sigma_z = sympy.Array([[1,0],[0,-1]])
        Oz = model.construct_observable(sigma_z)
        Oz_proj = np.array([Oz.mel(ev1,ev2) for ev1 in eigvecs[:2] for ev2 in eigvecs[:2]][::-1]).reshape((2,2)) #reversed order because evecs are sorted by eigenvalue (low to high)
        
        
        np.testing.assert_allclose(Oz_proj,np.array(sigma_z).astype(complex),atol=1e-3,rtol=1e-3)
        
        
        # check that the MEL of O work as intended.
        
        Ox = model.construct_observable(sympy.Array([[0,1],[1,0]]))
        flipped = Ox.apply(eigvecs[0])
        flipped = flipped/np.linalg.norm(flipped)
        
        ev0 = eigvecs[0]/np.linalg.norm(eigvecs[0])
        ev1 = eigvecs[1]/np.linalg.norm(eigvecs[1])
        # flipped must be orthogonal to eigvecs[0]
        np.testing.assert_allclose(np.einsum('ix,ix',flipped.conj(),ev0),0,atol=1e-3)
        np.testing.assert_allclose(np.abs(np.einsum('ix,ix',flipped.conj(),ev1)),1,rtol=1e-3)
        
        
    def test_make_dolfinx_functions(self):

        def numerical_func(x):
            return x[0]
        
        from nqcpfem.functions import NumericalFunction
        func = NumericalFunction(numerical_func,'f(x)',spatial_dependencies=[0])
        from nqcpfem.band_model import __MOMENTUM_NAMES__
        import sympy
        K = sympy.symbols(__MOMENTUM_NAMES__,commutative=False)
        self.band_model.independent_vars['preprocessed_array'] = self.band_model.independent_vars['preprocessed_array']+ sympy.Array([[func.symbol*K[0]*K[0],0],[0,0]])
        self.band_model.fix_k_arrangement('FEM',allow_placeholder_functions=True)
        self.band_model.function_dict[func.symbol] = func
        result = self.model.converted_functions()

        self.assertEqual(len(result[1]),2)
        
        
        
        
if __name__ == '__main__':
    unittest.main()
