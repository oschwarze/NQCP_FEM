import unittest
import numpy as np
import os,sys

src_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.append(src_path)

import nqcpfem.band_model


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
        topology, cell_types, x = dolfinx.plot.create_vtk_mesh(self.model.function_space())

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
        np.testing.assert_allclose(N_found,N_found.astype(int),rtol=1e-2,atol=5.2e-1)

        A_box = self.box_model.assemble_array()
        S_box = self.box_model.make_S_array()


        print('Solving box problem...')
        box_solution = eigsh(A_box,M=S_box,k=N,sigma=-1000)
        N_box_found = box_solution[0]/(_hbar**2*np.pi**2/(2*self.mass*self.Lx**2))*self.box_model.energy_scale()

        print('box_modes:\n',N_box_found)
        N_box_found[N_box_found<0.5] = 0
        np.testing.assert_allclose(N_box_found,N_box_found.astype(int),rtol=1e-2,atol=1e-3)

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
        self.assertIs(self.model.ufl_form(),old_form.value)
        self.assertEqual(self.model._saved_ufl_form._modified_time_,old_form._modified_time_)
        #del self.model._saved_assemble_array 
        new_A = self.model.assemble_array()
        import numpy as np
        np.testing.assert_allclose(old_A.todense(),4*new_A.todense()-3*self.model.infinite_boundary_vec().todense())
        
        self.model.band_model.independent_vars['parameter_dict'][sympy.symbols('omega')] = 0
        old_A = self.model.assemble_array()
        self.assertIs(self.model.ufl_form(),old_form.value)
        self.assertEqual(self.model._saved_ufl_form._modified_time_,old_form._modified_time_)
        

        self.model.band_model.independent_vars['parameter_dict'][sympy.symbols('m')] *= 2 
        new_A = self.model.assemble_array()
        self.assertIs(self.model.ufl_form(),old_form.value)
        self.assertEqual(self.model._saved_ufl_form._modified_time_,old_form._modified_time_)
        
    

if __name__ == '__main__':
    unittest.main()
