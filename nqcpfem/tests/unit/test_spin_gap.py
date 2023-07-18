from unittest import TestCase
import numpy as np

class Test(TestCase):
    def test_find_spin_gap(self):

        from nqcpfem.band_model import FreeFermion
        band_model = FreeFermion(1, 2)
        from nqcpfem.fenics import FEniCsModel
        from nqcpfem.envelope_function import RectangleDomain

        domain = RectangleDomain(1, 1, 1)
        domain.resolution = [10, 10]

        model = FEniCsModel(band_model, domain)

        n_points = model.mesh.geometry.x.shape[0] #number of x_points

        # Make mock spinor states where spin eigenstates are along x and they are either supported on the first or
        # second half of the points. Also add a boundary state that is oncly supported on the boundary the should therefore be discarded
        x_right = np.ones(n_points, dtype='complex')
        x_left = np.ones(n_points, dtype='complex')
        x_left[:int(n_points/2)] = 0
        x_right[int(n_points/2):] = 0
        singleton_x = np.zeros_like(x_left) # only supported at one of the boundary points
        singleton_x[0] = 1

        x_up_spinor = 1/np.sqrt(2)*np.array([1, 1], dtype='complex')
        x_down_spinor = 1/np.sqrt(2)*np.array([1, -1], dtype='complex')

        gs_state = np.tensordot(x_up_spinor, x_left, axes=0).flatten()
        gs_state = gs_state/np.linalg.norm(gs_state)
        ex_state = np.tensordot(x_down_spinor, x_left, axes=0).flatten()
        ex_state = ex_state/np.linalg.norm(ex_state)
        boundary_state = np.tensordot(x_down_spinor, singleton_x, axes=0).flatten()
        boundary_state = boundary_state/np.linalg.norm(boundary_state)
        false_positive = np.tensordot(x_down_spinor, x_right,axes=0).flatten()
        false_positive = false_positive/np.linalg.norm(false_positive)
        eigenstates = np.stack([gs_state,ex_state,false_positive,boundary_state], axis=1)
        energies = np.array([-1, 1, 0, -5])

        from models.spin_gap import find_spin_gap
        gap,states,has_intermediate = find_spin_gap((energies, eigenstates), model, bounded_state_tolerance=0.95)
        self.assertTrue(has_intermediate)
        self.assertAlmostEqual(gap,2)
        np.testing.assert_array_equal(states, np.stack([gs_state, ex_state],axis=0))

        #

