import unittest
from unittest import TestCase
import numpy as np
from scipy.stats import unitary_group


class MockEFM():
    def __init__(self):
        self.eigenvalues = np.arange(100)
        self.U = unitary_group.rvs(100)
        self.array = self.U @ np.diag(self.eigenvalues) @ self.U.conj().T

    def assemble_array(self, sparse=False):
        return self.array

    def make_S_array(self):
        return 1
class TestScipySolver(TestCase):
    def setUp(self) -> None:
        from nqcpfem.solvers import ScipySolver
        self.k = 30
        self.solver = ScipySolver(k=self.k, which='SA')



        self.model = MockEFM()

    def test_solve(self):
        results = self.solver.solve(self.model)
        np.testing.assert_allclose(results[0], self.model.eigenvalues[:self.k], atol=1e-14)


if __name__ == '__main__':
    unit.main()


class TestPETScSolver(TestCase):
    def test_solve(self):
        from .test_fenics import setup
        setup(self)




        from nqcpfem.solvers import PETScSolver
        solver = PETScSolver(k=10,sigma=0)
        solution = solver.solve(self.box_model)
        from nqcpfem import _hbar,_m_e
        import numpy as np
        Lx = 1e-6

        print(np.array(solution[0])/ (_hbar ** 2 * np.pi ** 2 / (2 * _m_e * Lx ** 2)))

