from unittest import TestCase
from nqcpfem.gridsearch import MinimizationSearch
class TestMinimizationSearch(TestCase):

    def setUp(self) -> None:
        self.mock_ef_function = lambda x: x
        self.default_params = {'1':1 ,'2': 2}
        class mockSolver():
            def solve(self,arg):
                return arg
        self.mock_solver = mockSolver()
        def post_processin_func(params):
            return (params['1']-10)**2 + (params['2']-20)**2

        self.post_processing_func = post_processin_func
    def test_find_minimum(self):
        minimization_func = MinimizationSearch(self.mock_solver,self.mock_ef_function,self.post_processing_func,self.default_params)
        res =minimization_func.find_minimum()
        print(res)
        import numpy.testing as testing
        testing.assert_allclose(res.x,[10,20],rtol=1e-4)
