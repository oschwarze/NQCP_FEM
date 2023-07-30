from unittest import TestCase
import sympy
class TestSymbolicFunctions(TestCase):
    
    def test_call(self):
        from nqcpfem.functions import SymbolicFunction,X,Y,Z

        F = X**2+Y**2+Z**2
        FF = SymbolicFunction(F,'F(x)')
        self.assertEqual(FF(2,0,0),4)
        self.assertEqual(FF(0,3,0),9)
        self.assertEqual(FF(0,0,4),16)
        self.assertEqual(FF(1,2,4),21)
        
        
        self.fail('call with arrays as arguments')
        
    def test_compute_deriative(self):

        from nqcpfem.functions import SymbolicFunction,X,Y,Z
        F = X**2+Y**2+Z**2
        FF = SymbolicFunction(F,'F(x)')
        
        self.assertEqual(FF.derivative(0).symbol,sympy.symbols('F_{(x)}(x)',commutative=False))
        self.assertEqual(FF.derivative(0).expression,2*X)
        
        self.assertEqual(FF.derivative(1).symbol,sympy.symbols('F_{(y)}(x)',commutative=False))
        self.assertEqual(FF.derivative(1).expression,2*Y)

        self.assertEqual(FF.derivative(2).symbol,sympy.symbols('F_{(z)}(x)',commutative=False))
        self.assertEqual(FF.derivative(2).expression,2*Z)
        
        
        self.assertEqual(FF.derivative([2,2]).expression,2)
        
    def test_project_to_box(self):
        # check that we can project simple expressions down to box modes,
        # check that it we can just do it for a single coordinate
        #TODO: rewrite function for box_mode stuff which uses numpy. Let it cache its results (since we will call it with same arguments often. Let arguments be hashable therefore)
        
        
        from nqcpfem.functions import SymbolicFunction,X,Y,Z
        F = X
        FF = SymbolicFunction(F,'F(x)')
        L = sympy.symbols('L')
        res = FF.project_to_basis(0,3,L=L)
        # diagonal should vanish
        self.assertEqual(res[0][0,0],0)
        self.assertEqual(res[0][1,1],0)
        self.assertEqual(res[0][2,2],0)
        
        
        res_array = res[0].subs(res[1]) 
        # off diagonal should be # off diagonal should be  ... (WHAT)
        self.assertEqual(res_array[0,1],-(4/(3*sympy.pi))**2*L)
        self.assertEqual(res_array[1,0],-(4/(3*sympy.pi))**2*L)
        self.assertEqual(res_array[1,0],-(2*3/(3*sympy.pi))**2*L)
        
        self.fail()
    

class TestNumericalFunction(TestCase):
    def test_project_to_box(self):
        self.fail()
        
        
        
class TestAnalyticalArrays(TestCase):
    def test_bipartition_box_array(self):
        from nqcpfem.functions import box_bipartition_matrix
        L = sympy.symbols('L')
        # we need to confirm some of the entries in this 
        self.fail()
        
    def test_linear_box_array(self):
        # we need to confirm some of the entries in this 
        self.fail()
        
    def test_harmonic_box_array(self):
        # we need to confirm some of the entries in this 
        self.fail()
