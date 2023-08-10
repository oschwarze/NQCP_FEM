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

        #self.assertEqual(res_array[0,1],-(4/(3*sympy.pi))**2*L)
        #self.assertEqual(res_array[1,0],-(4/(3*sympy.pi))**2*L)
        #self.assertEqual(res_array[1,0],-(2*3/(3*sympy.pi))**2*L)
        
        import numpy as np
        F = X+np.pi*Y
        FF = SymbolicFunction(F,'F(x)')
        L = sympy.symbols('L')
        res = FF.project_to_basis(0,3,L=L)
        # diagonal should not vanish here but be replaced by a function
        replacement = sympy.Symbol('(F)_{1}(x)',commutative=False)
        self.assertEqual(res[0][0,0],replacement)
        self.assertEqual(res[0][1,1],replacement)
        self.assertEqual(res[0][2,2],replacement)
        
        self.assertEqual(res[1][replacement].expression,np.pi*Y)

        
        self.fail('check off diagonals as well')
    

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



class TestPlaceHolderFunc(TestCase):
    def test_project_to_basis(self):
        from nqcpfem.functions import PlaceHolderFunction
        f_base_name = 'F_{z}'
        F = PlaceHolderFunction(f_base_name+'(x)')
        
        
        facit_funcs_suffix = ['_{00}(x)','_{01}(x)','_{10}(x)','_{11}(x)']
        facit_funcs = sympy.symbols([f'({f_base_name})'+ff for ff in facit_funcs_suffix],commutative=False)
        
        
        facit_array = sympy.Array([[facit_funcs[0],facit_funcs[1]],[facit_funcs[2],facit_funcs[3]]])
        
        F_proj,f_funcs = F.project_to_basis(2,2)
        
        for f in facit_funcs:
            self.assertIn(f,f_funcs)
        
        self.assertEqual(F_proj,facit_array)
        
        facit_funcs_suffix = []
        F_proj_proj,f_proj_proj_funcs = F.project_to_basis((0,1),(2,3))
        for i in range(2):
            for j in range(2):
                for k in range(3):
                    for l in range(3):
                        facit_funcs_suffix.append('_{'+f'{i}{j}{k}{l}'+'}(x)')
        facit_funcs = sympy.symbols([f'({f_base_name})'+ff for ff in facit_funcs_suffix],commutative=False)
        import numpy as np
        facit_array = sympy.Array(np.array(facit_funcs).reshape(2,2,3,3))
        
        for f in facit_funcs:
            self.assertIn(f,f_proj_proj_funcs)
        
        self.assertEqual(F_proj_proj,facit_array)
        
        
        
        
        
class TestFunctionName(TestCase):
    def test_decompose_name(self):
        
        testname_A = 'B_{xy(z)}(x)'
        facit_A = ('B_{xy}(x)',(2,),None)
        testname_B = '(F_{(xyz)})_{12}(x)'
        facit_B = ('F(x)',(0,1,2),(1,2))
        
        from nqcpfem.functions import decompose_func_name,assemble_func_name
        
        res_A = decompose_func_name(testname_A)
        self.assertEqual(res_A,facit_A)
        
        res_B = decompose_func_name(testname_B)
        self.assertEqual(res_B,facit_B)
        
        self.assertEqual(assemble_func_name(*res_A),testname_A)
        self.assertEqual(assemble_func_name(*res_B),testname_B)
        
    