import unittest
from unittest import TestCase
import sympy

from nqcpfem.symbolic import construct_target_signature, k_component_signature

class TestSymbolic(TestCase):
    def test_expand_term(self):
        from nqcpfem.symbolic import expand_term,Kx,Ky,Kz
        a,b,c = sympy.symbols('a,b,c')
        A = Kx**3*a*Ky*b
        res = expand_term(A)
        self.assertEqual((a,b,Kx,Kx,Kx,Ky),res)
        
    def test_derivative_of_function(self):
        
        from nqcpfem.symbolic import derivative_of_function
        F,FF  = sympy.symbols('f(x),f_{(zxy)}(x)',commutative=False)
        facit_F,facit_FF = sympy.symbols('f_{(y)}(x),f_{(xyyz)}(x)',commutative=False)
        Y = sympy.symbols('y',commutative=False)
        R = derivative_of_function(F,Y)
        RR = derivative_of_function(FF,Y)
        print(R.name == facit_F.name ,type(R),R==facit_F)
        self.assertEqual(R,facit_F)
        self.assertEqual(RR,facit_FF)

        a = sympy.symbols('a')
        mul_F =  a*F
        com_conj_F = sympy.conjugate(F)
        
        mul_conj_FF = sympy.conjugate(a*FF)
        
        self.assertEqual(derivative_of_function(mul_F,Y),a*facit_F)
        self.assertEqual(derivative_of_function(com_conj_F,Y),sympy.conjugate(facit_F))
        self.assertEqual(derivative_of_function(mul_conj_FF,Y),sympy.conjugate(a*facit_FF))
        
    def test_construct_target_structure(self):
        from nqcpfem.symbolic import Kx,Ky,Kz,construct_target_signature
        a = sympy.symbols('a')
        term = (Kx,Kx,a,2,Ky,Kz)
        
        self.assertEqual(construct_target_signature(term,'all left'),[True,True,True,True,False,False])
        self.assertEqual(construct_target_signature(term,'all right'),[False,False,True,True,True,True])
        self.assertEqual(construct_target_signature(term,'FEM'),[True,True,True,False,False,True])
    
    
    def test_k_component_signature(self):
        from nqcpfem.symbolic import Kx,Ky,Kz,k_component_signature
        term = (Kx,Kx,Ky,Ky,Kz,Kz,Kx,Kz,Ky)
        self.assertTrue(k_component_signature(term),[0,0,1,1,2,2,0,2,1])
        
    def test_extract_k_independent_part(self):
        from nqcpfem.symbolic import Kx,Ky,Kz,extract_k_independent_part,X,Y,Z
        a = sympy.symbols('a')
        F = sympy.symbols('F(x)',commutative=False)
        term_A = (Kx,Ky,Kz,a,F,X,Y,Z)
        term_B = (a,F,X,Y,Z,Kx,Ky,Kz)
        term_C = (Kz,Kz,a,F,X,Y,Z,Kx,Ky,Kz)
        
        self.assertEqual(extract_k_independent_part(term_A),(a,F,X,Y,Z))
        self.assertEqual(extract_k_independent_part(term_B),(a,F,X,Y,Z))
        self.assertEqual(extract_k_independent_part(term_C),(a,F,X,Y,Z))
        
    
    def test_commutator_map(self):
        from nqcpfem.symbolic import commutator_map,Kx,X
        a = sympy.symbols('a')
        V = sympy.symbols('F(x)',commutative=False)
        facit_V = sympy.symbols('F_{(x)}(x)',commutative=False)
        comm = commutator_map(Kx)
        self.assertEqual(comm(a*X),a)
        self.assertEqual(comm(a*X**2),2*a)
        self.assertEqual(comm(V),facit_V)
        self.assertEqual(comm(a*V),a*facit_V)
        self.assertEqual(comm(2*V),2*facit_V)
        self.assertEqual(comm(sympy.conjugate(V)),sympy.conjugate(facit_V))
        self.assertEqual(comm(sympy.conjugate(a*V)),sympy.conjugate(a*facit_V))

    def test_permute_factors(self):
        self.fail()