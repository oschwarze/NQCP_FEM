import unittest
from unittest import TestCase
import sympy

from nqcpfem.symbolic import construct_target_signature, k_component_signature

class TestSymbolic(TestCase):
    def test_expand_term(self):
        from nqcpfem.symbolic import expand_term,Kx,Ky,Kz,X,Y,Z
        a,b,c = sympy.symbols('a,b,c')
        A = Kx**3*a*Ky*b
        res = expand_term(A)
        self.assertEqual((a,b,Kx,Kx,Kx,Ky),res)
        
        self.assertEqual(expand_term(Kx**2),(Kx,Kx))
        
        self.assertEqual(expand_term(sympy.symbols('F(x)',commutative=False)),(sympy.symbols('F(x)',commutative=False),))
        
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
    
    def test_extract_valid_biparition_part(self):
        from nqcpfem.symbolic import X,Y,Z,extract_valid_bipartition_part
        a,b,c = sympy.symbols('a,b,c')
        X,Y,Z = sympy.symbols('x,y,z',commutative = True)
        atoms =(sympy.Piecewise((1,X>0),(0,True)),
                sympy.Piecewise((1,Y>0),(0,True)),
                sympy.Piecewise((1,Z>0),(0,True)))
        
        term =sympy.Piecewise((a,X>0),(0,True))
        self.assertEqual(extract_valid_bipartition_part(term),([sympy.Piecewise((1,X>0),(0,True))],a))
        
        term = sympy.Piecewise((a,X>0),(0,True))*sympy.Piecewise((b,Y>0),(0,True))
        term = term.simplify()
        res = extract_valid_bipartition_part(term)
        self.assertEqual(res[1],a*b)
        for r in res[0]:
            self.assertIn(r,[atoms[0],atoms[1]])

        
        term = sympy.Piecewise((a,X>0),(0,True))*sympy.Piecewise((b*Y,Y>0),(0,True))
        term = term.simplify()
        res = extract_valid_bipartition_part(term)
        
        
        self.assertEqual(extract_valid_bipartition_part(term),([atoms[0]],sympy.Piecewise((a*b*Y,Y>0),(0,True))))
        
        
        term = sympy.Piecewise((a*Z,X>0),(0,True))*sympy.Piecewise((b*Y,Y>0),(0,True))
        term = term.simplify()
        self.assertEqual(extract_valid_bipartition_part(term),([atoms[0]],sympy.Piecewise((Z*a*b*Y,Y>0),(0,True))))
        
        term = sympy.Piecewise((a*Y,X>0),(0,True))*sympy.Piecewise((b*Y,Y>0),(0,True))*sympy.Piecewise((1,Z>0),(0,True))
        term = term.simplify()
        res = extract_valid_bipartition_part(term)
        self.assertEqual(res[1],sympy.Piecewise((a*b*Y**2,Y>0),(0,True)))
        for r in res[0]:
            self.assertIn(r,[atoms[0],atoms[2]])
            
    
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
        self.assertEqual(comm(a*X**2),2*a*X)
        self.assertEqual(comm(V),facit_V)
        self.assertEqual(comm(a*V),a*facit_V)
        self.assertEqual(comm(2*V),2*facit_V)
        self.assertEqual(comm(sympy.conjugate(V)),sympy.conjugate(facit_V))
        self.assertEqual(comm(sympy.conjugate(a*V)),sympy.conjugate(a*facit_V))

    def test_permute_factors(self):
        from nqcpfem.symbolic import permute_factors,Kx,Ky,Kz,X,Y,Z
        term = [X,Y,Z,Kx,Ky,Kz]
        self.assertEqual(permute_factors(term,5,0),[[Kz,X,Y,Z,Kx,Ky],[X,Y,-1,Kx,Ky]])

        V=sympy.symbols('F(x)',commutative=False)
        facit_V = sympy.symbols('F_{(x)}(x)',commutative=False)
        term = [X,Y,V,Kx,Ky,Kz]
        self.assertEqual(permute_factors(term,3,0),[[Kx,X,Y,V,Ky,Kz],[X,Y,-facit_V,Ky,Kz],[-1,Y,V,Ky,Kz]])

        
        term = [Kx,X**2,Y,V,Ky,Kz]
        self.assertEqual(permute_factors(term,0,5),[[X**2,Y,V,Ky,Kz,Kx],[2*X,Y,V,Ky,Kz],[X**2,Y,facit_V,Ky,Kz]])

    def test_arange_ks(self):
        from nqcpfem.symbolic import arange_ks,Kx,Ky,Kz,X,Y,Z
        
        V=sympy.symbols('F(x)',commutative=False)
        Vx=sympy.symbols('F_{(x)}(x)',commutative=False)
        Vy=sympy.symbols('F_{(y)}(x)',commutative=False)
        Vz=sympy.symbols('F_{(z)}(x)',commutative=False)
        Vyz=sympy.symbols('F_{(yz)}(x)',commutative=False)
        term = [X,Kx,Y**2,V,Kz,Ky]
        target = [True,True,True,False,False,False]
        result = arange_ks(term,target_signature=target,signature_reduction_direction='left')
        facit = [[Kz,Ky,Kx,X,Y**2,V],
                [Kz,Kx,X,-2*Y,V],
                [Kz,Kx,X,Y**2,-Vy],
                [Ky,Kx,X,Y**2,-Vz],
                [Kx,X,-2*Y,-Vz],
                [Kx,X,Y**2,Vyz], # plus sign in front of Vyz because two - signs
                [Kz,Ky,-1,Y**2,V],
                [Kz,-1,-2*Y,V],
                [Kz,-1,Y**2,-Vy],
                [Ky,-1,Y**2,-Vz],
                [-1,-2*Y,-Vz],
                [-1,Y**2,Vyz],
                ]
        self.assertEqual(len(facit),len(result))
        for f in facit:
            self.assertIn(f,result)
    
    def test_arange_k_array(self):
        
        from nqcpfem.symbolic import arange_ks_array,Kx,Ky,Kz,X,Y,Z
        V=sympy.symbols('F(x)',commutative=False)
        Vx=sympy.symbols('F_{(x)}(x)',commutative=False)
        Vy=sympy.symbols('F_{(y)}(x)',commutative=False)
        Vz=sympy.symbols('F_{(z)}(x)',commutative=False)
        Vyz=sympy.symbols('F_{(yz)}(x)',commutative=False)
        term_00 = X*Kx*Y**2*V*Kz*Ky
        facit_00 = sum((sympy.Mul(*f) for f in [[Kz,Ky,Kx,X,Y**2,V],
                [Kz,Kx,X,-2*Y,V],
                [Kz,Kx,X,Y**2,-Vy],
                [Ky,Kx,X,Y**2,-Vz],
                [Kx,X,-2*Y,-Vz],
                [Kx,X,Y**2,Vyz], # plus sign in front of Vyz because two - signs
                [Kz,Ky,-1,Y**2,V],
                [Kz,-1,-2*Y,V],
                [Kz,-1,Y**2,-Vy],
                [Ky,-1,Y**2,-Vz],
                [-1,-2*Y,-Vz],
                [-1,Y**2,Vyz],
                ]),sympy.sympify(0))
        
        term_01 = V*Kx + V*Ky
        facit_01 = sum((sympy.Mul(*f) for f in [[Kx,V],[-Vx]]),sympy.sympify(0))+ sum((sympy.Mul(*f) for f in [[Ky,V],[-Vy]]),sympy.sympify(0))
        
        term_10 = V*Y**2*Ky
        facit_10 = sum((sympy.Mul(*f) for f in [[Ky,V,Y**2],[-Vy,Y**2],[V,-2*Y]]),sympy.sympify(0))
        
        
        term_11 = X**3*Kx*Kx
        facit_11 = sum((sympy.Mul(*f) for f in [[Kx,Kx,X**3],[Kx,-3*X**2],[Kx,-3*X**2],[6*X]]),sympy.sympify(0))
        array = sympy.Array([[term_00,term_01],[term_10,term_11]])
        
        result = arange_ks_array(array,'all left','left')
        facit_array = sympy.Array([[facit_00,facit_01],[facit_10,facit_11]])

        
        
        self.assertEqual(result,facit_array)