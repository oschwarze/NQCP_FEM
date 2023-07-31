from unittest import TestCase
import unittest
import numpy as np
import numpy.testing
import numpy.testing as testing
import sympy
# for locating the source files in all tests
import os,sys

src_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.append(src_path)


def __compare_sympy_arrays__(array_a,array_b):
    """
    This function compares two Sympy arrays and returns True if they are equal, False otherwise.
    
    :param array_a: The first Sympy array to be compared
    :param array_b: The second input array to be compared with array_a
    :return: a boolean value. It returns True if the shape of the two input arrays are the same and
    their difference after simplification is equal to a zero array, and False otherwise.
    """
    import numpy as np
    if array_a.shape != array_b.shape:
        return False
    
    zero_arr = sympy.Array(np.full(array_a.shape,sympy.sympify(0)))
    return (array_a-array_b).simplify() == zero_arr

class TestBandModel(TestCase):
    def setUp(self) -> None:

        from nqcpfem.band_model import BandModel,__MOMENTUM_NAMES__,__POSITION_NAMES__
        from nqcpfem.functions import NumericalFunction,SymbolicFunction
        x,y,z = sympy.symbols(__POSITION_NAMES__)
        kx,ky,kz = sympy.symbols(__MOMENTUM_NAMES__,commutative=False)
        self.constants = sympy.symbols('a,b,c')
        a,b,c = self.constants
        from unittest.mock import MagicMock
        self.mockV = MagicMock(side_effect =lambda x,y,z: 2*x )
        del self.mockV._modified_time_
        V1,V2,V3 = sympy.symbols('V1(x),V2(x),V3(x)',commutative=False)
        Vval1 = NumericalFunction(self.mockV,'V1(x)',[0,1,2])
        Vval2 = SymbolicFunction(x**2 + a*y**2 +c*z**2,'V2(x)')
        Vval3 = SymbolicFunction(sympy.Piecewise((1,x>0),(0,True)),'V3(x)')
        
        self.funcs = ((V1,Vval1),(V2,Vval2),(V3,Vval3))
        
        self.scalar_A= sympy.Array([[a*kx**2+ky**2+kz**2+b*kx+kx*ky+(c+1)*kz*ky-ky*kz+V1+V2+V3]])
        self.scalar_problem = BandModel(self.scalar_A,3)
        self.scalar_problem.independent_vars['parameter_dict'][a] = 1
        self.scalar_problem.independent_vars['parameter_dict'][b] = 2
        self.scalar_problem.independent_vars['parameter_dict'][c] = 3.14
        self.scalar_problem.independent_vars['function_dict'].update({v[0]:v[1] for v in self.funcs})
        from nqcpfem import ANGULAR_MOMENTUM
        sigma = ANGULAR_MOMENTUM['1/2'] * 2
        
        self.spinor_A_00_element = a*kx**2+kx*ky*kz+ky**2+kz**2+V1+V2+V3
        self.spinor_A_11_element =  kx**2+b*ky**2+c*b*kz**2+V1+V2+V3
        self.spinor_A = sympy.Array([[self.spinor_A_00_element,0],[0,self.spinor_A_11_element]])
        self.spinor_problem = BandModel(self.spinor_A,3)
        self.spinor_problem.independent_vars['parameter_dict'][a] = 1
        self.spinor_problem.independent_vars['parameter_dict'][b] = 2
        self.spinor_problem.independent_vars['parameter_dict'][c] = 3.14
        self.spinor_problem.independent_vars['function_dict'].update({v[0]:v[1] for v in self.funcs})

        self.tensor_A = sympy.Array([[[[kx**2+b**2*ky**2+V1+V2+V3,0],[0, kx**2+ky**2]],[[1,0],[0,0]]],[[[0,0],[0,0]],[[2*a*kx**2+2*ky**2,1],[a, c*2*kx**2+2*ky**2]]]])
        self.tensor_problem = BandModel(self.tensor_A,2)
        from functools import partial 
        self.tensor_problem.independent_vars['parameter_dict'][a] = 1
        self.tensor_problem.independent_vars['parameter_dict'][b] = 2
        self.tensor_problem.independent_vars['parameter_dict'][c] = 3.14
        self.tensor_problem.independent_vars['function_dict'].update({v[0]:v[1] for v in self.funcs})
        self.tensor_problem.independent_vars['function_dict'][self.funcs[0][0]] = self.funcs[0][1]
        
        
    def test_tensor_shape(self):
        self.assertEqual(self.scalar_problem.tensor_shape, (1, 1))
        self.assertEqual(self.spinor_problem.tensor_shape, (2, 2))
        self.assertEqual(self.tensor_problem.tensor_shape, (2,2,2,2))

    def test_n_bands(self):
        self.assertEqual(self.scalar_problem.n_bands, 1)
        self.assertEqual(self.spinor_problem.n_bands, 2)
        self.assertEqual(self.tensor_problem.n_bands, 4)
        

    def test_make_tensor_dict(self):
            a,b,c = self.constants

            scalar_r1 = np.zeros((1,1,3),dtype='O')
            scalar_r1[:,:,0] = 1*self.tensor_problem.independent_vars['parameter_dict'][b]
            scalar_r2 = np.zeros((1,1,3,3),dtype='O')
            scalar_r2[:,:,0,0] = 1 *self.tensor_problem.independent_vars['parameter_dict'][a]
            scalar_r2[:,:,1,1] = 1 
            scalar_r2[:,:,2,2] = 1 
            scalar_r2[:,:,0,1] = 1 
            scalar_r2[:,:,2,1] = 1*self.tensor_problem.independent_vars['parameter_dict'][c]+1
            scalar_r2[:,:,1,2] = -1
            result = self.scalar_problem.numerical_tensor_repr()
            print(result)
            self.assertEqual(len(result.keys()),3,msg=f'result keys: {tuple(result.keys())}')
            self.assertTrue(all(i in result.keys() for i in [1,2]))
            testing.assert_array_equal(scalar_r1,result[1])
            testing.assert_array_almost_equal(scalar_r2,result[2],decimal=10)
            self.mockV.assert_not_called()
            result = self.scalar_problem.numerical_tensor_repr()
            
            
            spinor_r2 = np.zeros((2,2,3,3),dtype='O')
            spinor_r2[0,0,0,0] = 1*self.spinor_problem.independent_vars['parameter_dict'][a]
            spinor_r2[1,1,0,0] = 1
            spinor_r2[0,0,1,1] = 1
            spinor_r2[1,1,1,1] = 1*self.spinor_problem.independent_vars['parameter_dict'][b]
            spinor_r2[0,0,2,2] = 1
            spinor_r2[1,1,2,2] = 1*self.spinor_problem.independent_vars['parameter_dict'][b]*self.spinor_problem.independent_vars['parameter_dict'][c]
            spinor_r3 = np.zeros((2,2,3,3,3),dtype='O')
            spinor_r3[0,0,0,1,2] = 1 *self.spinor_problem.independent_vars['parameter_dict'][a]
            result = self.spinor_problem.numerical_tensor_repr()
            self.assertEqual(len(result.keys()),3,msg=f'result keys: {tuple(result.keys())}')
            self.assertTrue(all(i in result.keys() for i in [2,3]))
            testing.assert_array_equal(spinor_r2,result[2])
            testing.assert_array_equal(spinor_r3,result[3])
            self.mockV.assert_not_called()
            self.mockV.reset_mock()
            
            x = sympy.symbols('x')
            tensor_r0 = np.zeros((2,2,2,2),dtype='O')
            tensor_r0[0,0,0,0] = self.funcs[0][1].symbol+self.funcs[1][1].expression.subs({s:v for s,v in zip(self.constants,(1,2,3.14))}) + self.funcs[2][1].expression
            tensor_r0[0,1,0,0] = 1
            tensor_r0[1,1,0,1] = self.tensor_problem.independent_vars['parameter_dict'][a]
            tensor_r0[1,1,1,0] = 1
            
            tensor_r2 = np.zeros((2,2,2,2,2,2),dtype='O')
            tensor_r2[0,0,0,0,0,0] = 1
            tensor_r2[0,0,1,1,0,0] = 1
            tensor_r2[1,1,0,0,0,0] = 2*self.tensor_problem.independent_vars['parameter_dict'][a]
            tensor_r2[1,1,1,1,0,0] = 2*self.tensor_problem.independent_vars['parameter_dict'][c]
            
            tensor_r2[0,0,0,0,1,1] = 1*self.tensor_problem.independent_vars['parameter_dict'][b]**2
            tensor_r2[0,0,1,1,1,1] = 1
            tensor_r2[1,1,0,0,1,1] = 2
            tensor_r2[1,1,1,1,1,1] = 2
            

            
            result = self.tensor_problem.numerical_tensor_repr()
            self.assertTrue(len(result.keys())==2)
            self.assertTrue(all(i in result.keys() for i in [0,2]))
            testing.assert_array_equal(tensor_r0,result[0])
            testing.assert_array_equal(tensor_r2,result[2])
            self.mockV.assert_not_called()
            self.mockV.reset_mock()
            
            
            #tesing that it also works if we use the symbolic array:
            symbolic_res = self.tensor_problem.symbolic_tensor_repr()
            self.assertTrue(all(s in sympy.Array(symbolic_res[2]).free_symbols for s in (a,b,c)))
            subs_dict = {a:1,b:2,c:3.14}
            subs_dict.update({self.funcs[0][0]:self.funcs[0][0],self.funcs[1][0]:self.funcs[1][1].expression,self.funcs[2][0]:self.funcs[2][1].expression})
            #subs_dict[self.funcs[0][0]] = self.funcs[0][1].expression)
            V1,V2,V3 = sympy.symbols('V1(x),V2(x),V3(x)',commutative=False)
            testing.assert_array_equal(tensor_r0,np.array(sympy.Array(symbolic_res[0]).subs(subs_dict)))
            testing.assert_array_equal(tensor_r2,np.array(sympy.Array(symbolic_res[2]).subs(subs_dict)))
            
            
    def test_make_numerical(self):
        a,b,c = self.constants
        from nqcpfem.band_model import BandModel,__MOMENTUM_NAMES__
        self.scalar_problem.independent_vars['parameter_dict'][a] = None 
        with self.assertRaises(ValueError):
            self.scalar_problem.numerical_array()
        
        self.scalar_problem.independent_vars['parameter_dict'][a] = 1
        result = self.scalar_problem.numerical_array()
        A_subs = {a:1,b:2,c:3.14}
        
        A_subs.update({v[0]:v[1].expression for v in self.funcs[1:]})
        x = sympy.symbols('x')
        #A_subs[self.funcs[0][0]] = self.funcs[0][1](*(x,0,0))
        #A_subs.update({sympy.symbols(k,commutative=False):sympy.symbols(k) for k in __MOMENTUM_NAMES__})
        testing.assert_array_equal(result,np.array(self.scalar_A.subs(A_subs)))
        
        result = self.spinor_problem.numerical_array()
        testing.assert_array_equal(result,np.array(self.spinor_A.subs(A_subs)))
        
        
        result = self.tensor_problem.numerical_array()
        testing.assert_array_equal(result,np.array(self.tensor_A.subs(A_subs)))
    
    def test_syncing(self):
        H = self.scalar_problem
        
        
        numerical_tensor = H.numerical_tensor_repr() # updates everything
        H.symbolic_tensor_repr()
        
        # extract status of attributes
        post_processed = H._saved_post_processed_array
        numerical = H._saved_numerical_array
        symbolic = H._saved_symbolic_tensor_repr
        numerical_tensor = H._saved_numerical_tensor_repr
        post_processor = H._saved_array_post_processor
        import copy
        pre_processed = copy.copy(H.independent_vars.get_stored_attribute('preprocessed_array'))
        
        
        
        H.independent_vars['preprocessed_array'] = H.independent_vars['preprocessed_array']+sympy.Array([[0]]) # setting to new should flag everything as out of sync and update everything
        H.numerical_tensor_repr()
        
    
        self.assertFalse( pre_processed ==  H.independent_vars.get_stored_attribute('preprocessed_array'))
        self.assertFalse( post_processed ==  H._saved_post_processed_array)
        self.assertFalse( numerical ==  H._saved_numerical_array)
        self.assertTrue( symbolic._modified_time_ ==  H._saved_symbolic_tensor_repr._modified_time_)
        H.symbolic_tensor_repr()
        self.assertFalse( symbolic._modified_time_ ==  H._saved_symbolic_tensor_repr._modified_time_)
        
        self.assertFalse( numerical_tensor._modified_time_ ==  H._saved_numerical_tensor_repr._modified_time_)
        self.assertTrue( post_processor ==  H._saved_array_post_processor)
        

        
        # extract status of attributes
        post_processed = H._saved_post_processed_array
        numerical = H._saved_numerical_array
        symbolic = H._saved_symbolic_tensor_repr
        numerical_tensor = H._saved_numerical_tensor_repr
        post_processor = H._saved_array_post_processor
        
        pre_processed = copy.copy(H.independent_vars.get_stored_attribute('preprocessed_array'))
        
        # changing the parameter dict should change numerical array and tensor repr but nothing else
        w = sympy.symbols('w')
        print(H.independent_vars.get_stored_attribute('parameter_dict')._modified_time_)
        H.independent_vars['parameter_dict'][w] = 0        
        print(H.independent_vars.get_stored_attribute('parameter_dict')._modified_time_)
        H.numerical_tensor_repr() # updates everything
        H.symbolic_tensor_repr()
        
        self.assertTrue( pre_processed._modified_time_ ==  H.independent_vars.get_stored_attribute('preprocessed_array')._modified_time_)
        self.assertTrue( post_processed ==  H._saved_post_processed_array)
        print({k:(v.snapshot_time,v.attribute._modified_time_) for k,v in numerical.dependencies.items()})
        print('\n')
        print({k:(v.snapshot_time,v.attribute._modified_time_) for k,v in H._saved_numerical_array.dependencies.items()})
        print('\n')
        print({k:H.independent_vars.get_stored_attribute(k)._modified_time_ for k in H.independent_vars.keys()})
        self.assertFalse( numerical._modified_time_ ==  H._saved_numerical_array._modified_time_)
        self.assertTrue( symbolic._modified_time_ ==  H._saved_symbolic_tensor_repr._modified_time_)
        self.assertFalse( numerical_tensor._modified_time_ ==  H._saved_numerical_tensor_repr._modified_time_)
        self.assertTrue( post_processor ==  H._saved_array_post_processor)


        H.independent_vars['postprocessing_function_specification']['test'] = lambda x,model:2*x
        
        # extract status of attributes
        post_processed = H._saved_post_processed_array
        numerical = H._saved_numerical_array
        symbolic = H._saved_symbolic_tensor_repr
        numerical_tensor = H._saved_numerical_tensor_repr
        post_processor = H._saved_array_post_processor
        pre_processed = H.independent_vars.get_stored_attribute('preprocessed_array')

        H.numerical_tensor_repr() # updates everything
        H.symbolic_tensor_repr()
        self.assertTrue( pre_processed ==  H.independent_vars.get_stored_attribute('preprocessed_array'))
        self.assertFalse( post_processed ==  H._saved_post_processed_array)
        self.assertFalse( numerical._modified_time_ ==  H._saved_numerical_array._modified_time_)
        self.assertFalse( symbolic._modified_time_ ==  H._saved_symbolic_tensor_repr._modified_time_)
        self.assertFalse( numerical_tensor._modified_time_ ==  H._saved_numerical_tensor_repr._modified_time_)
        self.assertFalse( post_processor ==  H._saved_array_post_processor)
        
        # test if the syncing is correctly updated when:
        # - just assembling the array
        # altering the underlying array (changes all)
        # altering a parameter in the dict (changes numerical and tensor)
        # altering the postprocessing func (changes numerical, tensor but nothing else)
    
    def test_array_post_processor(self):
        from unittest.mock import MagicMock
    
        func_A = MagicMock(return_value=1)
        del func_A._modified_time_
        func_B = MagicMock(return_value=2)
        del func_B._modified_time_
        func_C = MagicMock(return_value=3)
        del func_C._modified_time_
        H = self.scalar_problem
        
        H.independent_vars['postprocessing_function_specification']['test'] = func_A
        H.independent_vars['postprocessing_function_specification']['BdG'] = func_B
        
        res=H.post_processed_array()
        func_A.assert_called_once_with(H.independent_vars['preprocessed_array'],model=H) #model must be kwarg because this is how it is defined in partial
        func_B.assert_called_once_with(1,model=H)
        self.assertTrue(res==2)

        
        # test that updating also works
        H.independent_vars['postprocessing_function_specification']['A-field'] = func_C
        res = H.post_processed_array()
        func_C.assert_called_once_with(1,model=H)
        func_B.assert_called_with(3,model=H)
        
    def test_make_array(self): 
        # test that the array is correctly built with and without a post-processing step
        from unittest.mock import MagicMock
        
        H = self.scalar_problem
        
        res=H.post_processed_array()# makes the array regularly
        self.assertTrue(__compare_sympy_arrays__(res,self.scalar_A),msg=f'post_processed array was not correct: expected \n{self.scalar_A}\n got:\n{res}')
        
        # add post_processor (casting as nparray) and verify that we get the correct thing out again
        def func(x,model):
            pass
            return np.array(x)
        
        mock = MagicMock(side_effect=func)
        del mock._modified_time_
        
        H.independent_vars['postprocessing_function_specification']['test'] =  mock
        
        res = H.post_processed_array()
        mock.assert_called()
        self.assertTrue(isinstance(res,np.ndarray),msg=f'post processing function not called correctly')
        
        self.assertTrue(__compare_sympy_arrays__(sympy.Array(res),self.scalar_A),msg=f'post_processed array was not correct: expected \n{self.scalar_A}\n got:\n{res}')
    
    def test_equality(self):
        self.assertTrue(self.scalar_problem == self.scalar_problem)
        self.assertFalse(self.scalar_problem == self.spinor_problem)
        self.assertFalse(self.spinor_problem == 1)
        
    def test_saving_and_loading(self):
        from unittest.mock import mock_open,patch
        import pickle as pkl
        M = mock_open()
        self.scalar_problem.independent_vars['function_dict'][self.funcs[0][0]] = sympy.sympify(1) # overwrite since Mock is not pickleable
        self.spinor_problem.independent_vars['function_dict'][self.funcs[0][0]] = sympy.sympify(1) # overwrite since Mock is not pickleable
        with patch('builtins.open',M):
            with open('testsave','wb') as f:
                pkl.dump(self.scalar_problem,f)
                pkl.dump(self.spinor_problem,f)
            
        Mscalar = mock_open(read_data=pkl.dumps(self.scalar_problem))
        with patch('builtins.open',Mscalar):
            with open('testsave1','rb') as f:
                loaded_scalar = pkl.load(f)
                self.assertEqual(loaded_scalar,self.scalar_problem)
                
            
        Mspinor = mock_open(read_data=pkl.dumps(self.spinor_problem))
        with patch('builtins.open',Mspinor):
            with open('testsave2','rb') as f:
                loaded_spinor = pkl.load(f)
                self.assertEqual(loaded_spinor,self.spinor_problem)    
        
    def test_eig(self):
        rng_seed=np.random.default_rng().integers(0,1024)
        
        rng = np.random.default_rng(rng_seed)
        k_vec = rng.uniform(0, 10, 3)
        position = rng.uniform(low=0,high=1,size=3)
        result = self.spinor_problem.eig(k_vec, drop_eigenvectors=False,position=position)
        Ks = sympy.symbols('k_{x},k_{y},k_{z}',commutative=False)
        a,b,c = sympy.symbols('a,b,c')
        subs_dict = {a:1,b:2,c:3.14}
        subs_dict.update({Ks[i]:k_vec[i] for i in range(3)})
        x,y,z = sympy.symbols('x,y,z')
        subs_dict.update({v[0]:v[1].expression.subs({'x':x,'y':y,'z':z}) for v in self.funcs[1:]})
        subs_dict[self.funcs[0][0]] = self.funcs[0][1](*(x,y,z))
        subs_dict.update({s:v for s,v in zip((x,y,z),position)})
        facit = sorted((self.spinor_A_00_element.subs(subs_dict), self.spinor_A_11_element.subs(subs_dict)))
        sorted_res = sorted(result[0])
        self.assertAlmostEqual(sorted_res[0],facit[0],msg=f'seed = {rng_seed}')
        self.assertAlmostEqual(sorted_res[1],facit[1],msg=f'seed = {rng_seed}')
        
        
        
        #check shape of more vectors passed:
        N= 2
        k_vec = rng.uniform(0,10,(N,3))
        result = self.spinor_problem.eig(k_vec, drop_eigenvectors=False,position=position)
        self.assertTrue(isinstance(result[0],np.ndarray))
        self.assertTrue(result[0].shape==(N,2))
        for i in range(N):
            subs_dict.update({Ks[j]:k_vec[i,j] for j in range(3)})
            facit = sorted((self.spinor_A_00_element.subs(subs_dict), self.spinor_A_11_element.subs(subs_dict)))
            
            sorted_res = sorted(result[0][i])
            self.assertAlmostEqual(sorted_res[0],facit[0])
            self.assertAlmostEqual(sorted_res[1],facit[1])
        
    def test_spectrum(self):
        k_range = (-10, 10)
        n_points = 100
        facit = np.linspace(k_range[0], k_range[1], n_points)
        rng_seed = np.random.default_rng().integers(0,1024)
        rng_seed=692
        rng = np.random.default_rng(rng_seed)
        position = rng.uniform(low=0,high=1,size=3)
        Ks = sympy.symbols('k_{x},k_{y},k_{z}',commutative=False)
        a,b,c = sympy.symbols('a,b,c')
        subs_dict = {a:1,b:2,c:3.14}
        subs_dict.update({Ks[1]:0,Ks[2]:0}) 
        x,y,z = sympy.symbols('x,y,z')
        subs_dict.update({v[0]:v[1].expression for v in self.funcs[1:]})
        subs_dict[self.funcs[0][0]] = self.funcs[0][1](*(x,y,z))
        subs_dict.update({s:v for s,v in zip((x,y,z),position)})
        facit_0_func = sympy.lambdify(Ks[0],self.spinor_A_00_element.subs(subs_dict))
        facit_1_func = sympy.lambdify(Ks[0],self.spinor_A_11_element.subs(subs_dict))
        facits = np.sort(np.array([[facit_0_func(k) for k in facit],[facit_1_func(k) for k in facit]]),axis=0)
        
        k_res, res = self.spinor_problem.spectrum(k_range=k_range, k_direction=[1,0,0], n_points=n_points,position=position)
        res = np.sort(res,axis=1)
        
        np.testing.assert_array_equal(facit, k_res,err_msg=f'seed={rng_seed}')
        np.testing.assert_array_almost_equal(facits.T, res,err_msg=f'seed={rng_seed}')
        
        
        #test tensor problem to make sure that the array is reshaped correctly!
        tensor_matrix = sympy.Array(np.array(self.tensor_A).transpose((0,2,1,3)).reshape(4,4)).subs(subs_dict)
        mat_func = sympy.lambdify(Ks[0],tensor_matrix)
        spectrum = [np.linalg.eigh(mat_func(k))[0] for k in facit]
        
        facit = self.tensor_problem.spectrum(k_range,[1,0,0],n_points,position=position)
        np.testing.assert_allclose(spectrum,facit[1])
        
        

    def test_material_spec(self):
        # test that the parameters are correctly added to the parameter dict
        self.spinor_problem.material_spec('Ge')
        from nqcpfem import _m_e
        facit = {r'\Delta_{0}': 0.296,
                 r'\gamma_{1}': 13.38,
                 r'\gamma_{2}': 4.24,
                 r'\gamma_{3}': 5.69,
                 r'\kappa': 3.41,
                 r'q': 0.06,
                 r'D_{u}': 3.3195,
                 r"D'_{u}": 5.7158,
                 r'c_{11}': 12.40,
                 r'c_{12}': 4.13,
                 r'c_{44}': 6.83,
                 r'a': 5.65791,
                 r'\epsilon': 16.5,
                 r'm': _m_e}
        facit = {sympy.symbols(k):v for k,v in facit.items()}
        self.assertEqual({k:self.spinor_problem.independent_vars['parameter_dict'][k] for k in facit.keys()}, facit)

    def test_add_potential(self):
        
        x,y,z,W,C = sympy.symbols('x,y,z,W,C')
        pot = W*x*y+z+C
        H = self.scalar_problem
        H.add_potential(pot)
        self.assertTrue(W in H.independent_vars['parameter_dict'].keys(),msg='parameter not added to parameter dict') 
        self.assertTrue(C in H.independent_vars['parameter_dict'].keys(),msg='parameter not added to parameter dict') 
        self.assertFalse(any(x in H.independent_vars['parameter_dict'].keys() for x in ('x','y','z')),msg='coordinates wrongly added to parameter_dict')
        self.assertTrue(__compare_sympy_arrays__(H.independent_vars['preprocessed_array'],self.scalar_A+sympy.Array([[pot]])),msg='potential not correctly added to hamiltonian')
        
        H = self.spinor_problem
        H.add_potential(pot,{W:10},{C:5})
        self.assertTrue(W in H.independent_vars['parameter_dict'].keys(),msg='parameter not added to parameter dict') 
        self.assertTrue(H.independent_vars['parameter_dict'][W] == 10,msg='parameter not added to parameter dict') 
        self.assertFalse(C in H.independent_vars['parameter_dict'].keys(),msg='constant wrongly added to parameter dict') 
        self.assertTrue(C in H.independent_vars['constants'].keys(),msg='constnat not added to constants dict') 
        self.assertTrue(H.independent_vars['constants'][C] == 5,msg='constnat not added to constants dict') 
        self.assertFalse(any(x in H.independent_vars['parameter_dict'].keys() for x in ('x','y','z')),msg='coordinates wrongly added to parameter_dict')
        self.assertTrue(__compare_sympy_arrays__(H.independent_vars['preprocessed_array'],self.spinor_A+sympy.Array([[pot,0],[0,pot]])),msg='potential not correctly added to hamiltonian')
        
        with self.assertRaises(ValueError,msg='should complain that symbols was both parameter and constantnt'):
            H.add_potential(pot,{W:10},{W:15})
            
    def test_add_z_confinement(self):
        self.scalar_problem.independent_vars['preprocessed_array'] = self.scalar_problem.independent_vars['preprocessed_array'].subs({sympy.symbols('V1(x)',commutative=False):0}) # ignore V1 for now until we have numerical integration in place
        del self.scalar_problem.independent_vars['function_dict'][sympy.symbols('V1(x)',commutative=False)]
        self.scalar_problem.add_z_confinement(nz_modes=2,lz=0.1,z_confinement_type='box')
        self.assertTrue(self.scalar_problem.post_processed_array().shape== (1,1,2,2))
        self.assertTrue(all( s not in self.scalar_problem.post_processed_array().free_symbols for s in (sympy.symbols('k_{z}',commutative=False),sympy.symbols('z'))),msg='kz and/or z were not all replaced')
        V1,V2,V3 = [v[0] for v in self.funcs]
        res = self.scalar_problem.post_processed_array().subs({k:0 for k in self.scalar_problem.momentum_symbols})
        V2s = sympy.Symbol('(V2)_1(x)',commutative=False)
        facit = sympy.Array([[[[sympy.pi**2/sympy.symbols('l_z')**2+V2s+V3,0],[0,4*sympy.pi**2/sympy.symbols('l_z')**2+V2s+V3]]]])
        self.assertTrue(__compare_sympy_arrays__(res,facit))

        
    def test_BdG_extension(self):
        from nqcpfem.band_model import FreeBoson,FreeFermion
        
        
        # region scalar problem
        H = self.scalar_problem
        H.__time_reversal_change_of_basis__ = sympy.Array([[1]])
        
        H.BdG_extension()
        
        self.assertTrue('BdG' in H.independent_vars['postprocessing_function_specification'])
        
        A=H.post_processed_array() # verify that shape of array is 2 by 2 and ks are replaced by their minus sign
        self.assertTrue(A.shape==(2,2))
        self.assertTrue(__compare_sympy_arrays__(sympy.Array([[A[0,0]]]),self.scalar_A),msg='particle part was wrong')
        kx,ky,kz = sympy.symbols('k_{x},k_{y},k_{z}',commutative=False)
        self.constants = sympy.symbols('a,b,c')
        a,b,c = self.constants
        C = lambda x: sympy.conjugate(x)
        V1,V2,V3 = [v[0] for v in self.funcs]
        subs_dict = {a:C(a),b:C(b),c:C(c),kx:-kx,ky:-ky,kz:-kz,V1:C(V1),V2:C(V2),V3:C(V3)}
        self.assertTrue(__compare_sympy_arrays__(sympy.Array([[A[1,1]]]),self.scalar_A.subs(subs_dict)),msg='anti-particle part was wrong') #NB there may be a minus sign wrong here!
        #endregion
        
        # region spinor problem
        H = self.spinor_problem
        H.__time_reversal_change_of_basis__ = sympy.Array([[0,1],[-1,0]])
        
        H.BdG_extension()
        
        self.assertTrue('BdG' in H.independent_vars['postprocessing_function_specification'])
        
        A=H.post_processed_array() # verify that shape of array is 2 by 2 and ks are replaced by their minus sign
        self.assertTrue(A.shape==(4,4))
        self.assertTrue(__compare_sympy_arrays__(A[0:2,0:2],self.spinor_A),msg='particle part was wrong')
        kx,ky,kz = sympy.symbols('k_{x},k_{y},k_{z}',commutative=False)
        self.constants = sympy.symbols('a,b,c')
        a,b,c = self.constants
        C = lambda x: sympy.conjugate(x)
        time_reversed_version = sympy.Array([[self.spinor_A_11_element,0],[0,self.spinor_A_00_element]]).subs(subs_dict)
        
        
        self.assertTrue(__compare_sympy_arrays__(A[2:,2:],time_reversed_version),msg='anti-particle part was wrong') #NB there may be a minus sign wrong here!
        #endregion
        
        # region tensor problem
        H = self.tensor_problem # mock spinor problem with z-confinement
        H.__time_reversal_change_of_basis__ = sympy.Array([[0,1],[-1,0]])
        
        H.BdG_extension()
        
        self.assertTrue('BdG' in H.independent_vars['postprocessing_function_specification'])
        
        A=H.post_processed_array() # verify that shape of array is 2 by 2 and ks are replaced by their minus sign
        self.assertTrue(A.shape==(4,4,2,2))
        self.assertTrue(__compare_sympy_arrays__(A[0:2,0:2],self.tensor_A),msg='particle part was wrong')
        kx,ky,kz = sympy.symbols('k_{x},k_{y},k_{z}',commutative=False)
        self.constants = sympy.symbols('a,b,c')
        a,b,c = self.constants
        C = lambda x: sympy.conjugate(x)
        time_reversed_skeleton = np.full((2,2,2,2),sympy.sympify(0))
        time_reversed_skeleton[0,0,] = np.array(self.tensor_A[1,1,]) 
        time_reversed_skeleton[0,1,] = np.array(-self.tensor_A[1,0,]) 
        time_reversed_skeleton[1,0,] = np.array(-self.tensor_A[0,1,])
        time_reversed_skeleton[1,1,] = np.array(self.tensor_A[0,0,]) 
        
        time_reversed_version = sympy.Array(time_reversed_skeleton).subs(subs_dict)
        
        
        self.assertTrue(__compare_sympy_arrays__(A[2:,2:],time_reversed_version),msg='anti-particle part was wrong') #NB there may be a minus sign wrong here!
        #endregion
        
        
    
    def test_SC_order_parameter(self):
        self.scalar_problem.__time_reversal_change_of_basis__ = sympy.Array([[1]])
        
        
        scalar_order_parameter = sympy.Array([[10j]]) 
        self.scalar_problem.independent_vars['SC_Delta'] = scalar_order_parameter
        
        self.scalar_problem.BdG_extension()
        A=self.scalar_problem.post_processed_array() # verify that shape of array is 2 by 2 and ks are replaced by their minus sign
        self.assertTrue(__compare_sympy_arrays__(sympy.Array([[A[0,1]]]),scalar_order_parameter),msg='particle part was wrong')
        self.assertTrue(__compare_sympy_arrays__(sympy.Array([[A[1,0]]]),scalar_order_parameter.conjugate()),msg='particle part was wrong')
    
    
    def test_add_vector_field(self):

        from nqcpfem import constants,values
        hbar = constants['hbar']
        A_field = [values[constants['hbar']],0,0]
        self.scalar_problem.add_vector_potential(A_field,charge=np.pi)
        H=self.scalar_problem.post_processed_array()
        kx,ky,kz = self.scalar_problem.momentum_symbols
        ckx,cky,ckz = sympy.symbols('k_{x},k_{y},k_{z}',commutative=False)
        
        AA = self.scalar_A.copy()
        
        e = sympy.symbols('q_{c}')
        import nqcpfem
        A_sym = sympy.symbols(nqcpfem.band_model.__VECTOR_FIELD_NAMES__)
        facit = self.scalar_A.copy().subs({kx:(kx+e*A_sym[0]/hbar), ky:(ky+e*A_sym[1]/hbar), kz:(kz+e*A_sym[2]/hbar)})
        self.assertTrue(__compare_sympy_arrays__(H,facit),msg='scalar problem did not get correct substitution')

        H_num = self.scalar_problem.numerical_array()
        e = np.pi
        hbar = values[constants['hbar']]
        subs_dict = {s:n for s,n in zip(self.constants,(1,2,3.14))}
        subs_dict.update({f[0]:f[1].expression for f in self.funcs[1:]})
        facit_num= AA.copy().subs({kx:(ckx+e*A_field[0]/hbar), ky:(cky+e*A_field[1]/hbar), kz:(ckz+e*A_field[2]/hbar)}).subs(subs_dict)
        diff = facit_num - sympy.Array(H_num)
        diff = diff.subs({k:1 for k in (kx,ky,kz)})
        np.testing.assert_allclose(np.array(diff).astype(float),np.zeros(diff.shape),atol=1e-15)


        
    def test_fix_position(self):
        V1,V2,V3 = [f[0] for f in self.funcs]
        x,y,z = self.spinor_problem.position_symbols
        testarray = sympy.Array([[x,y,z,],[V1,V2,V3]])
        rng_seed = np.random.default_rng().integers(0,1024)
        rng = np.random.default_rng(rng_seed)
        pos = rng.uniform(0,10,3)
        res= self.spinor_problem.fix_position(testarray,*pos)
        subs_dict = {f[0]:f[1](*(pos)) for f in self.funcs}
        subs_dict.update({x:p for x,p in zip(self.spinor_problem.position_symbols,pos)})
        self.assertEqual(res,testarray.subs(subs_dict),msg=f'seed = {rng_seed}. Single sympy array not fixed correctly')
        
        
        test2 = sympy.Array([[V1*x+2,V2*y+3,V3*3],[x,y,z]])
        test_dict = {0:testarray,'a':np.array(test2)}
        facit_dict = {0:testarray.subs(subs_dict),'a':np.array(test2.subs(subs_dict))}
        res = self.spinor_problem.fix_position(test_dict,*pos)
        self.assertEqual(facit_dict[0],res[0],msg=f'seed = {rng_seed}. Dict of sympy array not fixed corectly')
        self.assertTrue((facit_dict['a']==res['a']).all(),msg=f'seed = {rng_seed}. Dict of numpy array not fixed correctly')
    
    
class TestFreeBoson(TestCase):
    def test_init(self):
        from nqcpfem.band_model import FreeBoson,__MOMENTUM_NAMES__
        instance = FreeBoson(mass=1,spatial_dim = 3)
        m = sympy.symbols('m')
        self.assertTrue(instance.independent_vars['parameter_dict'][m]==1)
        self.assertTrue(instance.spatial_dim==3)
        kx,ky,kz = sympy.symbols(__MOMENTUM_NAMES__,commutative=False)
        ckx,cky,ckz = sympy.symbols(__MOMENTUM_NAMES__,commutative=False)
        from nqcpfem import constants,values
        self.assertTrue(instance.post_processed_array() == sympy.Array([[constants['hbar']**2/(2*m) *(kx**2+ky**2+kz**2)]]))
        
        np.testing.assert_array_equal(instance.numerical_array(), np.array([[values[constants['hbar']]**2/(2*1)*(ckx**2+cky**2+ckz**2)]]))

class testFreeFermion(TestCase):
    def test_init(self):
        from nqcpfem.band_model import FreeFermion,__MOMENTUM_NAMES__
        from nqcpfem import constants,values
        instance = FreeFermion(mass=1,spatial_dim = 3)
        m = sympy.symbols('m')
        self.assertTrue(instance.independent_vars['parameter_dict'][m]==1)
        self.assertTrue(instance.spatial_dim==3)
        kx,ky,kz = sympy.symbols(__MOMENTUM_NAMES__,commutative=False)
        print('###############################')
        print(instance.post_processed_array())
        self.assertTrue(instance.post_processed_array() == sympy.Array([[constants['hbar']**2/(2*m) *(kx**2+ky**2+kz**2),0],[0,constants['hbar']**2/(2*m) *(kx**2+ky**2+kz**2)]]))

    def test_add_zeeman_term(self):
        #add zeemann term first with B-field specified.
        #  for scalar g_tensor (with and without tensorize)
        # tensor g-factor as well. Check that parameters are added correctly and that the term is also correct by comparing post-processed array.
        
        #make new instance where an A field is added (get name correct) and check that adding term with B specified raises error. Then specify without B-field and check that it is correct.
        from nqcpfem.band_model import FreeFermion,__MOMENTUM_NAMES__,__MAGNETIC_FIELD_NAMES__
        from nqcpfem import constants,values
        hbar = constants['hbar']
        mu_B = constants['mu_B']
        kx,ky,kz = sympy.symbols(__MOMENTUM_NAMES__,commutative=False)
        ckx,cky,ckz = sympy.symbols(__MOMENTUM_NAMES__,commutative=False)

        instance = FreeFermion(mass=2,spatial_dim=3)
        g_tensor = 3 
        Bvec = [4,5,6]
        g,m = sympy.symbols('g,m')
        kinetic_term =  sympy.Array([[hbar**2/(2*m) *(kx**2+ky**2+kz**2),0],[0,hbar**2/(2*m) *(kx**2+ky**2+kz**2)]])
        Bx,By,Bz = sympy.symbols(__MAGNETIC_FIELD_NAMES__)
        magnetic_term = mu_B*g*0.5* sympy.Array([[Bz,Bx-1j*By],[Bx+1j*By,-Bz]])
        instance.add_zeeman_term(g_tensor=g_tensor,Bvec=Bvec)
        
        #testing that the array is correct
        self.assertTrue(__compare_sympy_arrays__(instance.post_processed_array(),kinetic_term+magnetic_term),msg=f'Arrays did not match:\n result:\n {instance.post_processed_array}\n expected:\n {kinetic_term+magnetic_term}\n Difference:\n {(instance.post_processed_array()-(kinetic_term+magnetic_term)).simplify()}')

        #testing that the values are substituted correctly:
        numeric_facit = kinetic_term.subs({m:2,hbar:values[hbar]})+magnetic_term.subs({Bx:4,By:5,Bz:6,g:3,mu_B:values[mu_B]})
        numeric_facit = numeric_facit.subs({kx:ckx,ky:cky,kz:ckz}) # replace with commuting symbols
        self.assertTrue(__compare_sympy_arrays__(sympy.Array(instance.numerical_array()),numeric_facit),msg=f'Arrays did not match:\n result:\n {instance.numerical_array()}\n expected:\n {numeric_facit}\n Difference:\n {(sympy.Array(instance.numerical_array())-(numeric_facit)).simplify()}')
        
class TestLuttingerKohnHamiltonian(TestCase):
    def test_init(self):
        import nqcpfem
        from nqcpfem.band_model import FreeBoson,FreeFermion
        H = nqcpfem.band_model.LuttingerKohnHamiltonian()
        self.assertTrue(all(sympy.symbols(k) in H.independent_vars['parameter_dict'].keys() for k in (r'\gamma_{1}',r'\gamma_{2}',r'\gamma_{3}','m',)))
        self.assertTrue(__compare_sympy_arrays__(H.independent_vars['preprocessed_array'] ,nqcpfem.band_model.LuttingerKohnHamiltonian.__make_LK_Hamiltonian__()))

        K = sympy.symbols([s.name for s in H.momentum_symbols],commutative=False)
        g1,g2,g3 = sympy.symbols(r'\gamma_{1},\gamma_{2},\gamma_{3}')
        h = nqcpfem.constants['hbar']
        A = h**2/(2*sympy.symbols('m'))
        """SIGN IS WRONG HERE!
        P = A*g1*(K[0]**2+K[1]**2+K[2]**2)
        Q = A*g2*(K[0]**2+K[1]**2-2*K[2]**2)
        R = A*sympy.sqrt(3)*(-g2*(K[0]**2-K[1]**2)+sympy.I*g3*(K[0]*K[1]+K[1]*K[0]))
        S = A*sympy.sqrt(3)*g3*(K[0]*K[2]+K[2]*K[0]-sympy.I*(K[1]*K[2]+K[2]*K[1]))
        
        facit = sympy.Array([[P+Q,S,R,0],[-S.conjugate(),P-Q,0,R],[R.conjugate(),0,P-Q,S],[0,R.conjugate(),S.conjugate(),P+Q]])
        
        self.assertTrue(__compare_sympy_arrays__(H.preprocessed_array ,facit))
        """
        from nqcpfem import SP_ANGULAR_MOMENTUM as AM
        J = AM['3/2']
        Id = sympy.Matrix([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]])
        K2 = [k*k for k in K]
        
        facit = (g1+5/2*g2)*(K2[0]+K2[1]+K2[2])*Id
        facit = facit - 2*g2*sum([K2[i]*(J[i]@J[i]) for i in range(3)],0*Id) 
        facit = facit - g3*sum([(K[(i+1)%3]*K[(i+2)%3]+K[(i+2)%3]*K[(i+1)%3])*(J[(i+1)%3]*J[(i+2)%3]+J[(i+2)%3]*J[(i+1)%3]) for i in range(3)],Id*0) 
        facit = sympy.Array(A*facit)
        self.assertTrue(__compare_sympy_arrays__(H.independent_vars['preprocessed_array'] ,facit))

        # LK Hamiltonian construction is in agreement with WInkler book and https://link.aps.org/doi/10.1103/PRXQuantum.2.010348
    
    def test_add_zeeman_term(self):
        
        
        # setting all parameters to zero excep kappa should give us:
        import nqcpfem
        from nqcpfem.band_model import FreeBoson,FreeFermion
        from nqcpfem import constants,SP_ANGULAR_MOMENTUM
        import nqcpfem
        kappa,q,Bx,By,Bz = sympy.symbols(r'\kappa,q, B_{x}(x),B_{y}(x),B_{z}(x)')
        J = SP_ANGULAR_MOMENTUM['3/2']
        
        
        kappa_facit = -2*kappa*constants['mu_B']*(J[0]*Bx+J[1]*By+J[2]*Bz)
        kappa_facit = sympy.Array(kappa_facit)
        q_facit = -2*q*constants['mu_B']*(J[0] @J[0]@J[0]*Bx+J[1] @J[1]@J[1]*By+J[2] @J[2]@J[2]*Bz)
        q_facit = sympy.Array(q_facit)
        
        H = nqcpfem.band_model.LuttingerKohnHamiltonian().add_zeeman_term()
        
        array = H.post_processed_array()
        
        ss = lambda x: sympy.symbols(x)
        symbol_subs = {ss(r'\gamma_{1}'):0, ss(r'\gamma_{2}'):0 ,ss(r'\gamma_{3}'):0}
        symbol_subs[q] = 0
        kappa_result = array.subs(symbol_subs)
        self.assertTrue(__compare_sympy_arrays__(kappa_result,kappa_facit))
        
        del symbol_subs[q]
        symbol_subs[kappa] = 0
        q_result = array.subs(symbol_subs)
        self.assertTrue(__compare_sympy_arrays__(q_result,q_facit))
        

        symbol_subs[kappa]=10
        symbol_subs[q]=20
        symbol_subs[ss('m')] = constants['hbar']**2
        symbol_subs[ss('q_{c}')]=1
        x,y,z = sympy.symbols('x,y,z')
        A_field = nqcpfem.band_model.make_vector_field([1,2,3],[x,y,z])
        
        H.add_vector_potential(A_field)
        H.independent_vars['parameter_dict'].update(symbol_subs)
        result = sympy.Array(H.numerical_array())
        # make sure that the we get the correct B-field term. All k-terms should vanish since we have gammas set to zero
        facit = kappa_facit+q_facit
        facit=facit.subs({Bx:1,By:2,Bz:3,kappa:10,q:20,constants['mu_B']:nqcpfem.values[constants['mu_B']]})     
        np.testing.assert_array_almost_equal(facit,result,err_msg='vector field differentiation and numerical evaluation did not give the correct result')
        
        
    def test_rotate_crystal(self):
        import nqcpfem
        
        import nqcpfem
        from nqcpfem.band_model import FreeBoson,FreeFermion
        from nqcpfem import constants,SP_ANGULAR_MOMENTUM
        
        
        H = nqcpfem.band_model.LuttingerKohnHamiltonian().rotate_crystallographic_direction(2*sympy.pi,2*sympy.pi)#(theta=sympy.pi,phi=sympy.pi)
        
        # this rotation should not change anything so facit is regular array
        facit = H.__make_LK_Hamiltonian__()
        
        
        result = H.post_processed_array()
        theta,phi = sympy.symbols(r'\theta,\phi')
        
        self.assertTrue(theta in result.free_symbols and phi in result.free_symbols,msg='theta and phi where not in the free symbols of the postprocessed array')
        result = result.subs({theta:H.independent_vars['parameter_dict'][theta], phi:H.independent_vars['parameter_dict'][phi]})
        self.assertTrue(__compare_sympy_arrays__(facit,result),msg=f'not correct: difference: {(result-facit).simplify()}') # this rotation should leve everything unchanged
        
        
        
if __name__ == '__main__':
    unittest.main()