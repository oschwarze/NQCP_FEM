from unittest import TestCase

from scipy import constants

import os,sys

src_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.append(src_path)
print(src_path)


class TestIndependentAttribute(TestCase):
    
    def test_constant_dict(self):
        from nqcpfem.updatable_object import ConstantsDict
        
        test = ConstantsDict({'a':1,'b':2,'c':3})
        
        t = test.get_stored_attribute('a')._modified_time_
        self.assertFalse(test.get_stored_attribute('a')._was_used_)
        
        _ = test['a']
        self.assertTrue(test.get_stored_attribute('a')._was_used_)
        
        """
        used_dependencies = test.snapshot_used_dependencies()
        self.assertEqual(len(used_dependencies),1)
        self.assertEqual(used_dependencies.values()[0].attribute, test.get_stored_attribute('a'))
        self.assertEqual(used_dependencies.values()[0].snapshot_time, test.get_stored_attribute('a')._modified_time_)
        """
        test['a'] = 10
        attr = test.get_stored_attribute('a') 
        self.assertEqual(attr.value , 10)
        self.assertNotEqual(attr._modified_time_, t)
        """
        self.assertEqual(used_dependencies.values()[0].attribute, test.get_stored_attribute('a'))
        self.assertNotEqual(used_dependencies.values()[0].snapshot_time, test.get_stored_attribute('a')._modified_time_)
        """
        
        
        test['d'] = {'1':1}
        self.assertTrue(isinstance(test.get_stored_attribute('d').value,ConstantsDict))
        T = test.get_stored_attribute('d')._modified_time_
        import time
        time.sleep(1)
        
        test['d']['2'] = 2
        self.assertTrue('2' in test.get_stored_attribute('d').value.keys())
        self.assertEqual(test.get_stored_attribute('d').value['2'] , 2)
        test.__update_changeable_elements__()
        self.assertTrue(test.get_stored_attribute('d')._modified_time_>T,msg=f'modified_time not updated: {test.get_stored_attribute("d")._modified_time_} vs {T}')
        
        
    def test_AttributeSnapshot(self):
        from nqcpfem.updatable_object import IndependentAttribute,ComputedAttribute,AttributeSnapshot
        
        test = AttributeSnapshot(IndependentAttribute(1))
        
        old_t = test.snapshot_time
        import time
        test.attribute._modified_time_ = time.time()
        self.assertTrue(test.has_changed())
        
        new_test = test.update()
        self.assertFalse(new_test.has_changed())
        self.assertEqual(new_test.snapshot_time,new_test.attribute._modified_time_)
        
    def test_Computed_Attr(self):
        from nqcpfem.updatable_object import IndependentAttribute,ComputedAttribute,AttributeSnapshot
        dependency = AttributeSnapshot(IndependentAttribute(1))
        test = ComputedAttribute(10,{'a':dependency})
        
        
        O = test._modified_time_
        import time
        dependency.attribute.value = 101
        T = time.time()
        dependency.attribute._modified_time_ = T
        old_t = dependency.snapshot_time
        self.assertNotEqual(T,old_t)
        
        
        test_b=test.update('a')
        self.assertNotEqual(test.value,'a',msg='updating did not create a new instance')
        
        
        
        self.assertEqual(test_b.value,'a',msg='updating did not create a new instance')
        self.assertEqual(test_b.dependencies['a'].snapshot_time,T)
        
    
    def test_updatable_object(self):
        from nqcpfem.updatable_object import UpdatableObject
        test = UpdatableObject(**{'a':1,'b':2,'c':3})
        
        self.assertEqual(list(test.independent_vars.keys()),['a','b','c'])
        
        test.reset_dependency_flags()
        self.assertEqual(test.snapshot_used_dependencies(),{})       
        
        a_attr = test.independent_vars.get_stored_attribute('a')
        t = a_attr._modified_time_
        _ = test.independent_vars['a'] 
        
        old_snapshot = test.snapshot_used_dependencies()
        self.assertEqual(len(old_snapshot),1)       
        self.assertEqual(list(old_snapshot.values())[0].attribute,a_attr)       
        
        self.assertEqual(list(old_snapshot.values())[0].attribute._modified_time_,list(old_snapshot.values())[0].snapshot_time)
        import time
        T = time.time()
        test.independent_vars['a'] = 10
        new_a_attr = test.independent_vars.get_stored_attribute('a')
        self.assertEqual(new_a_attr.value, 10)
        self.assertTrue(new_a_attr._modified_time_>T)
        self.assertNotEqual(list(old_snapshot.values())[0].attribute._modified_time_,list(old_snapshot.values())[0].snapshot_time)
        
        self.assertNotEqual(new_a_attr._modified_time_,t)
        
        from nqcpfem.updatable_object import auto_update

        # Subclassing must also be tested
        class Example(UpdatableObject):
        
            def __init__(self, a,b,c) -> None:
                kwargs = locals()
                del kwargs['self']
                super().__init__(**kwargs)
            
            @auto_update
            def A(self):
                print('in A')
                return self.independent_vars['a']+1
            
            @auto_update
            def B(self):
                print('in B')
                return self.A()+10
            
            @auto_update
            def C(self):
                print('in C')
                return self.A()+ self.independent_vars['c']
        
        test = Example(a=1,b=2,c=3)
        
        self.assertEqual(test.A(),2)
        self.assertEqual(test.B(),12)

        self.assertTrue(hasattr(test,'_saved_A'))
        self.assertFalse(test._saved_A.dependencies['a'].has_changed())
        self.assertTrue(hasattr(test,'_saved_B'))
        self.assertFalse(test._saved_B.dependencies['a'].has_changed())
        # test that the dependencies also work:
        test.independent_vars['a'] = 100
        
        
        
        self.assertTrue(test._saved_A.dependencies['a'].has_changed())
        self.assertTrue(test._saved_B.dependencies['a'].has_changed())
        # overwrite value to assure that it is correctly updated
        test._saved_A.value = None 
        test._saved_B.value = None 
        
        # making B should make both A and B
        self.assertEqual(test.B(),111)
        
        
        self.assertFalse(test._saved_A.dependencies['a'].has_changed())
        self.assertFalse(test._saved_B.dependencies['a'].has_changed())
        
        self.assertFalse(test._saved_B.value is None)
        self.assertFalse(test._saved_A.value is None)

        # making A now should return the existing value of A, because is should not need recomputing 
        test._saved_A.value = None 
        self.assertTrue(test.A() is None)
        
        
        # after updating a it should be recomputed:
        test.independent_vars['a'] +=1
        self.assertFalse(test.A() is None)
        
        
        
        # Remake test and check that A has th correct dependencies when initialized inside C
        test2 = Example(a=1,b=2,c=3)
        self.assertEqual(test2.C(),5)
        print(test2._saved_C.dependencies)
        print(test2._saved_A.dependencies)
        self.assertTrue(len(test2._saved_A.dependencies) == 1)
        self.assertTrue('a' in test2._saved_A.dependencies.keys())
        self.assertTrue(len(test2._saved_C.dependencies) == 2,msg=f'method C did not have the correct dependencies: {test2._saved_C.dependencies.keys()}')
        self.assertTrue('a' in test2._saved_C.dependencies.keys())
        self.assertTrue('c' in test2._saved_C.dependencies.keys())
        
        
        test2 = Example(a=100,b=2,c=3)
        
        class example2(UpdatableObject):
            def __init__(self, a2,b2,ex):

                super(example2, self).__init__(a2=a2,b2=b2,ex=ex)

            
            @auto_update
            def A2(self):
                print('In A2')
                return self.independent_vars['a2'] + self.independent_vars['ex'].A()

            
        ex2 = example2(100,200,test2)
        
        self.assertEqual(ex2.A2(),201)

        self.assertIsInstance(ex2.get_current_dependency_flags()['ex'],dict)
        self.assertIn('ex',ex2.get_current_dependency_flags())
        
        self.assertIsInstance(ex2._saved_A2.dependencies['ex'],dict)
        self.assertEqual(len(ex2._saved_A2.dependencies['ex']),1,msg=f'wrong dependencies: {ex2._saved_A2.dependencies["ex"]}')
        self.assertIn('a',ex2._saved_A2.dependencies['ex'])
        
        # calling again should reuse saved_value of A2 so changing this should not affect anything
        test2._saved_A.value = -10000 
        self.assertEqual(ex2.A2(),201)
        
        #this should now need updating
        test2.independent_vars['a'] = -100
        self.assertEqual(ex2.A2(),1)
        
    
    
    def test_method_with_arguments(self):
        self.fail('todo: make version of auto_update which requires hasable arguments to be passed to the method but keeps a dict of the stored values and invalidates all of them (resets the dict) if something is updated')