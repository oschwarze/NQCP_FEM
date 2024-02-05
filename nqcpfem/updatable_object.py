"""This module creates and object where every attribute is updatable. 



# Wrapper for constant attributes.
# the constant attributes should be the all the things that are passed to the init.

# Wrapper for Computed Attributes





# An updatable object has a GLOBAL modified time which changes whenever one of the constants change. 




## CONSTANTS
# all constants are stored as values and modified times. in the attributes `_constant_[]_`. 
# For each constant, there is a get_[] method and a, `set_[]` method 
# the Getattr returns the value, and the setattr alters the value and updates the modified time of the constant and the entire object

## TRACEABLE CONSTANTS
# like a constant, but whenever the getattr.value is called, a flag called __was_used__ is set to True.


## DERIVED VALUES
# Derived values are retrieved using methods called "get_[]". they take no arguments and check that the dependencies of the computed value hasn't changed.
# If they have it computes it using the _make_[]_ method.
# The return of the _make_[]_ function is stored in an attribute called "_saved_[]_" as a value, a list of DEPENDENCIES: which are the constants and the time.
# The dependencies of each make function is determined at runtime, where the constants are replaced by TRACEABLE CONSTANTS. 



### COMPUTING THE DERIVED VALUES
# they are computed using methods called "_make_[]_" which take NO arguments and simply returns the result of the computation (SEE WRAPPER FOR THESE METHODS BELOW).




# WRAPPER FOR THE _MAKE_[]
# the wrapper: sets the _was_used_flag of all constants to False if the instance's __dependency_check_active__ is False, else it does nothing 
# run the method.
# check which constants where used and return a COMPUTEDATTRIBUTE with the return of the method along with the relevant dependencies, and set all flags to None for good measure

"""
from typing import Any
import time
import sympy

from collections import namedtuple,UserDict





# The IndependentAttribute class is used to represent an independent attribute in a data set.
class IndependentAttribute():
    def __init__(self,value:Any):
        self.value = value
        self.__modified_time__ = time.time()
        self.__was_used__ = False 
        self._was_changed_ = False

    @property 
    def _was_used_(self):
        return self.__was_used__
    
    @_was_used_.setter
    def _was_used_(self,value):
        self.__was_used__ = value
    
    
    @property
    def _modified_time_(self):
        return self.__modified_time__

    @_modified_time_.setter 
    def _modified_time_(self,value):
        self.__modified_time__ = value

# The AttributeSnapshot class is used to store and manage snapshots of object attributes.
class AttributeSnapshot():
    
    def __init__(self,attr:IndependentAttribute):
        self.attribute = attr
        self.snapshot_time = attr._modified_time_

    def has_changed(self):
        try:
            self.attribute.__update_changeable_elements__()
        except AttributeError:
            pass
        return self.attribute._modified_time_ != self.snapshot_time

    def update(self,new_value = None):
        """
        Returns a new snapshot with the updated snapshot time
        """

        if new_value is not None:
            try:
                self.attribute.update(new_value)
            except AttributeError:
                pass
        new = self.__class__(self.attribute)
        return new



class ComputedAttribute(IndependentAttribute):
    def __init__(self,value:Any,dependencies:dict[str,AttributeSnapshot]):
        
        super().__init__(value)
        self.dependencies = dependencies
        
    
    def update(self,new_value):
        new_dependencies = {k:dep.update() for k,dep in self.dependencies.items()}
        return ComputedAttribute(new_value,new_dependencies)
    
    @property
    def _modified_time_(self):
        return self.__modified_time__
    
    @_modified_time_.setter 
    def _modified_time_(self,value):
        self.__modified_time__ = value


# The ConstantsDict class is a subclass of UserDict that represents a dictionary of constants.
class ConstantsDict(UserDict):
    
    
    def __init__(self, *args, **kwargs):
        super(ConstantsDict,self).__init__(*args, **kwargs)
        self.__dependency_check_active__ = False
        for k in super(ConstantsDict,self).keys():
            if not isinstance(self.get_stored_attribute(k),(IndependentAttribute,ConstantsDict,UpdatableObject)):
                super().__setitem__(key=k,item=IndependentAttribute(self.get_stored_attribute(k)))
        self.value = self # allows this object to be treated as an IndependentAttribute as well
        self.__was_used__ = False

        self.__was_iterated_over__ = False

    @property
    def _modified_time_(self):
        if len(self.keys()):
            self.__update_changeable_elements__() # make sure everything is up to date
            return max(super(ConstantsDict,self).__getitem__(k)._modified_time_ for k in self.keys())
        else:
            return 0 # no keys means it has never been modified
    
    @property
    def _was_used_(self):
        
        was_used_dict = {k:v._was_used_ for k,v in self.meta_items()}
        #n_true = sum(was_used_dict.values())
        if any(v is not False for v in was_used_dict.values()):
            #if n_true == len(self):
            #   return True  # return total number of elements in array. If elements are added, we know that we need to update!
            return was_used_dict
        else:
            return False
        #if True: 
            #return self.__was_used__
    @_was_used_.setter
    def _was_used_(self,value:bool|dict):
        
        for k in super(UserDict,self).keys():
                    self.get_stored_attribute(k)._was_used_ = value if isinstance(value,bool) else value[k]
    
    @property
    def _dependency_check_active_(self):
        return self.__dependency_check_active__
    
    @_dependency_check_active_.setter
    def _dependency_check_active_(self,value):
        self.__dependency_check_active__ = value
        for v in (v for v in super(UserDict,self).values() if isinstance(v,(UpdatableObject,ConstantsDict))):
            v._dependency_check_active_ = value
    
    
    def __update_changeable_elements__(self,key=None):
        """Checks that all items in the dict that are dicts themselves get their updated modified times. Alternatively, if a key is passed only update said key
        """
        key_gen = (key,) if key is not None else super().keys() # update either specified key or all keys 
        for k in key_gen:
            d = super().__getitem__(k) # to avoid recursion as get_stored_attr calls this method
            try:
                #d.value.__update_changeable_elements__()
                #getting modified time also updates elements
                if not isinstance(d,(UpdatableObject,ConstantsDict)): # these handle their modified times themselves
                    d._modified_time_ = max(d.value._modified_time_,d._modified_time_)
            except AttributeError:
                pass
        
    def __getitem__(self, key: Any) -> Any:
        
        # update any dict element
        self.__update_changeable_elements__(key)
        
        item = super().__getitem__(key)
        if not isinstance(item,(UpdatableObject,ConstantsDict)):
            #Update flags on non-container items. Container Items handle their flag setting themselves
            item._was_used_ = True
        return item.value
    
    def __setitem__(self, key: Any, item: Any) -> None:
        if key in super(ConstantsDict,self).keys():
            wrapped_item = self.get_stored_attribute(key)
            wrapped_item.value = item
            wrapped_item._modified_time_ = time.time()
            wrapped_item._was_used_ = True
            
        else:
            if isinstance(item,dict):
                wrapped_item = self.__class__(item) # cast as constants dict so that we can detect changes of the dict even though setitem is not used to alter it.
            elif isinstance(item,(UpdatableObject,ConstantsDict)):
                wrapped_item = item
            else:
                wrapped_item = IndependentAttribute(item)
        
        super().__setitem__(key, wrapped_item)
    
    def get_stored_attribute(self,key):
        """
        Allows retireval of the IndependentAttribute instance of the item rather than just the value itself. does not raise used flag
        """
        
        self.__update_changeable_elements__(key) #only update the item to retrieeve
        return super().__getitem__(key)
    
    def meta_items(self):
        for k in super().keys():
            yield k,self.get_stored_attribute(k)


    #region dependency check for looping over
    def iter_keys(self):
        self.__was_iterated_over__ = True
        return super().keys()
    
    def iter_values(self):
        self.__was_iterated_over__ = True
        return super().values()
     
    def iter_items(self):
        self.__was_iterated_over__ = True
        return super().items()

    #endregion


    def as_dict(self,only_values=True):

        if only_values:
            return {k:self[k] for k in self.keys()}
        else:
            return {k:self.get_stored_attribute(k) for k in self.keys()}
    
    def get_updated_since(self,timestamp):
        """
        Returns an iterable over all the items (key,value) that have been changed more recently than `timestamp` (time since epoch)
        """

        def time_checker(time_stamp):
            for k in super(ConstantsDict,self).keys():
                stored_attr = super(ConstantsDict,self).__getitem__(k)
                if stored_attr._modified_time_ > time_stamp:
                    yield k,stored_attr.value
        return time_checker(timestamp)

    
    def reset_dependency_flags(self):
        for k in super(ConstantsDict,self).keys():
            stored_attr = self.get_stored_attribute(k)
            if isinstance(stored_attr.value,(UpdatableObject,ConstantsDict)):
                " We want to check dependencies of this as well to make Updatable objects nestable"
                stored_attr.value.reset_dependency_flags()
            else:
                stored_attr._was_used_ = False

        self.__was_iterated_over__ = False

    def raise_dependency_flags(self,dependency_dict):
        """
        Raises the dependency flag of all the dependencies in the dict that have value True. If the value is another dict with boolean values, 
        """
        if dependency_dict is False:
            return 
        
       

        if dependency_dict is True:
            for k,v in self.meta_items():
                if isinstance(v,(UpdatableObject,ConstantsDict)):
                    v.raise_dependency_flags(True)
                else:
                    v._was_used_ = True
        else:
            for k,v in dependency_dict.items():
                if k == '__ITERATION_LENGTH__':
                    self.__was_iterated_over__ = v < 1e23
                else:
                    val = self.get_stored_attribute(k)
                    if isinstance(val,(UpdatableObject,ConstantsDict)):
                        val.raise_dependency_flags(v)
                    else:
                        if v:
                            val._was_used_ = True

                

    def get_current_dependency_flags(self):
        self.__update_changeable_elements__()
        return_dict = {k:v._was_used_ for k,v in self.meta_items()}
        return_dict['__ITERATION_LENGTH__'] = len(self) if self.__was_iterated_over__ else 1e23
        return return_dict
    
    def snapshot_used_dependencies(self):
        """Returns snapshots of all independent vars that were used. For the case where one of the dependencies is itself a ConstantsDict  it returns a dict of dependency snapshots if that. If all elements have been used, we return as single snapshot stating that 

        :return: _description_
        :rtype: _type_
        """
        snapshot_dict = {}


        for k,val in self.meta_items():
            if val._was_used_:
                snapshot_dict[k] = val.snapshot_used_dependencies() if isinstance(val,(UpdatableObject,ConstantsDict)) else AttributeSnapshot(val )

        snapshot_dict['__ITERATION_LENGTH__'] = len(self) if self.__was_iterated_over__ else 1e23 # default value avoids forcing recomputing if element is added
        return snapshot_dict

    def update_dependency_flags(self,flags_dict):
        for k,flag in flags_dict.items():
            if k == '__ITERATION_LENGTH__':
                self.__was_iterated_over__ = flag < 1e23
            else:
                self.get_stored_attribute(k)._was_used_ = flag
        
    def unify_dependency_flags(self,flags_dict):
        """Combines two flag dicts by performing elementwise or on each entry"""
        to_update = {}
        for k,v in flags_dict.items():
            if isinstance(v,dict):
                self.independent_vars[k].unify_dependency_flags(v)
            else:
                if v:
                    to_update[k] = v
        self.update_dependency_flags(to_update)

    def check_dependency_status(self,snapshot_dict:dict[str,AttributeSnapshot|dict]|None):
        # compares the snapshot_dict with the current state of the UpdatableObject to see if it needs updating:
        if snapshot_dict is None: 
            return True # we need to update since this is the first time it was computed
        else:
            for k,snapshot in snapshot_dict.items():
                if k == '__ITERATION_LENGTH__':
                    if snapshot < len(self):
                        return True # we iterated over the dict and the length has changed so we must update
                    else:
                        continue # skip to next item
                    
                if isinstance(snapshot,dict):
                    if self[k].check_dependency_status(snapshot):
                        break
                elif snapshot.has_changed():
                    break
            else:
                return False # get here if we finish loop without finding an update
            return True
    def update_dependency_snapshots(self,dependency_snaps:dict[str,AttributeSnapshot]):
        new_dependencies = {}
        for k,dep in dependency_snaps.items():
            if k=='__ITERATION_LENGTH__':
                self.__was_iterated_over__ = dep <  1e23
            else:
                val = self[k]
                if isinstance(val,(UpdatableObject,ConstantsDict)):
                    new_dependencies[k]=val.update_dependency_snapshots(dep)
                else:
                    new_dependencies[k] = dep.update()
        return new_dependencies


class SympyConstantsDict(ConstantsDict):
    """ Allows accessing entries where the key is a symbol by jsut using the name of the symbol as key"""
    def __getitem__(self, key: Any) -> Any:

        try:
            return super().__getitem__(key)
        except KeyError as err:
            if isinstance(key,sympy.Symbol):
                raise err 
            try:
                return super().__getitem__(sympy.Symbol(key))
            except KeyError as err:
                return super().__getitem__(sympy.Symbol(key,commutative=False))
    
from functools import wraps

class UpdatableObject(IndependentAttribute):
    def __init__(self,use_sympy_dict=False,**kwargs) -> None:
        """ Wraps a constant dict with so that users have to show intent on changing the contents of the independent variables. Keeping track on updating and dependency is handled by the Constants dict"""
        if '__class__' in kwargs:
            del kwargs['__class__']
        super().__init__(value=self) # this allows it to act as an IndependentAttribute
        if use_sympy_dict:
            self._independent_vars_ = SympyConstantsDict(kwargs)
        else:
            self._independent_vars_ = ConstantsDict(kwargs) 
        
        #whether to return the computed value of a method (False) or return it as a computed attribute.
        #If Computed Attrs of other objects depend on computations of this instance we can use this to check for updates before recomputing anything
        self.__wrap_computed_attrs__ = False 
        

     
    
    @property
    def _was_used_(self):
        return self._independent_vars_._was_used_
       
    @_was_used_.setter 
    def _was_used_(self,flags_dict:dict|bool):
        self._independent_vars_._was_used_(flags_dict)
        
    @property
    def _dependency_check_active_(self):
        return self._independent_vars_.__dependency_check_active__
    
    @_dependency_check_active_.setter  
    def _dependency_check_active_(self,value):
        self._independent_vars_.__dependency_check_active__ = value
    
    @property
    def independent_vars(self):
        #self._independent_vars_.__update_changeable_elements__() Do we need this? just check the items that are actually retrieved
        return self._independent_vars_
    
    def reset_dependency_flags(self):
        self._independent_vars_.reset_dependency_flags()

    def raise_dependency_flags(self,dependency_dict:dict|bool):
        """
        Raises the dependency flag of all the dependencies in the dict that have value True. If the value is another dict with boolean values, 
        it is assumed that the corresponding independent var is another UpdatableObject and its dependency flags are raised analogously.
        """
        self._independent_vars_.raise_dependency_flags(dependency_dict)

    @property
    def _modified_time_(self):
        return self.independent_vars._modified_time_ # this will update the times when calling .independent_vars

    
    def get_current_dependency_flags(self):
        return self._independent_vars_.get_current_dependency_flags()

    def snapshot_used_dependencies(self):
        return self._independent_vars_.snapshot_used_dependencies()
            
    def update_dependency_flags(self,flags_dict):
        self._independent_vars_.update_dependency_flags(flags_dict)

    def __update_changeable_elements__(self,key=None):
        self._independent_vars_.__update_changeable_elements__(key)
    
    def unify_dependency_flags(self,flags_dict):
        self._independent_vars_.unify_dependency_flags(flags_dict)

            
            
    def check_dependency_status(self,snapshot_dict:dict[str,AttributeSnapshot|dict]):
        return self._independent_vars_.check_dependency_status(snapshot_dict)

    def update_dependency_snapshots(self,dependency_snaps:dict[str,AttributeSnapshot]):
        return self._independent_vars_.update_dependency_snapshots(dependency_snaps)

    def __getstate__(self):
        return dict(self.independent_vars).copy()
    
    def __setstate__(self,state):
        self.__init__(**state)


from copy import copy

def auto_update(computation_method):
    
    @wraps(computation_method)
    def set_flags(self:UpdatableObject,*args,**kwargs):

        attr_to_update = '_saved_'+computation_method.__name__[:] # the name of the saved attribute
        try:
            saved_res = getattr(self,attr_to_update)
        except AttributeError:
            saved_res = ComputedAttribute(None,None)
        
        
            
        must_recompute = self.check_dependency_status(saved_res.dependencies)
        
        if not must_recompute:
            if self._dependency_check_active_:
                # update the state of the dependencies since if A derives from B and B depends on C, A depends on C
                self.raise_dependency_flags(saved_res.dependencies)
                
            return saved_res.value if not self.__wrap_computed_attrs__ else saved_res   

        else:
            # see if we have to compute what the dependencies are
            if saved_res.dependencies is None:
                substitute_flags = self._dependency_check_active_
                if substitute_flags:
                    # another dependency check is active so we have to temporarily overwrite the flags
                    current_flags = self.get_current_dependency_flags()
                else:
                    self._dependency_check_active_ = True 

                # run func and check used dependencies. Verify also that values were not changed
                self.reset_dependency_flags()

                res = computation_method(self,*args,**kwargs)
                dependencies = self.snapshot_used_dependencies()
                computed_attr = ComputedAttribute(res,dependencies)
                setattr(self,attr_to_update,computed_attr)
                    
                if substitute_flags:
                    # revert the flags back
                    
                    # Or the flags together since if B uses A and A depends on C then B depends on C
                    
                    self.raise_dependency_flags(current_flags)
                    # do not alter dependency check status!
                else:
                    self._dependency_check_active_ = False
                
                return res if not self.__wrap_computed_attrs__ else computed_attr   
            # if we simply have to update the value and snapshots
            else:
                # run func:
                res = computation_method(self,*args,**kwargs)
                
                # update the saved version
                updated_dependencies = self.update_dependency_snapshots(saved_res.dependencies)
                setattr(self,attr_to_update,ComputedAttribute(res,updated_dependencies))
                
                return res if not self.__wrap_computed_attrs__ else computed_attr   
    return set_flags
        

    

if __name__ == '__main__':
    test = Example(100,2,3)
    
    test.independent_vars['a'] += 1  
    test.independent_vars['a'] = 200