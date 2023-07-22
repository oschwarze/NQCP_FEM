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
from locale import locale_alias
from tracemalloc import Snapshot
from turtle import update
from typing import Any
import time
from xml.dom.minidom import Attr






from collections import namedtuple,UserDict

from attr import attributes

# The ConstantsDict class is a subclass of UserDict that represents a dictionary of constants.
class ConstantsDict(UserDict):
    
    
    def __init__(self, *args, **kwargs):
        super(ConstantsDict,self).__init__(*args, **kwargs)
        for k in self.keys():
            if not isinstance(self.get_stored_attribute(k),(IndependentAttribute,ConstantsDict,UpdatableObject)):
                super().__setitem__(key=k,item=IndependentAttribute(self.get_stored_attribute(k)))
        self.value = self # allows this object to be treated as an IndependentAttribute as well
        self.__was_used__ = False
    @property
    def _modified_time_(self):
        if len(self.keys()):
            self.__update_changeable_elements__() # make sure everything is up to date
            return max(super(ConstantsDict,self).__getitem__(k)._modified_time_ for k in self.keys())
        else:
            return 0 # no keys means it has never been modified
    
    @property
    def _was_used_(self):
        """
        if len(self):
            return {k:self.get_stored_attribute(k)._was_used_ for k in self.keys()}
        """
        if True: 
            return self.__was_used__
    @_was_used_.setter
    def _was_used_(self,value):
        """
        if len(self):
                for k in self.keys():
                    self.get_stored_attribute(k)._was_used_ = value if isinstance(value,bool) else value[k]
        """
        if True:
            self.__was_used__ = value
    
    def __update_changeable_elements__(self):
        """Checks that all items in the dict that are dicts themselves get their updated modified times. 
        """
        for k in self.keys():
            d = super().__getitem__(k) # to avoid recursion as get_stored_attr calls this method
            try:
                d.value.__update_changeable_elements__()
            except AttributeError:
                pass
            try:
                # if the value itself has a _modified_time_ we use that
                d._modified_time_ = max(d.value._modified_time_,d._modified_time_)
            except AttributeError:
                pass
        
    def __getitem__(self, key: Any) -> Any:
        
        # update any dict elements
        self.__update_changeable_elements__()
        
        item = super().__getitem__(key)
        if not isinstance(item,(UpdatableObject)):
            
            item._was_used_ = True
        return item.value
    
    def __setitem__(self, key: Any, item: Any) -> None:
        if key in self.keys():
            wrapped_item = self.get_stored_attribute(key)
            wrapped_item.value = item
            wrapped_item._modified_time_ = time.time()
            
        else:
            if isinstance(item,dict):
                wrapped_item = ConstantsDict(item) # cast as constants dict so that we can detect changes of the dict even though setitem is not used to alter it.
            elif isinstance(item,UpdatableObject):
                wrapped_item = item
            else:
                wrapped_item = IndependentAttribute(item)
        
        super().__setitem__(key, wrapped_item)
    
    def get_stored_attribute(self,key):
        """
        Allows retireval of the IndependentAttribute instance of the item rather than just the value itself
        """
        
        self.__update_changeable_elements__()
        return super().__getitem__(key)
    
    def as_dict(self,only_values=True):

        self.__update_changeable_elements__()
        if only_values:
            return {k:self[k] for k in self.keys()}
        else:
            return {k:self.get_stored_attribute(k) for k in self.keys()}
    
    
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

from functools import wraps

class UpdatableObject(IndependentAttribute):
    def __init__(self,**kwargs) -> None:
        if '__class__' in kwargs:
            del kwargs['__class__']
        super().__init__(value=self) # this allows it to act as an IndependentAttribute
        self._independent_vars_ = ConstantsDict(kwargs)
        self.__dependency_check_active__ = False
        
        #whether to return the computed value of a method (False) or return it as a computed attribute.
        #If Computed Attrs of other objects depend on computations of this instance we can use this to check for updates before recomputing anything
        self.__wrap_computed_attrs__ = False 
        
        
    
    @property
    def _was_used_(self):
        """returns a dict of the independent attributes that were used

        :return: _description_
        :rtype: _type_
        """
        return_dict = {k:self.independent_vars.get_stored_attribute(k)._was_used_ for k in self.independent_vars.keys()}
        return return_dict if any(return_dict.values()) else False
    
    @_was_used_.setter 
    def _was_used_(self,flags_dict:dict|bool):
        if not isinstance(flags_dict,dict):
            for k in self.independent_vars.keys():
                self.independent_vars.get_stored_attribute(k)._was_used_ = flags_dict
        else:
            for k,flag in flags_dict.items():
                self.independent_vars.get_stored_attribute(k)._was_used_ = flag
        
    @property
    def _dependency_check_active_(self):
        return self.__dependency_check_active__
    
    @_dependency_check_active_.setter  
    def _dependency_check_active_(self,value):
        self.__dependency_check_active__ = value
        for indep_var in self.independent_vars.values():
            if isinstance(indep_var,UpdatableObject):
                indep_var._dependency_check_active_ = value
    
    
    
    @property
    def independent_vars(self):
        self._independent_vars_.__update_changeable_elements__()
        return self._independent_vars_
    
    def reset_dependency_flags(self):
        for k in self.independent_vars.keys():
            stored_attr = self.independent_vars.get_stored_attribute(k)
            if isinstance(stored_attr.value,UpdatableObject):
                " We want to check dependencies of this as well to make Updatable objects nestable"
                stored_attr.value.reset_dependency_flags()
            else:
                stored_attr._was_used_ = False
    def raise_dependency_flags(self,dependency_dict):
        """
        Raises the dependency flag of all the dependencies in the dict that have value True. If the value is another dict with boolean values, 
        it is assumed that the corresponding independent var is another UpdatableObject and its dependency flags are raised analogously.
        """
        if dependency_dict is False:
            return 
        
        for k,v in dependency_dict.items():
            if isinstance(self._independent_vars_.get_stored_attribute(k).value,UpdatableObject):
                self._independent_vars_[k].raise_dependency_flags(v)
            else:
                if v:
                    self._independent_vars_.get_stored_attribute(k)._was_used_ = True
                
    @property
    def _modified_time_(self):
        return self.independent_vars._modified_time_ # this will update the times when calling .independent_vars

    
    def get_current_dependency_flags(self):
        return {k:self.independent_vars.get_stored_attribute(k)._was_used_ for k in self.independent_vars.keys()}

    def snapshot_used_dependencies(self):
        """Returns snapshots of all independent vars that were used. For the case where one of the dependencies is itself an UpdatableOBject it returns a dict of dependency snapshots if that.

        :return: _description_
        :rtype: _type_
        """
        
        snapshot_dict={k:(self.independent_vars[k].snapshot_used_dependencies() if isinstance(self.independent_vars[k],UpdatableObject) 
                            else AttributeSnapshot(self.independent_vars.get_stored_attribute(k)))
                        for k in self.independent_vars.keys() 
                        if self.independent_vars.get_stored_attribute(k)._was_used_}
        return snapshot_dict
            
    def update_dependency_flags(self,flags_dict):
        for k,flag in flags_dict.items():
            self.independent_vars.get_stored_attribute(k)._was_used_ = flag

    def __update_changeable_elements__(self):
        self.independent_vars # this performs the update
        return
    
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
            
            
    def check_dependency_status(self,snapshot_dict:dict[str,AttributeSnapshot|dict]):
        # compares the snapshot_dict with the current state of the UpdatableObject to see if it needs updating:
        if snapshot_dict is None: 
            return True # we need to update
        
        for k,snapshot in snapshot_dict.items():
            if isinstance(snapshot,dict):
                if self.independent_vars[k].check_dependency_status(snapshot):
                    break
            elif snapshot.has_changed():
                break
        else:
            return False # get here if we finish loop without finding an update
        return True

    def update_dependency_snapshots(self,dependency_snaps:dict[str,AttributeSnapshot]):
        new_dependencies = {}
        for k,dep in dependency_snaps.items():
            if isinstance(self._independent_vars_[k],UpdatableObject):
                new_dependencies[k]=self._independent_vars_[k].update_dependency_snapshots(dep)
            else:
                new_dependencies[k] = dep.update()
        return new_dependencies
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