import sympy
from .updatable_object import UpdatableObject,auto_update

import logging

from nqcpfem import updatable_object
LOGGER = logging.getLogger(__name__)
LOGGER.addHandler(logging.NullHandler())
from abc import ABC,abstractmethod

class Function(ABC):
    """A class representing a function. This function specifies the available methods for working with functions.
    """
    def __init__(self,symbol,spatial_dependencies,is_constant=False):
        self.spatial_dependencies = spatial_dependencies
        self.is_constant = is_constant
        self.symbol = symbol # the symbol representing the function. Must end on (x)
        
        super(Function,self).__init__()
    @abstractmethod
    def __call__(self,x,y=None,z=None):
        """Functions should be callable. The call signature should take as many arguments as len(self.spatial_dependencies).
        They should return the value of the function at that position
        :param x: x-position
        :type x: Any
        :param y: y-position
        :type y: Any
        :param z: z-position
        :type z: Any
        """
        raise NotImplementedError()
        
    @abstractmethod
    def derivative(self,dir,order=1):
        """In order to permute K operators with the function we must be able to take derivatives of the function.
        This should return another Function.
        :param dir: integer representing the direction: 0,1 or 2
        :type dir: int
        :param order: the order of the derivate operatorion
        :type order: int
        """
        raise NotImplementedError()
    
    @abstractmethod
    def project_to_basis(directions,n_modes,type='box'):
        """Computes the matrix expression of the function when projected down to a manifold spanned by certain types of eigenstates.

        :param directions: which direction(s) to 'integrate out'
        :type directions: int|Iterable[int]
        :param n_modes: number of modes of the basis to include
        :type n_modes: int
        :param type: which kind of eigenstates to used, defaults to 'box', meaning eigenstates of a particle in a box system.
        :type type: str, optional
        
        returns a Sympy Array containing explicit parameters as well as new functions labeled as f'{self.symbol.name[:-3]}'+"_{i}(x)" for i an integer.
        The corresponding functions are also returned in the form of a dict mapping symbols to Functions.+
        """
        #This the return value should only have to be computed once so use the auto_update feature! this also ensures that we get the same Function instances every time.
        raise NotImplementedError()
    
    
    
class SymbolicFunction(Function,UpdatableObject):
    pass


class NumericalFunction(Function,UpdatableObject):
    pass