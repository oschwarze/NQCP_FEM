from abc import ABC,abstractmethod
from dataclasses import dataclass
from .band_model import BandModel
from .updatable_object import UpdatableObject,auto_update
from typing import Sequence
import numpy as np
import sympy
class Domain():
    def __post_init__(self):
        self.mesh = None
        self.__mesh_scale__ = None

    def bounded_scale(self):
        """
        returns a measure of how far away from 0,0 the boundary of the domain is
        :return:
        """
        raise NotImplementedError

    @property
    def mesh_scale(self):
        if self.__mesh_scale__ is None:
            self.__mesh_scale__ = 1
        return self.__mesh_scale__
    
class EnvelopeFunctionModel(UpdatableObject,ABC):
    def __int__(self, band_model:BandModel,domain=None,**independent_vars):
        """
        Model which is used to determine envelope function of som band model
        :param BandModel band_model: envelope_model discribing dispersion relation of system
        :param Domain domain: the domain on which the envelope function is to be defined
        """
        self._is_sparse = None
        
        independent_vars['band_model'] = band_model
        independent_vars['domain'] = domain
        super(EnvelopeFunctionModel, self).__init__(**independent_vars)
        
    @property
    def band_model(self)->BandModel:
        return self.independent_vars['band_model']
    @band_model.setter
    def band_model(self,value):
        self.independent_vars['band_model'] = value 

    @property
    def domain(self)->Domain:
        return self.independent_vars['domain']
    @domain.setter
    def domain(self,value):
        self.independent_vars['domain'] = value
    
    @property
    #@abstractmethod
    def k_signature(self) -> str:
        # return the desired K-ordering for this type of envelope function.
        pass
    
        
    @abstractmethod
    @auto_update
    def assemble_array(self,sparse=True):
        """
        Computes a Hamiltonian that can be passed to a solver.
        This is in fact just a special (very common) case of project_operator where the operator to project is the Hamiltonian.
        We make it a special method to allow for saving of the result (and only updating necessary parts)
        :return:
        """
        raise NotImplementedError

    @abstractmethod
    def positional_rep(self,vector,x_vals=None):
        """
        Compute the positional wave-function of a given vector.
        :param np.ndarray x_vals: array of shape [N,3] specifying specific points to evaluate the eigenfunction at.
        returns an error if the x_vals cannot be specified (like for FEniCs)
        :param Array vector:
        :return Tuple[x_points,function_values]: X values and Y values with Y being the value of the wave-function at the point X. The shape of Y is (N,tensor_shape) where N is the number of X points
        """
        raise NotImplementedError


    @abstractmethod
    def eigensolutions_to_eigentensors(self,eigensolutions):
        # converts a list (collumns are the eigensolutions) to a list of eigentensors (first index indexes the eigensolutions)
        #
        raise NotImplementedError

    
    @abstractmethod
    def solution_shape(self):
        raise NotImplementedError
    
    @abstractmethod
    @auto_update
    def make_S_array(self):
        """
        Method for making the S array in terms of the relevant basis functions.
        S_ij = <i|j>
        :return:
        
        # this is in fact just a special case of the project_operator,  where the operator to project is the identity
        """
        
        
        
        raise NotImplementedError

    #@abstractmethod
    def project_operator(self,operator):
        """
        Projects the operator down to the states used in the envelope function model.
        The operators is assumed to be a sympy array of number and the only free symbols being X,Y,Z kx,ky,kz
        
        in general the process is as follows:
            the ks are rearanged to the desired signature specified by self.k_signature
            
        
        """
        pass
    
    
    
    #@abstractmethod
    def construct_observable(self,operator:sympy.Array|np.ndarray):
        # Construct an AbstractObservable which can be aplied to the solutions of this model.
        pass
        
    
    def energy_scale(self):
        """
        If the entries of the assembled array have been scaled by the model to ensure better numerical stability,
        the actual eigenvalues con be recovered by multiplying with .energy_scale
        :return:
        """
        return 1

    def length_scale(self):
        """
        If the model internally uses scaled legths in order to improve numerical stability, the actual lengths can be
        recovered by multiplying with .length_scale
        :return:
        """
        return 1
    


def sort_eigenvalues(eigenvalues,return_index = False):
    E = eigenvalues 
    E_p = np.sort(E[E>0])
    E_n = np.sort(E[E<0])[::-1]
    return np.vstack([E_p,E_n])
    
    
@dataclass()
class RectangleDomain(Domain):
    Lx: float
    Ly: float
    Lz: float
    origin: tuple[float,3] = (0, 0, 0)

    def __post_init__(self):
        self.lower_left_corner = [-(self.Lx - self.origin[0]) / 2,
                                  -(self.Ly - self.origin[1]) / 2,
                                  -(self.Lz - self.origin[1]) / 2]

        self.upper_right_corner = [(self.Lx - self.origin[0]) / 2,
                                   (self.Ly - self.origin[1]) / 2,
                                   (self.Lz - self.origin[1]) / 2]
        super().__post_init__()
    def bounded_scale(self):
        # origin is (0,0,0) ad boundary is this far away from origin
        # fixme: correctly compute the minimum distance to the perimeter from the origin
        return min([self.Lx - self.origin[0], self.Ly - self.origin[1], self.Lz - self.origin[2]])/2

    @property
    def mesh_scale(self):
        if self. __mesh_scale__ is None:
            return max([self.Lx,self.Ly])
    @classmethod
    def from_corners(cls,lower_left_corner,upper_right_corner):
        dimensions = [uc - lc for uc, lc in zip(upper_right_corner, lower_left_corner)]
        center = tuple((2 * lc + d) for lc, d in zip(lower_left_corner, dimensions))
        new = cls(*dimensions,origin=center)
        new.lower_left_corner = lower_left_corner
        new.upper_right_corner = upper_right_corner
        return new
@dataclass()
class CircleDomain(Domain):
    R: float

    def bounded_scale(self):
        return self.R
