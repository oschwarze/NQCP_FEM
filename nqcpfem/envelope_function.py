from abc import ABC,abstractmethod
from dataclasses import dataclass
from .band_model import BandModel
from .updatable_object import UpdatableObject,auto_update
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
    def __init__(self, band_model:BandModel,domain,**independent_vars):
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

    @property
    def domain(self)->Domain:
        return self.independent_vars['domain']

        
    @abstractmethod
    @auto_update
    def assemble_array(self,sparse=True):
        """
        Computes a Hamiltonian that can be passed to a solver
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
    @auto_update
    def make_S_array(self):
        """
        Method for making the S array in terms of the relevant basis functions.
        S_ij = <i|j>
        :return:
        """
        raise NotImplementedError

    def __getstate__(self):
        state = self.__dict__.copy()
        state['_array'] = None
        return state

    @property
    def energy_scale(self):
        """
        If the entries of the assembled array have been scaled by the model to ensure better numerical stability,
        the actual eigenvalues con be recovered by multiplying with .energy_scale
        :return:
        """
        return 1
    @property
    def length_scale(self):
        """
        If the model internally uses scaled legths in order to improve numerical stability, the actual lengths can be
        recovered by multiplying with .length_scale
        :return:
        """
        return 1
    

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
