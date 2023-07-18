from . import band_modeling, envelope_function
import numpy as np
from . import _hbar
from typing import Callable
from contextlib import suppress

class Observable(np.ndarray):
    def __new__(cls, tensor, tensor_index=None):
        obj = np.asarray(tensor).view(cls)
        obj.tensor_index = tensor_index # which index of the eigentensor it acts on
        if isinstance(obj.tensor_index, int):
            obj.tensor_index = [obj.tensor_index]
        cls._check_array_shape_(obj.tensor_index,obj.shape)
        return obj

    def __array_finalize__(self, obj):
        if obj is None:
            return
        self.tensor_index = getattr(obj, 'tensor_index', 0)
        if isinstance(self.tensor_index, int):
            self.tensor_index = [self.tensor_index]


    @staticmethod
    def _check_array_shape_(tensor_index,shape):
        if 2 * len(tensor_index) != len(shape):
            raise ValueError(
                f'Observables tensor index did not match its shape ({tensor_index}, {shape}). length mismatch:'
                f'({2 * len(tensor_index)} vs. {len(shape)}). '
                f'maybe tensor_index is wrong or observable array is not in grouped ordering')
    def _get_summation_indices_(self,vector_shape):
        """
        Returns the indices that can be plugged into einsum, to compute image of vector under observable or
        matrix elements of the observable
        :param tuple vector_shape: the shape of the vector
        :return Tuple[list,list,list]: left_vector_index,observable_index,right_vector_index
        """

        r_vector_indices = list(range(2 * len(self.tensor_index), 2 * len(self.tensor_index) + len(
            vector_shape)))  # index 0 to 2*len(tensor_index) reserved for summing and observable
        l_vector_indices = r_vector_indices.copy()
        # assumes grouped ordering of tensor
        l_summation_indices = list(range(0, 2 * len(self.tensor_index), 2))
        r_summation_indices = list(range(1, 2 * len(self.tensor_index), 2))
        tensor_indices = list(range(0, 2 * len(self.tensor_index)))
        for i, t_i in enumerate(self.tensor_index):
            # assign summation indices to the right axes
            r_vector_indices[t_i] = r_summation_indices[i]
            l_vector_indices[t_i] = l_summation_indices[i]

        return l_vector_indices,tensor_indices,r_vector_indices

    def mel(self, vector, other_vector=None):
        if other_vector is None:
            other_vector = vector

        l_vector_indices,tensor_indices,r_vector_indices = self._get_summation_indices_(vector.shape)
        expval = np.einsum(vector.conj(), l_vector_indices, self, tensor_indices, other_vector, r_vector_indices)
        return np.real(expval)

    def apply(self, vector, normalize=False):
        _,tensor_indices,vector_indices = self._get_summation_indices_(vector.shape)
        outcome = np.einsum(self, tensor_indices, vector, vector_indices)
        if normalize:
            outcome = outcome / np.linalg.norm(outcome)
        return outcome


class VectorObservable(Observable):
    """
    Specifictype of observable that is a tensor. The first axis of this array indexes which component of the vector we look at
    """
    @staticmethod
    def _check_array_shape_(tensor_index,shape):
        # shape will always be one more than tensor index since it is a vector observable and first index is for 
        #vector component
        Observable._check_array_shape_(tensor_index,shape[1:])

    def _get_summation_indices_(self, vector_shape):
        l_vec_i,ten_i,r_vec_i = super()._get_summation_indices_(vector_shape)
        l_vec_i = [i+1 for i in l_vec_i]
        ten_i = [0]+[i+1 for i in ten_i] # first index of ten_i is now reserved for the vector components of the observ.
        r_vec_i = [i+1 for i in r_vec_i]
        return l_vec_i,ten_i,r_vec_i

    def get_component(self,direction):
        """
        Retrieve one component of the VectorObservable
        :param direction:
        :return:
        """
        return Observable(self[direction,...],self.tensor_index)


def HH_projection(band_model):
    """
    Projects eigenstates of the band model down onto the HH subspace
    :param band_model:
    :return:
    """


    valid_band_models = (band_modeling.LuttingerKohnHamiltonian,)
    if isinstance(band_model,band_modeling.ParticleHoleBandModel):
        if band_model.band_model_class not in valid_band_models:
            raise TypeError(f'ParticleHoleBandModel based on model {band_model.band_model_class} does not have a HH subspace')

        observable = HH_projection(band_model.band_model_class)
        return band_model.extend_observable(observable)



    if not (isinstance(band_model, valid_band_models) or band_model in valid_band_models):
        raise TypeError(f'band model does not have a HH subspace: got {band_model,type(band_model)}')

    if isinstance(band_model, band_modeling.LuttingerKohnHamiltonian) or band_model == band_modeling.LuttingerKohnHamiltonian:
        hh_projection = np.zeros((4, 4), dtype='complex')
        hh_projection[0, 0] = 1
        hh_projection[3, 3] = 1
        observable = Observable(hh_projection, 0)


    return observable



def LH_projection(band_model):
    """
    Projects eigenstates of the band model down onto the HH subspace
    :param band_model:
    :return:
    """
    valid_band_models = (band_modeling.LuttingerKohnHamiltonian,)

    if isinstance(band_model,band_modeling.ParticleHoleBandModel):
        if band_model.band_model_class not in valid_band_models:
            raise TypeError(
            f'ParticleHoleBandModel based on model {band_model.band_model_class} does not have a HH subspace')

        observable = LH_projection(band_model.band_model_class)
        return band_model.extend_observable(observable)

    if not (isinstance(band_model, valid_band_models) or band_model in valid_band_models):
        raise TypeError(f'band model does not have a HH subspace: got {band_model,type(band_model)}')

    if isinstance(band_model, band_modeling.LuttingerKohnHamiltonian)or band_model == band_modeling.LuttingerKohnHamiltonian:
        lh_projection = np.zeros((4, 4), dtype='complex')
        lh_projection[1, 1] = 1
        lh_projection[2, 2] = 1
        observable = Observable(lh_projection, 0)

    return observable

def band_angular_momentum(band_model):
    """
    constructor for angular momentum operator
    :param band_model:
    :return:
    """
    from models import ANGULAR_MOMENTUM
    valid_band_models = (band_modeling.LuttingerKohnHamiltonian, band_modeling.FreeFermion)


    if isinstance(band_model, band_modeling.ParticleHoleBandModel):
        if band_model.band_model_class not in valid_band_models:
            raise TypeError(
                f'ParticleHoleBandModel based on model {band_model.band_model_class} does not have a HH subspace')

        observable = band_angular_momentum(band_model.band_model_class)
        new_ops =band_model.extend_observable(observable)
        return new_ops


    if not (isinstance(band_model, valid_band_models) or band_model in valid_band_models):
        raise TypeError(f'band model does not have a HH subspac: got {type(band_model)}')



    if isinstance(band_model, band_modeling.LuttingerKohnHamiltonian)or band_model == band_modeling.LuttingerKohnHamiltonian:
        AM = _hbar * ANGULAR_MOMENTUM['3/2']
        return VectorObservable(AM, 0)
    if isinstance(band_model, band_modeling.FreeFermion) or band_model == band_modeling.FreeFermion:
        AM = _hbar * ANGULAR_MOMENTUM['1/2']
        return VectorObservable(AM, 0)


def spin(band_model):
    from . import ANGULAR_MOMENTUM
    valid_band_models = (band_modeling.LuttingerKohnHamiltonian, band_modeling.FreeFermion)

    if isinstance(band_model, band_modeling.ParticleHoleBandModel):
        if band_model.band_model_class not in valid_band_models:
            raise TypeError(
                f'ParticleHoleBandModel based on model {band_model.band_model_class} does not have a HH subspace')

        observable = spin(band_model.band_model_class)
        return band_model.extend_observable(observable)


    if not (isinstance(band_model, valid_band_models) or band_model in valid_band_models):
        raise TypeError(f'band model does not have a spin space subspac: got {type(band_model)}')

    if isinstance(band_model, band_modeling.LuttingerKohnHamiltonian) or band_model== band_modeling.LuttingerKohnHamiltonian:
        raise NotImplementedError(f'fixme. Spin operator wrongly defined.')

        sigmas = ANGULAR_MOMENTUM['1/2']
        mat = np.zeros((3, 4, 4), dtype='complex')
        mat[:,1:3, 1:3] = _hbar * sigmas
        mat[:,np.array([[0, 0], [3, 3]]), np.array([[0, 3], [0, 3]])] = _hbar * sigmas
        return VectorObservable(mat, 0)
    if isinstance(band_model, band_modeling.FreeFermion) or band_model == band_modeling.LuttingerKohnHamiltonian:
        AM = _hbar * ANGULAR_MOMENTUM['1/2']
        return VectorObservable(AM, 0)


def positional_probability_distribution(band_model) -> Callable:
    """
    Constructs a function that can convert an envelope function solution to it positional probability distribution
    :param band_model: The model whose eigenstate we want to project out
    :return Callable:
    """

    def probability_distribution_projector(vector):
        """
        Converts the vector into the positional probability distribution. The vector can be passed as either
        an eigentensor, or as an eigentensor in its positional representation. NB: it cannot take eigensolutions not cast as eigentensors!
        :param np.ndarray vector:
        :return np.ndarray:
        """
        bare_vector_shape = band_model.tensor_shape[::2]
        if vector.shape[:len(bare_vector_shape)] != bare_vector_shape:
            # check if we are in positional rep
            if vector.shape[1:] != bare_vector_shape:
                raise ValueError(f'vector shape did not match expected shape. {vector.shape} '
                                 f'could not be cast into positional axes + {bare_vector_shape}')

            squares = vector * vector.conj()
            # positional axis is only the first one, so we sum over all the other ones.
            probability_distribution = np.sum(squares.reshape(vector.shape[0], -1), axis=1)
        else:
            positional_shape = vector.shape[len(bare_vector_shape):]
            squares = vector * vector.conj()
            # sum out all the irrelevant axes.
            probability_distribution = np.sum(squares.reshape((-1,) + positional_shape), axis=0)
        return probability_distribution

    return probability_distribution_projector


def inner_product(vector_a,vector_b):
    """
    Computes the inner product of two eigenstensors of a model
    :param vector_a:
    :param vector_b:
    :return:
    """
    if not vector_a.shape == vector_b.shape:
        raise ValueError(f'vector did not have the same shape: {vector_a.shape} vs. {vector_b.shape}')
    summation_indices = list(range(len(vector_a.shape)))
    return np.einsum(vector_a.conj(),summation_indices,vector_b,summation_indices)


def gram_schmidt_orthogonalization(vectors, normalize=True):
    """
    Performs Gram-Schmidt orthogonalization on the set of vectors.
    :param np.ndarray vectors: Vectors to orthogonalize, the first index must index the vectors
    :param bool normalize=True: whether to normalize the vectors
    :return np.ndarray:
    """

    unnormalized_onb = [vectors[0]]
    # normalized vectors will always be needed when doing gram schmidt.
    normalized_onb = [vectors[0]/np.linalg.norm(vectors[0])]
    for vec in vectors[1:]:
        inner_products = (inner_product(onb_vec,vec)*onb_vec for onb_vec in normalized_onb)
        orthogonalized = vec-sum(inner_products)
        normalized = orthogonalized/np.linalg.norm(orthogonalized)
        unnormalized_onb.append(orthogonalized)
        normalized_onb.append(normalized)

    if normalize:
        return np.stack(normalized_onb,axis=0)
    else:
        return np.stack(unnormalized_onb,axis=0)


def particle_projector(band_model):
    """
    Projects onto subspace describing particle
    :param band_modeling.ParticleHoleBandModel band_model: the band model which the eigenvectors belong to
    :return:
    """
    pass
    relevant_axis = band_model.particle_hole_axis
    axis_dim = band_model.tensor_shape[2*relevant_axis]
    particle_dims = [d for d in range(axis_dim) if d not in band_model.hole_indices]
    projector_arr = np.zeros((axis_dim,axis_dim),dtype='complex')
    for pdim in particle_dims:
        projector_arr[pdim,pdim] = 1

    return Observable(projector_arr,relevant_axis)


def hole_projector(band_model):
    """
    Projects onto subspace describing particle
    :param band_modeling.ParticleHoleBandModel band_model: the band model which the eigenvectors belong to
    :return:
    """
    pass
    relevant_axis = band_model.particle_hole_axis
    axis_dim = band_model.tensor_shape[2 * relevant_axis]
    projector_arr = np.zeros((axis_dim, axis_dim), dtype='complex')
    for hdim in band_model.hole_indices:
        projector_arr[hdim, hdim] = 1

    return Observable(projector_arr, relevant_axis)


def flip_spins(band_model):
    """
    Defines an operator which flips the spins of the vector. This is usefull for when checking if two states are quivalent up to flipping of the spin
    :param band_modeling.BandModel band_model:
    :return:
    """
    valid_band_models = (band_modeling.LuttingerKohnHamiltonian,band_modeling.FreeFermion)
    if isinstance(band_model, band_modeling.ParticleHoleBandModel):
        if band_model.band_model_class not in valid_band_models:
            raise TypeError(
                f'ParticleHoleBandModel based on model {band_model.band_model_class} does not have a HH subspace')

        observable = flip_spins(band_model.band_model_class)
        return band_model.extend_observable(observable)

    if isinstance(band_model,band_modeling.LuttingerKohnHamiltonian) or band_model == band_modeling.LuttingerKohnHamiltonian:
        flip = band_modeling.time_reversal_operator(band_model)
        axis=0

    elif isinstance(band_model,band_modeling.FreeFermion) or band_model == band_modeling.FreeFermion:
        flip = band_modeling.time_reversal_operator(band_model)
        axis=0

    else:
        raise NotImplementedError(f'band model {band_model} does not have a spin flip operation')

    return Observable(flip,axis)
