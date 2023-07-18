from unittest import TestCase

import numpy as np
import numpy.testing as testing


class Test(TestCase):

    def setUp(self) -> None:
        self.mock_lk_shape = (4, 2, 16)  # 4 are lk band, 2 are something else anf 16 are positional
        self.mock_lk_state = np.arange(0, np.prod(self.mock_lk_shape), 1, dtype='complex').reshape(self.mock_lk_shape)
        self.mock_lk_state[:, :, ::2] *= 1j
        self.mock_lk_state = self.mock_lk_state / np.linalg.norm(self.mock_lk_state)

    def test_hh_projection(self):
        from nqcpfem.observables import HH_projection
        from nqcpfem.band_model import LuttingerKohnHamiltonian
        model = LuttingerKohnHamiltonian(3)
        HH_proj = HH_projection(model)
        result = HH_proj.apply(self.mock_lk_state)
        facit = np.copy(self.mock_lk_state)
        facit[1, ...] = 0
        facit[2, ...] = 0
        np.testing.assert_array_equal(result, facit)

    def test_lh_projcetion(self):
        from nqcpfem.observables import LH_projection
        from nqcpfem.band_model import LuttingerKohnHamiltonian
        model = LuttingerKohnHamiltonian(3)
        LH_proj = LH_projection(model)
        result = LH_proj.apply(self.mock_lk_state)
        facit = np.copy(self.mock_lk_state)
        facit[0, ...] = 0
        facit[-1, ...] = 0
        np.testing.assert_array_equal(result, facit)

    def test_band_angular_momentum(self):
        from nqcpfem.observables import band_angular_momentum
        from nqcpfem.band_model import LuttingerKohnHamiltonian
        from nqcpfem import ANGULAR_MOMENTUM, _hbar
        J = ANGULAR_MOMENTUM['3/2'] * _hbar
        model = LuttingerKohnHamiltonian(3)
        AM = band_angular_momentum(model)
        result = AM.mel(self.mock_lk_state)
        facit = np.einsum('jzx,ijk,kzx', self.mock_lk_state.conj(), J, self.mock_lk_state)
        np.testing.assert_array_equal(result, facit)

    def test_spin(self):
        from nqcpfem.observables import spin
        from nqcpfem.band_model import LuttingerKohnHamiltonian
        from nqcpfem import ANGULAR_MOMENTUM, _hbar
        J = ANGULAR_MOMENTUM['1/2'] * _hbar
        model = LuttingerKohnHamiltonian(3)
        AM = spin(model)
        AM_mat = np.array(AM)
        np.testing.assert_array_equal(AM_mat[:, 1:3, 1:3], J, verbose=True)
        hh_subsp_AM = np.array([[AM_mat[:, 0, 0], AM_mat[:, 0, 3]], [AM_mat[:, 3, 0], AM_mat[:, 3, 3]]]).transpose(
            [2, 0, 1])
        np.testing.assert_array_equal(hh_subsp_AM, J)

    def test_positional_probability_distribution(self):
        from nqcpfem.observables import positional_probability_distribution
        from nqcpfem.band_model import LuttingerKohnHamiltonian, FreeFermion

        model = LuttingerKohnHamiltonian(3).material_spec('Ge').add_z_confinement(0.1,
                                                                                  2)  # add z confinement modes just to make sure that the tensor shape is what we expect
        target_shape = (16,)
        positional_prop = positional_probability_distribution(model)
        result = positional_prop(self.mock_lk_state)
        self.assertEqual(result.shape, target_shape)
        np.testing.assert_array_almost_equal(result, np.linalg.norm(self.mock_lk_state.reshape(-1, 16), axis=0) ** 2)

        fermion_model = FreeFermion(1, 3)
        spinor_up = np.array([1, 0], dtype='complex')
        spinor_down = np.array([0, 1], dtype='complex')
        x_part = np.ones(10, dtype='complex')
        x_part[:5] = 0

        func_up = np.tensordot(spinor_up, x_part, axes=0)
        func_down = np.tensordot(spinor_down, x_part, axes=0)
        print(func_up.shape, func_down.shape)
        pos = positional_probability_distribution(fermion_model)
        np.testing.assert_array_almost_equal(pos(func_down), pos(func_up))

    def test_observable(self):
        from nqcpfem.observables import Observable
        mat = np.arange(4).reshape((2, 2))
        tensor = np.arange(4 * 16).reshape((4, 4, 2, 2))
        observable = Observable(mat, 1)
        tensor_observable = Observable(tensor, [0, 1])

        other_mock_vec = np.linspace(0, 1, 8 * 16, dtype='complex').reshape(4, 2, 16)
        other_mock_vec = other_mock_vec * 1j
        facit_mel_observable = np.einsum('ixk,xy,iyk', self.mock_lk_state.conj(), mat, self.mock_lk_state)
        np.testing.assert_array_equal(observable.mel(self.mock_lk_state), facit_mel_observable)
        facit_mel_observable = np.einsum('ixk,xy,iyk', other_mock_vec.conj(), mat, self.mock_lk_state)
        np.testing.assert_array_equal(observable.mel(other_mock_vec, self.mock_lk_state), facit_mel_observable)

        facit_mel_tensor = np.einsum('xli,xylk,yki', self.mock_lk_state.conj(), tensor, self.mock_lk_state)
        np.testing.assert_array_equal(tensor_observable.mel(self.mock_lk_state), facit_mel_tensor)
        facit_mel_tensor = np.einsum('xli,xylk,yki', other_mock_vec.conj(), tensor, self.mock_lk_state)
        np.testing.assert_array_equal(tensor_observable.mel(other_mock_vec, self.mock_lk_state), facit_mel_tensor)

    def test_vector_observable(self):
        from nqcpfem.observables import VectorObservable
        mat = np.arange(3 * 4).reshape((3, 2, 2))
        tensor = np.arange(5 * 4 * 16).reshape((5, 4, 4, 2, 2))
        observable = VectorObservable(mat, 1)
        tensor_observable = VectorObservable(tensor, [0, 1])

        other_mock_vec = np.linspace(0, 1, 8 * 16, dtype='complex').reshape(4, 2, 16)
        other_mock_vec = other_mock_vec * 1j
        facit_mel_observable = np.einsum('ixk,axy,iyk', self.mock_lk_state.conj(), mat, self.mock_lk_state)
        np.testing.assert_array_equal(observable.mel(self.mock_lk_state), facit_mel_observable)
        facit_mel_observable = np.einsum('ixk,axy,iyk', other_mock_vec.conj(), mat, self.mock_lk_state)
        np.testing.assert_array_equal(observable.mel(other_mock_vec, self.mock_lk_state), facit_mel_observable)

        facit_mel_tensor = np.einsum('xli,axylk,yki', self.mock_lk_state.conj(), tensor, self.mock_lk_state)
        np.testing.assert_array_equal(tensor_observable.mel(self.mock_lk_state), facit_mel_tensor)
        facit_mel_tensor = np.einsum('xli,axylk,yki', other_mock_vec.conj(), tensor, self.mock_lk_state)
        np.testing.assert_array_equal(tensor_observable.mel(other_mock_vec, self.mock_lk_state), facit_mel_tensor)
    def test_gram_schmidt_orthogonalization(self):
        from scipy.stats import unitary_group
        from nqcpfem.observables import gram_schmidt_orthogonalization
        U = unitary_group.rvs(16)

        b = U.T
        shuffled_basis = [b[0],
                          2*b[1],
                          b[0]+b[2],
                          b[1]-2*b[0]+b[3],
                          b[1]+b[3]+10*b[4],
                          b[5],b[6],b[7],b[8],b[9],b[10],b[11],b[12],b[13],b[14],b[15]]
        testing.assert_array_almost_equal(b,gram_schmidt_orthogonalization(shuffled_basis))
