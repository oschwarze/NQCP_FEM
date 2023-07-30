"""Integration tests for nqcpsim.band_model module
"""
import unittest
from unittest import TestCase
class TestBandModel(TestCase):
    def test_z_confinement_and_spectrum(self):
        # take FreeFermion and make sure that the spectrum is correctly determined.
        from nqcpfem.band_model import FreeFermion 
        from nqcpfem import constants,values
        H = FreeFermion().material_spec('Ge')
        K_range=(0,1e9)
        N_points=64
        dir = [1,0,0]
        regular_spectrum = H.spectrum(K_range,dir,N_points)
        
        import numpy as np
        lz = 10e-9
        hbar = values['hbar']
        m_e = values['m_e']
        z_confinement_coeff = hbar**2*np.pi**2/(2*m_e*lz**2)
        
        
        N_modes = 4
        facit_spectrum = np.hstack([regular_spectrum[1]+z_confinement_coeff*n**2 for n in range(1,N_modes+1)])
        
        H.add_z_confinement(N_modes,'box',lz)
        
        
        result_spectrum = H.spectrum(K_range,dir,N_points)
        
        np.testing.assert_allclose(facit_spectrum,result_spectrum[1])
        
    def test_k_FEM_reshuffle(self):
        self.fail('TODO')
        
class TestLuttingerKohnHamiltonian(TestCase):
    def test_crystal_rotation_and_spectrum(self):
        #test that rotating the crystal and computing spectrum along [1,0,0] and [0,1,0] and [0,0,1] is the same as checking spectrum along rotated K_directions!
        import numpy as np
        rng_seed = np.random.default_rng().integers(0,1024)
        
        rng = np.random.default_rng(rng_seed)
        
        import sympy
        theta,phi= sympy.symbols(r'\theta,\phi')
        rotation_mat = sympy.Array(
                [[sympy.cos(theta) * sympy.cos(phi), sympy.cos(theta) * sympy.sin(phi),
                -sympy.sin(theta)],
                [-sympy.sin(phi), sympy.cos(phi), 0],
                 [sympy.sin(theta) * sympy.cos(phi), sympy.sin(theta) * sympy.sin(phi),
                sympy.cos(theta)]])
        
        theta,phi = rng.uniform(0,2*np.pi,2) #pick random angles
        #theta = 0*np.pi
        #phi = np.pi/6
        R = np.array(rotation_mat.subs({r'\theta':theta,r'\phi':phi})).astype(np.complex128)#.transpose()
        
        N_k_vectors = 3
        K_vec = rng.uniform(0,1,(N_k_vectors,3))
        K_vec = K_vec/np.linalg.norm(K_vec) # unit k_vector
        
        from nqcpfem.band_model import LuttingerKohnHamiltonian
        H = LuttingerKohnHamiltonian().material_spec('Ge')
        # compute spectrum wrt rotated K
        
        rot_K = np.einsum('ij,kj->ki',R,K_vec)
        K_range=(1e17,1e19)
        N_points = (64)
        facit_spectrums = np.stack([H.spectrum(K_range,k,N_points)[1] for k in rot_K],axis=0)
        
        
        
        
        
        #rotate the Hamiltonian
        H.rotate_crystallographic_direction(theta,phi)
        res_spectrums = np.stack([H.spectrum(K_range,k,N_points)[1] for k in K_vec],axis=0)
        
        np.testing.assert_allclose(res_spectrums,facit_spectrums,err_msg=f'spectrum comparison failed for seed {rng_seed}')
        
        
