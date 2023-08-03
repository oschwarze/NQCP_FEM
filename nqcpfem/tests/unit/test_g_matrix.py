import unittest
from unittest import TestCase
import numpy as np
import numpy.testing as testing
import logging
import os,sys

src_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.append(src_path)

LOG = logging.getLogger(__name__)
LOG.setLevel(logging.DEBUG)
import sys
c_handler = logging.StreamHandler()
c_handler.setLevel(logging.DEBUG)
LOG.addHandler(c_handler)

class TestGMatrix(TestCase):

    def setUp(self):
        from nqcpfem.band_model import LuttingerKohnHamiltonian,FreeFermion
        from nqcpfem.envelope_function import RectangleDomain
        from nqcpfem.fenics import FEniCsModel
        from nqcpfem import _m_e, _hbar
        self.mass = _m_e
        self.SEED=2012023
        L = 4e-7
        self.Lx = L
        self.Ly = L
        self.Lz = L
        self.g_factor=2
        self.box_band_models= {'FreeParticle': FreeFermion(self.mass, 3).add_zeeman_term(g_tensor=self.g_factor, Bvec=[0,0,0])}
        self.box_band_models['LK'] = LuttingerKohnHamiltonian(3).material_spec('Ge').add_zeeman_term(B=[0,0,0])
        self.fenics_band_models= {'FreeParticle': FreeFermion(self.mass, 2).add_zeeman_term(self.g_factor,[0,0,0])}
        self.fenics_band_models['LK'] = LuttingerKohnHamiltonian(3).material_spec('Ge').add_zeeman_term(B=[0,0,0]).add_z_confinement(lz=25e-9,nz_modes=1,z_confinement_type='box')


        self.domain = RectangleDomain(self.Lx, self.Ly, self.Lz)
        self.domain.resolution = [200,200]
        self.function_class = ('CG', 1)
        self.fenics_models= {'FreeParticle': FEniCsModel(self.fenics_band_models['FreeParticle'], self.domain, 0, self.function_class)}
        self.fenics_models['LK'] = FEniCsModel(self.fenics_band_models['LK'], self.domain, 0, self.function_class)
        self.omega = 5e11
        import sympy
        omega = sympy.symbols(r'\omega')
        m,x,y = sympy.symbols('m,x,y')
        ho_potential = 0.5*m*omega**2*(x**2+y**2)
        
        self.fenics_models['FreeParticle'].band_model.add_potential(ho_potential)
        self.fenics_models['FreeParticle'].band_model.independent_vars['parameter_dict'][m] = self.mass
        self.fenics_models['FreeParticle'].band_model.independent_vars['parameter_dict'][omega] = self.omega
        self.fenics_models['LK'].band_model.add_potential(ho_potential)
        self.fenics_models['LK'].band_model.independent_vars['parameter_dict'][m] = self.mass
        self.fenics_models['LK'].band_model.independent_vars['parameter_dict'][omega] = self.omega

        from nqcpfem.box_modes import BoxEFM
        n_modes = 5
        self.box_models={'FreeParticle': BoxEFM(self.box_band_models['FreeParticle'],self.domain,n_modes,n_modes,n_modes),
                        'LK':BoxEFM(self.box_band_models['LK'],self.domain,n_modes,n_modes,n_modes)}

        self.box_models['FreeParticle'].band_model.add_potential(ho_potential)
        self.box_models['FreeParticle'].band_model.independent_vars['parameter_dict'][m] = self.mass
        self.box_models['FreeParticle'].band_model.independent_vars['parameter_dict'][omega] = self.omega
        self.box_models['LK'].band_model.add_potential(ho_potential)
        self.box_models['LK'].band_model.independent_vars['parameter_dict'][m] = self.mass
        self.box_models['LK'].band_model.independent_vars['parameter_dict'][omega] = self.omega


        from nqcpfem.solvers import PETScSolver,ScipySolver
        self.fenicssolver = PETScSolver(k=5,sigma=0)
        self.fenicssolver.sparse = True
        self.boxsolver = ScipySolver()
        self.boxsolver.sparse = False
        self.N_B_sample_points = 4

    def random_B_vector(self,n_vectors=1,max_strength=1.0,seed=None):
            rng = np.random.default_rng(seed)
            angles = rng.uniform(0,2*np.pi,n_vectors)
            strengths = rng.uniform(0,max_strength,n_vectors)
            z_coords = rng.uniform(-1,1,n_vectors)
            square = np.sqrt(1-z_coords**2)*strengths
            vectors = np.stack([square*np.cos(angles),square*np.sin(angles),z_coords*strengths],axis=1)
            return vectors

    def test_compute_g_matrix(self):
        from nqcpfem.g_matrix import GMatrix

        # For free particle we should just recover the magnetic field term of the Hamiltonian
        from nqcpfem import  ANGULAR_MOMENTUM,_mu_B
        sigma = ANGULAR_MOMENTUM['1/2'] # Winklers definiton of the sigmas
        M = 2*self.g_factor*sigma # factor 2/mu_B in definition of g-matrix, gives factor 2 and no mu_B here
        facit= np.array([[np.real(M[0, 0, 1]), np.real(M[1, 0, 1]), np.real(M[2, 0, 1])],
                         [np.imag(M[0, 0, 1]), np.imag(M[1, 0, 1]), np.imag(M[2, 0, 1])],
                         [np.real(M[0, 1, 1]), np.real(M[1, 1, 1]), np.real(M[2, 1, 1])],
                         ])
        facit_U,facit_sigma,facit_Vdagger = np.linalg.svd(facit)

        for efm_dict,name in zip([self.box_models,self.fenics_models],['box_efm','fenics_efm']):
            # if name == 'box_efm':

            #    continue
            LOG.info(f'gmatrix test for FreeFermion using {name}:')
            solver = self.boxsolver if name == 'box_efm' else self.fenicssolver
            bound_state_tol = None if name == 'box_efm' else 0.7

            gmatrix = GMatrix(efm_dict['FreeParticle'],solver)

            result=gmatrix.compute_g_matrix(bounded_state_tolerance=bound_state_tol)
            # assert that result is just a change of basis away from facit:
            result_U,result_sigma,result_Vdagger = np.linalg.svd(result)
            np.testing.assert_allclose(np.sort(facit_sigma),np.sort(result_sigma),err_msg=f'FreeParticle for {name} failed to compute correct eigenvalues:')
            LOG.info(f'{name}: g-matrix singular values: {result_sigma}')
            # change of basis is unitary we are good
            change_of_basis_U = result_U@facit_U.conj().T # maps facit evecs to result_evecs
            change_of_basis_V = result_Vdagger@facit_Vdagger.conj().T # maps facit evecs to result_evecs
            testing.assert_allclose(np.eye(3,dtype='complex'),change_of_basis_U@change_of_basis_U.conj().T,atol=1e-15,err_msg=f'FreeParticle for {name} failed to compute unitarily equivalent set of eigenvectors')
            np.testing.assert_allclose(np.eye(3,dtype='complex'),change_of_basis_V@change_of_basis_V.conj().T,atol=1e-15,err_msg=f'FreeParticle for {name} failed to compute unitarily equivalent set of eigenvectors')
            LOG.info(f'FreeFermion {name} test successful.')


        # For Luttinger kohn Hamiltonian, since we have spin-orbit coupling we can only really verify correctness by
        # Checking that it works for randomly chosen magnetic fields
        random_B_vectors = self.random_B_vector(self.N_B_sample_points,max_strength=0.1,seed=self.SEED)
        skip_box = True
        from nqcpfem.spin_gap import find_spin_gap
        for efm_dict,name in zip([self.box_models,self.fenics_models],['box_efm','fenics_efm']):
            if skip_box:
                skip_box=False
                continue
            LOG.info(f'gmatrix test for Luttinger Kohn Model using {name}:')
            solver = self.boxsolver if name == 'box_efm' else self.fenicssolver
            efm_model = efm_dict['LK']
            gmatrix = GMatrix(efm_model,solver)
            bound_state_tol = None if name == 'box_efm' else 0.7
            result=gmatrix.compute_g_matrix(bounded_state_tolerance=bound_state_tol)
            print(result,np.linalg.svd(result)[1])
            for n,Bvec in enumerate(random_B_vectors):
                # compute the gap using the GMatrix
                u_vec = result @ Bvec.T
                gmatrix_gap = np.linalg.norm(u_vec)*_mu_B

                LOG.info(f'verifying B-vector result for explicit B-vector {Bvec} ({n+1}/{len(random_B_vectors)}):')
                # compute spin_gap splitting numerically
                efm_model.band_model.parameter_dict['Bvec'] = Bvec
                print(efm_model.band_model.parameter_dict)
                #efm_model.band_model.__build_model__(force_build=True)
                #print(efm_model.band_model._tensor_constructors_[-1](efm_model.band_model.parameter_dict))
                eigensolution = solver.solve(efm_model,sparse=True)
                #from nqcpfem.plotting import plot_eigenvector
                #from nqcpfem.observables import band_angular_momentum,HH_projection
                #from nqcpfem import _hbar
                #AM = band_angular_momentum(efm_model.band_model)
                #HH = HH_projection(efm_model.band_model)

                #print([AM.mel(es)/_hbar for es in eigensolution[1]])
                #print(eigensolution[0]/(_hbar*self.omega))
                #print([HH.mel(es) for es in eigensolution[1]])
                #from matplotlib import pyplot as plt
                """
                for i in range(5):
                    positional,x = efm_model.positional_rep(eigensolution[1][i],50)
                    fig,ax=density_plot_3d(positional,x)
                    fig.suptitle(f'{i}')
                    plt.show()
                """
                gap,_,has_intermediate = find_spin_gap(eigensolution, efm_model,bounded_state_tolerance=bound_state_tol,positional_max_tv_dist=1e-3)
                print(gap,_,has_intermediate)

                self.assertAlmostEqual(gmatrix_gap,gap,msg=f'LK model for {name}: magnetic field {Bvec} ({n}/{self.N_B_sample_points}) did not produce correct energy gap.')
                from nqcpfem import UNIT_CONVENTION as U
                print(f'{name}: expected gap {gmatrix_gap*U["J to eV"]}, determined gap {gap*U["J to eV"]} eV')
            
    def test_matrix(self):
        self.fail()
    def test_derivative(self):
        self.fail()

if __name__ =='__main__':
    unittest.main()