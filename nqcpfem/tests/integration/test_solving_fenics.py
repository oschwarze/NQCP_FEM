from unittest import TestCase
import numpy as np


class TestFenicsFEMSolve(TestCase):
    
    def test_FEM_and_box_Fermion(self):
        from nqcpfem import envelope_function,band_model,fenics,solvers,box_modes,_m_e
        rng_seed = np.random.default_rng().integers(0,1024) 
        
        rng = np.random.default_rng(rng_seed)
        
        Lx,Ly,Lz = rng.uniform(100e-9,400e-9,3)

        
        
        domain = envelope_function.RectangleDomain(Lx=Lx,Ly=Ly,Lz=Lz)
        domain.resolution=[1000,1000]
        m = rng.uniform(1,10)*_m_e
        
        
        bm = band_model.FreeFermion(spatial_dim=2,mass=m)
        
        fem_model = fenics.FEniCsModel(bm,domain,0,('CG',1))
        
        box_model = box_modes.BoxEFM(bm,domain,nx=10,ny=10)
        
        
        fem_solver =solvers.PETScSolver(k=10,which='SM',sigma=0)
        
        box_solver = solvers.ScipySolver(k=20)
        
        fem_solution = fem_solver.solve(fem_model)
        
        box_solution = box_solver.solve(box_model)
        
        
        for fem_evec in fem_solution[0]:
            diff = np.min(np.abs(box_solution[0]-fem_evec))
            np.testing.assert_allclose(diff,0,atol=1e-4,),err_msg=f'seed: {rng_seed}' 
        
        
        