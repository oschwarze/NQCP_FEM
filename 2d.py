"""
This script runs a gridsearch and saves the results to a CSV file. This can be used on SLURM
"""

import os
import sys
try:
    task_id = int(os.environ['SLURM_ARRAY_TASK_ID'])-1 # index starts from 1!
    n_tasks = int(os.environ['SLURM_ARRAY_TASK_COUNT'])
except Exception as err:
    print(err)
    task_id = 0
    n_tasks = 1   


TEMP_SAVE= f'sweet_spot_high.pkl' #MP


try:
    import nqcpfem
except ModuleNotFoundError:
    import sys
    sys.path.extend([r'/nbi/user-scratch/k/ksh820/NQCP/NQCP_FEM'])
    import nqcpfem
import pandas as pd
import logging
LOGGER = logging.getLogger()

LOGGER.setLevel(logging.DEBUG)

file_formatting =logging.Formatter(fmt='- %(asctime)s: %(processName)s:%(name)s >> %(levelname)s:%(message)s', datefmt= '%Y-%m-%d %H:%M:%S')
if False:
    ch= logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    ch.setFormatter(file_formatting)
    LOGGER.addHandler(ch)

file_handler = logging.FileHandler(f'./logs/sweet_spot_high.log') #MP
file_handler.setFormatter(file_formatting)

logging.getLogger('spawn').setLevel(logging.ERROR)
logging.getLogger('dist').setLevel(logging.ERROR)
logging.getLogger('build_ext').setLevel(logging.ERROR)

LOGGER.addHandler(file_handler)

import numpy as np
import sympy
E0 = 1/(nqcpfem.UNIT_CONVENTION['J to eV']*1000000) # unit_scale: 1muev
from nqcpfem.solvers import PETScSolver
from nqcpfem.systems import PositionalState,DefiniteTensorComponent
from nqcpfem.envelope_function import RectangleDomain
from nqcpfem.fenics import FEniCsModel
from nqcpfem.band_model import LuttingerKohnHamiltonian
from nqcpfem.systems.dot_sc_dot import *



def syst_init():
    H = LuttingerKohnHamiltonian(spatial_dim=3).material_spec('Ge').add_z_confinement(1,'box',25e-9)
    H.BdG_extension()
    B = np.sqrt(2)/2
    H.add_zeeman_term(B=[B/np.sqrt(2),B/np.sqrt(2),0])
    

    #H.add_potential(-sympy.symbols('\mu'))

    domain =RectangleDomain(200e-9,100e-9,0)
    domain.resolution = [100,50]

    model = FEniCsModel(H,None, boundary_condition=0,function_class=('CG',1))
    omega = 5e11
    L_sc = 300e-9
    ldot = Dot(-(86e-9+L_sc/2),omega,0,150e-9,1.44*omega,0,150e-9,)
    rdot = Dot((86e-9+L_sc/2),1.44*omega,0,150e-9,omega,0,150e-9,)
    barr = Barrier(10e-9,1000*E0) # 1 meV barrier
    #rbarr = Barrier(15e-9,3/(nqcpfem.UNIT_CONVENTION['J to eV']*1000))
    sc = Superconductor(100*E0,L_sc,100e-9,0,5*E0)
    syst = DotSCDot(model,ldot,barr,sc,barr,rdot,domain_resolution=[200,100])

    mu,mu_sc = sympy.symbols('\mu,\mu_{sc}')
    mu_R = sympy.symbols('\mu_{R}')
    mu_L = sympy.symbols('\mu_{L}')
    chemical_potential = SymbolicFunction(sympy.Piecewise((-mu_sc,syst.domains['sc_in']),(-mu_L,syst.domains['ld_in']),(-mu_R,syst.domains['rd_in']),(0,True)),'\mu(x)')
    H.add_potential(chemical_potential)

    H.parameter_dict[mu_L] = 0*E0 # will be set to something other than zero on determining couping
    H.parameter_dict[mu_R] = 0*E0 # will be set to something other than zero on determining couping
    H.parameter_dict[mu_sc] = 0

    #minimizations = syst.determine_all_couplings(0,3,solver,425*E0,(-50*E0,50*E0))
    return syst

syst = syst_init() # initialize before to get easier access.

from nqcpfem.solvers import PETScSolver,IterativeSolver
solver = PETScSolver(k=16,which='SM',sigma=0)

def system_update(mu_sc_val,mL,mR):
    mu_R = sympy.symbols('\mu_{R}')
    mu,mu_sc = sympy.symbols('\mu,\mu_{sc}')
    mu_L = sympy.symbols('\mu_{L}')
    syst.envelope_model.band_model.parameter_dict[mu_L] = mL
    syst.envelope_model.band_model.parameter_dict[mu_R] = mR
    syst.envelope_model.band_model.parameter_dict[mu_sc] = mu_sc_val
    return syst.envelope_model

first_solver = PETScSolver(k=8,sigma=0,which='SM')
second_solver = PETScSolver(k=8,sigma=0,which='SM',method='GD') # GD uses the the entire subspaces of the previous guess!
second_solver =None
solver = IterativeSolver(first_solver,second_solver)

from nqcpfem.band_model import covariant_sorting
from nqcpfem.majorana import majorana_overlap
def eval_func(model,res):

    try:
        #l_dot,r_dot,sc = syst.__make_system_classes__()
        #X_points = syst.envelope_model.positional_rep(res[1][0])[1]
        #selection=syst.select_subspace((l_dot,r_dot),res[1],4,x_points=X_points) # discard the states in the superconductor as we do not care about them
        
        overlaps = [majorana_overlap(res[1][i],model) for i in range(len(res[0]))]
        
        return res[0],overlaps
    except Exception as err:
        print(err)
        LOGGER.exception(err)
        LOGGER.info(err)
        raise err
        return res[0],None


from nqcpfem.parameter_search import IterativeModelSolver
it_solver = IterativeModelSolver(system_update,solver,eval_func)




if __name__ == '__main__':
    
    # construct parameter set for this worker
    mu_sc_values = np.linspace(3e3,9e3,16)[::4]*E0 
    #mu_sc_values = np.linspace(9.8e3,10.2e3,9)*E0
    #mu_sc_values = np.linspace(9.9e3,10.6e3,16)*E0
    #mu_sc_values = [9.5e3*E0,10e3*E0,10.5e3*E0]
    #mu_sc_values = np.linspace(4200,5200,8)*E0#+list(np.linspace(5700,6200,8)*E0)
    E0_values = np.linspace(4000,4120,1)*E0
    E0_values = [4120*E0,4150*E0]
    
    
    #high mu sweet spot
    mu_sc_values = np.linspace(6100,6200,8)*E0 # High mu sweet_spot zoom
    mR_values = np.linspace(4602.75,4604.5,48)*E0 #+list(np.linspace(4512,4614,48)*E0)
    mL_values = np.linspace(4597.5,4601,64)*E0
    
    # low mu sweet spot
    mu_sc_values = np.linspace(4340,4490,8)*E0 # Log mu sweet_spot zoom
    mR_values = np.linspace(4603.25,4604.5,48)*E0 #+list(np.linspace(4512,4614,48)*E0)
    mL_values = np.linspace(4597.5,4600.5,64)*E0
    
    parameter_set = [{'mu_sc_val':mu_val,'mL':ml,'mR':mr} for  mu_val in mu_sc_values for mr in mR_values for ml in mL_values]
    
    from nqcpfem.parameter_search import MPParameterSearch,ParameterSearch,DBMPParameterSearch
    import pickle as pkl
    with open(f'param_{TEMP_SAVE}','wb') as f:
        pkl.dump(parameter_set,f)
    
    try:
        raise FileNotFoundError
        print(f'loading {TEMP_SAVE}')
        search = MPParameterSearch.load(TEMP_SAVE)
        search.evaluation_function = system_builder
        print(f'found saved search resuming from: {len(search.results)}')        
    except FileNotFoundError:
        search =  DBMPParameterSearch(parameter_set,it_solver,TEMP_SAVE) #MP
    
    
    n_workers = len(os.sched_getaffinity(0))
    
    print(f'running search with number of workers: {n_workers}')
    #search.run(n_workers,True,False)
    search.run(n_workers,True,False)
    
    #store result
   



