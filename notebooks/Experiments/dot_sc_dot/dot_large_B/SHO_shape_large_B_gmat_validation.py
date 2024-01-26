try:
    import nqcpfem
except ModuleNotFoundError as err:
    import sys
        # put path to nqcpfem module here   
    src_path = '/mnt/c/Users/olisc/NQCP/NQCP_FEM/'
    sys.path.append(src_path)
    import nqcpfem
import sympy
sympy.init_printing(use_latex='mathjax')
import IPython.display as disp
from matplotlib import pyplot as plt
import numpy as np
import os

import logging
LOG = logging.getLogger()
LOG.setLevel(logging.INFO)
from nqcpfem.parameter_search import DBMPParameterSearch, ParameterSearch
from nqcpfem.band_model import LuttingerKohnHamiltonian
from nqcpfem.solvers import PETScSolver
from nqcpfem.envelope_function import RectangleDomain
from nqcpfem.fenics import FEniCsModel
from nqcpfem.g_matrix import GMatrix
from nqcpfem.band_model import __MAGNETIC_FIELD_NAMES__ as _B_
_B_ = sympy.symbols(_B_,commutative=False) # cast to symbols
from nqcpfem.functions import SymbolicFunction
from nqcpfem.spin_gap import find_spin_gap




def sho_eval_gmat(lx):
    #omega_x_val = bm.constants[sympy.symbols('\hbar')]/(2*bm.parameter_dict[sympy.symbols('m')]*(lx/10)**2)
    sho_gmat.envelope_model.band_model.parameter_dict[omega_x] = lx#omega_x_val
    mat=sho_gmat.matrix()#min_energy=0) # drop negative energy solutions when working with BdG
    return mat



def sho_eval_b_field(lx,b):
    #make band_model and fem model
    bm = LuttingerKohnHamiltonian(spatial_dim=3).material_spec('Ge')
    bm.add_z_confinement(1,'box',25e-9)
    bm.add_zeeman_term()#B=[1,1,1])



    solver = PETScSolver(k=10,which='SM',sigma=0)
    domain = RectangleDomain(Lx=250e-9,Ly=150e-9,Lz=0) 
    domain.resolution = [300,180]
    fem_model = FEniCsModel(bm,domain,0,('CG',1))


    x,y,z = bm.position_symbols
    omega_x,omega_y = sympy.symbols(r'\omega_x,\omega_y')
    m = sympy.symbols('m')

    V = 0.5*m*( (omega_x*x)**2 + (omega_y*y)**2 ) 

    bm.add_potential(V)

    bm.material_spec('Ge')

    E0 = 1/(nqcpfem.UNIT_CONVENTION['J to eV']*1000000) # unit_scale: 1muev
    omega_y_val = 2e12


    fem_model.band_model.parameter_dict[omega_y] = omega_y_val

    fem_model.band_model.parameter_dict[omega_x] = 1e12 #omega_x_val


    sho_gmat = GMatrix(fem_model,solver)
    sho_gmat.envelope_model.band_model.parameter_dict[omega_y]


    for bname in _B_:
    
        bm.function_dict[bname] = SymbolicFunction(sympy.sympify(0),bname)
    
    fem_model = sho_gmat.envelope_model
    fem_model.band_model.parameter_dict[omega_x] = lx#omega_x_val
    fem_model.band_model.function_dict[_B_[0]]= SymbolicFunction(sympy.sympify(b),_B_[0])
    fem_model.band_model.function_dict[_B_[1]]= SymbolicFunction(sympy.sympify(0),_B_[1])
    #compute Bx energies
    
    
    Bx_energies = find_spin_gap(solver.solve(fem_model),fem_model)
    LOG.debug(Bx_energies)

    
    #compute By energies
    fem_model.band_model.function_dict[_B_[0]]= SymbolicFunction(sympy.sympify(0),_B_[0])
    fem_model.band_model.function_dict[_B_[1]]= SymbolicFunction(sympy.sympify(b),_B_[1])
    By_energies = find_spin_gap(solver.solve(fem_model),fem_model)
    LOG.debug(By_energies)
    
    fem_model.band_model.function_dict[_B_[1]]= SymbolicFunction(sympy.sympify(0),_B_[1])
    
    del fem_model
    
    return Bx_energies,By_energies
    
from nqcpfem.parameter_search import ParameterSearch,DBMPParameterSearch
sho_save = 'sho_fem_b_field.save'

#Lx_values = np.linspace(50e-9,200e-9,64)

B_field_magnitudes = np.linspace(0.1,3,32)
Lx_values = np.linspace(1,3.2,64)*1e12 # range from 1e12 to 4e12
sho_search  = None 

param_dict = [{'b':b,'lx':lx} for b in B_field_magnitudes for lx in Lx_values]

if os.path.exists(sho_save):
    try:
        sho_search = ParameterSearch.load(sho_save)
    except Exception as err:
        print(err)
#sho_search=None
#sho_search = None
if sho_search is None:
    sho_search = DBMPParameterSearch(parameter_sets = param_dict,evaluation_function=sho_eval_b_field,save_file=sho_save)
    
    
    
sho_search.run(n_workers = 2,skip_errors=False)