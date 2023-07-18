import numpy as np
import logging
LOG = logging.getLogger(__name__)
LOG.addHandler(logging.NullHandler())
try:
    # Asserting that we are using the correct version of PETSc built to use complex numbers
    from petsc4py import PETSc # type: ignore
    if np.dtype(PETSc.ScalarType).kind != 'c':

        import os,importlib,sys
        os.environ['PETSC_DIR'] = '/usr/lib/petscdir/petsc-complex'
        os.environ['SLEPC_DIR'] = '/usr/lib/slepcdir/slepc-complex'
        importlib.reload(sys.modules['petsc4py'])
        from petsc4py import PETSc

        if np.dtype(PETSc.ScalarType).kind != 'c':
            LOG.warning(f'unable to load PETSc with complex scalartype. This will cause errors when using PETSc.'
                        f'To fix this specify PETSC_DIR, SLEPC_DIR environment variables to be the complex builds of PETSc and SLEPc') # unable to load PETSc using complex build
            class PETSc(): # type: ignore
                ScalarType = np.complex128
except ModuleNotFoundError as e:
    LOG.info('PETSc module not found. Proceeding __init__ with PETSc.Scalartype = np.complex128')
    class PETSc():
        """Placeholder class for using this module if PETSc is not installed"""
        ScalarType = np.complex128
# Physical Constants used for definining the unit convention used in the model.
_hbar = 1.054571817e-34  # J*s
_m_e = 9.1093837015e-31  # Kg
_L = 50e-9  # m.    Size of a Germanium quantum dot according to Hendrickx et. al 2018
_e = 1.60217663e-19  # Coulombs
_mu_B = _e*_hbar/(2*_m_e) # bohr magneton
_varepsilon_0 =  8.8541878128e-12 # permittivity of vacuum  in farad per meter



import sympy
constants = {'hbar': sympy.symbols('\hbar'),
    'pi': sympy.symbols('\pi'),
    'm_e':  sympy.symbols(r'm_{e}'),
    'varepsilon_0':  sympy.symbols(r'\varepsilon_{0}'),
    'e':  sympy.symbols(r'e'),
    'mu_B':  sympy.symbols(r'\mu_{B}'),
}

values =  {
    constants['hbar']: _hbar,
    sympy.symbols('hbar'): _hbar, # \hbar name breaks lambdify so we often use the hbar name instead
    'hbar': _hbar,
    constants['pi']: np.pi,
    'pi': np.pi,
    constants['m_e']:  _m_e,
    'm_e':  _m_e,
    constants['varepsilon_0']: _varepsilon_0,
    'varepsilon_0': _varepsilon_0,
    constants['e']:  _e,
    'e':  _e,
    constants['mu_B']:  _mu_B,
    'mu_B':  _mu_B,
}

UNIT_CONVENTION = {'E': _hbar**2/(_m_e*_L**2),  # energy scale (in Joules)
                   't': (_m_e*_L**2)/_hbar,  # time=hbar/energy_scale (in seconds). in these units if omega=1 a
                                             # regualr SHO has postional variance _L**2/2
                   'B': _hbar/(_L**2 * _e),  # magnetic field (in Tesla)
                   'x': _L,  # length scale (in meters)
                   'Q': _e,  # typical charge (in Coulombs)
                   'J to eV': 1/1.602176634e-19,
                   }

sq = np.sqrt(3)
Jx_c = 1 / 2 * np.array([[0, sq, 0, 0], [sq, 0, 2, 0], [0, 2, 0, sq], [0, 0, sq, 0]], dtype=PETSc.ScalarType)
Jy_c = 1j / 2 * np.array([[0, -sq, 0, 0], [sq, 0, -2, 0], [0, 2, 0, -sq], [0, 0, sq, 0]],
                         dtype=PETSc.ScalarType)
Jz_c = 1 / 2 * np.array([[3, 0, 0, 0], [0, 1, 0, 0], [0, 0, -1, 0], [0, 0, 0, -3]], dtype=PETSc.ScalarType)

sigma_x = np.array([[0,1],[1,0]], dtype=PETSc.ScalarType)
sigma_y = np.array([[0,-1j],[1j,0]], dtype=PETSc.ScalarType)
sigma_z = np.array([[1,0],[0,-1]], dtype=PETSc.ScalarType)
ANGULAR_MOMENTUM = {'1/2': 0.5*np.array([sigma_x,sigma_y,sigma_z]),
                    '3/2': np.array([Jx_c,Jy_c,Jz_c])}

sq = sympy.sqrt(3)
Jx = sympy.Matrix([[0, sq, 0, 0], [sq, 0, 2, 0], [0, 2, 0, sq], [0, 0, sq, 0]]) * 1 / sympy.Integer(2)
Jy = sympy.Matrix([[0, -sq, 0, 0], [sq, 0, -2, 0], [0, 2, 0, -sq], [0, 0, sq, 0]]) * sympy.I / sympy.Integer(2)
Jz = sympy.Matrix([[3, 0, 0, 0], [0, 1, 0, 0], [0, 0, -1, 0], [0, 0, 0, -3]]) / sympy.Integer(2)
SP_ANGULAR_MOMENTUM= {'3/2':[Jx,Jy,Jz]}