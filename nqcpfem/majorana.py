from nqcpfem.envelope_function import EnvelopeFunctionModel
from nqcpfem.observables import AbstractObservable
import sympy
import numpy as np

import logging
LOGGER = logging.getLogger(__name__	)
LOGGER.addHandler(logging.NullHandler())


def find_majoranas(solution,model:EnvelopeFunctionModel):
    evals = solution[0]
    evecs = solution[1]
    
    tr = model.band_model.__time_reversal_change_of_basis__ 
    # parity operator is [0,-i*O,i*O,0] where O is the time-reversal operator (for fermions. For bosons there is no minus sign because O^-1 = O)
    
    if tr is None:
        raise ValueError(f'No time-reversal change of basis has been defined for the band model: {type(model.band_model)}')
    
    n= tr.shape[0]
    U=sympy.Array([[0]*n + l.tolist() for l in -1j*tr]+[l.tolist()+[0]*n for l in 1j*tr ]) 
    
    # the operator to construct the majorana pairs are 1/sqrt(2)*(1 \pm U) 
    I = sympy.Array(sympy.eye(n))
    U_p = 1/sympy.sqrt(2)*(I+U)
    U_m = 1/sympy.sqrt(2)*(I-U)
    
    
    
    U_p_op = model.construct_observable(U_p)

    U_m_op = model.construct_observable(U_m)
    
    
    # find the two eigenvectors closest to the chemical potential
    
    E_Is = np.argsort(np.abs(evals))[:2]
    relevant_vectors = evecs[E_Is]
    
    # project the parity operator down to ti subspace an diagonaize it:
    
    U_proj = U_op.mel(relevant_vectors,relevant_vectors)
    LOGGER.debug(f'U_proj: {U_proj}')
    parity_evals,parity_evecs = np.linalg.eig(U_proj) 
    LOGGER.debug(f'eigen decomposition: {parity_evals}, {parity_evecs}')
    
    disentangled_vectors = np.einsum('ij,i... -> j...',parity_evecs,relevant_vectors)
    return disentangled_vectors,U

from nqcpfem.fenics import FEniCsModel
import dolfinx,ufl
def majorana_overlap(state,model:FEniCsModel):
    # given a state \psi compute the Kolmogorov distance between the positional probability distributions of \phi_{\pm} where \phi_\pm = 1/sqrt(2)*(\psi \pm P\psi)
    # where P is the particle-hole operator
    
    V = model.function_space()
    f = dolfinx.fem.Function(V)
    tr = model.band_model.__time_reversal_change_of_basis__ 
    # parity operator is [0,-i*O,i*O,0] where O is the time-reversal operator (for fermions. For bosons there is no minus sign because O^-1 = O)
    if tr is None:
        raise ValueError(f'No time-reversal change of basis has been defined for the band model: {type(model.band_model)}')
    n= tr.shape[0]
    U=sympy.Array([[0]*n + l.tolist() for l in -1j*tr]+[l.tolist()+[0]*n for l in 1j*tr ]) 
    # the operators to construct the majorana pairs are 1/sqrt(2)*(1 \pm U) 
    P = ufl.as_tensor(np.array(U).astype('complex')/np.sqrt(2))
    sq = np.complex128(1/np.sqrt(2))
    
    i,j,k,l,m,n = ufl.indices(6)
    phi_p = sq*f[i,k]+P[i,j]*ufl.conj(f[j,k])
    phi_m = sq*f[l,n]-P[l,m]*ufl.conj(f[m,n])
    
    prob_p = phi_p*ufl.conj(phi_p)
    prob_m = phi_m*ufl.conj(phi_m)
    
    ufl_form = 0.5*ufl.sqrt((prob_p-prob_m)**2)*ufl.dx # L1 norm
    
    
    
    
    f.x.array[:] = model.flatten_eigentensors(state)
    
    dolfinx_form = dolfinx.fem.form(ufl_form)
    return dolfinx.fem.assemble_scalar(dolfinx_form)



def majorana_overlap_bound(state_0,state_1,model:FEniCsModel):
    V = model.function_space()
    f = dolfinx.fem.Function(V)
    g = dolfinx.fem.Function(V)

    i,j = ufl.indices(2)
    ip = f[i,j]*ufl.conj(g[i,j])
    
    ufl_form= ufl.sqrt(ip*ufl.conj(ip))*ufl.dx # integrate |<f|g>| over all of space. different from usual inner produt because we have absolute values!
     
    f.x.array[:] = model.flatten_eigentensors(state_0)
    g.x.array[:] = model.flatten_eigentensors(state_1)
    
    dolfinx_form = dolfinx.fem.form(ufl_form)
    return dolfinx.fem.assemble_scalar(dolfinx_form)

    