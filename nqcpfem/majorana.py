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
    # parity operator is [[0,-O],[O,0]]*K where O is the time-reversal operator and K is the complex conjugation operator(for fermions. For bosons there is no minus sign because O^-1 = O)
    
    if tr is None:
        raise ValueError(f'No time-reversal change of basis has been defined for the band model: {type(model.band_model)}')
    
    n= tr.shape[0]
    U = E = sympy.Array([[0]*n + l.tolist() for l in -1*tr]+[l.tolist()+[0]*n for l in tr ])
    
    U_op = model.construct_observable(U)
    
    
    # find the two eigenvectors closest to the chemical potential
    
    E_Is = np.argsort(np.abs(evals))[:2]
    relevant_vectors = evecs[E_Is]
    
    # project the parity operator down to ti subspace an diagonaize it:
    
    U_proj = U_op.mel(relevant_vectors,np.conj(relevant_vectors)) # we complex conjugate the right hand vectors as dictated by complex conjugation operator
    LOGGER.debug(f'U_proj: {U_proj}')
    parity_evals,parity_evecs = np.linalg.eig(U_proj) 
    LOGGER.debug(f'eigen decomposition: {parity_evals}, {parity_evecs}')
    
    disentangled_vectors = np.einsum('ij,i... -> j...',parity_evecs,relevant_vectors)
    return disentangled_vectors
    