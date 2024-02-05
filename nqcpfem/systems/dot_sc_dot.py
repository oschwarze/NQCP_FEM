from nqcpfem.solvers import ModelSolver
from . import SpinorProjection, System,StateClass
import sympy
from collections import namedtuple
import numpy as np
from typing import Callable, Type
from copy import copy
from ..envelope_function import EnvelopeFunctionModel,RectangleDomain
from ..band_model import BandModel
from ..functions import SymbolicFunction,X,Y,Z

import logging
LOGGER = logging.getLogger(__name__)
LOGGER.addHandler(logging.NullHandler())
Dot = namedtuple('Dot',['x','omega_x','Omega_x','w_x','omega_y','Omega_y','w_y'])
Barrier = namedtuple('Barrier',['length','V']) # barrier is always fixed to be at the end of the SC
Superconductor = namedtuple('Superconductor',['Delta','length','width','V_in','V_out']) #superconductor center is always in the center

from ..functions import X,Y,Z



class DotSCDot(System):
    def __init__(self,envelope_model:EnvelopeFunctionModel,left_dot:Dot,left_barrier:Barrier,superconductor:Superconductor,right_barrier:Barrier|None = None,right_dot:Dot|None = None,domain_resolution =None):
        envelope_model = copy(envelope_model)     
        self.left_dot = left_dot
        self.left_barrier = left_barrier
        
        
        self.superconductor = superconductor

        # mirror the the left side if right side was not specified
        self.right_barrier = left_barrier if right_barrier is None else right_barrier
        self.right_dot = self.left_dot._replace(x=-self.left_dot.x ) if right_dot is None else right_dot
        
        #define the domains of the system
        self.domains = self.__setup_domains__()
        
        
        # set the EFM domain
        x_range = (self.left_dot.x-self.left_dot.w_x/2, self.right_dot.x+self.right_dot.w_x/2)
        y_dist = max((self.left_dot.w_y,self.right_dot.w_y,self.superconductor.width))
        self.enf_domain = RectangleDomain.from_corners((x_range[0]*1.01,-y_dist*1.01/2,0),(x_range[1]*1.01,y_dist*1.01/2,0))
        if domain_resolution is not None:
            self.enf_domain.resolution = domain_resolution
        envelope_model.domain = self.enf_domain
        
        
        # add potential defining the system
        envelope_model.band_model.add_potential(self.__make_EM_potential__(envelope_model.band_model.parameter_dict['m']))
        
        # add SC order parameter: 
        
        # create identity_array and reshape into tensor shape
        #transposition = tuple(list(range(0,len(envelope_model.band_model.tensor_shape),2))) +  tuple(list(range(1,len(envelope_model.band_model.tensor_shape),2))) 
        #I = np.eye(envelope_model.band_model.n_bands).reshape([s for s in envelope_model.band_model.tensor_shape[::2]]*2).transpose(transposition)
        I = sympy.eye(envelope_model.band_model.independent_vars['preprocessed_array'].shape[0])
        Delta_func = self.make_SC_Delta()
        envelope_model.band_model.independent_vars['Delta_SC'] = I*Delta_func.symbol
        envelope_model.band_model.function_dict[Delta_func.symbol] = Delta_func
        
        super().__init__(envelope_model,self.__make_system_classes__())
        
        

    def __setup_domains__(self):
        
        ld = self.left_dot
        rd = self.right_dot
        lb = self.left_barrier
        rb = self.right_barrier
        sc = self.superconductor
        
        # check that stuff fits:
        if ld.x+ld.w_x/2 > -sc.length/2 -lb.length:
            raise ValueError(f'left dot clipping into left barrier by amount: {np.abs(ld.x+ld.w_x/2  +sc.length/2 +lb.length)}')
        
        
        if rd.x-rd.w_x/2 < sc.length/2+rb.length:
            raise ValueError(f'right dot clipping into right barrier by amount: { np.abs(rd.x-rd.w_x/2 - sc.length/2-rb.length)}')
        
        s = lambda x:sympy.sympify(x)
        
        
        ldot_domain =sympy.And(( s(ld.x-ld.w_x/2) <= X),(X <= s(ld.x + ld.w_x/2)),( s(-ld.w_y/2) <=Y),(Y<=s(ld.w_y/2)))
        rdot_domain =sympy.And(( s(rd.x-rd.w_x/2) <= X),(X <= s(rd.x + rd.w_x/2)),( s(-rd.w_y/2) <=Y),(Y<=s(rd.w_y/2)))
        
        left_side_domain = sympy.And(X<=s(-sc.length/2 - lb.length), sympy.Not(ldot_domain))#, sympy.Not(ldot_domain))
        right_side_domain = sympy.And(X>=s(sc.length/2 + rb.length), sympy.Not(rdot_domain))

        
        left_B_domain = sympy.And(X<=s(-sc.length/2),s(-sc.length/2-lb.length)<X)
        right_B_domain = sympy.And(X>=s(sc.length/2),s(sc.length/2+rb.length)>X)
        
        
        sc_x_domain = sympy.And(X>=s(-sc.length/2),X<=s(sc.length/2))
        
        sc_y_domain =sympy.And( s(-sc.width/2)<=Y, Y<=s(sc.width/2))
        
        in_SC_domain = sympy.And(sc_y_domain,sc_x_domain)#
        out_SC_domain = sympy.And(sc_x_domain, sympy.Not(sc_y_domain))
        return {'ld_in':ldot_domain,'rd_in':rdot_domain,'ld_out':left_side_domain,'rd_out':right_side_domain,'lb':left_B_domain,'rb':right_B_domain,'sc_in':in_SC_domain,'sc_out':out_SC_domain}

        
    def __make_EM_potential__(self,mass):
        
        ld = self.left_dot
        rd = self.right_dot
        lb = self.left_barrier
        rb = self.right_barrier
        sc = self.superconductor
        
        
        m = sympy.Symbol('m')
        symbols = r'\omega_{Lx},\omega_{Ly},\Omega_{Lx},\Omega_{Ly},V_{BL}, V_{BR},V_{SC},V_{out},\omega_{Rx},\omega_{Ry},\Omega_{Rx},\Omega_{Ry}'
        olx,oly,Olx,Oly,Bl,Br,Vsc,Vout,orx,ory,Orx,Ory= sympy.symbols(symbols)
        
        olx = ld.omega_x
        oly = ld.omega_y
        Olx = ld.Omega_x
        Oly = ld.Omega_y
        
        orx = rd.omega_x
        ory = rd.omega_y
        Orx = rd.Omega_x
        Ory = rd.Omega_y
        
        m=mass
        
        Vd_left = 1/2 * m * (olx**2 * (X-ld.x)**2 + oly**2 * (Y)**2 + (Olx * (X-ld.x)**2 + Oly* Y**2)**2)  
        
        
        Vd_right= 1/2 * m * (orx**2 * (X-rd.x)**2 + ory**2 * Y**2 + (Orx * (X-rd.x)**2 + Ory* Y**2)**2)  


        l_dot_outside = Vd_left.subs({X:(ld.x+ld.w_x/2),Y:ld.w_y/2})
        r_dot_outside = Vd_right.subs({X:(rd.x+rd.w_x/2),Y:rd.w_y/2})
        
        values = {'ld_in':Vd_left,'ld_out':l_dot_outside, 'rd_in':Vd_right,'rd_out':r_dot_outside, 'lb':lb.V,'rb':rb.V, 'sc_in':sc.V_in,'sc_out':sc.V_out}
        
        return SymbolicFunction(self.assemble_piecewise(values),sympy.Symbol('V_{sys}(x)',commutative=False))
    
    def make_SC_Delta(self):
        sc_values = {k:0 for k in self.domains.keys()}
        sc_values['sc_in'] = self.superconductor.Delta
        return SymbolicFunction(self.assemble_piecewise(sc_values),sympy.Symbol('\Delta_{sc}(x)',commutative=False))
        
    
    def assemble_piecewise(self,domain_values):
        arg = [ (domain_values[k],dom) for k,dom in self.domains.items()]
        
        return sympy.Piecewise(*arg)
    
    def __make_system_classes__(self):
        # Inside Left dot,
        from . import PositionalState
        left_cond = self.domains['ld_in'] 
        left_dot = PositionalState(left_cond,'Left dot')
        
        
        # Inside Right dot
        right_cond = self.domains['rd_in']
        right_dot = PositionalState(right_cond,'Right dot')

        sc_cond = self.domains['sc_in']
        superconductor = PositionalState(sc_cond,'Superconductor')
        
        return left_dot,right_dot,superconductor
        
    
    
    def perturbative_selection_couplings(self,basis_states,solver,E0,parameter_range,method,**crossing_finder_kwargs):
        """
        determine T,Tso,D and Dso coupling between the dots of the system, by taking a dict of states as the reference states.
        :param basis_states: dict with strings: "plu,pru,prd,hru,hrd,"
        """
            
        mu_L = sympy.Dummy(r'\mu_{L}') # make them dummies to avoid overwriting them 
        mu_R = sympy.Dummy(r'\mu_{R}')
        mu_detuning = sympy.Dummy('\mu_{detuning}(x)',commutative=False)
        detuning = SymbolicFunction(sympy.Piecewise((-mu_L,self.domains['ld_in']),(-mu_R,self.domains['rd_in']),(0,True)),mu_detuning)
        
        self.envelope_model.band_model.add_potential(detuning)
        self.envelope_model.band_model.parameter_dict[mu_L] = E0
        self.envelope_model.band_model.parameter_dict[mu_R] = E0
        
        def model_update(mu_R_val):
            params = {mu_R:mu_R_val}
            self.envelope_model.band_model.parameter_dict[mu_R] = mu_R_val
            #self.envelope_model.band_model.parameter_dict.update(params)
            return self.envelope_model
        
        
        
        parameter_range = tuple(p+E0 for p in parameter_range)
        if method == 'signed':
            
            from nqcpfem.parameter_search import IterativeModelSolver
            
            simple_it_solver=IterativeModelSolver(model_update,solver)

            #cast energies to numerically reasonable values
            # apply preprocessing: divide x_range into 100 pieces and make this the unit scale of energy
            E_scale = (parameter_range[1]-parameter_range[0])/100
            LOGGER.debug(f'E_scale: {E_scale}')
            xL = parameter_range[0]/E_scale
            xR = parameter_range[1]/E_scale

            #normalize basis states and construct subspace projections
            if isinstance(basis_states,list):
                subspaces = [(0,i) for i in range(1,len(basis_states))]

                for i,b in enumerate(basis_states):
                    norm = np.linalg.norm(b)
                    if not np.isclose(norm,1):
                        basis_states[i]= b/norm
                    
            elif all(w in basis_states for w in ('plu','pru','prd','hru','hrd')):
                subspaces = (('plu','pru'),('plu','prd'),('plu','hrd'),('plu','hru')) # T, Tso,D,Dso
                #normalize the states
                for k,v in basis_states.items():
                    norm= np.linalg.norm(v)
                    if not np.isclose(norm,1):
                        basis_states[k] = v/norm

            else:
                    raise ValueError('basis_states must either be a dict with keys "plu","pru","prd","hru,"hrd" or a list')
            
            #this contains the subspaces we wish to compute the overlaps wrt. They are already conjugated!
            subspace_stacks_conj = {N: np.stack([basis_states[N[0]],basis_states[N[1]]]).conj() for N in subspaces}

            def overlap_func(evecs,target_states):
                
                #compute in the regular way
                sum_index = tuple(range(2,len(evecs.shape)+1))
                weights = np.abs(np.einsum(subspace_stacks_conj[target_states],(0,)+sum_index,evecs,(1,)+sum_index))**2
                return weights
                
                
            
            def eval_func(x):
                #convert energy back
                x_real = x*E_scale
                LOGGER.debug(f'solving with model params: {x_real}')
                evals,evecs = simple_it_solver(x_real)
                #cast the energies to to the specified energy_scale
                evals = np.real(evals)/E_scale #drop imaginary part
                
                #normalisze the eigenvectors:
                norms = np.linalg.norm(evecs.reshape(evecs.shape[0],-1),axis=1)
                # add newaxis untill we have the right shape:
                while len(norms.shape)<len(evecs.shape):
                    norms = norms[...,np.newaxis] 
                evecs = evecs/norms
                return evals,evecs
            
            S = SignedSearch(xL,xR,eval_func,1e-2,subspaces,overlap_func) 
            res = S.minimize()
            
            
            # set the system back to what it was before (to avoid mysterious behaviour when working with the system again)
            self.envelope_model.band_model.independent_vars['preprocessed_array'] = self.envelope_model.band_model.independent_vars['preprocessed_array'].subs(mu_detuning,0)
            del self.envelope_model.band_model.function_dict[mu_detuning] 
            del self.envelope_model.band_model.parameter_dict[mu_L]
            del self.envelope_model.band_model.parameter_dict[mu_R]
            
            
            #cast the result back to real units
            result = [r['val']*E_scale/2 for r in res]
            return result
            
            
            
    def determine_all_couplings(self,spin_up_I,spin_down_I,solver,E0,parameter_range,verbose_result=False,method='custom',energy_method=False,**crossing_finder_kwargs):
        """
        Determine the T,Tso,D and Dso coupling of the two dots in the system
        :param int|np.ndarray spin_up_I: either the index of the spin up component or a spinor specifying the spin up state 
        :param int|np.ndarray spin_down_I: either the index of the spin down component or a spinor specifying the spin down state 
        
        """
        #region class constuction
        #make the Classes
        from . import PositionalState,DefiniteTensorComponent
        
        
        if isinstance(spin_up_I,int) and isinstance(spin_down_I,int):
            mask_template = lambda x: np.zeros(self.envelope_model.solution_shape()[:-1])
            up_p_mask = mask_template(None)
            up_p_mask[spin_up_I] = 1
            up_p_state = DefiniteTensorComponent(up_p_mask,'spin up particle')

            down_p_mask = mask_template(None)
            down_p_mask[spin_down_I] = 1
            down_p_state = DefiniteTensorComponent(down_p_mask,'spin down particle')
            
            up_h_mask = mask_template(None)
            ph_shift = int(up_h_mask.shape[0]/2)
            up_h_mask[ph_shift+spin_up_I] = 1
            up_h_state = DefiniteTensorComponent(up_h_mask,'spin up hole')
            
            
            down_h_mask = mask_template(None)
            down_h_mask[ph_shift+spin_down_I] = 1
            down_h_state = DefiniteTensorComponent(down_h_mask,'spin down hole')
        
        
        
        else:
            mask_template = lambda x: np.zeros(self.envelope_model.solution_shape()[:-1],dtype='complex')
            N = spin_up_I.shape[0]

            up_p_mask = mask_template(None)
            up_p_mask[:N] = spin_up_I
            up_p_state = SpinorProjection(up_p_mask,'spin up particle',axis=0)
            
            
            down_p_mask = mask_template(None)
            up_p_mask[:N] = spin_down_I
            down_p_state = SpinorProjection(down_p_mask,'spin down particle',axis=0)
            
            up_h_mask = mask_template(None)
            ph_shift = int(up_h_mask.shape[0]/2)
            up_h_mask[ph_shift:] = spin_up_I.conj()
            
            up_h_state = SpinorProjection(up_h_mask,'spin up hole',axis=0)
            
            
            down_h_mask = mask_template(None)
            up_h_mask[ph_shift:] = spin_down_I.conj()
            down_h_state = SpinorProjection(down_h_mask,'spin down hole',axis=0)
            
        
        
        mu_L = sympy.Dummy(r'\mu_{L}') # make them dummies to avoid overwriting them 
        mu_R = sympy.Dummy(r'\mu_{R}')
        mu_detuning = sympy.Dummy('\mu_{detuning}(x)',commutative=False)
        detuning = SymbolicFunction(sympy.Piecewise((-mu_L,self.domains['ld_in']),(-mu_R,self.domains['rd_in']),(0,True)),mu_detuning)
        
        self.envelope_model.band_model.add_potential(detuning)
        self.envelope_model.band_model.parameter_dict[mu_L] = E0
        self.envelope_model.band_model.parameter_dict[mu_R] = E0
        
        left,right,sc = self.__make_system_classes__()
        
        
        # step 2: Determine Tuu by detuning around mL = mR with s(pick spin up lef and spin up right stateClasses)
        left_up = left.combine_state_cls(up_p_state,'left up particle')
        right_up = right.combine_state_cls(up_p_state,'right up particle')
        
        right_down = right.combine_state_cls(down_p_state,'right down particle')
        
        right_h_down = right.combine_state_cls(down_h_state,'right down anit-particle')
        right_h_up = right.combine_state_cls(up_h_state,'right up anit-particle')
        #endregion
        
        # region iterative_model_solver setup
        def model_update(mu_R_val):
            params = {mu_R:mu_R_val}
            self.envelope_model.band_model.parameter_dict.update(params)
            return self.envelope_model
        
        
        prev_energies = [None,None]
        def energy_continuity(energies):
            delta1 = np.abs(energies[0]-prev_energies)     
            delta2 = np.abs(energies[1]-prev_energies)
            # find the energies that minimize the difference and 
            
            min_Is = (np.argmin(delta1),np.argmin(delta2))
            if min_Is[0] == min_Is[1]:
                l = delta1 if delta1[min_Is[0]]>delta2[min_Is[0]] else delta2
                sort_l = np.argsort(l)
                min_Is = (min_Is[0],sort_l[1])
                
            return np.array(min_Is)
        
        def evaluation_func(model,res):
            #determine all couplings here:
            X_arr = self.envelope_model.positional_rep(res[1][0])[1]
            vals = []
            subspaces = ((left_up,right_up),(left_up,right_down),(left_up,right_h_down),(left_up,right_h_up)) # T, Tso,D,Dso
            for subsp in subspaces:
                
                subspace_I = self.select_subspace(subsp,res[1],2,x_points=X_arr)
                vals.append(np.abs(res[0][subspace_I[0]]-res[0][subspace_I[1]]))

            other = res if verbose_result else None
                
            return (vals,res)
        
        iterative_model_solver = IntermediateResSave(model_update,solver,evaluation_func,return_func=lambda x:x)
        #endregion
        
        
        parameter_range = tuple(p+E0 for p in parameter_range)

        if method == 'signed':
            
            from nqcpfem.parameter_search import IterativeModelSolver
            
            simple_it_solver=IterativeModelSolver(model_update,solver)
            subspaces = ((left_up,right_up),(left_up,right_down),(left_up,right_h_down),(left_up,right_h_up)) # T, Tso,D,Dso
            # apply preprocessing: divide x_range into 100 pieces and make this the unit scale of energy
            
            E_scale = (parameter_range[1]-parameter_range[0])/100

            xL = parameter_range[0]/E_scale
            xR = parameter_range[1]/E_scale

            def overlap_func(evecs,target_states):
                X_arr = self.envelope_model.positional_rep(evecs[0])[1]
                return self.subspace_weights(target_states,evecs,x_points=X_arr).T
            
            def eval_func(x):
                #convert energy back
                x_real = x*E_scale
                evals,evecs = simple_it_solver(x_real)
                #cast the energies to to the specified energy_scale
                evals = evals/E_scale
                
                #rotate the basis of the spin DOF to make it parallel and anti-parallel wrt B
                
                return evals,evecs
            
            S = SignedSearch(xL,xR,eval_func,1e-2,subspaces,overlap_func) 
            res = S.minimize()
            
            #cast the result back to real units
            result = [r['val']*E_scale/2 for r in res] 
            return result
            
            
            
        
        if method == 'new':
            
            ranges = crossing_preprocessing(E0*1.01,left_up,right_up,right_down,right_h_up,right_h_down,iterative_model_solver,self)

            #determine the values at the boundaries to begin with so we don't have to compute these ourselves
            iterative_model_solver.return_func = lambda x:x[0]
            boundaries = {}
            for r in ranges:
                for v in r:
                    if v not in boundaries:
                        boundaries[v]= iterative_model_solver(v)
            
            solutions = []
            for i,r in enumerate(ranges):
                LOGGER.debug(f'run {i}')
                xL,fL= r[0],boundaries[r[0]][i]
                xR,fR = r[1],boundaries[r[1]][i]
                print(xL,fL,xR,fR)
                iterative_model_solver.return_func = lambda x: x[0][i]
                finder = CrossingFinder(iterative_model_solver,r,**crossing_finder_kwargs)
                solutions.append(finder.minimize(xL,fL,xR=xR,fR=fR,verbose_res = True ))
            
            
            return tuple(solutions)
        
        
        if method == 'custom':
            #find T
            LOGGER.info('finding T')
            
            iterative_model_solver.return_func = lambda x:x[0][0] # T number
            t_crossing_finder = CrossingFinder(iterative_model_solver,parameter_range,**crossing_finder_kwargs)
            
            T_sol = t_crossing_finder.minimize(verbose_res=True)
            LOGGER.debug(f'T_found. N_iter = {len(T_sol.derivs)}')

            
            
            #find T_so
            LOGGER.info('finding T_so')
            iterative_model_solver.return_func = lambda x:x[0][1] #Tso number
            x_vals = [s[0] for s in iterative_model_solver.saved_results]
            y_vals = [s[-1][0][1] for s in iterative_model_solver.saved_results]
            tso_crossing_finder,points = t_crossing_finder.warm_start(T_sol.derivs,iterative_model_solver,x_vals,y_vals,**crossing_finder_kwargs)
            points = tso_crossing_finder.find_starting_points(points)
            if isinstance(points,Result):
                Tso_sol = points
            else:
                lp,rp = points 
                Tso_sol = tso_crossing_finder.minimize(*lp[:3],*rp[:3],verbose_res = True)
            
            LOGGER.debug(f'Tso_found. N_iter = {len(Tso_sol.derivs)-len(T_sol.derivs)}')

            #find D
            LOGGER.info('finding D')
            iterative_model_solver.return_func = lambda x:x[0][2] #D number
            x_vals = [s[0] for s in iterative_model_solver.saved_results]
            y_vals = [s[-1][0][2] for s in iterative_model_solver.saved_results]
            D_crossing_finder,points = t_crossing_finder.warm_start(Tso_sol.derivs,iterative_model_solver,x_vals,y_vals,**crossing_finder_kwargs)
            points = D_crossing_finder.find_starting_points(points)
            if isinstance(points,Result):
                D_sol = points
            else:
                lp,rp = points 
                D_sol = D_crossing_finder.minimize(*lp[:3],*rp[:3],verbose_res = True)
            
            LOGGER.debug(f'D_found. N_iter = {len(D_sol.derivs)-len(Tso_sol.derivs)}')

            #find D_so
            
            LOGGER.info('finding D_so')
            iterative_model_solver.return_func = lambda x:x[0][3] #Dso number
            x_vals = [s[0] for s in iterative_model_solver.saved_results]
            y_vals = [s[-1][0][3] for s in iterative_model_solver.saved_results]
            Dso_crossing_finder,points = t_crossing_finder.warm_start(D_sol.derivs,iterative_model_solver,x_vals,y_vals,**crossing_finder_kwargs)
            points = Dso_crossing_finder.find_starting_points(points)
            if isinstance(points,Result):
                Dso_sol = points
            else:
                lp,rp = points 
                Dso_sol=Dso_crossing_finder.minimize(*lp[:3],*rp[:3],verbose_res = True)
            
            
            LOGGER.debug(f'Dso_found. N_iter = {len(Dso_sol.derivs)-len(D_sol.derivs)}')

            LOGGER.info(f'total number of function calls: {len(iterative_model_solver.saved_results)}')
            
            # set the sytem back to what it was before (to avoid mysterious behaviour when working with the system again)
            self.envelope_model.band_model.independent_vars['preprocessed_array'] = self.envelope_model.band_model.independent_vars['preprocessed_array'].subs(mu_detuning,0)
            del self.envelope_model.band_model.function_dict[mu_detuning] 
            
            
            return T_sol,Tso_sol,D_sol,Dso_sol

        elif method =='brent':
            
            
            
            
            
            # evaluate the model in the left-most, right-most and center points .
            iterative_model_solver.return_func = lambda x:x[0][0] # T number
            init_bracket = []
            for p in (parameter_range[0],2*parameter_range[0]/3+parameter_range[1]/3, parameter_range[1]):
                init_bracket.append(p)
                init_bracket.append(iterative_model_solver(p))
            
            
            #find T
            LOGGER.info('finding T')
            
            t_crossing_finder = GoldenCrossingFinder(iterative_model_solver,parameter_range,**crossing_finder_kwargs)
            
            T_sol = t_crossing_finder.minimize(*init_bracket)
            LOGGER.debug(f'T_found. N_iter = {len(T_sol.derivs)}')

            
            
            #find T_so
            LOGGER.info('finding T_so')
            iterative_model_solver.return_func = lambda x:x[0][1] #Tso number
            x_vals = [s[0] for s in iterative_model_solver.saved_results]
            y_vals = [s[-1][0][1] for s in iterative_model_solver.saved_results]
            
            
            tso_crossing_finder = GoldenCrossingFinder(iterative_model_solver,parameter_range,**crossing_finder_kwargs)
            init_bracket = tso_crossing_finder.determine_bracket(x_vals,y_vals)
            Tso_sol=tso_crossing_finder.minimize(*init_bracket)
            LOGGER.debug(f'Tso_found. N_iter = {len(Tso_sol.derivs)}')

            

            #find D
            LOGGER.info('finding D')
            iterative_model_solver.return_func = lambda x:x[0][2] #Tso number
            x_vals = [s[0] for s in iterative_model_solver.saved_results]
            y_vals = [s[-1][0][2] for s in iterative_model_solver.saved_results]
            
            D_crossing_finder = GoldenCrossingFinder(iterative_model_solver,parameter_range,**crossing_finder_kwargs)
            init_bracket = D_crossing_finder.determine_bracket(x_vals,y_vals)
            D_sol = D_crossing_finder.minimize(*init_bracket)
            LOGGER.debug(f'D found. N_iter = {len(D_sol.derivs)}')

            #find Dso
            LOGGER.info('finding D_so')
            iterative_model_solver.return_func = lambda x:x[0][3] #Tso number
            x_vals = [s[0] for s in iterative_model_solver.saved_results]
            y_vals = [s[-1][0][3] for s in iterative_model_solver.saved_results]
            
            
            Dso_crossing_finder = GoldenCrossingFinder(iterative_model_solver,parameter_range,**crossing_finder_kwargs)
            init_bracket = Dso_crossing_finder.determine_bracket(x_vals,y_vals)
            Dso_sol=Dso_crossing_finder.minimize(*init_bracket)
            LOGGER.debug(f'Dso fund. N_iter = {len(Dso_sol.derivs)}')
            LOGGER.info(f'total number of function calls: {len(iterative_model_solver.saved_results)}')
            
            # set the system back to what it was before (to avoid mysterious behaviour when working with the system again)
            self.envelope_model.band_model.independent_vars['preprocessed_array'] = self.envelope_model.band_model.independent_vars['preprocessed_array'].subs(mu_detuning,0)
            del self.envelope_model.band_model.function_dict[mu_detuning] 
            
            
            return T_sol,Tso_sol,D_sol,Dso_sol
        
        elif method =='GD':
            prev_energies = [None,None]
            def energy_continuity(energies,prev_energies):
                delta1 = np.abs(energies[0]-prev_energies)     
                delta2 = np.abs(energies[1]-prev_energies)
                # find the energies that minimize the difference and 
                
                min_Is = (np.argmin(delta1),np.argmin(delta2))
                if min_Is[0] == min_Is[1]:
                    l = delta1 if delta1[min_Is[0]]>delta2[min_Is[0]] else delta2
                    sort_l = np.argsort(l)
                    min_Is = (min_Is[0],sort_l[1])
                    
                return np.array(min_Is)
            spin_I= 0
            class evaluation_func():
                def __init__(e_self,prev_energies=[None,None]):
                    e_self.prev_energies = prev_energies
            
                def __call__(e_self,model,res):
                    #determine all couplings here:
                    vals = []
                    subspaces = ((left_up,right_up),(left_up,right_down),(left_up,right_h_down),(left_up,right_h_up)) # T, Tso,D,Dso
                    subsp = subspaces[spin_I] 
                    if energy_method and e_self.prev_energies[0] is not None:
                        subspace_I = energy_continuity(res[0],e_self.prev_energies)
                    else:
                        X_arr = self.envelope_model.positional_rep(res[1][0])[1]
                        subspace_I = self.select_subspace(subsp,res[1],2,x_points=X_arr)
                
                    e_self.prev_energies = (res[0][subspace_I[0]],res[0][subspace_I[1]])
                    return np.abs(res[0][subspace_I[0]]-res[0][subspace_I[1]])
            
            
            
            # evaluate the model in the left-most, right-most and center points .
            evalfunc = evaluation_func()
            iterative_model_solver.evaluation_func =  evalfunc   
            
            
            
            def make_init_bracket():
                init_bracket = []
                for p in (parameter_range[0],2*parameter_range[0]/3+parameter_range[1]/3, parameter_range[1]):
                    init_bracket.append(p)
                    init_bracket.append(iterative_model_solver(p))
                return init_bracket
            
            init_bracket = make_init_bracket()
            
            #find T
            LOGGER.info('finding T')
            
            t_crossing_finder = GoldenCrossingFinder(iterative_model_solver,parameter_range,**crossing_finder_kwargs)
            
            T_sol = t_crossing_finder.minimize(*init_bracket)
            LOGGER.debug(f'T_found. N_iter = {len(T_sol.derivs)}')

            
            
            #find T_so
            LOGGER.info('finding T_so')
            spin_I=1
            evalfunc.prev_energies = [None,None]
            
            tso_crossing_finder = GoldenCrossingFinder(iterative_model_solver,parameter_range,**crossing_finder_kwargs)
            init_bracket = make_init_bracket()
            Tso_sol=tso_crossing_finder.minimize(*init_bracket)
            LOGGER.debug(f'Tso_found. N_iter = {len(Tso_sol.derivs)}')

            

            #find D
            LOGGER.info('finding D')
            spin_I=2
            evalfunc.prev_energies = [None,None]
            
            D_crossing_finder = GoldenCrossingFinder(iterative_model_solver,parameter_range,**crossing_finder_kwargs)
            init_bracket =  make_init_bracket()
            D_sol = D_crossing_finder.minimize(*init_bracket)
            LOGGER.debug(f'D found. N_iter = {len(D_sol.derivs)}')

            #find Dso
            LOGGER.info('finding D_so')
            spin_I=3
            evalfunc.prev_energies = [None,None]
            
            
            Dso_crossing_finder = GoldenCrossingFinder(iterative_model_solver,parameter_range,**crossing_finder_kwargs)
            init_bracket =  make_init_bracket()
            Dso_sol=Dso_crossing_finder.minimize(*init_bracket)
            LOGGER.debug(f'Dso fund. N_iter = {len(Dso_sol.derivs)}')
            LOGGER.info(f'total number of function calls: {len(iterative_model_solver.saved_results)}')
            
            # set the system back to what it was before (to avoid mysterious behaviour when working with the system again)
            self.envelope_model.band_model.independent_vars['preprocessed_array'] = self.envelope_model.band_model.independent_vars['preprocessed_array'].subs(mu_detuning,0)
            del self.envelope_model.band_model.function_dict[mu_detuning] 
            
            
            return T_sol,Tso_sol,D_sol,Dso_sol
        """
        # evaluate the model at the golden ratio points so that we can determine the initial brcaket
        golden_mean = 0.5*(3-np.sqrt(5))
        xL = parameter_range[0]+E0
        xR = parameter_range[1]+E0
        point_finder =lambda l,r: l+golden_mean*(r-l)
        l0 = point_finder(xL,xR)
        l1 = point_finder(l0,xR)
        l2 = point_finder(l1,xR)
        l3 = point_finder(xL,l1)
        
        bracket_sols = [iterative_model_solver({mu_R:l}) for l in (l0,l1,l2,l3)]
        
        
        def bracket_finder(i):
            ls = np.array([l0,l1,l2,l3])

        
        return T_sol,Tso_sol,D_sol,Dso_sol

        
        # evaluate the model at the golden ratio points so that we can determine the initial brcaket
        golden_mean = 0.5*(3-np.sqrt(5))
        xL = parameter_range[0]+E0
        xR = parameter_range[1]+E0
        point_finder =lambda l,r: l+golden_mean*(r-l)
        l0 = point_finder(xL,xR)
        l1 = point_finder(l0,xR)
        l2 = point_finder(l1,xR)
        l3 = point_finder(xL,l1)
        
        bracket_sols = [iterative_model_solver({mu_R:l}) for l in (l0,l1,l2,l3)]
        
        
        def bracket_finder(i):
            ls = np.array([l0,l1,l2,l3])
            
            fu = np.array([f[0][i] for f in bracket_sols])[np.argsort(ls)]
            ls = np.sort(ls)
            
            # there are only 4 possibilities 
            if  fu[1]<fu[2]and fu[1]<fu[0]:
                return ((ls[0],ls[1],ls[2]),(fu[0],fu[1],fu[2]))
            if  fu[1]<fu[3]and fu[1]<fu[0]:
                return ((ls[0],ls[1],ls[3]),(fu[0],fu[1],fu[3]))
            if  fu[2]<fu[3]and fu[2]<fu[0]:
                return ((ls[0],ls[2],ls[3]),(fu[0],fu[2],fu[3]))
            if  fu[2]<fu[3]and fu[2]<fu[1]:
                return ((ls[1],ls[2],ls[3]),(fu[1],fu[2],fu[3]))
        
        
        bracket = bracket_finder(0)
        LOGGER.debug(f'T bracket: {bracket}')
        iterative_model_solver.return_func = lambda x:x[0][0]
        LOGGER.info('finding T:')
        T_sol = self.find_avoided_crossing(solver,(left_up,right_up),(mu_R,),{mu_R:parameter_range},iterative_solving=iterative_model_solver,bracket=bracket)
        
        
        #use the previous results to determine the bounds for the next one
        def bounds_determining(results,res_i):
            return parameter_range # workaround
            if len(results)<3:
                return parameter_range #we do not have enough info to go with
            
            results = [r[:-1] for r in results]
                
            x_vals = np.array([r[0][0][mu_R] for r in results])
            y_vals = np.array([r[-1][res_i] for r in results])
            #find the smallest two y_vals and return the corresponding x_vals
            LOGGER.debug(x_vals,y_vals)
            
            
            sorted_x_vals = x_vals[np.argsort(y_vals)] 
            for x_middle_i in range(0,len(x_vals)):
                
                
                greater = sorted_x_vals[sorted_x_vals > sorted_x_vals[x_middle_i]]    
                less = sorted_x_vals[sorted_x_vals < sorted_x_vals[x_middle_i]]    

                if len(greater) and len(less):
                    left = less[0]
                    right = greater[0]
                    min_x_vals = np.array([left,right])
                    break
            
            else:
                raise ValueError('unable to find bounds')
            
            
            return min_x_vals - E0 # substract to get bounds relative to E0
        
        new_bounds = bounds_determining(iterative_model_solver.saved_results,1)
        bracket = bracket_finder(1)
        LOGGER.debug(f'T_so bracket: {bracket}')
        iterative_model_solver.return_func = lambda x:x[0][1]
        
        # step 3: Determine Tud by etuning around mL = mR with symmetric detuning (pick spin_up left and spin down right)
        self.envelope_model.band_model.parameter_dict[mu_R] = E0

        LOGGER.info('finding T_so')       
        Tso_sol = self.find_avoided_crossing(solver,(left_up,right_down),(mu_R,),{mu_R:new_bounds},iterative_solving=iterative_model_solver,bracket=bracket)
        
        new_bounds = bounds_determining(iterative_model_solver.saved_results,2)
        iterative_model_solver.return_func = lambda x:x[0][2]
        bracket = bracket_finder(2)
        LOGGER.debug(f'D bracket: {bracket}')
        
        # step 5: determin Delta_ud by detuing around mL=E0 (fixed) and vary mR around mR=E0-2Eu (spin up particle left adn spin down anti-particle right)
        self.envelope_model.band_model.parameter_dict[mu_R] = E0
        
        LOGGER.info('finding D')       
        D_sol = self.find_avoided_crossing(solver,(left_up,right_h_down),(mu_R,),{mu_R:new_bounds},iterative_solving=iterative_model_solver,bracket=bracket)
        
        new_bounds = bounds_determining(iterative_model_solver.saved_results,3)
        iterative_model_solver.return_func = lambda x:x[0][3]
        bracket = bracket_finder(3)
        LOGGER.debug(f'D_so bracket: {bracket}')
        
        # step 4: Determine Delta_uu by detuning around mL=E0 (fixed) and vary mR around mR=E0-2Eu (spin up particle left and anti-particle right)
        self.envelope_model.band_model.parameter_dict[mu_R] = E0
        LOGGER.info('finding D_so')       
        Dso_sol = self.find_avoided_crossing(solver,(left_up,right_h_up),(mu_R,),{mu_R:new_bounds},iterative_solving=iterative_model_solver,bracket=bracket)
        
        
        return_tup =  T_sol, Tso_sol,D_sol,Dso_sol
        if verbose_result:
            return return_tup + (iterative_model_solver.saved_results,)
        return return_tup
        """
        #return



from nqcpfem.parameter_search import IterativeModelSolver
from typing import Any
class IntermediateResSave(IterativeModelSolver):
    def __init__(self, construction_func: Callable[..., Any], solver: ModelSolver, evaluation_func: Callable[..., Any] | None = None, start_from_prev=True,return_func=None):
        super().__init__(construction_func, solver, evaluation_func, start_from_prev)
        self.return_func =return_func
        self.saved_results = []
    def __call__(self, *args, **kwargs):
        res =super().__call__(*args, **kwargs)
        self.saved_results.append((args,kwargs,res))
        return self.return_func(res)
    


from collections import namedtuple
Result = namedtuple('Result',['x','f','xvals','fvals','derivs'])

class CrossingFinder():
    def __init__(self,func,x_range,deriv_step_factor=1e-3,deriv_tol=1,max_iter=18,x_tol=1e-3):

        self.func = func
        self.x_range = x_range
        self.deriv_step_factor = deriv_step_factor
        self.deriv_tol = deriv_tol # how small should the derivative be before we believe that ii
        self.max_iter = max_iter
        self.x_tol = x_tol
        self._fL_bound = None
        self._fR_bound = None
        self.derivs = []
        
    @property
    def deriv_step(self):
        return (self.x_range[1]-self.x_range[0])*self.deriv_step_factor
    
    
    @property
    def fL_bound(self):
        if self._fL_bound is None:
            self._fL_bound = self.func(self.x_range[0])
        return self._fL_bound
    
    @property
    def fR_bound(self):
        if self._fR_bound is None:
            self._fR_bound = self.func(self.x_range[1])
        return self._fR_bound
    
    def minimize(self,xL=None,fL=None,dL=None,xR=None,fR=None,dR=None,verbose_res=False):
        class func_wrap():
            def __init__(self,func):
                self.xs = []
                self.fs = []
                self.func = func
            def __call__(self,x):
                self.xs.append(x)
                y = self.func(x)
                self.fs.append(y)
                return y
        func = func_wrap(self.func)
        if xL is None or xR is None:
            xL,xR = self.x_range[0],self.x_range[1]
            fL,fR = self.fL_bound,self.fR_bound
            dL = (-1,-1)
            dR = (1,1)
        if dR is None:
            dR = (1,1)
        if dL is None:
            dL = (-1,-1)
        i  = 0
        

        
        returner = lambda x,f: Result(x,f,func.xs,func.fs,self.derivs) if verbose_res else (x,f) 
        
        maxima = []
        x = (xR+xL)/2
        while i<self.max_iter:
            i+=1
            LOGGER.debug((i,xR,xL))
            x_old = x
            x = (fL-fR +xL+xR)/2
            # if outside interval: reflect it into the interval
            
            if x<xL:
                LOGGER.debug(f'reflect around xL: {x,2*(xL-x)}')
                x = x + 2*(xL-x)
            if x>xR:
                LOGGER.debug(f'reflect around xR: {x,-2*(x-xR)}')
                x = x - 2*(x-xR)
            
            # if too close to the edge: move it further in
            if np.isclose(x,xL,atol=10*self.deriv_step):
                LOGGER.debug(f'shift right: {x,10*self.deriv_step}')
                x += 10*self.deriv_step
            if np.isclose(x,xR,atol=10*self.deriv_step):
                LOGGER.debug(f'shift left: {x,-10*self.deriv_step}')
                x += -10*self.deriv_step
            
            
            # if it is still too close, outside or interval is too small. We do the alternative point distinguising

            if x<xL or x>xR or any(np.isclose(x,[xL,xR],atol=10*self.deriv_step)) or np.abs(xR-xL)<2*self.x_tol*(self.x_range[1]-self.x_range[0]):
                # we get here if: linear intersection guess moves us further away, is too close to previous points, or search interval is already small
                # find the closest of xL and xR and move 10*self.deriv_step towards the center
                
                # be closest to the point with smallest derivative 
                left_deriv = np.abs(dL[1]) if np.abs(dL[1])<self.deriv_tol else np.abs(dL[0])
                right_deriv = np.abs(dR[0]) if np.abs(dR[0])<self.deriv_tol else np.abs(dR[1])
                delta = left_deriv-right_deriv
                sigma = left_deriv+right_deriv
                # delta = 0 -> middle. delta = sigma -> right point. delta = -sigma left_point

                x_new= (0.5+delta/(2*sigma))*xR + (0.5-delta/(2*sigma))*xL  # shift delta according to the smallest derivative.
                LOGGER.debug(f'alternative:{x,x_new}. reason:{x<xL ,x>xR,np.isclose(x,[xL,xR],atol=10*self.deriv_step),np.abs(xR-xL)<2*self.x_tol*(self.x_range[1]-self.x_range[0])} ')

                LOGGER.debug((left_deriv,right_deriv,delta,sigma,(0.5+delta/(2*sigma)),(0.5-delta/(2*sigma))))
                x = x_new
            prev_xs = np.array([m[0] for m in maxima] + [m[0] for m in self.derivs])
            if any(np.isclose(x,prev_xs,atol=self.deriv_step)) and not np.abs(xR-xL)<2*self.x_tol*(self.x_range[1]-self.x_range[0]):
                # if point is too close to an already evalated point (except for case where left and right are super close)
                point = prev_xs[np.isclose(x,prev_xs,atol=self.deriv_step)]
                LOGGER.debug(f'already seen: {point}. shifting the point')
                if not isinstance(point,float):
                    point = point[0]
                if x-xL < xR-x:
                    x += 2*self.deriv_step
                else:
                    x += -2*self.deriv_step

            
            
            LOGGER.debug(f'relative {(x-x_old)/(self.x_range[1]-self.x_range[0])}')
            deriv=self.derivative_check(x,return_derivs=True)
            LOGGER.debug(f'deriv: {(i,deriv,x)}')

            self.derivs.append((x,)+deriv[1:])
            direc = deriv[0]
            fx = deriv[1]
            if direc is None:
                maxima.append((x,deriv[1:]))
                # pick a point between the maxima and the furthest of the points xL, xR
                if (x-xL) < (xR-x) :
                    xL = x
                    fL = fx
                else:
                    xR = x
                    fR = fx
                
            elif (xR-xL)<self.x_tol*(self.x_range[1]-self.x_range[0]):
                LOGGER.debug(f'converged in {i} steps. Reason: x is within ({xL},{xR})')
                break
            elif direc == 0:
                LOGGER.debug(f'converged in {i} steps. Reason: valid local minimum.')

                break
            else:
                #look at all derivs and pick the next guess
                left,right = self.find_starting_points(self.derivs)
                LOGGER.debug((left,right))

                xL,fL,dL = left[:3]
                xR,fR,dR = right[:3]
                
        if i<self.max_iter:
            return returner(x,fx)
        else:
            LOGGER.debug('did not converge')

            return  returner(x,fx)

    def derivative_check(self,x0,f0=None,dfs=None,return_derivs=False):
        # check left first and if it is within tol, check right as well
        if dfs is None or f0 is None:
            f0 = self.func(x0)
            fl = self.func(x0-self.deriv_step)
            fr = self.func(x0+self.deriv_step)
            
            dfl = (f0-fl)/self.deriv_step 
            dfr = (fr-f0)/self.deriv_step
        else:
            dfl,dfr = dfs
        returner = lambda x: (x,f0,(dfl,dfr)) if return_derivs else (x,f0)
        if np.sign(dfl) == np.sign(dfr):
            return returner(-1*np.sign(dfl)) # go in the direction of of downhill slope
        else: # np.sign(dfl)<=0 and np.sign(dfr)>=0:
            if np.abs(dfl)>self.deriv_tol and np.abs(dfr)>self.deriv_tol:
                # go in direction with lowest value:
                return returner(None) # invalid local extremum? 
                return returner(1 if np.abs(dfl) < dfr else -1)
            elif np.abs(dfl)>self.deriv_tol and dfr<self.deriv_tol:
                return returner(-1*np.sign(dfr))# go in the direction specified by the one below self.deriv_tol
            elif np.abs(dfl)<self.deriv_tol and dfr>self.deriv_tol:    
                return returner(-1*np.sign(dfl))# go in the direction specified by the one below self.deriv_tol
            elif dfl>-self.deriv_tol and dfr<self.deriv_tol:
                return returner(0) # we converged
            else:
                LOGGER.debug('local maxima')
                return returner(None) # pick any i guess

    def find_starting_points(self,derivative_points,return_minimum=True):
        """ based on the derivatives at the points, estimate the best starting x-points. 
        These are chosen to be the two points that are closest to eachother,while still having the opposite sign of the derivatives"""
        left_points = []
        right_points = []
        for point in derivative_points:
            grad = self.derivative_check(*point)[0]
            LOGGER.debug((point,grad))

            if grad is None:
                continue # skip local maxima
            if grad > 0: # go to right i.e. this point is to the left:
                left_points.append(point)
            elif grad < 0:
                right_points.append(point)
            elif return_minimum:
                LOGGER.debug('minimum_found')

                return Result(point[0],point[1],[],[],self.derivs)
        
        
        def outermost_point_I(points,dir):
            xs =  np.array([p[0] for p in points])
            func = np.argmax if dir=='left' else np.argmin
            return points[func(xs)]
        
        
        l_point = outermost_point_I(left_points,'left') if len(left_points) else (self.x_range[0],self.fL_bound,(-1,-1))
        r_point = outermost_point_I(right_points,'right') if len(right_points) else (self.x_range[1],self.fR_bound,(1,1))
        
        if l_point > r_point:
            raise Exception('left was to the right of right...')
        return l_point,r_point

    
    @classmethod
    def warm_start(cls,old_points,new_func,x_vals,new_values,deriv_step_factor=None,**finder_kwargs):
        """create a new crossing finder and a set of points which can be passed to find_starting_points of the crossing finder for a warm start"""
        
        # assumes x_range is in x_vals as min and max
        xLi = np.argmin(x_vals)
        xRi = np.argmax(x_vals)
        
        if deriv_step_factor is None:   
            new = CrossingFinder(new_func,(x_vals[xLi][0],x_vals[xRi][0]),**finder_kwargs)
        else:
            new = CrossingFinder(new_func,(x_vals[xLi][0],x_vals[xRi][0]),deriv_step_factor = deriv_step_factor,**finder_kwargs)
        # set bounds
        LOGGER.debug(new.x_range)
        LOGGER.debug(new.deriv_step_factor)
        LOGGER.debug(new.deriv_step)
        
        new._fL_bound = new_values[xLi]
        new._fR_bound = new_values[xRi]

        
        new_values = np.asarray(new_values)
        x_vals = np.asarray(x_vals)
        
        # create new point list:
        new_points = []
        for p in old_points:
            x = p[0]
            f = new_values[np.where(x_vals==x)[0]][0]
            fl = new_values[np.where(x_vals==x-new.deriv_step)[0]][0]
            fr = new_values[np.where(x_vals==x+new.deriv_step)[0]][0]
            dfl = (f-fl)/new.deriv_step
            dfr = (fr-f)/new.deriv_step

            LOGGER.debug((x,f,(dfl,dfr)))

            new_points.append((x,f,(dfl,dfr)))
        new.derivs = new_points
        return new,new_points



class GoldenCrossingFinder():
    
    
    def __init__(self,func,x_range,max_iter=18,x_tol=1e-3):
        self.func = func
        self.x_range = x_range
        self.max_iter = max_iter
        self.x_tol = x_tol
        self.y_scale = 1

    def minimize(self,xL,fL,xM,fM,xR,fR):
        from scipy.optimize._optimize import Brent

        self.y_scale=np.average([fL,fR,fM])
        
        
        
        xL = self.x_to_scipy(xL)
        xR = self.x_to_scipy(xR)
        xM = self.x_to_scipy(xM)
        fL = fL/self.y_scale
        fR = fR/self.y_scale
        fM = fM/self.y_scale
        
        brent = Brent(self.wrap_func,maxiter=self.max_iter,full_output=True,disp=3,tol=self.x_tol)
        
        def override_bracket_constructor(*args):
            return (xL,xM,xR,fL,fM,fR,0) # 0 is func_calls
        
        brent.get_bracket_info = override_bracket_constructor
        print(brent.get_bracket_info())
        
        brent.optimize()
        
        res = Result(self.scipy_to_x(brent.xmin),self.y_scale*brent.fval,[],[],list(range(brent.funcalls)))
        return res

    
    
    @staticmethod
    def determine_bracket(x_vals,f_vals):
        x_vals = np.asarray(x_vals)
        f_vals = np.asarray(f_vals)
        
        
        if len(x_vals)<3:
            raise ValueError('we need at least 3 points to determine a bracket')
            
            
        #find the smallest two y_vals and return the corresponding x_vals
        
        sorted_x_vals = x_vals[np.argsort(f_vals)] 
        sorted_f_vals = np.sort(f_vals)
        for x_middle_i in range(0,len(x_vals)):
            greater = np.where(sorted_x_vals > sorted_x_vals[x_middle_i])[0]    
            less = np.where(sorted_x_vals < sorted_x_vals[x_middle_i])[0]
            if len(greater) and len(less):
                left = (float(sorted_x_vals[less[0]]),sorted_f_vals[less[0]])
                middle = (float(sorted_x_vals[x_middle_i]),sorted_f_vals[x_middle_i])
                right = (float(sorted_x_vals[greater[0]]),sorted_f_vals[greater[0]])
                break
        
        else:
            raise ValueError('unable to find bounds for points: {x_vals}, {f_vals}')
        
        return left+middle+right
    
    
    
    def x_to_scipy(self,x):
        x_scale = (self.x_range[1]-self.x_range[0])/1000

        # shift so that center of range is at zero
        x_shift = np.average(self.x_range)
        
        return (x-x_shift)/x_scale
    
    def scipy_to_x(self,sc):
        
        x_scale = (self.x_range[1]-self.x_range[0])/1000

        # shift so that center of range is at zero
        x_shift = np.average(self.x_range)

        return (sc*x_scale)+x_shift
            
    def wrap_func(self,x):
        
        # because scipy does not hanl ver sall numbes well we have to scale it. The scale is so tha the x_range is divided into 100 pieces
        scaled_x= self.scipy_to_x(x)
        res = self.func(scaled_x)
        print(scaled_x,res)
        return res/self.y_scale
    
    


class BracketMinimizer():
    def __init__(self,xa,xb,xc,f,target_states,xtol=1e-3,max_iter=100):
        #xa<xb<xc and f(xb)<f(xa), f(xb)<f(xc)
        
        if xb>xc or xa>xb:
            raise ValueError(f'invalid brakcet {a,b,c}')
        self.a = RunInfo(xa,None,None,None)
        self.b = RunInfo(xb,None,None,None)
        self.c = RunInfo(xc,None,None,None)
        self.xtol=xtol
        self.target_states = target_states
        self.f = f
        self.max_iter=max_iter
        self.hist = []

    
    
    def pair_ordering(self,states):
        # states indexed by first index!
        projection_fidelity = []
        indices = []
        #generate the list
        # project all the states down to the subspace
        proj = np.abs(np.einsum('aj,bj->ab',self.target_states.conj(),states))**2# should be 2xN
        # loop over collumn for all pairs and compute their projection fidelity
        for i in range(states.shape[0]):
            for j in range(i+1,states.shape[0]):
                projection_fidelity.append(np.sum(proj[:,i])+np.sum(proj[:,j]))
                indices.append((i,j))
                
        return [indices[i] for i in np.argsort(projection_fidelity)[::-1]] # return pairs sorted accoridng to projection fidelity
    
    
    def find_selection(self,x,res,left_point,right_point):
        left_delta = x-left_point.x
        left_diff = np.abs(left_point.Es[left_point.sel[0]]- left_point.Es[left_point.sel[1]])

        right_delta = right_point.x-x
        right_diff = np.abs(right_point.Es[right_point.sel[0]]- right_point.Es[right_point.sel[1]])
        
        
        
        # pick the first pair of states that give valid differences
        E = res[0]
        psi = res[1].T
        selection= None
        state_ordering = self.pair_ordering(psi)
        for (i,j) in state_ordering:
            diff = np.abs(E[i]-E[j])
            if (diff-left_diff > left_delta) or  (diff-right_diff>right_delta):
                continue
            
            selection = (i,j)
            break # 
        if selection is None:
            #raise ValueError('did not find valid pair')
            print('did not find valid pair')
            selection = state_ordering[0]
        result = RunInfo(x,E,psi,selection) 
        return result
    def update_iter(self,x,res):

        
        #determine left and right bounds from xa,xb,xc
        b_dist = x-self.b.x
        if b_dist>0: # this point to the right of b
            left_point = self.b
            right_point = self.c
        else:
            left_point = self.a
            right_point = self.b
        
        
        result = self.find_selection(x,res,left_point,right_point)
        
        right_diff = np.abs(right_point.Es[right_point.sel[0]]- right_point.Es[right_point.sel[1]])
        left_diff = np.abs(left_point.Es[left_point.sel[0]]- left_point.Es[left_point.sel[1]])
        res_diff = np.abs(result.Es[result.sel[0]]- result.Es[result.sel[1]])
        if b_dist>0: # ordering is abrc and left is b and right is c
            if res_diff > left_diff:
                # new bracket is abr
                self.c = result
            else:
                #new bracket is brc
                self.a = self.b
                self.b = result
        else: # ordering is arbc and left is a and right is c
            if res_diff > right_diff:
                # new bracket is rbc
                self.a = result
            else:
                #new bracket is arb
                self.c = self.b
                self.b = result
        
        
        EE = np.sort([result.Es[result.sel[0]],result.Es[result.sel[1]]])
        self.hist.append((x,EE))
        
        return result
    
    
    def __call__(self,x):
        res = self.f(x)
        r=self.update_iter(x,res)
        diff = np.abs(r.Es[r.sel[0]]-r.Es[r.sel[1]]) 
        LOGGER.debug(x,diff)
        return diff
    def run(self):
        
        
        if self.a.Es is None:
            res = self.f(self.a.x)
            # this must just match the target states as good as possible
            state_ordering = self.pair_ordering(res[1].T)
            self.a = RunInfo(self.a.x,res[0],res[1].T,state_ordering[0])
        if self.c.Es is None:
            res = self.f(self.c.x)
            # this must just match the target states as good as possible
            state_ordering = self.pair_ordering(res[1].T)
            self.c = RunInfo(self.c.x,res[0],res[1].T,state_ordering[0])
        
        if self.b.Es is None:
            res = self.f(self.b.x)
            self.b = self.find_selection(self.b.x,res,self.a,self.c)
        
        for r in (self.a,self.c,self.b):
            EE = np.sort([r.Es[r.sel[0]],r.Es[r.sel[1]]])
            self.hist.append((r.x,EE))
        
        from scipy.optimize._optimize import Brent

        
        brent = Brent(self,maxiter=self.max_iter,full_output=True,disp=3,tol=self.xtol)
        
        
        f_vals = [np.abs(r.Es[r.sel[0]]-r.Es[r.sel[1]]) for r in (self.a,self.b,self.c)]
        
        def override_bracket_constructor(*args):
            return (self.a.x,self.b.x,self.c.x,*f_vals,0) # 0 is func_calls
        
        brent.get_bracket_info = override_bracket_constructor
        
        brent.optimize()
        res = (brent.xmin,brent.fval) 
        return res
    
    

class SignedSearch():
    def __init__(self,xL,xR,evalfunc,xtol,state_sets,overlap_func,method='brentq'):
        """
        Find the Coupling coefficient between state sets efficiently. The underlying idea is to view the voided crossing between states as not a minum of the 
        energy splitting of the states, but as a discrete transition in the signed energy difference (E1(x)-E2(x)). At xmin-delta E1 > E2 but at xmin+delta E1 <E2
        This insight allows us to find the crossing by finding where the signed energy crossing changes sign. 
        """
        self.xL =xL
        self.xR =xR
        self.f = evalfunc
        self.overlap_func=overlap_func # func that given output of evalfunc (N vectors) and state_sets (L vectors) gives and LxN array specifying the overlap
        self.xtol = xtol
        self.target_states=state_sets
        self.__current_i__=None
        self.runs = [{} for _ in range(len(self.target_states))] # faster lookup using the x value
        self.calls=0
        self.__reused_calls__ = 0
        self.method = method
        self.initial_brackets = [None]*len(self.target_states)

    def energy_diff(self,overlaps,evals):
        #pick the two states with highest overlapwith highest overlap
        fidelity=np.sum(overlaps,axis=0)
        sort_I=np.argsort(fidelity)
        selection = sort_I[-2:]
        #purity is based the fidelity of the discarded states
        sorting = np.argsort([overlaps[0,s] for s in selection]) # sorted from low to high
        total_fidelity = np.sum(fidelity)
        return evals[selection[sorting[1]]]-evals[selection[sorting[0]]],1-np.sum(fidelity[sort_I[:-2]])/total_fidelity
    
    def make_overlap_mat(self,selected_state,other_state,evecs):
        overlaps =np.abs(np.einsum('ki,ji->kj',np.stack([selected_state,other_state]).conj(),evecs))**2
        return overlaps    
    def signed_diff(self,x):
        if x in self.runs[self.__current_i__]:
            #reuse previous calls if they are available
            self.__reused_calls__ +=1
            return self.runs[self.__current_i__][x][1]
        evals,evecs=self.f(x)
        #determine the two relevant states
        results = []
        log_info = []
        for i,target_states in enumerate(self.target_states):
            overlaps = self.overlap_func(evecs,target_states)
            diff,fidelity=self.energy_diff(overlaps,evals)
            log_info.append((diff,fidelity))
            self.runs[i][x]=((x,diff,fidelity))
            results.append(diff)

        if LOGGER.level <= logging.DEBUG:
            log_string = f'Run for {self.__current_i__},x={x}:'
            for i,log_entry in enumerate(log_info):
                log_string += f'\nPair {i}: Ediff: {log_entry[0]}. F={log_entry[1]}.'
            LOGGER.debug(log_string)
        return results[self.__current_i__]
    
    
    @classmethod
    def inspire(cls,old,res_x,percentile=0.5):
        new = cls(old.xL,old.xR,old.f,old.xtol,old.target_states,old.overlap_func,old.method)

        
        spacing = percentile*(new.xR-new.xL)
        brackets = []
        for i,R in enumerate(old.runs):
            # assume last element is the final guess.
            center=res_x[i]
            #brackets.append((center-spacing,center+spacing))

            N = len(R)
            included = int(np.ceil(N*percentile))
            X_pos = np.array([r[0] for r in R])
            neighbors= np.argsort(np.abs(X_pos-center))[:included]
            # find the two neighbors that are left and rightmost
            neighbor_pos = X_pos[neighbors]
            brackets.append((np.min(neighbor_pos),np.max(neighbor_pos)))
        
        new.initial_brackets=brackets
        
        return new
            
        
    
    def find_bracket(self,i,min_fidelity=0.8):
        X = np.array([x for x,v in self.runs[i].items() if v[2]>min_fidelity])
        V = np.array([v[1] for v in self.runs[i].values() if v[2]>min_fidelity])
        if len(X)<2:
            return (self.xL,self.xR) # fallback to default bracket
        L_sign = np.sign(V[np.argmin(X)])
        R_sign = np.sign(V[np.argmax(X)])
        
        # pick largest x_value with same sign as L_sign:
        X_L= X[np.sign(V)==L_sign]
        V_L= V[np.sign(V)==L_sign]
        X_R= X[np.sign(V)==R_sign]
        V_R= V[np.sign(V)==R_sign]
        if not(len(X_L)) or not(len(X_L)):
            return (self.xL,self.xR) # fallback to default bracket
        LOGGER.debug((V_L[np.argmax(X_L)],np.max(X_L),V_R[np.argmin(X_R)],np.min(X_R),i))
        
        new_bracket=(np.max(X_L),np.min(X_R))
        if new_bracket[0]>new_bracket[1]:
            LOGGER.debug('fallback to old bracket')
            return (self.xL,self.xR) # fallback to default bracket
        return new_bracket
        
    def minimize(self):
        results = []
        bracket =None
        for i in range(len(self.target_states)):
            self.__current_i__ = i
            if self.initial_brackets[i] is None:
                if bracket is None:
                    bracket = (self.xL,self.xR)
                else:
                    bracket=self.find_bracket(i) 
                    if bracket is None: bracket= (self.xL,self.xR) #default if we do not find a bracket
                    #bracket = (self.xL,self.xR)
            else:
                bracket = self.initial_brackets[i] # allows specifying bracket for each run manually
            setup_kwargs = {'method':self.method}
            if self.method == 'halley':
                x0 = 0.5*(bracket[1]+bracket[0] )
                setup_kwargs['x0'] = x0
                setup_kwargs['fprime'] = False
                setup_kwargs['fprime2'] = False
            elif self.method in ('secant','newton'):   
                x0 = 0.5*(bracket[1]+bracket[0] )
                setup_kwargs['x0'] = x0
            from scipy.optimize import root_scalar
            
            try:
                res = root_scalar(self.signed_diff,xtol=self.xtol,bracket=bracket,**setup_kwargs)
            except Exception as err:
                LOGGER.debug('ERROR: '+ str(err))
                LOGGER.debug(((err,bracket,self.runs)))
                if bracket == (self.xL,self.xR):
                    #determine energy values of the left and the right parts of the bracket for easier printing
                    run_brackets = self.runs[self.__current_i__]
                    left_e = [run_brackets[self.xL]]
                    right_e = [run_brackets[self.xR]]
                    raise Exception(f'big bracket failed for coupling with index {self.__current_i__}. Bracket values was:\n {left_e}\n{right_e}.') from err
                #fallback to default bracket
                res = root_scalar(self.signed_diff,xtol=self.xtol,bracket=(self.xL,self.xR),**setup_kwargs)

            x = res.root
            res['val'] = np.abs(self.signed_diff(x))
            results.append(res)
            LOGGER.debug(bracket,res)
        calls=sum(r.function_calls for r in results)-self.__reused_calls__
        LOGGER.info(f'all couplings determined using {calls} function calls. ({self.__reused_calls__} reused)')
        self.calls=calls
        return results 

    
def crossing_preprocessing(E0,Vsweep,Pup,Pdown,Aup,Adown,solution_func,system):
    init_run= solution_func(E0)
    
    # determine Ec and Emax
    
    Emax = np.max(init_run[0]) # get maximum detuning reltive to E0
    X_points= system.envelope_model.positional_rep(init_run[1][0])[1]
    Vsweep_i = system.select_subspace((Vsweep,),init_run[1],1,x_points=X_points)[0]
    
    Ec = E0-init_run[0][Vsweep_i]
    
    ranges = []
    for vec in (Pup,Pdown,Adown,Aup):
        vec_i = system.select_subspace((vec,),init_run[1],1,x_points=X_points)[0]
        Esign = np.sign(init_run[0][vec_i])
        if Esign > 0 :
            # we have to increase energy in order to get crossing
            ranges.append((Ec,Ec+Emax*0.99))
        elif Esign < 0:
            # we have to decrease 
            ranges.append((Ec-Emax*0.99,Ec))

    LOGGER.debug(ranges)
    return ranges 

    
            
            

def make_B_eigenstates(Bx,By,Bz):           
    """
    Construct the spin eigenstates of the Zeeman Hamiltonian with the specific B. Returns a 2x2 unitary for transforming into the spin eigenstates
    """
    norm_B = np.linalg.norm([Bx,By,Bz])

    
    alpha_0 = np.sqrt( (Bx**2+By**2)/(2*norm_B*(norm_B+Bz)))
    alpha_1 = np.sqrt( (Bx**2+By**2)/(2*norm_B*(norm_B-Bz)))
    up_0 = 1/(alpha_0*(Bx+1j*By)) * (Bz+norm_B)
    up_1 = 1/alpha_0
    
    down_0 = 1/(alpha_1*(Bx+1j*By)) * (Bz-norm_B)
    down_1 = 1/alpha_1
    U=np.array([[up_0,down_0],[up_1,down_1]])
    return U/np.linalg.norm(U,axis=0)[np.newaxis,:]
        