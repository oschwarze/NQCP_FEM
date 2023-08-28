from nqcpfem.solvers import ModelSolver
from . import System,StateClass
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
        
    
    
    def determine_all_couplings(self,spin_up_I,spin_down_I,solver,E0,parameter_range):
        pass
        #make the Classes
        from . import PositionalState,DefiniteTensorComponent
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
        
        
        mu_L,mu_R = sympy.symbols('\mu_{L},\mu_{R}')
        detuning = SymbolicFunction(sympy.Piecewise((-mu_L,self.domains['ld_in']),(-mu_R,self.domains['rd_in']),(0,True)),'\mu_{detuning}(x)')
        
        self.envelope_model.band_model.add_potential(detuning)
        self.envelope_model.band_model.parameter_dict[mu_L] = E0
        self.envelope_model.band_model.parameter_dict[mu_R] = E0
        
<<<<<<< HEAD
        left,right,sc = self.__make_system_classes__()
=======
        
>>>>>>> 5f5529c2432c8138eb4e469ba42585498fac03f2
        
        
        # step 2: Determine Tuu by detuning around mL = mR with s(pick spin up lef and spin up right stateClasses)
        left_up = left.combine_state_cls(up_p_state,'left up particle')
        right_up = right.combine_state_cls(up_p_state,'right up particle')
        
        right_down = right.combine_state_cls(down_p_state,'right down particle')
        
        right_h_down = right.combine_state_cls(down_h_state,'right down anit-particle')
        right_h_up = right.combine_state_cls(up_h_state,'right up anit-particle')
        
        
        def model_update(params):
            self.envelope_model.band_model.parameter_dict.update(params)
            return self.envelope_model
        
        
        def evaluation_func(model,res):
            #determine all couplings here:
            X_arr = self.envelope_model.positional_rep(res[1][0])[1]
            vals = []
            subspaces = ((left_up,right_up),(left_up,right_down),(left_up,right_h_down),(left_up,right_h_up)) # T, Tso,D,Dso
            for subsp in subspaces:

                subspace_I = self.select_subspace(subsp,res[1],2,x_points=X_arr)
                vals.append(np.abs(res[0][subspace_I[0]]-res[0][subspace_I[1]]))
            
            return vals
            
        iterative_model_solver = CrossingsFinder(model_update,solver,evaluation_func,return_func=lambda x:x[0])
        
            
        T_sol = self.find_avoided_crossing(solver,(left_up,right_up),(mu_R,),{mu_R:parameter_range},iterative_solving=iterative_model_solver)
        
        
        #use the previous results to determine the bounds for the next one
        def bounds_determining(results,res_i):
            if len(results)<3:
                return parameter_range #we do not have enough info to go with
                
            x_vals = np.array([r[0][0][mu_R] for r in results])
            y_vals = np.array([r[-1][res_i] for r in results])
            #find the smallest two y_vals and return the corresponding x_vals
            print(x_vals,y_vals)
            
            
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
        print(new_bounds)
        iterative_model_solver.return_func = lambda x:x[1]
        
        # step 3: Determine Tud by etuning around mL = mR with symmetric detuning (pick spin_up left and spin down right)
        self.envelope_model.band_model.parameter_dict[mu_R] = E0
        Tso_sol = self.find_avoided_crossing(solver,(left_up,right_down),(mu_R,),{mu_R:new_bounds},iterative_solving=iterative_model_solver)
        
        new_bounds = bounds_determining(iterative_model_solver.saved_results,2)
        iterative_model_solver.return_func = lambda x:x[2]
        
        # step 5: determin Delta_ud by detuing around mL=E0 (fixed) and vary mR around mR=E0-2Eu (spin up particle left adn spin down anti-particle right)
        self.envelope_model.band_model.parameter_dict[mu_R] = E0
        D_sol = self.find_avoided_crossing(solver,(left_up,right_h_down),(mu_R,),{mu_R:new_bounds},iterative_solving=iterative_model_solver)
        
        new_bounds = bounds_determining(iterative_model_solver.saved_results,3)
        iterative_model_solver.return_func = lambda x:x[3]
        
        # step 4: Determine Delta_uu by detuning around mL=E0 (fixed) and vary mR around mR=E0-2Eu (spin up particle left and anti-particle right)
        self.envelope_model.band_model.parameter_dict[mu_R] = E0
        Dso_sol = self.find_avoided_crossing(solver,(left_up,right_h_up),(mu_R,),{mu_R:new_bounds},iterative_solving=iterative_model_solver)
        
        
        
        return T_sol, Tso_sol,D_sol,Dso_sol
        #return



from nqcpfem.parameter_search import IterativeModelSolver
from typing import Any
class CrossingsFinder(IterativeModelSolver):
    def __init__(self, construction_func: Callable[..., Any], solver: ModelSolver, evaluation_func: Callable[..., Any] | None = None, start_from_prev=True,return_func=None):
        super().__init__(construction_func, solver, evaluation_func, start_from_prev)
        self.return_func =return_func
        self.saved_results = []
    def __call__(self, *args, **kwargs):
        res =super().__call__(*args, **kwargs)
        self.saved_results.append((args,kwargs,res))
        return self.return_func(res)
    

# region state Classes
#endregion