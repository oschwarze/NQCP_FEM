from abc import ABC,abstractmethod
import numpy as np 
import sympy
from ..envelope_function import EnvelopeFunctionModel
from ..observables import inner_product
from typing import Iterable


"""The purpose of this module is to streamline the construction of specific device architectures and studying the energy eigenstates within them. A System is just a specific envelope model, along with methods for characterizing/classifying the eigenstates of the system. 
    """


class StateClass(ABC):
    """
    State Classes are just an array of bools specifying which coordinates are relevant. This is fast, but not very general. 
    Use with caution
    
    """
    pass
    def __init__(self,name):
        self.name = name

    @abstractmethod
    def project(self,state):
        # projects the passes state down to the subspace 
        pass

    def __hash__(self) -> int:
        return hash(self.name)
    
    def __repr__(self) -> str:
        return f'StateClass: {self.name}'
    
from functools import reduce
class BoolState(StateClass):
    def __init__(self,name,projector):
        self.projector = projector
        super().__init__(name=name)
    
    def project(self,vector,**kwargs):
        return self.projector(vector,**kwargs)
    
    
    def combine_state_cls(self,other_state_cls,new_name = None):
        def chain(*funcs):
            def chained_call(arg,**kwargs):
                return reduce(lambda r, f: f(r,**kwargs), funcs, arg)
            return chained_call
        new_proj = chain(self.projector, other_state_cls.projector)
        
        if new_name is None:
            new_name = self.name + ' AND ' + other_state_cls.name
        return BoolState(new_name,new_proj)


class DefiniteTensorComponent(BoolState):
    # the HH_components of the tensor are one, the rest is zero.
    def __init__(self,tensor_index,name):
        
        def spinor_projection(vector,**kwargs):
            new = np.zeros_like(vector)
            new[np.where(tensor_index)] = vector[np.where(tensor_index)]
            return new
        
        return super().__init__(name,spinor_projection)
    

class SpinorProjection(BoolState):
    def __init__(self,spinor_state,name,axis=0):
        self.spinor_state= spinor_state
        self.axis=axis

        return super().__init__(name,self.spinor_basis_projection)
    
    def spinor_basis_projection(self,state,**kwargs):
        state_index = tuple(range(len(state.shape)))
        sum_index = state_index[self.axis:self.axis+len(self.spinor_state.shape)]
        if state.shape[self.axis:self.axis+len(self.spinor_state.shape)] != self.spinor_state.shape:
            raise ValueError(f'shape mismatch for spinor projection and state {self.spinor_state.shape} vs {state.shape[self.axis:self.axis+len(self.spinor_state.shape)]}')
        return np.einsum(self.spinor_state.conj(),sum_index,state,state_index)


class PositionalState(BoolState):
    def __init__(self,positional_condition,name,x_points=None):
        # convert the positional_condition to 
        X,Y,Z = sympy.symbols('x,y,z')
        self.x_mask_func = sympy.lambdify((X,Y,Z),positional_condition)
        
        
        super().__init__(name,projector=self.positional_mask)
        
    def __make_x_mask__(self, x_points=None):
        
        if x_points is None:
            raise ValueError('no x points were passed')
        
        arr =np.array(self.x_mask_func(x_points[:,0],x_points[:,1],x_points[:,2] )).astype(bool)
        return arr if any(arr) else None
    
    
    def positional_mask(self,vector,x_points=None,**kwargs): 
        mask = self.__make_x_mask__(x_points)
        if mask is None:
            return np.zeros_like(vector)
        return np.einsum('...x,x->...x',vector,mask)



class System(ABC):
    def __init__(self,envelope_model:EnvelopeFunctionModel,state_classes:tuple[StateClass]):
        self.__envelope_model__ = envelope_model
        self.state_classes = state_classes

    def classify_states(self,eigenstates):
        """
        given a set of eigenstates, groups them into sets according to the state_class of the system that they belong to. The return is a tuple of tuples where each element of the tuple is an eigenstate, the ordering of the tuples is the same as the orderrign of self.state_classes.
        Possible methods for classification are:
        """
        
        X_array = self.envelope_model.positional_rep(eigenstates[0])[0]
        
        classification = {c:[] for c in self.state_classes}
        
        for c in self.state_classes:
            if isinstance(c,PositionalState):
                c.x_points = X_array
        
        for psi in eigenstates:
            projections = [np.abs(inner_product(psi,c.project(psi)))
                           for c in self.state_classes]
            
            classification[self.state_classes[np.argmax(projections)]].append(psi)
            
        return classification
    
    
    
    def subspace_weights(self,subspace_classes,eigenstates,**proj_kwargs):
        """
        Given a subspace compute the proection magnitue of the eigenstates.
        Returns an array 
        
        """              
        weights = []
        for psi in eigenstates:
            
            projections = np.array([np.linalg.norm(c.project(psi,**proj_kwargs))
            for c in subspace_classes])
            weights.append((projections/np.linalg.norm(psi))**2)   
        return np.stack(weights)
    def select_subspace(self,subspace_classes,eigenstates,subspace_dim,**proj_kwargs):
        """ Given a set of Stateclasses defining a certain subspace of the eigenstates of the system, 
        determine the `subspace_dim` number of eigenstates in the supplied `eigenstates` that lie the most in this subspace. 
        """
        # determiene wei
        
        weights = self.subspace_weights(subspace_classes,eigenstates,**proj_kwargs)
        
        sort_I = np.argsort(np.linalg.norm(weights,axis=1))
        
        
        return sort_I[-subspace_dim:]
    
    
    @property
    def envelope_model(self):
        return self.__envelope_model__
    
    
    

    
    def find_avoided_crossing(self,solver,subspace_classes,parameters,parameter_bounds=None,return_minimization_instance=False,iterative_solving=True,energy_method=False,**minimization_kwargs):
        """ Find an avoided crossing of energy levels in the system, by varying the prameters and minimizing the energy difference. The initial guess for the minimization is the parameter_dict of the model
        :param solver: the solver to use to determine the eigenstates of the system
        :param subspace_classes: tuple of StateClasses which define the 2D subspace of interest
        :param parameters: tuple of symbols or a single symbol which define the parameters that can be altered in order to minimize the energy difference between the levels.
        :param parameter_bounds: tuple of tuple or single tuple describing the allowed ranges of the parameters.
        :param minimization_kwargs: Kwargs that are passed to `scipy.optimize.minimize`.
        """
        
        prev_energies = [None,None] # stores the (sorted) energies of the two states in focus between runs. If energy_method is true, the next states are chosen as to minimize the difference in energies between runs
        
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
                
                
            
            
            
        
        
        def model_update(parameters):
            self.envelope_model.band_model.parameter_dict.update(parameters)
            return self.envelope_model
        
        def res_post_processing(model,res):
    
            X_arr = self.envelope_model.positional_rep(res[1][0])[1]
            if energy_method and prev_energies[0] is not None:
                subspace_I = energy_continuity(res[0])
            else:    
                subspace_I = self.select_subspace(subspace_classes,res[1],2,x_points=X_arr)
            prev_energies = (res[0][subspace_I[0]],res[0][subspace_I[0]])
            
            return np.abs(res[0][subspace_I[0]]-res[0][subspace_I[1]])
        
        from ..parameter_search import IterativeModelSolver
        if isinstance(iterative_solving,IterativeModelSolver):
            energy_diff = iterative_solving
        
        elif iterative_solving:
            energy_diff = IterativeModelSolver(model_update,solver,res_post_processing)
        else:
            def energy_diff(parameters):
                # update values solve model, determine energy difference 
                self.envelope_model.band_model.parameter_dict.update(parameters)
                
                solution = solver.solve(self.envelope_model)
                X_arr = self.envelope_model.positional_rep(solution[1][0])[1]
                if energy_method and prev_energies[0] is not None:
                    subspace_I = energy_continuity(solution[0])
                else:    
                    subspace_I = self.select_subspace(subspace_classes,solution[1],2,x_points=X_arr)
                
                prev_energies = (solution[0][subspace_I[0]],solution[0][subspace_I[0]])
                return np.abs(solution[0][subspace_I[0]]-solution[0][subspace_I[1]])

        
        default_vals = {p:self.envelope_model.band_model.parameter_dict[p] for p in parameters} 
        

        if parameter_bounds is None:
            parameter_bounds = {k:(-0.25*d,0.25*d) for k,d in default_vals.items()}
            
        # create Minimization and run it
        from ..parameter_search import MinimizationSearch
        minimization = MinimizationSearch(energy_diff,default_vals,parameter_bounds,**minimization_kwargs)
        
        min=minimization.find_minimum()
        if return_minimization_instance:
            return minimization
        return min
        




# Class which can be passed to scipy's minimize and acts like a function in order to determine the crossing
