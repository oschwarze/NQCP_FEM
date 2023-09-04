import pandas as pd
import pandas as pd
from sklearn.model_selection import ParameterGrid
from .solvers import ModelSolver
import logging
import numpy as np
from scipy.optimize import minimize,minimize_scalar
import pickle as pkl
import os 


from . import UNIT_CONVENTION
E0 = 1/(UNIT_CONVENTION['J to eV']*100000) # unit_scale: 10muev
from typing import Iterable
LOGGER = logging.getLogger(__name__)
LOGGER.addHandler(logging.NullHandler())


class NotLoadedError(Exception):
    pass

#TODO: figure out whch type id indexable
class ParameterSearch():
    def __init__(self,parameter_sets:Iterable,evaluation_function,save_file:str|None=None):
        """
        Class which runs a for-loop over the specified parameters, uses them to evaluate a function, and stores the return value of
        the function. The ParameterSearch can save the result after each iteration which allows interuption of the loop without having to start all over again.
        :param parameter_sets: tuple containing args (tuple) and kwargs (dict). If the `parameter_sets` is a dict it is interpreted as kwargs only and if `parameter_sets` is a tuple of length >2 
        """
        self.parameter_sets = parameter_sets
        self.evaluation_function = evaluation_function
        self.save_file = save_file
        self._results_ = []
    
    def save(self,overwrite=False):
        """
        saving of the instance, possibly overwriting the file
        Args:
            filename: str
            overwrite: bool
        Returns:
        """
        if self.save_file is None:
            raise ValueError(f'save_file name was not passed to parametersearch. Unable to save.')
        
        import pickle as pkl
        if not overwrite:
            import os
            if os.path.exists(self.save_file):
                raise FileExistsError(f'file {self.save_file} already exists and overwrite was set to False')

        with open(self.save_file,'wb') as f:
            pkl.dump(self,f)
        
            
    def __raise_not_loaded__(self,*args,**kwargs):
        return NotLoadedError('Evaluation function cannot be saved, so the .evaluation_function attribute must be set to the evaluation funciton manually')
    
    def __getstate__(self):
        state = self.__dict__.copy()
        #state['evaluation_function']= self.__raise_not_loaded__
        return state
    
    @property
    def results(self):
        return self._results_
    
    @classmethod
    def load(cls,save_file,evaluation_function=None):
        with open(save_file,'rb') as f:
            new = pkl.load(f)
        
        if evaluation_function is not None:
            new.evaluation_function = evaluation_function
        return new

    
    def _param_set_preprocessing_(self,param_set):
        if isinstance(param_set,dict):
            param_set = (tuple(),param_set) # cast as kwargs
        elif not (isinstance(param_set,tuple)):
            param_set = ((param_set,),{})
        return param_set
        
    def run(self,save_results=True,skip_errors=True):
        
        # resume from here
        start_i = len(self._results_)
        
        for i,param_set in enumerate(self.parameter_sets[start_i:]):
            LOGGER.info(f'evaluating at grid point {i+1+start_i}/{len(self.parameter_sets)}')
            print(f'evaluating at grid point {i+1+start_i}/{len(self.parameter_sets)}')
            print(f'parameters: {param_set}')
            param_set = self._param_set_preprocessing_(param_set)
            try:
                result = self.evaluation_function(*param_set[0],**param_set[1])
                self._results_.append(result)
                if save_results: 
                    self.save(overwrite=True)
    
            except Exception as err:
                if (not skip_errors) or isinstance(err,NotLoadedError):
                    raise err
                else:
                    LOGGER.warn(f'error occured:{err}')
                    self._results_.append(np.nan)


class MPParameterSearch(ParameterSearch):
    def run(self,n_workers,save_results=True,skip_errors=True):
        import multiprocessing as mp
        
        
 
        pool = mp.Pool(n_workers)
        
        results = pool.map(MPParameterSearch.run_func,((p,self,skip_errors) for p in self.parameter_sets))
        self._results_ = results
        if save_results:
            self.save(overwrite=True)
    
    @staticmethod
    def run_func(args):#inst,param_set,skip_errors):
        param_set,inst,skip_errors = args
        param_set = inst._param_set_preprocessing_(param_set)

        try:
            result = inst.evaluation_function(*param_set[0],**param_set[1])
            print(f'result: {result}')    
            return result
        except Exception as err:
            if not skip_errors:
                raise err
            LOGGER.info(f'error occured: {err}')
            return None
    
class GridSearch(ParameterSearch):
    """
    Specific ParameterSearch where the parameters sets are formed by a ParameterGrid (checking all combinations of values for the different arguments)
    """
    def __init__(self,parameter_grid,evaluation_function,save_file=None):
        """
        :param dict|list parameter_grid:
        :param ModelSolver solver:
        :param Callable ef_construction:
        :param bool exact_point_set: Whether the specified parameter grid should be interpreted as a list of the exact parameter specification, rather
        a list of parameters and posible values to combine.
        :param Callable post_processing_func: the function to apply to the outcome of the model and save the return of this function.
        The function must take the EF model, the Parameter set passed to the ef_construction function and the model results
        """
        self.parameter_grid = parameter_grid # the possible parameters which to run the model for


        parameter_sets = list(ParameterGrid(parameter_grid))
        super(GridSearch,self).__init__(parameter_sets,evaluation_function,save_file)
        


    def to_dataframe(self,results_formatter = None)-> pd.DataFrame:
        df=pd.DataFrame(self.parameter_sets)
        res = self._results_ + [None]*(len(self.parameter_sets)-len(self._results_))
        if results_formatter is not None:
            res = [results_formatter(r) for r in res]
        df['results'] = res
        """
        evals = [r[0] for r in self.results]
        evecs = [r[1] for r in self.results]
        n_gridpoints = len(self.grid_points)
        evals = evals + [np.nan]*(n_gridpoints-len(evals))
        evecs = evecs + [np.nan]*(n_gridpoints-len(evecs))
        df['evals'] = evals
        df['evecs'] = evecs
        """
        return df


class VerboseGridSearch(GridSearch):
    def __init__(self,parameter_grid,evaluation_func,exact_point_set = False):
        """
        Particlular GridSearch where the evaluation function is simply run with the parameters as input and the return value of the function is the result
        :param parameter_grid:
        :param evaluation_func:
        :param exact_point_set:
        """
        super().__init__(parameter_grid,None,None,None,exact_point_set)
        self.evaluation_func = evaluation_func


    def __getstate__(self):
        state = self.__dict__.copy()
        state['evaluation_func'] = None

        return state
    def run_gridsearch(self,sparse=True,save=None,skip_errors=False):

        """
        :param skip_errors:
        :param int n_saves: How many eigenvectors of each gridpoint to store
        :param bool skip_errors: Whether to ingore any errors raised
        :return:
        """
        resume_from = len(self.results)
        for i,parameter_set in enumerate(self.grid_points):
            if i<resume_from:
                continue # skip the ones we have seen
            LOGGER.info(f'evaluating at grid point {i+1}/{len(self.grid_points)}')
            try:

                result = self.evaluation_func(parameter_set)
                self.results.append(result)
            except Exception as err:
                if skip_errors:
                    LOGGER.warning(err)
                    self.results.append(np.nan)
                else:
                    raise err

            if save is not None:
                LOGGER.info(f'saving results to {save}')
                self.save(save, overwrite=True)


class GridSearchDB(GridSearch):
    """
    Wraps a shelf for storing large amounts of data from each grid-point. recording the data into a pandas dataframe,
    the results are not put in directly but rather, functions applied to the results can be applied
    """

    class DictDatabase():
        def __init__(self,filename):
            self.filename = filename

        def __setitem__(self, key, value):
            if isinstance(value, np.ndarray):
                value_type = 'np.ndarray'
                value = value.tolist()
            else:
                value_type = str(type(value))
            import pysos
            db=pysos.Dict(self.filename)
            db[key] = (value,value_type)

        def __getitem__(self, item):

            import pysos
            db = pysos.Dict(self.filename)
            value,val_type = db[item]
            if val_type == 'np.ndarray':
                value = np.array(value)
            return value

        def __iter__(self):

            import pysos
            db = pysos.Dict(self.filename)
            for k in db.keys():
                yield k

        def items(self):

            import pysos
            db = pysos.Dict(self.filename)
            for k in db.keys():
                yield k,db[k]

        def keys(self):

            import pysos
            return pysos.Dict(self.filename).keys()

    def __init__(self, parameter_grid, solver, ef_construction, evaluation_function,db_path):
        self.db_path = db_path
        self.__db__ = self.DictDatabase(db_path)
        super().__init__(parameter_grid, solver, ef_construction, evaluation_function)
        self.results = None
        self.n_results = 0


    def save(self,filename,overwrite=True):
        """
        saves everything possible for later retrieval
        """
        state = self.__dict__.copy()
        del state['__db__']
        del state['construction']
        del state['evaluation_function']
        del state['_grid_points_']
        del state['solver']
        self.__db__['__meta__'] = state
        pass

    @classmethod
    def load(cls,solver,ef_construction,evaluation_function,filename):
        db = cls.DictDatabase(filename)
        if '__meta__' not in db.keys():
            raise AttributeError(f'database was not saved and gridsearch can therefore not be loaded')
        state = db['__meta__']
        parameter_grid = state['parameter_grid']
        new = cls(parameter_grid,solver,ef_construction,evaluation_function,filename)
        new.__dict__.update(state)
        return new
    def run_gridsearch(self, sparse=True, save=None, skip_errors=False):

        resume_from = self.n_results
        for i,parameter_set in enumerate(self.grid_points):
            if i<resume_from:
                continue # skip the ones we have seen
            LOGGER.info(f'evaluating at grid point {i+1}/{len(self.grid_points)}')
            ef_model = self.ef_construction_func(**parameter_set)
            result = self.solver.solve(ef_model,sparse)
            evaluation = self.post_processing_func(ef_model, parameter_set, result)
            LOGGER.info(f'saving results to {self.db_path}')
            self.__db__[i] = evaluation
            self.n_results +=1
    def to_dataframe(self):
        paramgrid = ParameterGrid(self.parameter_grid)
        df = pd.DataFrame(paramgrid)
        if self.__n_computed_results__:
            for i in range(self.__n_computed_results__):
                df[f'result_{i}'] = self.__db__[f'result_{i}']
        else:
            # if no result is computed, try to place all results in the dataframe
            try:
                results = [self.__db__[i] for i in range(self.n_results)]+ [np.nan]*(len(self.grid_points)-self.n_results)
                df['results'] = results
            except MemoryError as err:
                LOGGER.info(f'MemoryError: {err}. Unable to add results to dataframe')

        return df


    def process_results(self,function):
        """
        Applies funciton to every entry in the database and adds the result to the database
        Args:
            function:

        Returns:

        """
        computed = [function(r) for k,r in self.__db__.items() if isinstance(k,int)]
        name = f'result_{self.__n_computed_results__}'
        self.__db__[name] = computed
    @property
    def __n_computed_results__(self):
        res_number = 0
        for k in self.__db__.keys():
            if isinstance(k, str) and k[0] == 'r':
                res_number = max((res_number,k[-1]))

        return res_number


class GridSearchCSV(GridSearch):

    def save(self,filename,overwrite=False):

        if not overwrite:
            import os
            if os.path.exists(filename):
                raise FileExistsError(f'file {filename} already exists and overwrite was set to False')

        df = self.to_dataframe()
        df.to_csv(filename)


def detuning_parameter_set(default_values,detuning_range,n_points,signature=(-1,+1)):
    """
    Constructs a parameter_grid list for plotting bound state energies as functions of detuning
    :param dict default_values: dictionary of length 2 where the parameter names are the keys and their default values
    are the values.
    :param tuple[float,2]|float detuning_range: how far the detuning has to go away from the default value
    :param int n_points: Number of points in the set
    :return:
    """

    if isinstance(detuning_range,float):
        detuning_range = (-detuning_range,detuning_range)
    elif len(detuning_range)>2 or (detuning_range[0]>0 == detuning_range[1]>0):
        raise ValueError(f'detuning range: {detuning_range} not understood, should be a tuple of two elements of different sign')

    detuning_vals = np.linspace(detuning_range[0],detuning_range[1],n_points)
    # we probably always want to evaluate at the exact midpoint as well so we add that to the list
    detuning_vals= np.sort(np.array(detuning_vals.tolist()+[0]))
    if len(default_values)!=2:
        raise SyntaxError(f'default values must be a dict of length 2')

    val_1,val_2 = default_values.keys()

    parameter_grid = [{val_1: default_values[val_1]+signature[0]*delta/2,
                       val_2: default_values[val_2]+signature[1]*delta/2} for delta in detuning_vals]
    return parameter_grid


def quadrant_parameter_set(parameter_names,E0,detuning_range,n_points,quadrant):
    """
    Constructs parameter sets for detuning around (E,E), (-E,E), (-E,-E) or (E,-E)
    :param parameter_names:
    :param E0:
    :param detuning_range:
    :param n_points:
    :param quadrant:
    :return:
    """
    if quadrant == 1:
        default_values = {parameter_names[0]: E0,
                          parameter_names[1]: E0}
        signature = (-1,+1)

    elif quadrant == 2:
        default_values = {parameter_names[0]: E0,
                          parameter_names[1]: -E0}
        signature = (-1,-1)
    elif quadrant == 3:
        default_values = {parameter_names[0]: -E0,
                          parameter_names[1]: -E0}
        signature = (+1,-1)
    elif quadrant == 4:
        default_values = {parameter_names[0]: E0,
                          parameter_names[1]: -E0}
        signature = (+1,+1)
    return detuning_parameter_set(default_values,detuning_range,n_points,signature)



class MinimizationSearch():
    __SCALING_CONSTANT__ = 100  # When normalizing parameters in the solver search they will be set to this

    def __init__(self,func,default_values,parameter_bounds,**minimization_kwargs):
        """
        Searches the parameter_space for where the post_processing function is minimized.
        :param func: function to minimize
        :param default_values: Dictionary containing parameters to alter and their default value (initial guess)
        :param parameter_bounds: Dictionary with same keys ad default_values specifying the min and max range of each parameter
        :param minimization_kwargs: Kwargs to pass to scipy.optimize.minimize
        """
        self.minimization_kwargs = minimization_kwargs
        self.default_param_values = default_values

        
        self.is_scalar =  len(self.default_param_values) == 1
        if 'method' not in self.minimization_kwargs.keys():
            if self.is_scalar:
                if 'bracket' in self.minimization_kwargs:
                    self.minimization_kwargs['method'] = 'brent'
                else:
                    self.minimization_kwargs['method'] = 'bounded'
            else:
                self.minimization_kwargs['method'] = 'SLSQP' # default minimization method

        
        
        
        

        self.func = func

        self.parameter_bounds = parameter_bounds

        # define a specific key ordering which specifies how to convert the tuple x in the optimization function to
        self.key_ordering = tuple(self.default_param_values.keys())
        # the keyword parameter_set
        self.minimization_func = None
        self.param_normalization = {}
        for k,v in self.default_param_values.items():
            try:
                bound = parameter_bounds[k]
                scale = (bound[1]-bound[0])/(2*MinimizationSearch.__SCALING_CONSTANT__)
                shift = (bound[0]+bound[1])/2 + v 
                self.param_normalization[k] = (scale,shift)
                
                
            except TypeError as err:
                raise TypeError(f'unable to set norm for parameter {k} with default value {k} and bounds {self.parameter_bounds[k]}') from err

        self.minimum=None

    def __make_optimization_func__(self):
        """
        Constructs function which can be passed to scipy minimiztion function
        :return:
        """

        """
        class optimization_runner():
            def __init__(self,input_scale,output_scale=None,):
                self.input_scale = input_scale
                self.output_scale=output_scale
                self.gridsearch_instance = gridsearch_self
                self.first_run = None

            def __call__(self,x):

                def single_run(xval):
                    if gridsearch_self.__key_ordering__:
                        arg_dict = {k: xx * self.input_scale[k] for k, xx in zip(self.gridsearch_instance.__key_ordering__, xval)}

                    else:
                        arg_dict = xval * self.input_scale

                    LOGGER.debug(f'costructing model with arguments: {arg_dict}\n constructed from {xval}')
                    ef_model = self.gridsearch_instance.ef_construction_func(arg_dict)
                    LOGGER.debug(f'solving model')
                    solution = self.gridsearch_instance.solver.solve(ef_model)

                    if self.first_run is None:
                        LOGGER.debug(f'setting first run results for this')
                        self.first_run = (xval, solution)

                    LOGGER.debug(f'processing solution')
                    return_val = self.gridsearch_instance.post_processing_func(ef_model, arg_dict, solution)
                    if self.output_scale is None:
                        LOGGER.debug(f'setting output scale to {return_val/MinimizationSearch.__SCALING_CONSTANT__}')
                        self.output_scale = return_val/MinimizationSearch.__SCALING_CONSTANT__

                    return_val = return_val/self.output_scale
                    LOGGER.debug(f'evaluated to: {return_val}\n')

                    return return_val

                if isinstance(x, Iterable):
                    return_val = [single_run(xval) for xval in x]
                else:
                    return_val = single_run(x)

                return return_val
        """

        runner = MinimizationFuncWrap(self.func,self.param_normalization,ordering=self.key_ordering)
        if self.minimization_func is None:
            self.minimization_func = runner
        return runner

    def find_minimum(self,raise_on_fail=True):
        if self.minimum is not None:
            return self.minimum

        #convert everython to what scipy wants
        bounds = tuple([(-100,100)]*len(self.key_ordering))

        x0 =tuple([0]*len(self.key_ordering))
        
        minimization_func = self.__make_optimization_func__()

        LOGGER.info(f'starting minimization with\n x0:{x0},\n bounds: {bounds}')
        
        
        if self.is_scalar:
            
            if 'bracket' in self.minimization_kwargs and self.minimization_kwargs['method'] == 'brent':
                # convert the bracket:
                b0,b1,b2 =self.minimization_kwargs['bracket'][0]
                f0,f1,f2 = self.minimization_kwargs['bracket'][1]
                
                normalization = np.max([f0,f1,f2])/100
                minimization_func.output_scale=normalization
                
                scale,shift = list(self.param_normalization.values())[0]
                convert = lambda b: (b-shift)/scale
                from scipy.optimize._optimize import OptimizeResult,Brent
                brent = Brent(minimization_func,disp=3,tol=0.1)
                
                def ret_brack(*args,**kwargs):
                    return convert(b0),convert(b1),convert(b2),f0,f1,f2,0 #xa,xb,xc,fa,fb,fc,funcalss
                #owervrite bracket info func to avoid recomputing bracket vals
                brent.get_bracket_info = ret_brack
                brent.optimize()
                x,fval,nit,nfev, = brent.get_result(full_output=True)
                minimization = OptimizeResult(fun=fval,x=x,nit=nit,nfev=nfev,success=True,message='success')
                
                # We have to make the brent stuff ourselves because the we have alredy computed the bracket in full and do not need to evaluate the function at the bracket points
                
                
                
                
                
            else:
                bracket = None
                bound= bounds[0] = None
                minimization = minimize_scalar(minimization_func,bounds=bound,bracket=bracket,method=self.minimization_kwargs['method'],options={'disp':3,'xtol':0.1})

        else:
            minimization = minimize(minimization_func,x0,bounds=bounds,**self.minimization_kwargs)
            LOGGER.debug(f'Minimization finished:\n {minimization}')

        # convert outcome to paramdict
        if not minimization.success and raise_on_fail:
            raise Exception(f'minization did not converge')
        final_val = minimization.fun*minimization_func.output_scale
        if self.is_scalar:
            minimization.x = (minimization.x,)
        final_x = {k:xx*self.param_normalization[k][0] + self.param_normalization[k][1] for k,xx in zip(self.key_ordering,minimization.x)}


        minimization.update(x=final_x,fun=final_val)
        LOGGER.info(f'minimization successful with final value:\n {final_val}\n achieved for arguments:\n {final_x}')
        self.minimum= minimization
        self.func_wrap = minimization_func
        return minimization
    
    
class MinimizationFuncWrap():
    def __init__(self,func,input_scale,output_scale=None,ordering=None):
        """ This class wraps a function to make finding the minimium easier by scaling the input and output to more reasonable values (makes them order unity)"""
        self.input_scale = input_scale
        self.output_scale = output_scale 
        self.first_run = None # save the first run
        self.func =func
        self.ordering = ordering if ordering is not None else tuple(self.input_scale.keys())
        self.results_log = []
    
    def __call__(self,x):
        # sometimes x is an array meaninig we have to run it for each x_value
        if isinstance(x,Iterable):
            return_val = [self.single_run(xx) for xx in x]
        else:
            return_val = self.single_run(x)
        return return_val
    
    def single_run(self,x):
        # cast x as dict:
        from . import UNIT_CONVENTION
        E0 = 1/(UNIT_CONVENTION['J to eV']*100000) # unit_scale: 10muev
        if isinstance(x,float):
            scaling = self.input_scale[self.ordering[0]]
            x_dict = {self.ordering[0]:x*scaling[0] + scaling[1]}
        else:
            x_dict = {k:xx*self.input_scale[k][0] + self.input_scale[k][1] for k,xx in zip(self.ordering,x)}
        p = {k:v/E0 for k,v in x_dict.items()}
        LOGGER.debug(f'evaluating function with x={p}')
        result = self.func(x_dict) 
        
        if self.first_run is None:
            self.first_run = (x,result)
        
        if self.output_scale is None:
            self.output_scale = result/100 
        
        normalized_result = result / self.output_scale
        LOGGER.debug(f'result: {normalized_result}')
        self.results_log.append((x_dict,result))
        
        return normalized_result
    



from typing import Callable
class IterativeModelSolver():
    def __init__(self,construction_func:Callable,solver:ModelSolver,evaluation_func:None|Callable=None,start_from_prev =True):
        self.construction_func = construction_func
        self.solver = solver
        self.evaluation_func = evaluation_func
        self.prev_result = None
        self._i = 0
        
    def __call__(self,*args,**kwargs):
        self._i += 1
        print(self._i)
        model = self.construction_func(*args,**kwargs)
        
        if self.prev_result is not None:
            # make initial guess as linear combination of all the previous ones 
            initial_vec = np.sum(self.prev_result[1],axis=0)
            
            self.solver.solver_kwargs['initial_guess'] = initial_vec
            
        res = self.solver.solve(model)
        
        self.prev_result=res
        if self.evaluation_func is None:
            return res
        return self.evaluation_func(model,res)