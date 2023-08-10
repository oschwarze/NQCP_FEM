import pandas as pd
from sklearn.model_selection import ParameterGrid
from .solvers import ModelSolver
import logging
import numpy as np
from scipy.optimize import minimize
import pickle as pkl
import os 

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
        state['evaluation_function']= self.__raise_not_loaded__
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

    
    def run(self,save_results=True,skip_errors=True):
        
        # resume from here
        start_i = len(self._results_)
        
        for i,param_set in enumerate(self.parameter_sets[start_i:]):
            LOGGER.info(f'evaluating at grid point {i+1+start_i}/{len(self.parameter_sets)}')
            if isinstance(param_set,dict):
                param_set = (tuple(),param_set) # cast as kwargs
            elif not (isinstance(param_set,tuple)):
                param_set = ((param_set,),{})
            try:
                result = self.evaluation_function(*param_set[0],**param_set[1])
                self._results_.append(result)
                if save_results: 
                    self.save(overwrite=True)
    
            except Exception as err:
                if (not skip_errors) or isinstance(err,NotLoadedError):
                    raise err
                else:
                    LOGGER.warn(err)
                    self._results_.append(np.nan)
                
    
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

    import pysos
    class DictDatabase():
        def __init__(self,filename):
            self.filename = filename

        def __setitem__(self, key, value):
            if isinstance(value, np.ndarray):
                value_type = 'np.ndarray'
                value = value.tolist()
            else:
                value_type = str(type(value))

            db=pysos.Dict(self.filename)
            db[key] = (value,value_type)

        def __getitem__(self, item):
            db = pysos.Dict(self.filename)
            value,val_type = db[item]
            if val_type == 'np.ndarray':
                value = np.array(value)
            return value

        def __iter__(self):
            db = pysos.Dict(self.filename)
            for k in db.keys():
                yield k

        def items(self):
            db = pysos.Dict(self.filename)
            for k in db.keys():
                yield k,db[k]

        def keys(self):
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


class MinimizationSearch(ParameterSearch):
    __SCALING_CONSTANT__ = 100  # When normalizing parameters in the solver search they will be set to this

    def __init__(self,solver, ef_construction_func,post_processing_func,default_values,parameter_bounds=None,**minization_kwargs):
        """
        Searches the parameter_space for where the post_processing function is minimized.
        :param Solver solver:
        :param Callable ef_construction_func: function taking either a float or a dict and returning a EFM for solving
        :param post_processing_func: function taking outcome of a EFM solution and returning a float
        :param minization_kwargs: arguments passed to the minization function (see scipy.optimize.minimize)
        """
        super().__init__(solver,ef_construction_func,post_processing_func)
        self.minimization_kwargs = minization_kwargs
        if 'method' not in self.minimization_kwargs.keys():
            self.minimization_kwargs['method'] = 'SLSQP' # default minimization method
        self.default_param_values = default_values

        if isinstance(parameter_bounds,float):
            parameter_bounds = [(-1*parameter_bounds,parameter_bounds)]

        self.parameter_bounds = parameter_bounds

        # define a specific key ordering which specifies how to convert the tuple x in the optimization function to
        # the keyword parameter_set
        self.minimization_func = None
        self.normalized_bounds = []
        if isinstance(self.default_param_values, dict):
            self.__key_ordering__ = self.default_param_values.keys()
            self.param_normalization = {}
            for i,(k,v) in self.default_param_values.items():
                norm = v if not np.isclose(v,0)/MinimizationSearch.__SCALING_CONSTANT__ else np.abs(parameter_bounds[i][0])/MinimizationSearch.__SCALING_CONSTANT__
                self.param_normalization[k] = norm
                self.normalized_bounds.append((self.parameter_bounds[i][0]/norm,self.parameter_bounds[i][1]/norm))

        else:
            self.__key_ordering__ = None
            self.param_normalization = self.default_param_values if not np.isclose(self.default_param_values,0) else np.abs(self.parameter_bounds[0][0])/MinimizationSearch.__SCALING_CONSTANT__

            self.normalized_bounds.append((self.parameter_bounds[0][0] / self.param_normalization, self.parameter_bounds[0][1] / self.param_normalization))

    def __make_optimization_func__(gridsearch_self):
        """
        Constructs function which can be passed to scipy minimiztion function
        :return:
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

        runner = optimization_runner(gridsearch_self.param_normalization)
        if gridsearch_self.minimization_func is None:
            gridsearch_self.minimization_func = runner
        return runner

    def find_minimum(self,raise_on_fail=True):


        if self.parameter_bounds is not None:
            if self.__key_ordering__:
                bounds = tuple(self.normalized_bounds[k] for k in self.__key_ordering__)
            else:
                bounds = self.normalized_bounds
        else:
            bounds = None

        minimization_func = self.__make_optimization_func__()

        LOGGER.info(f'starting minimization with\n x0:{self.__x0__()},\n bounds: {bounds}')
        minimization = minimize(minimization_func,self.__x0__(),bounds=bounds,**self.minimization_kwargs)
        LOGGER.debug(f'Minimization finished:\n {minimization}')

        # convert outcome to paramdict
        if not minimization.success and raise_on_fail:
            raise Exception(f'minization did not converge')
        final_val = minimization.fun*minimization_func.output_scale
        if self.__key_ordering__:
            final_x = {k:xx*minimization_func.gridsearch_instance.param_normalization[k] for k,xx in zip(minimization_func.gridsearch_instance.__key_ordering__,minimization.x)}
        else:
            final_x = minimization.x*self.param_normalization


        minimization.update(x=final_x,fun=final_val)
        LOGGER.info(f'minimization successful with final value:\n {final_val}\n achieved for arguments:\n {final_x}')
        return minimization


    def __x0__(self):


        if self.__key_ordering__:
            x0 = tuple(self.default_param_values[k] / self.param_normalization[k] for k in self.__key_ordering__)
        else:
            x0 = self.default_param_values / self.param_normalization

        return x0