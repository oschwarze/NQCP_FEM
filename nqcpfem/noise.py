import numpy as np
from .derivative import derivative
from . import UNIT_CONVENTION as UNIT
import copy
from .io import ResultsContainer
import shelve


def T2_star_one_over_f(band_model, parameter, variance, deriv_tolerance=1e-2, ir_cutoff=1 * UNIT['t'],
                       solver_kwargs={},log_result_func=None):
	"""
	Compute $T_2^*$ for a band model due to some fluctuation of a parameter in the band model. Formula is from Makhlin et. al. 2004
	:param float ir_cutoff: the Lowest frequency in the noise model default is 1 Hz.
	:param float deriv_tolerance: The step size to use when computing the derivatives. Default is 1% of value
	:param models.band_model_sim.BandModel band_model: Band model to compute the dephasing time of. The relevant default value of the parameter
	must already be defined in the band model.
	:param str|tuple[str,int] parameter: parameter which fluctuates. If correspnding value is a vector parameter should be a tuple with name and index
	corresponding to which coordinate the derivative is to be taken with respect to.
	:param float variance: the variance of the fluctuation
	:return:
	"""
	default_params = band_model.parameters.copy()
	intermediate_states_present = False
	if isinstance(parameter, tuple):
		parameter_i = parameter[1]
		parameter = parameter[0]
		default_val = default_params[parameter][parameter_i]

		def func(x):

			val = copy.copy(default_params[parameter])
			val[parameter_i] = x
			band_model.parameters[parameter] = val
			res =band_model.spin_gap(**solver_kwargs)
			global intermediate_states_present # change the intermediate states argument
			intermediate_states_present = intermediate_states_present or res[2]
			return res[0]
	else:
		if hasattr(band_model.parameters[parameter], '__len__'):
			raise TypeError(
				f'parameter {parameter} was not a scalar. Specify index of the vector to differentiate wrt.')
		default_val = default_params[parameter]

		def func(x):
			# compute spin gap for the specified value of the parameter.
			band_model.parameters[parameter] = x
			res = band_model.spin_gap(**solver_kwargs)
			global intermediate_states_present # change the intermediate states argument
			intermediate_states_present = intermediate_states_present or res[2]
			return res[0]

	gradient = np.abs(derivative(func, default_val, tolerance=deriv_tolerance))

	# hbar in our units is just 1

	return gradient * np.sqrt(variance) * np.sqrt(1 / (2 * np.pi) * np.log(gradient * np.sqrt(variance) / ir_cutoff))


class T2Result(ResultsContainer):

	def __try_reload__(self, mesh_coords, parameter_grid, model_specification, compressed):
		dictionary = self.__load_from_file__(self.file_name)
		self.__dict__.update(dictionary)
		self.__compare_with_load__(mesh_coords, parameter_grid, model_specification, compressed)

	def __init_save_file__(self):
		with shelve.open(self.file_name) as f:
			f['mesh_coords'] = self.mesh_coords
			f['parameter_grid'] = self.parameter_grid
			f['model_specification'] = self.model_specification
			f['results'] = {}

	@staticmethod
	def __load_meta_files__(file_name):
		return_dict = {}
		with shelve.open(file_name) as f:
			return_dict['mesh_coords'] = f['mesh_coord']
			return_dict['parameter_grid'] = f['parameter_grid']
			return_dict['model_specification'] = f['model_specification']
		return return_dict

	@classmethod
	def __load_from_file__(cls, file_name):
		return cls.__load_meta_files__(file_name)

	def __load_save_dict__(self) -> dict:
		with shelve.open(self.file_name) as f:
			res = f['results']
		return res

	@property
	def _get_computed_results_indices(self):
		res_dict = self.__load_save_dict__()
		return set(res_dict.keys())

	def __len__(self):
		return len(self.__load_save_dict__())

	def __getitem__(self, item):
		if isinstance(item, dict):
			try:
				# if dicts is passed return the corresponding result
				i = self.param_grid_list.index(dict)
				return self.__getitem__(i)
			except ValueError as err:
				raise ValueError(f' unable to find the following dictionary in the parameter grid:{item} ')
		f = self.__load_save_dict__()
		if isinstance(item, slice):
			return [f[i] for i in range(item.start, item.stop, item.step)]
		elif isinstance(item, int):
			return f[item]

	def __iter__(self):
		f = self.__load_save_dict__()
		for i in range(len(self)):
			yield f[f'{i}']

	def add_result(self, result, index=None):
		if isinstance(index, int):
			i = index
		elif isinstance(index, dict):
			try:
				i = self.param_grid_list.index(index)
			except ValueError as err:
				raise ValueError(f' unable to locate parameter dict in parameter grid: {index}')
		elif index is None:
			i = min(self.unsolved_grid_points)
		else:
			raise ValueError(f' "index" can only be int,dict or None, recieved {index}')

		res_dict = self.__load_save_dict__()
		res_dict[i] = result
		with shelve.open(self.file_name) as f:
			res = f['results']
			res[i] = result
			f['results'] = res

	def export_dataframe(self):
		import pandas as pd
		import itertools
		results_dict = {'T2star': [i for _, i in self.__load_save_dict__().items()]}
		default_params = self.model_specification[-1]


		for param_dict in self.parameter_grid:
			for key in set(default_params.keys()).union(param_dict.keys()): # key is either in default dict or in param_dict
				if f'param_{key}' not in results_dict:
					results_dict[f'param_{key}']  = []
				append_val = param_dict.get(key, default_params.get(key,None))
				if append_val is None:
					raise KeyError(f'{key} was not found in neither default_parameter dict nor param_dict ')
				results_dict[f'param_{key}'].append(append_val)

		df = pd.DataFrame(results_dict)
		if 'param_omega' in df.columns:
			def expand_omega(x):
				xval = x['param_omega']
				return [xval,xval,xval] if isinstance(xval,float) else xval
			df[['ox', 'oy', 'oz']] = df[['param_omega']].apply(expand_omega, axis=1,result_type='expand')
			df['omega_is_iso'] = df['param_omega'].apply(lambda x: (isinstance(x,float) or x[0] == x[1]))
		if 'param_B' in df.columns:
			df[['Bx', 'By', 'Bz']] = pd.DataFrame(df['param_B'].to_list())
			df['B_in_plane'] = df['param_B'].apply(lambda x:  x[-1] == 0 )
			df['B_norm'] = df['param_B'].apply(lambda x: np.linalg.norm(x))

		return df
