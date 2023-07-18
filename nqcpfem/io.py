import numpy as np
from sklearn.model_selection import ParameterGrid
from abc import ABC,abstractmethod
import os
import io
import zipfile
import pickle as pkl
import logging
LOGGER = logging.getLogger(__name__)
LOGGER.addHandler(logging.NullHandler())



class ResultsContainer(ABC):

	def __init__(self,file_name,mesh_coords=None,parameter_grid=None,model_specification=None,compressed=False):
		self._file_name = file_name
		if os.path.exists(self._file_name):
			self.__try_reload__(mesh_coords,parameter_grid,model_specification,compressed=False)
		else:
			self.mesh_coords = mesh_coords
			self.parameter_grid = ParameterGrid(parameter_grid) if not isinstance(parameter_grid, ParameterGrid) else parameter_grid
			self.model_specification = model_specification
			self.is_compressed = compressed
			self.__init_save_file__()

	@abstractmethod
	def __try_reload__(self,mesh_coords,parameter_grid,model_specification,compressed):
		pass

	def __compare_with_load__(self,mesh_coords,parameter_grid,model_specification,compressed):
		if mesh_coords is not None and not np.allclose(mesh_coords, self.mesh_coords):
			raise ValueError(
				f'the file_name already existed and the mesh coordinates within it did not match the mesh coordinates passed to init')

		if parameter_grid is not None:
			p_init = ParameterGrid(parameter_grid) if not isinstance(parameter_grid, ParameterGrid) else parameter_grid
			error_msg = 'the file_name already existed and the parameter grid within it did not match the parameter grid passed to init'
			if p_init.param_grid != self.parameter_grid.param_grid:
				raise ValueError(error_msg)
		if model_specification is not None:
			error_msg = 'the file_name already existed and the model specification within it did not match the model specification passed to init'

			if len(model_specification) != len(self.model_specification) or len(model_specification[-1]) != len(
					self.model_specification[-1]):
				raise ValueError(error_msg)

			for d_init, d_load in zip(model_specification[1:-1], self.model_specification[1:-1]):
				if d_init['name'] != d_load['name']:
					raise ValueError(error_msg)
			if model_specification[-1] != self.model_specification[-1]:
				raise ValueError(error_msg)

			LOGGER.warning(f'SpinGapResult file {self._file_name} already existed, but model specification was'
			               f' passed to __init__. Model builder function names matched, but it is not possible'
			               f' to verify matching of all args and kwargs in construction,'
			               f' so the models may be different.')

	@abstractmethod
	def __init_save_file__(self):
		pass

	@staticmethod
	@abstractmethod
	def __load_meta_files__(file_name):
		pass

	@classmethod
	@abstractmethod
	def __load_from_file__(cls,file_name):
		# loads  results from file
		pass

	@property
	def file_name(self):
		return self._file_name

	@property
	@abstractmethod
	def _get_computed_results_indices(self):
		pass


	@property
	def unsolved_grid_points(self):
		"""
		returns a list of indices corresponding to the grid points that have yet to be solved
		:return:
		"""
		has_computed = self._get_computed_results_indices
		return [i for i in range(len(self.parameter_grid)) if i not in has_computed]


	@property
	def box_dims(self):
		""" Compute box dimensions from the mesh coordinates"""
		side_lengths = np.abs(np.amax(self.mesh_coords,axis=0))+np.abs(np.amin(self.mesh_coords, axis=0))
		return [l for l in side_lengths[side_lengths != 0]]


	@property
	def param_grid_list(self):
		"""
		Get the parameter grid as a list of all the parameter settings
		:return:
		"""
		if not hasattr(self,'_param_grid_list'):
			self._param_grid_list = [p for p in self.parameter_grid]
		return self._param_grid_list

	@abstractmethod
	def __len__(self):
		pass

	@abstractmethod
	def __getitem__(self, item):
		pass

	@abstractmethod
	def __iter__(self):
		pass

	@abstractmethod
	def add_result(self, result,index=None):
		pass

	def as_dict(self):
		return_dict = self.__dict__.copy()
		if '_param_grid_list' in return_dict.keys():
			del return_dict['_param_grid_list']
		return_dict['results'] = [res for res in self]
		return return_dict

	def delete(self):
		os.remove(self.file_name)

	def __wipe_results__(self):
		self.delete()
		self.__init_save_file__()