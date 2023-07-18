import numpy as np
import pickle as pkl
import io
from .io import ResultsContainer
import os
import zipfile
import itertools
from sklearn.model_selection import ParameterGrid
import logging

from .envelope_function import EnvelopeFunctionModel
from .plotting import compute_angular_momentum, expec_val

LOGGER = logging.getLogger(__name__)
LOGGER.addHandler(logging.NullHandler())

from . import ANGULAR_MOMENTUM as AM
J=AM['3/2']

"""
DESCRIPTION: This module handles working with results of a GridSearch and spin gaps of `band_model_sim`. 
It is in a separate module, such that this module can be loaded in python version that do not have FEniCSx installed.
"""



class SpinGapResult(ResultsContainer):
	"""
	Class responsible for handling results of af GridSearch. Enables working with the results on systems that have
	not installed FEniCSx.
	:param str file_name: file name to save the file
	:param np.array mesh_coords: the coordinates specifying the mesh of the problem
	:param dict|sklearn.model_selection.ParameterGrid parameter_grid:
	:param list model_specification: list specifying the model that was used to produce the results
	"""

	def __try_reload__(self,mesh_coords,parameter_grid,model_specification,compressed):
		if not zipfile.is_zipfile(self._file_name):
			raise ValueError(f'file name {self.file_name} already exists but is not a zip file')
		dictionary = self.__load_from_file__(self.file_name)
		self.__dict__.update(dictionary)
		self.__compare_with_load__(mesh_coords,parameter_grid,model_specification,compressed)
		return

	def __init_save_file__(self):
		with zipfile.ZipFile(self._file_name,'w') as f:
			bio = io.BytesIO()
			np.save(bio,self.mesh_coords)
			f.writestr('mesh_coords.dat', data=bio.getbuffer().tobytes())
			f.writestr(f'parameter_grid.pkl', data=pkl.dumps(self.parameter_grid))
			f.writestr('model_specification.pkl',data=pkl.dumps(self.model_specification))

	@staticmethod
	def __load_meta_files__(file_name):
		if not os.path.exists(file_name):
			raise FileNotFoundError(f'file named {file_name} was not found.')
		return_dict = {}

		with zipfile.ZipFile(file_name,'r') as f:
			with f.open('mesh_coords.dat','r') as coords:
				return_dict['mesh_coords'] = np.load(coords)
			with f.open('parameter_grid.pkl','r') as paramgrid:
				return_dict['parameter_grid'] = pkl.load(paramgrid)
			#with f.open('parameter_grid.pkl','r') as paramgrid:
			#	return_dict['parameter_grid'] = pkl.load(paramgrid)
			with f.open('model_specification.pkl','r') as modelspec:
				return_dict['model_specification'] = pkl.load(modelspec)
		return return_dict

	@classmethod
	def __load_from_file__(cls, file_name):
		return_dict = cls.__load_meta_files__(file_name)
		with zipfile.ZipFile(file_name,'r') as f:
			file_info = f.infolist()
			return_dict['is_compressed'] = False
			for f_info in file_info:
				if f_info.filename[-4:] == '.npz' and f_info.filename != 'coords_file.npy':
					if f_info.compress_type != 0:
						return_dict['is_compressed'] = True
					elif return_dict['is_compressed']:
						raise ValueError(' the .npy files in the save file were a mixture of compressed or uncompressed')
		return return_dict


	@property
	def file_name(self):
		return self._file_name

	@property
	def _get_computed_results_indices(self):
		with np.load(self.file_name) as file_dict:
			return {int(f.split('_')[0]) for f in file_dict.files if f[-4:] not in ('.dat','.pkl')} #retruns set since elements only appear once and lookup is quick

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

	def __len__(self):
		with np.load(self.file_name) as f:
			return int((len(f.files)-3)/2)

	def __getitem__(self, item):
		if isinstance(item,dict):
			try:
				# if dicts is passed return the corresponding result
				i = self.param_grid_list.index(dict)
				return self.__getitem__(i)
			except ValueError as err:
				raise ValueError(f' unable to find the following dictionary in the parameter grid:{item} ')

		with np.load(self.file_name) as f:
			if isinstance(item, slice):
				return [(f[f'{i}_evals'], f[f'{i}_evects']) for i in range(item.start, item.stop, item.step)]
			elif isinstance(item, int):
				return f[f'{item}_evals'], f[f'{item}_evects']

	def __iter__(self):
		with np.load(self.file_name) as f:
			for i in range(len(self)):
				yield f[f'{i}_evals'], f[f'{i}_evects']

	def add_result(self, result,index=None):
		"""
		Adds a new result to the container. By default the results are associated with the first parameter grid point
		that isn't solved yet.
		:param tuple result: result of a get_eigenvalues call
		:param int|dict|None index: which parameter grid point it belongs to. integer index references the paramter grid
		point with the same index. dict index specifies the parameter_grid
		:return:
		"""
		if isinstance(index,int):
			i = index
		elif isinstance(index,dict):
			try:
				i = self.param_grid_list.index(index)
			except ValueError as err:
				raise ValueError(f' unable to locate parameter dict in parameter grid: {index}')
		elif index is None:
			i = min(self.unsolved_grid_points)
		else:
			raise ValueError(f' "index" can only be int,dict or None, recieved {index}')

		bio_evects = io.BytesIO()
		bio_evals = io.BytesIO()
		np.save(bio_evects, result[1])
		np.save(bio_evals, result[0])
		with zipfile.ZipFile(self.file_name, 'a') as zipf:
			# careful, this file below must be .npy
			zipf.writestr(f'{i}_evals.npy', data=bio_evals.getbuffer().tobytes())
			zipf.writestr(f'{i}_evects.npy', data=bio_evects.getbuffer().tobytes())

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

	def overwrite_files(self,files):
		raise NotImplementedError(f'not yet implemented, and resource intensive due to having to rewrite the entire zip archive. avoid doing this')
		# todo: create new zipfile as tempfile and copy all the old and replaced files into that before deleting the old file and replaeing it with the new.

	def find_spin_gap(self,bounded_state_tolerance=1,wave_func_infidelity=1e-4,drop_eigenvectors=False,compute_J=False):
		"""
		Determines the spin gaps of the results and returns a dict of the results and the correspinding grid point
		parameters. see find_spin_gap function for documentation
		:param bounded_state_tolerance:
		:param wave_func_infidelity:
		:return:
		"""
		return find_spin_gap(self, bounded_state_tolerance, wave_func_infidelity)


	def export_dataframe(self,compute_J=True):
		import pandas as pd
		result_dict = self.find_spin_gap(drop_eigenvectors=True,compute_J=compute_J)
		df = pd.DataFrame.from_dict(result_dict)
		if 'param_omega' in df.columns:
			df[['ox', 'oy', 'oz']] = pd.DataFrame(df['param_omega'].to_list())
			df['omega_is_iso'] = df['param_omega'].apply(lambda x: True if x[0] == x[1] else False)
		df[['Bx', 'By', 'Bz']] = pd.DataFrame(df['param_B'].to_list())
		df['B_in_plane'] = df['param_B'].apply(lambda x: True if x[0] > 0 else False)
		df['B_norm'] = df['param_B'].apply(lambda x: np.linalg.norm(x))

		df['g_factor'] = df['spin_gaps'] / df['B_norm']
		if compute_J:
			df[['Jx', 'Jy', 'Jz']] = pd.DataFrame(
				[[J[0].tolist(), J[1].tolist(), J[2].tolist()] for J in df['J'].to_list()])
			df[['Jx_gs', 'Jx_ex']] = pd.DataFrame(df['Jx'].to_list())
			df[['Jy_gs', 'Jy_ex']] = pd.DataFrame(df['Jy'].to_list())
			df[['Jz_gs', 'Jz_ex']] = pd.DataFrame(df['Jz'].to_list())

			def func(x):
				return [max([abs(z) for z in y]) for y in x]

			df['maxes'] = df[['Jx', 'Jy', 'Jz']].apply(func).idxmax(axis=1)

			def func2(x):
				return [int(y) for y in np.real(np.sign(x[x['maxes']]))]

			df[['max_signs_gs', 'max_signs_ex']] = df[['maxes', 'Jx', 'Jy', 'Jz']].apply(func2, axis=1,
		                                                                             result_type='expand')

			markers = {'Jx': ("*", r">", r"<"), 'Jy': ("*", "x", "o"), 'Jz': ("*", "^", "v")}
			df['markers_gs'] = df[['max_signs_gs', 'maxes']].apply(lambda x: markers[x['maxes']][x['max_signs_gs']], axis=1)
			df['markers_ex'] = df[['max_signs_ex', 'maxes']].apply(lambda x: markers[x['maxes']][x['max_signs_ex']], axis=1)
		return df


def find_spin_gap(results,bounded_state_tolerance=1,wave_func_infidelity=1e-4,param_grid = None,default_params = None,mesh_coords=None,box_dims=None,drop_eigenvectors=False,compute_J=False):
	"""
	Find the spin gap between ground state and spin excited state from a set of computed eigenvalues.
	Determines the ground state as the lowest energy eigenstate that is bounded
	(has expected distance from origin = `bounded_state_tolerance*box_dimension` ). Determines spin excited state as
	the lowest excited energy eigenstate with high positional wave-function overlap (higher than ´1-wave_func_infidelity´)
	with respect to the ground statximum allowed expectation of the distance from (0,0,0) for a state to be considered bounded
	:param float wave_func_infidelity: Tolerance for the overlap of positional wave-functions before we consider them the same
	:param ParameterGrid param_grid: parameter grid for returning the results as a pandas dataframe
	:param dict default_params: Default parameters to use in case of a disjoint parametergrid. has to contain all parameters used in the model
	:return:
	"""
	if isinstance(results,tuple):
		if len(results) != 2:
			raise ValueError(f'unsupported value for "results argument. Got {results}"')
		results = [results]
		if mesh_coords is None or box_dims is None:
			raise ValueError(f'If a single result is passed, both `mesh_coords` and `box_dims` should be supplied.')
	else:
		mesh_coords = results.mesh_coords
		box_dims = results.box_dims

	if param_grid is not None and default_params is None:
		raise ValueError(f'param_grid was given but no default_params dict was passed')

	spin_gaps = []
	wave_functions = []
	has_intermediate_state = []
	angular_momentum = []
	for res_i,result in enumerate(results):
		LOGGER.info(f'evaluating results ({res_i+1}/{len(results)})')
		if result[0] is None:
			LOGGER.info('no complete solution...?')
			continue
		eigen_vals = result[0]
		eigen_vects = result[1]
		sorting = np.argsort(eigen_vals)
		eigen_vects = eigen_vects[sorting]
		eigen_vals = eigen_vals[sorting]

		ground_state_wave_func = None
		ground_state_i = None
		has_found_spin_gap = False
		log_overlaps = [] # for logging overlaps of unused states
		for i,e in enumerate(eigen_vals):
			positional_wave_func = np.linalg.norm(eigen_vects[i].reshape((eigen_vects[i].shape[0],-1)),axis=1) #figure out shape and do
			if ground_state_i is None:
				# check if ground state is a bound state.
				dist = np.linalg.norm(mesh_coords, axis=1) #distance from origin of DoD
				avg_dist = np.inner(dist,positional_wave_func**2)
				if avg_dist < max(box_dims)/2*bounded_state_tolerance:
					ground_state_i = i
					ground_state_wave_func = positional_wave_func
					LOGGER.debug(f'Found ground state {i} with distance {np.real(avg_dist)}.')
					continue
			else:
				wave_function_overlap = np.abs(np.inner(positional_wave_func,ground_state_wave_func))
				log_overlaps.append(wave_function_overlap)
				if 1-wave_function_overlap < wave_func_infidelity:
					LOGGER.debug(f'Found excited state {i} with overlap {wave_function_overlap}.\n other overlaps: {log_overlaps[:-1]}')
					spin_gaps.append(e-eigen_vals[ground_state_i])
					if not drop_eigenvectors:
						wave_functions.append((eigen_vects[ground_state_i],eigen_vects[i]))
					if compute_J:

						vects=np.stack((eigen_vects[ground_state_i],eigen_vects[i]))
						"""
						jvects = np.einsum('ijl,axj...->iaxl...',J,vects)
						indicies = [_ for _ in range(1,len(vects.shape)+1)]
						angular_momentum.append(np.einsum(np.conj(vects),indicies,jvects,[0]+indicies,[0,1]))
						jvects_sq = np.einsum('ijl,axj...->iaxl...',J @ J,vects)
						"""
						angular_momentum.append(compute_angular_momentum(vects))
						with np.printoptions(precision=3):
							debug_string = f'Angular Momenta: {angular_momentum[-1]},\n squares: {expec_val(vects, np.einsum("aji,aik->ajk", J, J))}'
						LOGGER.debug(debug_string)
					has_intermediate_state.append(not(ground_state_i+1 == i) )
					has_found_spin_gap = True
					break

		if ground_state_i is None:
			LOGGER.info(f'unable to find ground state....')
			spin_gaps.append(np.nan)
			if not drop_eigenvectors:
				wave_functions.append((None,None))
			has_intermediate_state.append(None)
		elif not has_found_spin_gap:
			LOGGER.info(f'unable to spin-excited state')
			spin_gaps.append(np.nan)
			if not drop_eigenvectors:
				wave_functions.append((None, None))
			has_intermediate_state.append(None)

	if isinstance(results, SpinGapResult):
		param_grid = results.parameter_grid
		default_params = results.model_specification[-1]
	if param_grid is not None :
		spin_gaps = np.array(spin_gaps)
		wave_functions = np.array(wave_functions, dtype=object)
		has_intermediate_state = np.array(has_intermediate_state, dtype=object)
		discards = np.isnan(spin_gaps)

		results_dict = {'spin_gaps': spin_gaps[~discards],
		                'has_intermediate_state': has_intermediate_state[~discards].astype(bool)}
		if not drop_eigenvectors: # If eigenvectors are not wanted (to save memory)
			results_dict['wave_function'] = np.stack(wave_functions[~discards], axis=0),

		if compute_J:
			results_dict['J'] = angular_momentum



		for blank,param_dict in zip(discards,param_grid):
			if not blank:
				for key in set(default_params.keys()).union(param_dict.keys()):
					if f'param_{key}' not in results_dict:
						results_dict[f'param_{key}'] = []
					results_dict[f'param_{key}'].append(param_dict.get(key,default_params[key]))

		"""		
		for key in default_params.keys(): #cast to numpy arrays
			results_dict[f'param_{key}'] = np.array(results_dict[f'param_{key}'])
		"""
		return results_dict
	else:
		wave_functions = [np.stack(w) for w in wave_functions] # combine to single array
		if len(results) ==1:
			to_return = (spin_gaps[0], wave_functions[0], has_intermediate_state[0])
		else:
			to_return=(spin_gaps, wave_functions, has_intermediate_state)
		if compute_J:
			if len(results) == 1:
				to_return = to_return + (angular_momentum[0],)
			else:
				to_return = to_return + angular_momentum
		return to_return



def find_spin_gap(results,envelope_model,bounded_state_tolerance=None,positional_max_tv_dist=1e-4):
	"""
	Find the spin gap between ground state and spin excited state from a set of computed eigenvalues.
	Determines the ground state as the lowest energy eigenstate that is bounded
	(has expected distance from origin = `bounded_state_tolerance*box_dimension` ). Determines spin excited state as
	the lowest excited energy eigenstate with high positional wave-function overlap (higher than ´1-wave_func_infidelity´)
	with respect to the ground state
	:param tuple|list[tuple]|SpinGapResult results: the results. Either a single result, a list of results or a SpinGapResult
	:param EnvelopeFunctionModel envelope_model: The model from which the results came.
	:param float bounded_state_tolerance: Maximum allowed expectation of the distance from (0,0,0) for a state to be considered bounded
	:param float positional_max_tv_dist: Tolerance for the overlap of positional wave-functions before we consider them the same
	:return:
	"""
	if isinstance(results,tuple):
		if len(results) != 2:
			raise ValueError(f'unsupported value for "results argument. Got {results}"')
		results = [results]

	spin_gaps = []
	wave_functions = []
	has_intermediate_state = []
	angular_momentum = []
	for res_i,result in enumerate(results):
		LOGGER.info(f'Determining Spin gap ({res_i+1}/{len(results)})')
		if result[0] is None:
			LOGGER.info('no complete solution... skipping')
			continue
		eigen_vals = result[0]
		eigen_vects = result[1]
		sorting = np.argsort(eigen_vals)
		eigen_vals = eigen_vals[sorting]
		eigen_vects = eigen_vects[sorting]#eigen_vects[:,sorting]

		ground_state_wave_func = None
		ground_state_i = None
		has_found_spin_gap = False
		avg_dists = []
		log_overlaps = [] # for logging overlaps of unused states
		from .observables import positional_probability_distribution
		position_prob_projector = positional_probability_distribution(envelope_model.band_model)
		eigentensors = eigen_vects#envelope_model.eigensolutions_to_eigentensors(eigen_vects)
		for i,e in enumerate(eigen_vals):
			eigentensor = eigentensors[i]
			if ground_state_i is None:
				positional_wave_func, x = envelope_model.positional_rep(eigentensor)

				positional_prob_distribution = position_prob_projector(positional_wave_func)
				if bounded_state_tolerance is None:
					LOGGER.debug(f'ground state was set to index {i} since boundedness condition is ignored')
					ground_state_i = i
					ground_state_wave_func = position_prob_projector(eigentensor)# save wave,function
					continue
				else:
					if x is None:
						raise ValueError(
							f'positional coordinates, x, returned by envelope_model.positional_rep method was None. '
							f'Either specifcy the points to positioal_rep (todo) or set `bounded_state_tolerance` to None.')
					# check if ground state is a bound state.
					dist = np.linalg.norm(x, axis=1) #  distance from origin of DoD
					avg_dist = np.inner(dist,positional_prob_distribution)
					if avg_dist < bounded_state_tolerance*envelope_model.domain.bounded_scale()/envelope_model.length_scale:
						# The average distance from the origin is within the tolerance to be considered a bound state
						ground_state_i = i
						ground_state_wave_func = positional_prob_distribution # save gs state to compare with exited states
						LOGGER.debug(f'Found ground state {i} with distance {np.real(avg_dist)}.')
						continue
					else:
						avg_dists.append(avg_dist)
			else:
				positional_wave_func = position_prob_projector(eigentensor)
				positional_prob_tv_distance = 0.5*np.linalg.norm((positional_wave_func-ground_state_wave_func).flatten(),1)

				log_overlaps.append(positional_prob_tv_distance)
				if positional_prob_tv_distance < positional_max_tv_dist:
					LOGGER.debug(f'Found excited state {i} with positional total variation distance {positional_prob_tv_distance}.\n other overlaps: {log_overlaps[:-1]}')
					spin_gaps.append(e-eigen_vals[ground_state_i])
					wave_functions.append((eigen_vects[ground_state_i], eigen_vects[i]))
					has_intermediate_state.append(not(ground_state_i+1 == i))
					has_found_spin_gap = True
					break

		if ground_state_i is None:
			LOGGER.info(f'unable to find ground state. average_dists were: {avg_dists}')
			print('what')
			spin_gaps.append(np.nan)
			wave_functions.append((None, None))
			has_intermediate_state.append(None)
		elif not has_found_spin_gap:
			LOGGER.info(f'unable to spin-excited state, tv distances were {log_overlaps}')
			print(f'unable to spin-excited state, tv distances were {log_overlaps}')
			spin_gaps.append(np.nan)
			wave_functions.append((None, None))
			has_intermediate_state.append(None)

		wave_functions = [np.stack(w) for w in wave_functions] # combine to single array
		if len(results) ==1:
			to_return = (spin_gaps[0], wave_functions[0], has_intermediate_state[0])
		else:
			to_return=(spin_gaps, wave_functions, has_intermediate_state)
		return to_return


""" 
		def plot_solution(self):
		eigen_vals, eigen_vects = self.get_eigenvalues(10)
		from matplotlib.colors import SymLogNorm

		colormap = SymLogNorm(linthresh=0.03, linscale=0.03,
		                      vmin=-1.0, vmax=1.0, base=10)

		for i in range(len(eigen_vals)):
			n_tol = 1e-9
			vector = eigen_vects[i]
			z_expec_norm = np.linalg.norm(vector, axis=1)
			scaled_vector = vector / np.max(np.abs(z_expec_norm))
			# vector = scaled_vector
			normalization = np.linalg.norm(vector, axis=1)
			normalization  # make it so that spinors for zero wave-vector points are zero
			spinor_normalized = vector / normalization[:, np.newaxis]  # normalize the spinor components
			vector = spinor_normalized
			vector[normalization < n_tol] = np.nan
			z_expec = 3 / 2 * np.abs(vector[:, 0]) ** 2 + 1 / 2 * np.abs(vector[:, 1]) ** 2 - 1 / 2 * np.abs(
				vector[:, 2]) ** 2 - 3 / 2 * np.abs(vector[:, 3]) ** 2
			hh_lh_expec = np.abs(vector[:, 0]) ** 2 + np.abs(vector[:, 1]) ** 2 + np.abs(vector[:, 2]) ** 2 + np.abs(
				vector[:, 3]) ** 2
			p = pyvista.Plotter()
			topology, cell_types, x = dolfinx.plot.create_vtk_mesh(self.F)
			grid = pyvista.UnstructuredGrid(topology, cell_types, x)
			# hh_grid = pyvista.UnstructuredGrid(topology, cell_types, x)
			# lh_grid = pyvista.UnstructuredGrid(topology, cell_types, x)
			# hh_grid["u"] = 2+(np.abs(vector[:,0])**2 +np.abs(vector[:,3])**2)*10
			# lh_grid["u"] = -2-(np.abs(vector[:, 1]) ** 2 + np.abs(vector[:, 2]) ** 2)*10
			grid["u"] = np.linalg.norm(scaled_vector, axis=1) * 10  # positional wave-function shape
			grid['hh_lh'] = hh_lh_expec
			# print(np.average(hh_lh_expec))
			actor_0 = p.add_mesh(grid, scalars='hh_lh', nan_color='black', clim=[-1, 1])
			warped = grid.warp_by_scalar("u")
			actor_1 = p.add_mesh(warped, opacity=0.9)  # ,cmap=colormap) #hh-lh character color
			spin_vec = np.zeros(
				(len(warped.points[:, 0]), 3))  # arrows indicating angular momentum (z-expectation value)
			spin_vec[:, 2] = z_expec
			warped['arrows'] = spin_vec * 3
			arrows = warped.glyph(scale='arrows', orient='arrows')
			p.add_mesh(arrows, color='black')
			# hh_warp=hh_grid.warp_by_scalar('u')
			# lh_warp=lh_grid.warp_by_scalar('u')
			# actor_2 = p.add_mesh(hh_warp)
			# actor_3 = p.add_mesh(lh_warp)
			p.show()
	"""

