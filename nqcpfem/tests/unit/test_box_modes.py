from unittest import TestCase


class TestBoxEFM(TestCase):
	def setUp(self) -> None:
		# test that assembled array has the correct eigenvalues for the particle in box problem
		from nqcpfem.band_model import FreeBoson
		from nqcpfem.box_modes import BoxEFM
		from nqcpfem.envelope_function import RectangleDomain

		self.mass = 1
		self.n_modes = (3,3,3)
		L = 1e-8
		self.Lx = L
		self.Ly = L
		self.Lz = L
		self.band_model = FreeBoson(self.mass, 3)
		self.domain = RectangleDomain(self.Lx, self.Ly, self.Lz)
		self.model = BoxEFM(self.band_model, self.domain, *self.n_modes)

	def test_positional_rep(self):
		n_points = 100
		import numpy as np
		coords = [np.linspace(-L,L,n_points) for L in [self.Lx,self.Ly,self.Lz]]
		x_vals = np.stack(coords,axis=1)
		vector = np.zeros((1,)+self.n_modes)
		# pick random element
		rng = np.random.default_rng()
		coord = tuple(rng.integers(0,3,size=3))
		vector[(0,)+coord] = 1 # pick random basis function
		def basis_func(x_vals):
			psi = 1
			for i,(n,l) in enumerate(zip(coord,[self.Lx,self.Ly,self.Lz])):
				psi = np.tensordot(psi,np.sqrt(2/l)*np.sin((n+1)*np.pi*(x_vals[:,i]/l+0.5)),axes=0)
			return psi
		positional_rep,X = self.model.positional_rep(vector,x_vals=x_vals)
		print(positional_rep.shape,x_vals.shape)
		np.testing.assert_allclose(positional_rep[0],basis_func(x_vals).flatten(),err_msg=f'positional wave-function with index {coord} wrong')

	def test_assemble_array(self):
		# test that assembled array has the correct eigenvalues for the particle in box problem
		import numpy as np

		A = self.model.assemble_array(sparse=False)
		evals, evecs = np.linalg.eigh(A)

		# check that the eigenvalues are correct
		
		# region plotting the solutions
		""" #
		from matplotlib import pyplot as plt
		for i in range(len(evals)):
			print(i,evals[i])
			x_vals = 100
			z_val = int(x_vals/2) # z val at origin
			im = np.abs(model.positional_rep(evecs[:,i],x_vals))[0,:,:,z_val]
			if np.count_nonzero(im)>0.25*im.shape[0]*im.shape[1]:
				plt.figure()
				plt.imshow(im)
				plt.title(f'{i,evals[i]*U["J to eV"]}')
				plt.show()
		"""
		# endregion



		# region evaluating
		from nqcpfem import _hbar
		def exact_energies(nx,ny,nz):
			NL = ((nx+1)/self.Lx)**2+((ny+1)/self.Ly)**2+((nz+1)/self.Lz)**2
			E = np.pi**2*_hbar**2*NL/(2*self.mass)
			return E

		exact_values = np.sort(np.fromfunction(exact_energies,self.n_modes).flatten())
		np.testing.assert_allclose(evals,exact_values)
		
		# check that that the evecs are just a permutation of the identity (box-modes are eigenstates of this Hamiltonian)
		x_i,y_i = np.where(evecs!=0)
		np.testing.assert_array_equal(x_i,np.arange(x_i.shape[0]))
		np.testing.assert_array_equal(np.sort(y_i),np.arange(y_i.shape[0]))

	def test_eigensolutions_to_eigentensors(self):
		import numpy as np
		vector = np.zeros((27,1))
		rng = np.random.default_rng()
		val = rng.integers(0,3,3)
		index = val[0]*9+val[1]*3+val[2]*1
		vector[index,0] = 1
		reshaped_vector = self.model.eigensolutions_to_eigentensors(vector)
		np.testing.assert_array_equal(reshaped_vector,vector.T.reshape((1,1,3,3,3)))
		
		# multiple vectors
		vector = np.zeros((27,2))
		val2 = rng.integers(0,3,3)
		index2 = val2[0]*9+val2[1]*+val2[2]*1
		vector[index,0] = 1
		vector[index2,1] = 1
		reshaped_vector = self.model.eigensolutions_to_eigentensors(vector)
		np.testing.assert_array_equal(reshaped_vector,vector.T.reshape((2,1,3,3,3)),err_msg=f'vectors of indices {val,val2} were not '
		                                                                                    f'rebroadcast correctly')
