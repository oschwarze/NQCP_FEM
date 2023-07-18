def derivative(function,x0,step_size,method='regular',**diff_kwargs):
	"""
	Numerical differentiation of some function wrt. a single variable.
	:param function: The funciton to differentiate. Differentiates wrt the first argument.
	:param x0: the x-value where the derivative is computed.
	:param step_size: step size used when computing the derivative.
	:return:
	"""
	if method == 'regular':
		from scipy.misc import derivative
		return derivative(function,param,dx=step_size,**diff_kwargs)
	elif method == 'custom':
		left = function(param-step_size)
		right = function(param+step_size)
		deriv = (right-left)*1/(2*step_size)
		if diff_kwargs.get('verbose',False):
			return left,right,deriv
		return deriv
	else:
		# todo: implement automatic differentiation routine
		raise NotImplementedError(f'sepcified method {method} is not implemented')