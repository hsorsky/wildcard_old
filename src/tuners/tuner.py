""" Runs the model to find optimal parameters and writes these to file.
"""
import time
from abc import ABC, abstractmethod

import numpy as np
from scipy.optimize import minimize

from src.logger import logger
from src.tuners.tuner_params import TunerParams
from src.utils import timer, multioptimiser


class Tuner(ABC):

	init_params = NotImplemented

	def __init__(self, data, fixed_params, only_do, method, tol, use_multi_grad, save_output):

		self.data = data
		self.use_multicore_gradient = use_multi_grad
		self.method = method
		self.tol = tol

		self.tuner_params = TunerParams(self.init_params, fixed_params, only_do)

		# This represents the last logged set of parameters so that if the optimiser stops
		# prematurely we dont write some speculative (possibly bad) set of parameters to file
		self.to_save_params = self.tuner_params.__copy__()
		self.save_output = save_output
		self.bounds_dict = self.tuner_params.get_bounds_dict()

	def run_tuner(self):
		with timer('Optimising with method {}'.format(self.method), __file__):
			logger.info("Null model likelihood: {:.4E}".format(self._get_null_model_likelihood()))
			self.tuner_params.log_initial()
			minimise_kwargs = self.minimize_args()

			try:
				if self.use_multicore_gradient:
					optimal = multioptimiser(**minimise_kwargs)
				else:
					optimal = minimize(**minimise_kwargs)
			except (KeyboardInterrupt, SystemExit) as e:
				time.sleep(2)
				logger.info('Cancelling optimisation........')
				self.teardown_params()
				raise e

			logger.info('Finished having run {} evaluations over {} iterations'.format(optimal.nfev, optimal.nit))
			self.tuner_params.update_using_opt_array(optimal.x)
			self.teardown_params()
			return self.tuner_params.nested_params

	def teardown_params(self):
		""" Performs the housekeeping on the parameters after the optimiser has run.
		"""
		self.to_save_params.log_output()
		if self.save_output:
			self.to_save_params.save_to_disk()

	@abstractmethod
	def _get_null_model_likelihood(self):
		return NotImplemented

	def minimise_me(self, opt_array):
		""" The cost function which maps optimiser parameter values to negative average log likelihood.
		"""
		self.tuner_params.update_using_opt_array(opt_array)
		params = self.tuner_params.all_params
		emll, pen_str = self.compute_emll(params)
		if np.isnan(emll):
			logger.info('Error running with params:')
			self.tuner_params.log_output()
			raise ValueError
		# null_likelihood = self._get_null_model_likelihood() if self.calc_null else None
		self.tuner_params.log_params_row(emll, pen_str) # , null_likelihood)

		# Only update the to_save params after we have logged the internal params so that we can see what we are
		# writing to file
		self.to_save_params.update_using_opt_array(opt_array)
		return -emll

	def minimize_args(self):
		kwargs = dict(
			fun=self.minimise_me,
			x0=self.tuner_params.x0,
			method=self.method,
			tol=self.tol,
			bounds=self.tuner_params.optimise_bounds
		)
		return kwargs

	@property
	def bounded_mode(self):
		try:
			return {
				'Nelder-Mead': False, 'Powell': False, 'CG': False, 'BFGS': False,
				'Newton-CG': False, 'L-BFGS-B': True, 'TNC': True, 'COBYLA': True,
				'SLSQP': False
			}[self.method]
		except KeyError:
			raise ValueError('Unknown optimiser method {}'.format(self.method))

	def penalise_boundaries(self, cost, params, pen_str='', scaling_value=1):
		""" Implements a rudimentary penalisation for use with optimisers like Nelder-Mead that cannot understand boundary
			conditions.

				:param cost:            Original cost returned by the optimiser
				:param params:          Current parameters being tested
				:param scaling_value:   Amount by which to penalise overstepping bounds
				:return:                Penalised cost
		"""
		for param, value in params.items():
			if param not in self.bounds_dict:
				continue
			bounds = self.bounds_dict[param]
			if bounds[0] is not None:
				if value < bounds[0]:
					penalty_factor = scaling_value * (bounds[0] - value)
					cost -= penalty_factor
					pen_str = '\t\t(penalising {}: {} < {} by a factor of {})'.format(param, value, bounds[0], penalty_factor)
			if bounds[1] is not None:
				if value > bounds[1]:
					penalty_factor = scaling_value * (value - bounds[1])
					cost -= penalty_factor
					pen_str = '\t\t(penalising {}: {} > {} by a factor of {})'.format(param, value, bounds[1], penalty_factor)
		return cost, pen_str

	# def penalise_constraints(self, cost, params, pen_str, scaling_value=1):
	# 	""" Rudimentary penalties for tier related constraints.
	# 	"""
	# 	surfaces = ('clay', 'hard', 'grass')
	# 	for s in surfaces:
	# 		first, second = params['db_ratings_1'][s], params['db_ratings_2'][s]
	# 		constraint_value = second - first
	# 		if constraint_value > 0:
	# 			penalty_factor = scaling_value * constraint_value
	# 			cost -= penalty_factor
	# 			pen_str = '\t\t(penalising {}: {} < {} by a factor of {})'.format(s, first, second, penalty_factor)
	# 	return cost, pen_str

	@abstractmethod
	def compute_emll(self, params):
		""" Should return the exp-mean-log-likelihood.
		"""
		return NotImplemented
