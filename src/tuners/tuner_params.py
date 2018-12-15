import collections
import json

import numpy as np

from params.league_kf_params import league_kf_params
from params.team_kf_params import team_kf_params
from src.logger import logger
from src.paths import paths
from src.utils import write_nested_dict_to_file

PARAM_ACRONYMS = {
	"gir_player": "gir_player",
	"gir_hole": "gir_hole",
	"dd_player": "dd_player",
	"dd_hole": "dd_hole",
}

LOGGER_ABBREVIATIONS = {
	"team": "tm",
	"league": "lg",
	"player": "pl",
	"away": "a",
	"home": "h",
	"initial": "init",
	"_rating": "",
	"variance": "var"
}


class TunerParams:

	def __init__(self, init_params, fixed_params, only_do):
		""" Manipulates parameters to be used by a tuners. They are stored in two forms:

				self.flat_params:       Flattened dictionary with long unique names for each parameter
				self.nested_params:     Nested dictionary which looks nicer, used for writing to file
										and by the Kalman filter
		"""

		self.initial_params = init_params

		self.only_do = only_do

		self.fixed_params, self.optimise_params = self.validate_args(init_params, fixed_params, only_do)

		params = flatten_dict(load_params())
		self.flat_params = {k: v for k, v in params.items() if k in self.initial_params}
		self.x0 = self.opt_value_array()

	def validate_args(self, init_params, fixed, only_do):
		""" Validates the TunerParams arguments, parses only_do and returns updated fixed_params.
		"""
		for p in fixed:
			assert p in init_params, \
				'fixed_params contains: {} which does not belong in initial params: {}'.format(p, init_params)
		for p in only_do:
			assert p in init_params, \
				'only_do contains: {} which does not belong in initial params: {}'.format(p, init_params)

		if only_do:
			assert not fixed, 'Supplied fixed_params and only_do, make one of these empty'
			fixed = [x for x in init_params if x not in only_do]
		opt = [x for x in init_params if x not in fixed]
		return fixed, opt

	@property
	def nested_params(self):
		""" Creates a nested dict version of the internal params.
		"""
		return unflatten_dict(self.flat_params)

	def __copy__(self):
		return TunerParams(self.initial_params, self.fixed_params, only_do=[])

	def _get_optimise_variables(self, fixed):
		output = {k: v for k, v in self.flat_params.items() if k not in fixed}
		if not output:
			raise ValueError('You have tried to run the Kalman tuners with every variable fixed')
		return output

	def log_initial(self):
		""" Logs the header row of the tuners output. Applies the abbreviations defined in LOG_ABBREVIATIONS.
		"""

		def abbrev(x):
			for k, v in LOGGER_ABBREVIATIONS.items():
				x = x.replace(k, v)
			return x.replace('__', '_')

		keys = list(self.flat_params.keys())
		logger.info('Out of possible parameters:')
		logger.info('\t\t{}'.format(keys))
		logger.info('keeping the following fixed:')
		logger.info('\t\t{}'.format(self.fixed_params))

		title_string = ' '.join(['{:>18}'] * len(keys)).format(
			*[abbrev(x) + ' (f)' if x in self.fixed_params else abbrev(x) for x in keys])
		title_string += '{:>20}'.format('L\'hood')
		logger.info('')
		logger.info('')
		logger.info(title_string)
		logger.info('')

	def get_bounds_dict(self):
		""" If a parameter doesn't exist it is defaulted to (None, None).
		"""
		bd = dict(

		)
		for what in (		# non-negative

		):
			bd[what] = (1e-7, None)
		for what in (		# in (0,1)

		):
			bd[what] = (1e-7, 1 - 1e-7)
		return bd

	@property
	def optimise_bounds(self):
		""" The bounds for the parameters being optimised. Stored as a list of tuples of
			upper and lower bounds.
		"""
		return [self.get_bounds_dict().get(k, (None, None)) for k in self.optimise_params]

	def update_using_opt_array(self, opt_params_array):
		""" Updates the internal params using a params array.
		"""
		for opt_key, opt_value in zip(self.optimise_params, opt_params_array):
			self.flat_params[opt_key] = opt_value

	def opt_value_array(self):
		""" Creates an array of values associated to the optimise parameters.
		"""
		output_list = []
		for k in self.optimise_params:
			if k not in self.fixed_params:
				output_list.append(self.flat_params[k])
		return np.array(output_list)

	def log_output(self):
		""" Logs the current output_params to st_out.
		"""
		pretty_output = json.dumps(self.nested_params, sort_keys=False, indent=4, separators=(',', ': '))
		print(pretty_output)

	def save_to_disk(self):
		""" Saves the internal array of params to a python dictionary.
		"""
		while True:
			response = input('Would you like to write optimal parameters to file? (y/n)')
			if response in 'yn':
				break
		if response == 'y':
			logger.info('Writing optimal parameters to their respective files')

			flat_params = flatten_dict(load_params())
			flat_params.update(self.flat_params)
			nested_params = unflatten_dict(flat_params)
			save_params(nested_params)

	def log_params_row(self, likelihood, pen_str):
		""" Logs the likelihood corresponding to the current set of parameter values.
		"""
		logging_string = ' '.join(['{:18.3g}'] * len(self.flat_params))
		likelihood_string = '{:>20.7f}'.format(likelihood)
		# if null_likelihood is not None:
		# 	likelihood_string += ' {:>20.7f}'.format(likelihood - null_likelihood)
		logger.info(logging_string.format(*list(self.flat_params.values())) + likelihood_string + pen_str)

	@property
	def all_params(self):
		""" Returns a nested dictionary of the tuners params combined with those from file.
			Used by Tuner.minimise_me() to prepare to run whichever cost function it needs
			to.
		"""
		all_params = load_params()
		flat_params = flatten_dict(all_params)
		flat_params.update(self.flat_params)
		return unflatten_dict(flat_params)


def load_params():
	""" Loads all the params from all the parameter files for a particular gender.
	"""
	team_kf = team_kf_params
	league_kf = league_kf_params
	output = {**team_kf, **league_kf}
	assert len(output) == len(team_kf) + len(league_kf), \
		'Duplicate parameter values found in different parameter files'
	return output


def save_params(all_params, sig_figs=5):
	""" Saves all params to their correct respective python files.
	"""
	ramified_params = ramify_params(all_params)
	for pricer_acronym, params in ramified_params.items():
		name = '{}_params'.format(pricer_acronym)
		write_nested_dict_to_file(paths[name], name, params, sig_figs)


def ramify_params(params):
	""" Splits a dictionary of params into smaller dictionaries, which each correspond
		to their logical location on file.
	"""
	output = collections.defaultdict(collections.OrderedDict)
	for k, v in params.items():
		matched = False
		for acronym, full_name in PARAM_ACRONYMS.items():
			a_len = len(acronym)
			if k[:a_len] == acronym:
				if matched:
					raise ValueError('Parameter {} matches two different acronyms'.format(k))
				output[full_name][k] = v
				matched = True
		if not matched:
			raise ValueError('Parameter {} matches no acronym'.format(k))
	return output


def flatten_dict(nested_dict, flattening_key='.'):
	""" Converts a two level nested dictionary into a flat dictionary.
	"""
	output = {}
	for k, v in nested_dict.items():
		if isinstance(v, dict):
			for k2, v2 in v.items():
				if isinstance(v2, dict):
					raise NotImplementedError('Cannot flatten triple nested dicts')
				flat_key = k + flattening_key + k2
				output[flat_key] = v2
		else:
			output[k] = v
	return output


def unflatten_dict(flat_dict, flattening_key='.'):
	""" Converts a flattened dict back into its nested form.
	"""
	output = {}
	for k, v in flat_dict.items():
		if flattening_key in k:
			split = k.split(flattening_key)
			assert len(split) == 2, 'flattening key found twice in {}'.format(k)
			k1, k2 = split
			output.setdefault(k1, {})[k2] = v
		else:
			output[k] = v
	return output


