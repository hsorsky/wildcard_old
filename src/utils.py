import re
import scipy

import contextlib
import datetime
import hashlib
import math
import multiprocessing
import os
import time

import numpy as np
import scipy.optimize

from config import project_directory, BURN_IN_RATIO
from src.logger import logger, validate_path, TimerLogger


@contextlib.contextmanager
def timer(msg='', file=None):
	""" Log how long something takes to run.
	"""
	if file is None:
		raise ValueError('Must pass __file__ to timer')
	module = file.split('/')[-1].split('.')[0]
	timer_logger = TimerLogger()
	if msg:
		timer_logger.info(msg, module)
	took = -time.time()
	yield
	took += time.time()
	timer_logger.info('{} took {:.2f}s'.format(msg, took), module)


def filter_data(data, msg, keep_condition=None, drop_indices=None, keep_indices=None, quiet_mode=False):
	""" Filters a pandas data frame and logs the result with 'msg'. This runs in two modes,
		either supply a condition under which data is to be *kept*, or supply a list of indices
		to drop.
	"""
	n_before = len(data)
	if drop_indices is not None:
		assert keep_condition is None
		assert keep_indices is None
		data = data.drop(drop_indices)
	elif keep_indices is not None:
		assert keep_condition is None
		assert drop_indices is None
		data = data.loc[keep_indices]
	else:
		data = data[keep_condition]
	n_after = len(data)
	if not quiet_mode:
		logger.info('\t\tRemoved {} data ({:.1f}%) which '.format(
			n_before - n_after,
			100 * (n_before - n_after) / n_before
		) + msg)
	data.is_copy = False

	return data


def is_variance_matrix(A, allow_zero_e_values=False):
	""" Tests to see if a real matrix is a variance matrix by first testing if it is symmetric and then
		that its eigenvalues are all non-negative.
	"""
	if len(A.shape) != 2:
		raise ValueError('Matrix is not two dimensional: shape is {}'.format(A.shape))

	if A.shape[0] != A.shape[1]:
		raise ValueError('Matrix is not square: shape is {}'.format(A.shape))

	symm_diff = np.abs(A - A.transpose())

	if (symm_diff > 1e-5).any():
		logger.info('Matrix is not symmetric: {}'.format(A))
		return False

	e_values = np.linalg.eigvals(A)

	if (e_values < 0).any():
		logger.info('Negative e-value found {}'.format(e_values))
		return False
	elif (e_values == 0).any() and not allow_zero_e_values:
		logger.info('Zero e-value found {}'.format(e_values))
		return False
	else:
		logger.info('All e-values are positive {}'.format(e_values))
		return True


def md5(fname):
	hash_md5 = hashlib.md5()
	with open(fname, "rb") as f:
		for chunk in iter(lambda: f.read(4096), b""):
			hash_md5.update(chunk)
	return hash_md5.hexdigest()


def logit_float(x):
	return math.log(x / (1 - x))


def logit_np_array(x):
	return np.log(np.divide(x, 1 - x))


def logistic_float(x):
	return 1 / (1 + math.exp(-x))


def logistic_np_array(x):
	return np.divide(1, 1 + np.exp(-x))


def delete_files_in_folder(folder, extension):
	""" Deletes all .h5 files in a folder.
	"""

	assert isinstance(extension, str)
	assert 1 < len(extension) < 4

	if extension[0] != '.':
		extension = '.' + extension

	if folder[-1] != '/':
		folder += '/'

	deleted = []
	for root, dirs, filenames in os.walk(folder):
		if root[-1] != '/':
			root += '/'
		for f in filenames:
			if f[-3:] == extension:
				full_path = root + f
				deleted.append(full_path)
				os.remove(full_path)

	if deleted:
		logger.info('Deleted {} files with the extension {}:'.format(len(deleted), extension))
		for d in deleted:
			logger.info('\t\t{}'.format(d))
	else:
		logger.info('Did not delete any files in folder {}'.format(folder))


def filter_burn_in(data, filter_tier):
	""" Filters out data corresponding to a burn in period as defined by config.BURN_IN_RATIO.
		If include_tier is True then also filters out lower tier data.
	"""
	date_ordinals = data.date_ordinal.values
	last_burn_in_ordinal = date_ordinals[int(len(date_ordinals) * BURN_IN_RATIO)]
	logger.info('Burn in date threshold is {}'.format(datetime.date.fromordinal(last_burn_in_ordinal)))
	keep_indices = (data.date_ordinal > last_burn_in_ordinal)
	if filter_tier:
		keep_indices & (data.tier == 1)
	data = filter_data(
		data,
		'correspond to lower tier and burn in data',
		keep_indices
	)
	logger.info('We now have {} matches between {} and {}'.format(len(data.match_id.unique()), data.date.iloc[0], data.date.iloc[-1]))
	return data


def multioptimiser(fun, bounds, method, x0, tol):
	return scipy.optimize.minimize(fun, jac=lambda x: _jac_multicore(x, fun), bounds=bounds, method=method, x0=x0, tol=tol)


def _jac_multicore(x, fun):
	n_cores = len(x)
	manager = multiprocessing.Manager()
	shared_memory_dict = manager.dict()
	epsilon = 1e-5

	f0 = fun(x)
	grad = np.zeros((len(x),), float)
	ei = np.zeros((len(x),), float)

	jobs = []
	for i in range(n_cores):
		ei[i] = 1.
		d = epsilon * ei
		job = multiprocessing.Process(
			target=_grad_multicore,
			args=(fun, x, d, f0, i, shared_memory_dict)
		)
		jobs.append(job)
		job.start()
		ei[i] = 0.

	for job in jobs:
		job.join()

	for k, g in shared_memory_dict.items():
		grad[k] = g
	return grad


def _grad_multicore(f, x, d, f0, batch_number, shared_memory_dict):
	shared_memory_dict[batch_number] = (f(x + d) - f0) / d[batch_number]
	return shared_memory_dict


def test_for_nans(data, name):
	if not len(data):
		raise ValueError('Data is empty!')
	nan_rows = data[data.isnull().values.any(axis=1)]

	if not len(nan_rows):
		logger.info('\t\tNo Nans found in {}'.format(name))
		return

	logger.info('\t\tFound {} rows ({:.1f}%) and {} columns (out of {}) with Nans in {}'.format(
		len(nan_rows),
		100 * len(nan_rows) / len(data),
		data.isnull().values.any(axis=0).sum(),
		len(data.columns),
		name
	))

	nan_counts = {}
	for source, source_data in data.groupby('source'):
		source_nans = {}
		for column in source_data.columns:
			col_nans = source_data[column].isnull().values.sum()
			if col_nans:
				source_nans[column] = col_nans
		nan_counts[source] = source_nans

	for k, v in nan_counts.items():
		logger.error('\t\tFor source {}, found NaNs in {} columns:'.format(k, len(v)))
		for l, c in v.items():
			logger.error(
				'\t\t\t{:20} {:8} ({:.1f}%)'.format(l, c, 100 * c / len(data)))

	raise ValueError


def save_plot(figure, name, extension='pdf'):
	now = datetime.datetime.now()
	path = project_directory + '/analysis/plots/{}/{}-{}-{}_{}:{}:{}.{}'.format(
		name,
		now.year,
		now.month,
		now.day,
		now.hour,
		now.minute,
		now.second,
		extension
	)
	validate_path(path)
	figure.savefig(path)
	logger.info('Saved plot to {}'.format(path))


def check_data_for_duplicates(data, column):
	""" Checks a dataframe with a match_id column for duplicates.
	"""
	duplicates = data[getattr(data, column).duplicated(keep=False)]
	if len(duplicates):
		raise ValueError('Duplicates found in {} data, {} out of {} ({:.1f}%)'.format(
			column,
			len(duplicates),
			len(data),
			100 * len(duplicates) / len(data)
		))


def write_nested_dict_to_file(path, dict_name, d, sig_figs):
	""" Writes a nested dictionary to file, rounding values to sig_fig significant figures.
		If make_ordered is True, will import collections and make the dict an ordered dict.
	"""

	def _write_sub_dict(file, name, sub_dict, level, sig_figs):
		""" Writes a dictionary and, recursively, all nested dictionaries
			to the provided file.

			Also rounds all values to sig_figs number of significant figures.
		"""
		if level:
			file.write('\t' * level + '"{}": '.format(name) + '{\n')
		else:
			file.write('{} = '.format(name) + '{\n')

		for k, v in sub_dict.items():
			if isinstance(v, dict):
				_write_sub_dict(file, k, v, level + 1, sig_figs)
			else:
				_write_level_item(file, k, v, level + 1, sig_figs)

		if level:
			file.write('\t' * level + '},\n')
		else:
			file.write('}\n')

	def _write_level_item(file, key, value, level, sig_figs):
		""" Writes the key:value pair with the correct formatting to fit the nesting level.
		"""
		file.write('\t' * level + '"{0}": {1:.{2}g},\n'.format(key, value, sig_figs))

	with open(path, 'w') as file:
		_write_sub_dict(file, dict_name, d, 0, sig_figs)


def clip_float(value, min_value, max_value):
	return max(min(value, max_value), min_value)


class CrazyParameters(Exception):
	pass


def residual_resample(weights):
	N = len(weights)
	indexes = np.zeros(N, 'i')

	# take int(N*w) copies of each weight, which ensures particles with the
	# same weight are drawn uniformly
	num_copies = (np.floor(N*weights)).astype(int)
	k = 0
	for i in range(N):
		for _ in range(num_copies[i]): # make n copies
			indexes[k] = i
			k += 1

	# use multinormal resample on the residual to fill up the rest. This
	# maximizes the variance of the samples
	residual = weights - num_copies     # get fractional part
	residual /= sum(residual)           # normalize
	cumulative_sum = np.cumsum(residual)
	cumulative_sum[-1] = 1. # avoid round-off errors: ensures sum is exactly one
	indexes[k:N] = np.searchsorted(cumulative_sum, np.random.random(N-k))

	return indexes


def stratified_resample(weights):
	N = len(weights)
	# make N subdivisions, and chose a random position within each one
	positions = (np.random.random(N) + np.arange(N)) / N

	indexes = np.zeros(N, 'i')
	cumulative_sum = np.cumsum(weights)
	i, j = 0, 0
	while i < N:
		if positions[i] < cumulative_sum[j]:
			indexes[i] = j
			i += 1
		else:
			j += 1
	return indexes


def systematic_resample(weights):
	N = len(weights)

	# make N subdivisions, and choose positions with a consistent random offset
	positions = (np.random.random() + np.arange(N)) / N

	indexes = np.zeros(N, 'i')
	cumulative_sum = np.cumsum(weights)
	i, j = 0, 0
	while i < N:
		if positions[i] < cumulative_sum[j]:
			indexes[i] = j
			i += 1
		else:
			j += 1
	return indexes


def reverse_dict(forward_dict):
	return dict((reversed(item) for item in forward_dict.items()))


def camel_to_snake(CamelString):
	s1 = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', CamelString)
	return re.sub('([a-z0-9])([A-Z])', r'\1_\2', s1).lower()


def calc_poisson_lhood(lamda, k):
	return (math.exp(-lamda) * lamda ** k) / math.factorial(k)


vcalc_poisson_lhood = np.vectorize(calc_poisson_lhood)