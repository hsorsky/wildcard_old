from abc import ABC, abstractmethod

import numpy as np
from collections import defaultdict


class PlayerRatings(ABC):

	def __init__(self, params):
		self.current_player_ratings = defaultdict(dict)
		self.historical_player_ratings = defaultdict(lambda: defaultdict(dict))
		self.params = params

		# -- hidden state variables -- #
		self.xk_minus = None
		self.xk = None
		self.Qk = None
		self.Pk_minus = None
		self.Pk = None

		# -- observation state variables -- #
		self.observations = None
		self.predictions = None
		self.yk = None
		self.Rk = None
		self.Hk = None
		self.Sk = None
		self.Kk = None

		# -- initial variables -- #
		self._initialise_params()

		self.tot_log_lhood = 0
		self.n_obs = 0

	@abstractmethod
	def _initialise_params(self):
		return NotImplemented

	def _update_current_ratings(self, pid, obs_time, rating, variance):
		"""
		Updates the PlayerRatings object's current ratings for the given player

		Parameters
		----------
		pid: int
			player id
		obs_time: any
			some sort of observation time identifier
		rating: float
			the rating we are updating
		variance: float
			the error variance of that rating

		Returns
		-------
		None

		"""

		self.current_player_ratings[pid]['rating'] = rating
		self.current_player_ratings[pid]['variance'] = variance
		self.current_player_ratings[pid]['last_obs'] = obs_time

	def _update_historical_ratings(self, pid, obs_time, rating, variance, r_type):
		"""
		Adds to the PlayerRatings object's historical ratings for the given player

		Parameters
		----------
		pid: int
			player id
		obs_time: any
			some sort of observation time identifier
		rating: float
			the rating we are updating
		variance: float
			the error variance of that rating
		r_type: str
			either 'posterior' or 'prior', indicating what type of rating is being recorded

		Returns
		-------
		None

		"""

		self.historical_player_ratings[pid][obs_time][r_type + '_rating'] = rating
		self.historical_player_ratings[pid][obs_time][r_type + '_variance'] = variance

	@abstractmethod
	def run_update_step(self, time, pids, hole_rating, observations):
		"""
		Performs the actual update step of the Kalman Filter

		Parameters
		----------
		time: np.datetime64[ns]
			time of the observation
		pids: ndarray
			1D numpy array, of type `int`, of player IDs
		hole_rating: float
			the previous estimate of the hole rating (difficulty)
		observations: ndarray
			1D numpy array of type `bool`, of the observations per player for the hole (i.e. did they GIR or not)

		Returns
		-------

		"""
		return NotImplemented

	@abstractmethod
	def _drift_variance(self, previous_variance, time_delta):
		"""
		Function that drifts P_{k|k-1} by scaling Q_k by the number of days since the last observation

		Parameters
		----------
		previous_variance: np.array
			a 1D numpy array of our previous estimate of error covariance, i.e. P_{k-1|k-1}
		time_delta

		Returns
		-------
		float
			our prior estimate of error covariance, P_{k|k-1}
		"""
		return NotImplemented

	def get_player_ratings(self, pids):
		return self._get_player_data(pids)[0]

	def _get_player_data(self, pids):
		"""
		Returns the current player data (rating, error covariance and last observation time).
		If the player does not exist yet, creates a new player using our initialisation values.
		"""
		player_ratings = np.full(len(pids), self.x0, dtype=float)
		player_variances = np.full(len(pids), self.P0, dtype=float)
		player_obs_times = np.full(len(pids), np.datetime64('2010-01-01'))
		for i, pid in enumerate(pids):
			if pid in self.current_player_ratings.keys():
				player_ratings[i] = self.current_player_ratings[pid]['rating']
				player_variances[i] = self.current_player_ratings[pid]['variance']
				player_obs_times[i] = self.current_player_ratings[pid]['last_obs']
		return player_ratings, player_variances, player_obs_times
