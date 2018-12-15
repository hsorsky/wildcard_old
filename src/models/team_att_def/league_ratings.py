import math
from collections import defaultdict

import numpy as np

from utils import vcalc_poisson_lhood


class LeagueRatings:

	def __init__(self, params):
		self.current_ratings = {}
		# self.historical_ratings = defaultdict(dict)
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

		self.x0 = None
		self.Qk = self.params['league_rating_variance']
		# self.P0 = self.params['team_initial_error_var']

		# -- lhood tracking -- #
		self.tot_log_lhood = 0
		self.n_observations = 0

	def _update_current_ratings(self, home, away):
		self.current_ratings['home'] = home
		self.current_ratings['away'] = away

	def _update_historical_ratings(self, team_id, gameweek, attack, defence, variance, r_type, ishome):
		pass

	def get_ratings(self):
		try:
			home = self.current_ratings['home']
			away = self.current_ratings['away']
			home_var = self.current_ratings['home_variance']
			away_var = self.current_ratings['away_variance']
		except KeyError:
			home = self.params['league_home_init']
			away = self.params['league_away_init']
			home_var = self.params['league_home_variance_init']
			away_var = self.params['league_away_variance_init']

		return home, away, home_var, away_var

	def run_update_step(self, home_att, home_def, away_att, away_def, home_goals, away_goals, gw):
		l_h, l_a, l_h_var, l_a_var = self.get_ratings()

		# -- predict -- #
		self.xk_minus = np.array([l_h, l_a])
		self.Pk_minus = np.diag([l_h_var, l_a_var])

		# -- update -- #
		self.Hk = self._generate_Hk(home_att, home_def, away_att, away_def)
		self.predictions = np.dot(self.Hk, self.xk_minus)
		self.Rk = np.diag(self.predictions)
		self.observations = np.array([home_goals, away_goals]).T.ravel()
		self.yk = self.observations - self.predictions
		self.Sk = np.dot(np.dot(self.Hk, self.Pk_minus), self.Hk.T) + self.Rk
		self.Kk = np.dot(np.dot(self.Pk_minus, self.Hk.T), np.linalg.inv(self.Sk))

		self.xk = self.xk_minus + np.dot(self.Kk, self.yk)
		self.Pk = (np.eye(2) - np.dot(self.Kk, self.Hk)).dot(self.Pk_minus)

		log_lhoods = np.log(vcalc_poisson_lhood(self.predictions, self.observations))
		self.tot_log_lhood += np.sum(log_lhoods)
		self.n_observations += len(log_lhoods)

	def _generate_Hk(self, home_att, home_def, away_att, away_def):
		home_ratings = np.array([home_att * away_def, np.zeros(len(home_att))]).T.ravel()
		away_ratings = np.array([np.zeros(len(home_att)), home_def * away_att]).T.ravel()
		return np.array([home_ratings, away_ratings]).T

	@property
	def likelihood(self):
		return math.exp(self.tot_log_lhood / self.n_observations)

