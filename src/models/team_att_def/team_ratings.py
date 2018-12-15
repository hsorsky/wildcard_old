import math
from collections import defaultdict

import numpy as np

from utils import vcalc_poisson_lhood


class TeamRatings:

	def __init__(self, params):
		self.current_ratings = defaultdict(dict)
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
		self.Qk = self.params['team_rating_variance'] ** 2
		# self.P0 = self.params['team_initial_error_var']

		# -- lhood tracking -- #
		self.tot_log_lhood = 0
		self.n_observations = 0

	def _update_current_ratings(self, team_id, att_rat, def_rat, att_var, def_var, ishome):
		ha = 'h' if ishome else 'a'
		self.current_ratings[team_id]['{}_att_rating'.format(ha)] = att_rat
		self.current_ratings[team_id]['{}_def_rating'.format(ha)] = def_rat
		self.current_ratings[team_id]['{}_att_variance'.format(ha)] = att_var
		self.current_ratings[team_id]['{}_def_variance'.format(ha)] = def_var

	def _update_historical_ratings(self, team_id, gameweek, attack, defence, variance, r_type, ishome):
		pass

	def get_ratings(self, h_id, a_id):
		try:
			h_att = self.current_ratings[h_id]['h_att_rating']
			h_def = self.current_ratings[h_id]['h_def_rating']
			h_att_var = self.current_ratings[h_id]['h_att_variance']
			h_def_var = self.current_ratings[h_id]['h_def_variance']
		except KeyError:
			h_att = 1  # self.params['team_initial_home_att_rating']
			h_def = 1  # self.params['team_initial_home_def_rating']
			h_att_var = self.params['team_initial_home_att_rating_var']
			h_def_var = self.params['team_initial_home_def_rating_var']

		try:
			a_att = self.current_ratings[a_id]['a_att_rating']
			a_def = self.current_ratings[a_id]['a_def_rating']
			a_att_var = self.current_ratings[a_id]['a_att_variance']
			a_def_var = self.current_ratings[a_id]['a_def_variance']
		except KeyError:
			a_att = 1  # self.params['team_initial_away_att_rating']
			a_def = 1  # self.params['team_initial_away_def_rating']
			a_att_var = self.params['team_initial_away_att_rating_var'] ** 2
			a_def_var = self.params['team_initial_away_def_rating_var'] ** 2

		return h_att, h_def, a_att, a_def, h_att_var, h_def_var, a_att_var, a_def_var

	def run_update_step(self, h_id, a_id, l_h, l_a, h_goals, a_goals, gw):
		h_att, h_def, a_att, a_def, h_att_var, h_def_var, a_att_var, a_def_var = self.get_ratings(h_id, a_id)

		# -- predict -- #
		self.xk_minus = np.array([h_att, h_def, a_att, a_def])
		self.Pk_minus = np.diag(np.array([h_att_var, h_def_var, a_att_var, a_def_var]) + self.Qk)

		# -- update -- #
		self.predictions = self._predict(l_h, l_a)
		self.Rk = np.diag(self.predictions)
		self.observations = np.array([h_goals, a_goals])
		self.yk = self.observations - self.predictions

		self.Hk = self._generate_Hk(l_h, l_a)
		self.Sk = self.Hk.dot(self.Pk_minus).dot(self.Hk.T) + self.Rk
		self.Kk = self.Pk_minus.dot(self.Hk.T).dot(np.linalg.inv(self.Sk))

		self.xk = self.xk_minus + self.Kk.dot(self.yk)
		self.Pk = (np.eye(4) - self.Kk.dot(self.Hk)).dot(self.Pk_minus)

		h_att, h_def, a_att, a_def = self.xk
		h_att_var, h_def_var, a_att_var, a_def_var = np.diag(self.Pk)

		self._update_current_ratings(
			team_id=h_id,
			att_rat=h_att,
			def_rat=h_def,
			att_var=h_att_var,
			def_var=h_def_var,
			ishome=True
		)
		self._update_current_ratings(
			team_id=a_id,
			att_rat=a_att,
			def_rat=a_def,
			att_var=a_att_var,
			def_var=a_def_var,
			ishome=False
		)

		log_lhoods = np.log(vcalc_poisson_lhood(self.predictions, self.observations))
		self.tot_log_lhood += np.sum(log_lhoods)
		self.n_observations += len(log_lhoods)

	def _predict(self, l_h, l_a):
		h_att, h_def, a_att, a_def = self.xk_minus
		return np.array([
			h_att * l_h * a_def,
			h_def * l_a * a_att,
		])

	def _generate_Hk(self, l_h, l_a):
		h_att, h_def, a_att, a_def = self.xk_minus
		return np.array([
			[l_h * a_def, 0, 0, l_h * h_att],
			[0, l_a * a_att, l_a * h_def, 0]
		])

	@property
	def likelihood(self):
		return math.exp(self.tot_log_lhood / self.n_observations)



