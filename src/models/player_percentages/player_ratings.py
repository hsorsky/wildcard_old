import math
from collections import defaultdict
from abc import ABC, abstractmethod

class PlayerRatings:

	def __init__(self, params, pid_to_position_map):
		self.current_ratings = defaultdict(dict)
		self.historical_ratings = defaultdict(lambda: defaultdict(dict))
		self.params = params
		self.pid_pos_map = pid_to_position_map

		# -- hidden state variables -- #
		self.xk_minus = None
		self.xk = None
		self.Qk = None
		self.Pk_minus = None
		self.Pk = None

		# -- observation state variables -- #
		# self.observation = None
		self.prediction = None
		self.yk = None
		self.Rk = None
		self.Hk = None
		self.Sk = None
		self.Kk = None

		self.tot_log_lhood = 0
		self.n_obs = 0

	@abstractmethod
	def run_update_step(self, pid, obs, n_goals_or_assists):
		pass

	def _get_player_data(self, pid, position):
		try:
			rating = self.current_ratings[pid]['rating']
			var = self.current_ratings[pid]['variance']
		except KeyError:
			rating = {
				1: self.x0_gks,
				2: self.x0_def,
				3: self.x0_mid,
				4: self.x0_att,
			}[position]
			var = self.P0
		return rating, var

	def _update_current_ratings(self, pid, rating, var):
		self.current_ratings[pid]['rating'] = rating
		self.current_ratings[pid]['variance'] = var


class PlayerGoalRatings(PlayerRatings):

	def __init__(self, params, pid_to_position_map):
		super().__init__(params, pid_to_position_map)

		self.x0_gks = self.params['player_goal_x0_gks']
		self.x0_def = self.params['player_goal_x0_def']
		self.x0_mid = self.params['player_goal_x0_mid']
		self.x0_att = self.params['player_goal_x0_att']
		self.P0 = self.params['player_goal_P0']
		self.Q = self.params['player_goal_Q']

	def run_update_step(self, pid, obs, n_goals):
		position = self.pid_pos_map[pid]

		# -- predict -- #
		self.xk_minus, self.Pk_minus = self._get_player_data(pid, position)

		# -- save prior -- #
		pass  # TODO: implement

		# -- update -- #
		# self.Hk = players_team_att * opp_def * league_avg
		self.Hk = n_goals
		self.prediction = self.Rk = self.Hk * self.xk_minus
		self.yk = obs - self.prediction
		self.Sk = self.Hk * self.Pk_minus * self.Hk + self.Rk
		self.Kk = self.Pk_minus * self.Hk / self.Sk

		self.xk = self.xk_minus + self.Kk * self.yk
		self.Pk = (1 - self.Kk * self.Hk) * self.Pk_minus

		# -- save prior -- #
		pass  # TODO: implement
		self._update_current_ratings(pid, self.xk, self.Pk)

		# -- calc lhood -- #
		self.tot_log_lhood += -self.prediction + obs * math.log(self.prediction) - math.log(math.factorial(obs))
		self.n_obs += 1


class PlayerAssistRatings(PlayerRatings):

	def __init__(self, params, pid_to_position_map):
		super().__init__(params, pid_to_position_map)

		self.x0_gks = self.params['player_assist_x0_gks']
		self.x0_def = self.params['player_assist_x0_def']
		self.x0_mid = self.params['player_assist_x0_mid']
		self.x0_att = self.params['player_assist_x0_att']
		self.P0 = self.params['player_assist_P0']
		self.Q = self.params['player_assist_Q']

	def run_update_step(self, pid, obs, n_assists):
		position = self.pid_pos_map[pid]

		# -- predict -- #
		self.xk_minus, self.Pk_minus = self._get_player_data(pid, position)

		# -- save prior -- #
		pass  # TODO: implement

		# -- update -- #
		self.Hk = n_assists
		self.prediction = self.Rk = self.Hk * self.xk_minus
		self.yk = obs - self.prediction
		self.Sk = self.Hk * self.Pk_minus * self.Hk + self.Rk
		self.Kk = self.Pk_minus * self.Hk / self.Sk

		self.xk = self.xk_minus + self.Kk * self.yk
		self.Pk = (1 - self.Kk * self.Hk) * self.Pk_minus

		# -- save posterior -- #
		pass  # TODO: implement
		self._update_current_ratings(pid, self.xk, self.Pk)

		# -- calc lhood -- #
		self.tot_log_lhood += -self.prediction + obs * math.log(self.prediction) - math.log(math.factorial(obs))
		self.n_obs += 1





