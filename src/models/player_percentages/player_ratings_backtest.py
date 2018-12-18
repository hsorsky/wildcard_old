import math

from src.models.player_percentages.player_ratings import PlayerGoalRatings, PlayerAssistRatings


class PlayerRatingsBacktest:
	# TODO: split goals and assists backtests as they're completely seperate (should speed up tuning)

	def __init__(self, params, pids, player_goals, player_assists, team_goals, team_assists, positions):
		self.params = params
		self.pids = pids
		self.player_goals = player_goals
		self.player_assists = player_assists
		self.team_goals = team_goals
		self.team_assists = team_assists
		self.positions = positions

		self.goal_ratings = PlayerGoalRatings(self.params)
		self.assist_ratings = PlayerAssistRatings(self.params)

	def run_backtest(self):
		for pid, player_goals, player_assists, team_goals, team_assists, player_position in \
				zip(self.pids, self.player_goals, self.player_assists, self.team_goals, self.team_assists, self.positions):

			if team_goals > 0:
				self.goal_ratings.run_update_step(pid, player_goals, team_goals, player_position)
				if team_assists > 0:
					self.assist_ratings.run_update_step(pid, player_assists, team_assists, player_position)

	@property
	def goal_prop(self):
		return self.goal_ratings.n_obs / (self.goal_ratings.n_obs + self.assist_ratings.n_obs)

	@property
	def assist_prop(self):
		return self.assist_ratings.n_obs / (self.goal_ratings.n_obs + self.assist_ratings.n_obs)

	@property
	def cost(self):
		goals_cost = self.goal_ratings.tot_log_lhood / self.goal_ratings.n_obs
		assits_cost = self.assist_ratings.tot_log_lhood / self.assist_ratings.n_obs
		cost = math.exp(self.goal_prop * goals_cost + self.assist_prop * assits_cost)
		return cost


