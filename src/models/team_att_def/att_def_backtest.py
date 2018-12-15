import math

import numpy as np
import pandas as pd

from src.load.load import load
from src.models.team_att_def.team_ratings import TeamRatings
from src.models.team_att_def.league_ratings import LeagueRatings
from tuners.tuner_params import load_params


class AttDefBacktest:

	def __init__(self, params, home_goals, away_goals, home_ids, away_ids, groupby_dict):
		self.home_goals = home_goals
		self.away_goals = away_goals
		self.home_ids = home_ids
		self.away_ids = away_ids
		self.team_ratings = TeamRatings(params)
		self.league_ratings = LeagueRatings(params)
		self.groupby_dict = groupby_dict
		self.groupby_list = sorted(self.groupby_dict.keys())

		self.cum_team_log_lhood = None
		self.n_team_obs = None
		self.cum_league_log_lhood = None
		self.n_league_obs = None

	def run_backtest(self):
		for gw in self.groupby_list:
			gw_ind = self.groupby_dict[gw]
			gw_h_goals = self.home_goals[gw_ind]
			gw_a_goals = self.away_goals[gw_ind]
			gw_h_ids = self.home_ids[gw_ind]
			gw_a_ids = self.away_ids[gw_ind]

			l_h, l_a, _, __ = self.league_ratings.get_ratings()

			home_att_ratings = []
			home_def_ratings = []
			away_att_ratings = []
			away_def_ratings = []

			for h_goals, a_goals, h_id, a_id in zip(gw_h_goals, gw_a_goals, gw_h_ids, gw_a_ids):
				h_att, h_def, a_att, a_def, h_att_var, h_def_var, a_att_var, a_def_var = \
					self.team_ratings.get_ratings(h_id, a_id)

				home_att_ratings.append(h_att)
				home_def_ratings.append(h_def)
				away_att_ratings.append(a_att)
				away_def_ratings.append(a_def)

				self.team_ratings.run_update_step(h_id, a_id, l_h, l_a, h_goals, a_goals, gw)

			home_att_ratings = np.array(home_att_ratings)
			home_def_ratings = np.array(home_def_ratings)
			away_att_ratings = np.array(away_att_ratings)
			away_def_ratings = np.array(away_def_ratings)

			self.league_ratings.run_update_step(
				home_att_ratings,
				home_def_ratings,
				away_att_ratings,
				away_def_ratings,
				gw_h_goals,
				gw_a_goals,
				gw
			)

		# -- store likelihoods -- #
		self.cum_team_log_lhood = self.team_ratings.tot_log_lhood
		self.n_team_obs = self.team_ratings.n_observations
		self.cum_league_log_lhood = self.league_ratings.tot_log_lhood
		self.n_league_obs = self.league_ratings.n_observations

	@property
	def team_prop(self):
		return self.n_team_obs / (self.n_team_obs + self.n_league_obs)

	@property
	def league_prop(self):
		return self.n_league_obs / (self.n_team_obs + self.n_league_obs)

	@property
	def cost(self):
		team_cost = self.cum_team_log_lhood/ self.n_team_obs
		league_cost = self.cum_league_log_lhood/ self.n_league_obs
		cost = math.exp(self.team_prop * team_cost + self.league_prop * league_cost)
		return cost


def run_backtest():
	data = load()['match_scores']
	home_ids = data.home_id.values
	away_ids = data.away_id.values
	home_goals = data.fthg.values
	away_goals = data.ftag.values

	groupby_dict = data.groupby('gw').indices

	params = load_params()

	bt = AttDefBacktest(
		params=params,
		home_goals=home_goals,
		away_goals=away_goals,
		home_ids=home_ids,
		away_ids=away_ids,
		groupby_dict=groupby_dict
	)

	bt.run_backtest()

	print(bt.cost)


if __name__ == '__main__':
	run_backtest()