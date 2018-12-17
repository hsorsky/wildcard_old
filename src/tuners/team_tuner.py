import numpy as np

from src.load.load import load
from src.models.team_ratings.team_ratings_backtest import TeamRatingsBacktest
from src.logger import logger
from src.tuners.tuner import Tuner
from src.utils import CrazyParameters


class TeamTuner(Tuner):

	init_params = (
		"league_home_init",
		"league_away_init",
		"league_home_variance_init",
		"league_away_variance_init",
		"league_rating_variance",

		"team_initial_home_att_rating_var",
		"team_initial_home_def_rating_var",
		"team_initial_away_att_rating_var",
		"team_initial_away_def_rating_var",
		"team_rating_variance",
	)

	def __init__(self, data, fixed_params, only_do, method, tol, use_multi_grad, save_output):
		super().__init__(data, fixed_params, only_do, method, tol, use_multi_grad, save_output)

		self.home_ids = data.home_id.values
		self.away_ids = data.away_id.values
		self.home_goals = data.fthg.values
		self.away_goals = data.ftag.values

		self.groupby_dict = data.groupby('gw').indices

	def compute_emll(self, params):
		try:
			bt = TeamRatingsBacktest(
				params=params,
				home_goals=self.home_goals,
				away_goals=self.away_goals,
				home_ids=self.home_ids,
				away_ids=self.away_ids,
				groupby_dict=self.groupby_dict
			)
			bt.run_backtest()
		except CrazyParameters:
			logger.info('Following params produced math error, change param bounds!!')
			for k, v in params.items():
				logger.info('\t\t{:20} {}'.format(k, v))

			raise ValueError

		cost = bt.cost

		# -- update cost wrt boundaries -- #
		cost, pen_str = self.penalise_boundaries(cost, params, pen_str='')
		return cost, pen_str

	def _get_null_model_likelihood(self):
		return np.NaN


def optimise_teams(method='Nelder-Mead', only_do=[], fixed_params=[], tol=1e-7, use_multigrad=False):
	data = load()['match_scores']

	tuner = TeamTuner(
		data=data,
		fixed_params=fixed_params,
		only_do=only_do,
		method=method,
		tol=tol,
		use_multi_grad=use_multigrad,
		save_output=True
	)

	tuner.run_tuner()

	return tuner


if __name__ == '__main__':
	optimise_teams()