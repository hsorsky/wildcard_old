import numpy as np

from src.load.load import load
from src.models.player_percentages.player_ratings_backtest import PlayerRatingsBacktest
from src.logger import logger
from src.tuners.tuner import Tuner
from src.utils import CrazyParameters


class PlayerTuner(Tuner):

	init_params = (
		"player_assist_x0_gks",
		"player_assist_x0_def",
		"player_assist_x0_mid",
		"player_assist_x0_att",
		"player_assist_Q",
		"player_assist_P0",

		"player_goal_x0_gks",
		"player_goal_x0_def",
		"player_goal_x0_mid",
		"player_goal_x0_att",
		"player_goal_Q",
		"player_goal_P0",
	)

	def __init__(self, data, fixed_params, only_do, method, tol, use_multi_grad, save_output):
		super().__init__(data, fixed_params, only_do, method, tol, use_multi_grad, save_output)

		self.pids = data.player_id.values
		self.team_goals = data.player_team_goals_scored.values
		self.team_assists = data.player_team_assists.values
		self.player_goals = data.goals_scored.values
		self.player_assists = data.assists.values
		self.positions = data.position_id.values

	def compute_emll(self, params):
		try:
			bt = PlayerRatingsBacktest(
				params=params,
				pids=self.pids,
				player_goals=self.player_goals,
				player_assists=self.player_assists,
				team_goals=self.team_goals,
				team_assists=self.team_assists,
				positions=self.positions
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


def optimise_players(method='Nelder-Mead', only_do=[], fixed_params=[], tol=1e-7, use_multigrad=False):
	data = load()['all_player_data']

	tuner = PlayerTuner(data, fixed_params, only_do, method, tol, use_multigrad, save_output=True)

	tuner.run_tuner()

	return tuner


if __name__ == '__main__':
	optimise_players()