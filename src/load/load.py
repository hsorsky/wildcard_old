import json
import time

import pandas as pd
import numpy as np

from src.models.team_ratings.team_ratings_backtest import TeamRatingsBacktest
from src.load import get_gameweek_start_dates
from src.utils import camel_to_snake, timer
from src.get_id_name_mappings import generate_id_name_mappings

from config import MATCH_DATA_PATH, FPL_DATA_PATH
from src.tuners.tuner_params import load_params


class Loader:

	def __init__(self, add_team_ratings):
		self.data = {}
		self.maps = {}
		self.add_team_ratings = add_team_ratings
		# TODO implement cache path

	def run_loader(self):
		# TODO implement search of cache for data
		with timer("loading fpl summary", __file__):
			self.load_fpl_summary()

		with timer("adding maps", __file__):
			self.add_maps()

		with timer("loading scores", __file__):
			self.load_match_scores()

		with timer("adding player IDs", __file__):
			self.add_player_id_list()

		with timer("loading player data", __file__):
			self.load_player_data()

		with timer("adding team IDs", __file__):
			self.add_player_team_id_to_player_data()

		# TODO: implement player % filtering

		if self.add_team_ratings:
			with timer("calculating attack and defence scores", __file__):
				self.add_att_def_scores_to_data()
			with timer("merging att def scores", __file__):
				self.merge_att_def_ratings_to_all_player_data()

		return self.data

	def merge_att_def_ratings_to_all_player_data(self):

		self.data['all_player_data'] = pd.merge(
			self.data['all_player_data'],
			self.data['team_att_def_scores'].add_prefix('opposition_'),
			left_on=['gameweek', 'opponent_team_id'],
			right_on=['opposition_gameweek', 'opposition_team_id'],
			how='left'
		)

		self.data['all_player_data'] = pd.merge(
			self.data['all_player_data'],
			self.data['team_att_def_scores'].add_prefix('player_team_'),
			left_on=['gameweek', 'player_team_id'],
			right_on=['player_team_gameweek', 'player_team_team_id'],
			how='left'
		)

		cols = [c for c in self.data['all_player_data'].columns if 'posterior' not in c]
		self.data['all_player_data'] = self.data['all_player_data'][cols]

	def get_att_def_ratings(self):
		pass

	def load_fpl_summary(self):
		# import data
		with open(FPL_DATA_PATH + '2017_18_Data/main_JSON.json', 'r') as file:
			data = json.load(file)

		self.data['fpl_summary_json'] = data


	def load_match_scores(self):
		# import data
		match_scores = pd.read_csv(MATCH_DATA_PATH + 'E0.csv')

		# remove unwanted columns
		COLS_TO_KEEP = ['Date', 'HomeTeam', 'AwayTeam', 'FTHG', 'FTAG']
		match_scores = match_scores[COLS_TO_KEEP]

		# rename columns
		match_scores.columns = list(map(camel_to_snake, match_scores.columns))
		renaming_map = {
			'home_team': 'home_name',
			'away_team': 'away_name',
		}
		match_scores = match_scores.rename(renaming_map, axis=1)

		# add in IDs
		match_scores['home_id'] = match_scores.home_name.map(self.maps['team_to_id'])
		match_scores['away_id'] = match_scores.away_name.map(self.maps['team_to_id'])

		# add in gameweeks
		match_scores.date = pd.to_datetime(match_scores.date, format='%d/%m/%y')
		match_scores['gw'] = match_scores.date.astype(str).map(self.maps['date_to_gw'])

		self.data['match_scores'] = match_scores

	def load_player_data(self):
		list_all_player_data = []

		for player_id_i in self.player_id_list:
			with open(FPL_DATA_PATH + '2017_18_Data/' + '{}.txt'.format(player_id_i)) as file:
				player_i_data_df = pd.DataFrame(json.load(file)['history'])
				list_all_player_data.append(player_i_data_df)

		all_player_data_df = pd.concat(list_all_player_data, ignore_index=True)
		renaming_map = {
			"element": 'player_id',
			"bonus": 'bonus_points',
			"offside": 'offsides',
			"opponent_team": 'opponent_team_id',
			"round": 'gameweek',
			"team_a_score": 'away_team_goals',
			"team_h_score": 'home_team_goals',
			"transfers_balance": 'transfers_net',
		}
		all_player_data_df = all_player_data_df.rename(renaming_map, axis=1)

		all_player_data_df['player_team_goals_scored'] = np.where(
			all_player_data_df['was_home'],
			all_player_data_df['home_team_goals'],
			all_player_data_df['away_team_goals']
		)
		all_player_data_df['opposition_team_goals_scored'] = np.where(
			~all_player_data_df['was_home'],
			all_player_data_df['home_team_goals'],
			all_player_data_df['away_team_goals']
		)

		all_player_data_df.kickoff_time = pd.to_datetime(all_player_data_df.kickoff_time, format='%Y-%m-%dT%H:%M:%SZ')
		all_player_data_df.kickoff_time = pd.to_datetime(all_player_data_df.kickoff_time.dt.date)		# strip times out


		self.data['all_player_data'] = all_player_data_df

	def add_maps(self):
		self.maps['gw_to_deadline'] = get_gameweek_start_dates.get_gameweek_to_deadline_map(self.data['fpl_summary_json'])
		temp_series = pd.Series(self.maps['gw_to_deadline'], name='Date')
		temp_series = (
			pd.to_datetime(temp_series.dt.date, format='%Y-%m-%d')
				.reset_index()
				.set_index('Date')
				.asfreq('D', method='ffill')
		)
		temp_series.index.freq = None
		temp_series.index = temp_series.index.astype(str)
		self.maps['date_to_gw'] = temp_series.to_dict()['index']
		self.maps['team_to_id'], \
		self.maps['id_to_team'] = generate_id_name_mappings(self.data['fpl_summary_json'])

	def add_att_def_scores_to_data(self):
		data = self.data['match_scores']
		home_ids = data.home_id.values
		away_ids = data.away_id.values
		home_goals = data.fthg.values
		away_goals = data.ftag.values

		groupby_dict = data.groupby('gw').indices

		params = load_params()

		bt = TeamRatingsBacktest(
			params=params,
			home_goals=home_goals,
			away_goals=away_goals,
			home_ids=home_ids,
			away_ids=away_ids,
			groupby_dict=groupby_dict
		)

		bt.run_backtest()

		historical_ratings = bt.team_ratings.historical_ratings
		historical_ratings = (
			pd.DataFrame.from_dict(historical_ratings).T
				.fillna(method='ffill')
				.reset_index()
				.rename({
					"level_0": "team_id",
					"level_1": "gameweek"
			}, axis=1)
		)

		self.data['team_att_def_scores'] = historical_ratings

	def add_player_id_list(self):
		self.player_id_list = pd.DataFrame(self.data['fpl_summary_json']['elements']).id.unique()

	def add_player_team_id_to_player_data(self):
		date_and_opponent_map_1 = self.data['match_scores'].set_index(['date', 'home_id'])['away_id'].to_dict()
		date_and_opponent_map_2 = self.data['match_scores'].set_index(['date', 'away_id'])['home_id'].to_dict()
		date_and_opponent_map = {**date_and_opponent_map_1, **date_and_opponent_map_2}
		temp_df = self.data['all_player_data']
		self.data['all_player_data']['player_team_id'] = list(map(date_and_opponent_map.get, list(zip(temp_df.kickoff_time, temp_df.opponent_team_id))))

	def get_player_team_id(self, match_date, opponent_id):
		subset_df = self.data['match_scores']
		subset_df = subset_df[subset_df.date == match_date]
		for match_row in subset_df.itertuples():
			if match_row.home_id == opponent_id:
				return match_row.away_id
			elif match_row.away_id == opponent_id:
				return match_row.home_id
		return np.NaN

	def add_home_away_ids(self):
		self.data['all_player_data']['home_team_id'] = np.where(
			self.data['all_player_data']['was_home'],
			self.data['all_player_data']['player_team_id'],
			self.data['all_player_data']['opponent_team_id']
		)
		self.data['all_player_data']['away_team_id'] = np.where(
			~self.data['all_player_data']['was_home'],
			self.data['all_player_data']['player_team_id'],
			self.data['all_player_data']['opponent_team_id']
		)


def load():
	loader = Loader()
	loader.run_loader()
	return loader.data

if __name__ == '__main__':
	load()