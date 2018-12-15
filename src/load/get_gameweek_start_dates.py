import datetime
import src.utils as utils

def get_gameweek_to_deadline_map(json):
	gameweek_to_deadline_map = {}
	for gameweek_json_i in json['events']:
		gameweek_number = int(gameweek_json_i['id'])
		gameweek_deadline_str = gameweek_json_i['deadline_time']
		gameweek_deadline_dt = datetime.datetime.strptime(gameweek_deadline_str, "%Y-%m-%dT%H:%M:%SZ")
		gameweek_to_deadline_map[gameweek_number] = gameweek_deadline_dt
	return gameweek_to_deadline_map


def get_gameweek_to_date_map(gameweek_to_deadline_map):
	gameweek_to_date_map = {}
	for gameweek, date_time in gameweek_to_deadline_map.items():
		date = date_time.date()
		gameweek_to_date_map[gameweek] = date
	return gameweek_to_date_map


def get_date_to_gamewek_map(gameweek_to_date_map):
	date_to_gamewek_map = utils.reverse_dict(gameweek_to_date_map)
	return date_to_gamewek_map


if __name__ == '__main__':

	pass