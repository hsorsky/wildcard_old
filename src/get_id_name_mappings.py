from mappings.name_mappings import name_mappings

def generate_id_name_mappings(summary_json):
	team_to_id_dict = {}
	id_to_team_dict = {}

	for team in summary_json['teams']:
		team_id = team['id']
		team_name = team['name']
		if team_name in name_mappings.keys():
			team_name = name_mappings[team_name]

		team_to_id_dict[team_name] = team_id
		id_to_team_dict[team_id] = team_name

	return team_to_id_dict, id_to_team_dict