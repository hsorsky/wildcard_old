from pathlib import Path

project_directory = str(Path.home()) + '/dev/fpl/'

DATA_PATH = project_directory + 'data/'
MATCH_DATA_PATH = DATA_PATH + 'match_data/'
FPL_DATA_PATH = DATA_PATH + 'fpl_data/'

BURN_IN_RATIO = 0.3