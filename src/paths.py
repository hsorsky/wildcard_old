from config import project_directory

# Dictionary of paths relative to project home
paths = dict(
	cache='cache/',
	# output='output/',
	logging_base='logging/',
	plots='analysis/plots',
)
# Make the paths dictionary relative to root
paths = {k: project_directory + v for k, v in paths.items()}