""" Configures all the logging bullshit.
"""
import datetime
import logging
import sys
from os import makedirs
from os.path import exists

from src.paths import paths

LOGGING_LEVEL = 'INFO'


def validate_path(path):
	""" Checks to see if the folder exists and, if not, creates it.
	"""
	if path[-1] == '/':
		folder = path
	else:
		folder = ''
		for x in path.split('/')[:-1]:
			if x:
				folder += '/' + x
	if not exists(folder):
		print('\nlogger: \t\tFolder {} doesnt yet exist, creating it\n'.format(folder))
		makedirs(folder)


def get_logging_path(base_path=paths['logging_base']):
	""" Calculates the filepath to save the logging output to from the system arguments and time. Returns the
		logging path and the folder path.
	"""
	script_name = sys.argv[0].split('/')[-1]
	now = datetime.datetime.now()
	script_folder_name = script_name.split('.')[0]
	folder_name = script_folder_name + '/{}_{}/'.format(now.year, now.month)
	folder_path = base_path + folder_name

	return folder_path + '/' + script_name + '_{}-{}-{}_{}:{}:{}.log'.format(
		now.year,
		now.month,
		now.day,
		now.hour,
		now.minute,
		now.second
	), folder_path


logging_path, folder_path = get_logging_path()
validate_path(folder_path)

logger = logging.getLogger('test')
stream_logger = logging.getLogger('stream')

logger.setLevel(LOGGING_LEVEL)

__stream_handler__ = logging.StreamHandler(
	stream=sys.stdout
)
__stream_handler__.setLevel(LOGGING_LEVEL)
__file_handler__ = logging.FileHandler(
	filename=logging_path
)
__formatter__ = logging.Formatter(
	fmt='%(asctime)s  %(module)-30s \t %(message)s',
	datefmt='%d/%m/%Y %H:%M:%S',
)

for handler in (__stream_handler__, __file_handler__):
	handler.setLevel(LOGGING_LEVEL)
	handler.setFormatter(__formatter__)
	logger.addHandler(handler)

stream_logger.addHandler(__stream_handler__)
stream_logger.info('Logging is being written to {}'.format(logging_path))


class TimerLogger:

	def __init__(self):
		self.timer_logger = logging.getLogger('timer')
		self.timer_logger.setLevel('INFO')
		if not self.timer_logger.handlers:
			sh = logging.StreamHandler(
				stream=sys.stdout
			)
			sh.setLevel('INFO')
			fh = logging.FileHandler(
				filename=logging_path
			)
			timer_formatter = logging.Formatter(
				fmt='%(asctime)s %(message)s',
				datefmt='%d/%m/%Y %H:%M:%S',
			)
			sh.setFormatter(timer_formatter)
			fh.setFormatter(timer_formatter)
			self.timer_logger.addHandler(sh)
			self.timer_logger.addHandler(fh)
		assert len(self.timer_logger.handlers) == 2

	def info(self, msg, override):
		msg = ' {:36}'.format(override) + msg
		return self.timer_logger.info(msg)
