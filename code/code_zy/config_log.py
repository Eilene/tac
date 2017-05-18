import os
from os.path import join as fjoin
import logging
import time

def config(file):
	root = os.path.splitext(os.path.basename(file))[0]
	# config logger, create 2 handler(file, console)
	logger = logging.getLogger('BEST.{}'.format(root))
	logger_fh = logging.FileHandler(fjoin('logs', 'BEST-{}.log'.format(root)))
	logger_ch = logging.StreamHandler()
	logger_formatter = logging.Formatter('[%(levelname)s] %(asctime)s %(filename)s [line:%(lineno)d]: %(message)s' )
	logger_fh.setFormatter(logger_formatter)
	logger_ch.setFormatter(logger_formatter)
	logger.addHandler(logger_fh)
	logger.addHandler(logger_ch)
	logger.setLevel(logging.DEBUG)

	return logger