import logging
import os
from datetime import datetime


def setup_logger(log_file=None):
    # Check if log directory exists, if not, create it
    log_dir = 'logs'
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # If log_file is not provided, generate a default name with timestamp
    if not log_file:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_file = f'genesis_mind_{timestamp}.log'

    # Set up logging
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s [%(levelname)s] [%(name)s.%(funcName)s]: %(message)s',
                        handlers=[logging.FileHandler(os.path.join(log_dir, log_file)),
                                  logging.StreamHandler()])

    logger = logging.getLogger()
    return logger


# Initialize the logger
logger = setup_logger()
