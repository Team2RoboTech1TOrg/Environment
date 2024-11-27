import logging
from config import logging_log


logging.basicConfig(
    filename=logging_log,
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
