import logging
import sys

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(message)s',
        stream=sys.stdout,
    )
    logging.getLogger('allensdk').setLevel(logging.WARNING)
