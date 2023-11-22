import logging

from .geometry import rot_around_vec, get_close_coords, get_angle


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
c_handler = logging.StreamHandler()
c_handler.setLevel(logging.INFO)
c_format = logging.Formatter('%(asctime)s | %(levelname)s | %(filename)s:%(lineno)d | %(funcName)s | %(message)s',
                             '%H:%M:%S')
c_handler.setFormatter(c_format)
logger.addHandler(c_handler)

__all__ = [
    "logger", "rot_around_vec", "get_close_coords", "get_angle",
]
