import numpy as np
from scipy.spatial import distance_matrix
from scipy.spatial.transform import Rotation as R


def get_distances(coords1: np.ndarray, coords2: np.ndarray):
    return distance_matrix(coords1, coords2)


def get_close_coords(coords1: np.ndarray, coords2: np.ndarray, cutoff: float):
    dist_matrix = get_distances(coords1, coords2)
    idx1, idx2 = np.where(dist_matrix <= cutoff)
    return idx1, idx2


def get_angle(v1, v2):
    dot = (v1 * v2).sum(axis=-1)
    norm = np.linalg.norm(v1, axis=-1) * np.linalg.norm(v2, axis=-1)
    return np.degrees(np.arccos(np.clip(dot / norm, -1, 1)))


def rot_around_vec(points, vector, centroid, angle):
    vector = vector / np.linalg.norm(vector)
    rotation = R.from_rotvec(np.radians(angle) * vector)
    translated_points = points - centroid
    rotated_points = rotation.apply(translated_points)
    rotated_points = rotated_points + centroid
    return rotated_points
