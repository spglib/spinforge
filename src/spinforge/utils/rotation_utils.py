from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation
from spgrep.utils import NDArrayFloat, NDArrayInt


def to_cartesian_rotations(
    lattice: NDArrayFloat,
    rotations: NDArrayInt | NDArrayFloat,
) -> NDArrayFloat:
    """Convert fractional rotation matrices to Cartesian ones: A^T @ R @ (A^T)^-1."""
    inv_lattice_t = np.linalg.inv(lattice.T)
    return np.array([lattice.T @ rotation @ inv_lattice_t for rotation in rotations])


def find_rotation_conjugator(
    src_axis: NDArrayFloat,
    dst_axis: NDArrayFloat,
    atol: float = 1e-5,
) -> NDArrayFloat:
    """Find a proper rotation matrix `U` such that U @ R(src_axis) @ U^-1 = R(dst_axis) such that
        - U @ src_axis = dst_axis
        - U @ R(src_axis, theta) @ U^-1 = R(dst_axis, theta) for any theta

    Be careful with the signs of the axes.
    """
    cos_angle = np.dot(src_axis, dst_axis) / (np.linalg.norm(src_axis) * np.linalg.norm(dst_axis))
    if np.allclose(cos_angle, 1, atol=atol):
        # src_axis and dst_axis are parallel
        return np.eye(3)
    elif np.allclose(cos_angle, -1, atol=atol):
        # src_axis and dst_axis are anti-parallel
        for basis in [
            np.array([1, 0, 0]),
            np.array([0, 1, 0]),
            np.array([0, 0, 1]),
        ]:
            axis = np.cross(src_axis, basis)
            if np.allclose(axis, 0, atol=atol):
                continue
            axis /= np.linalg.norm(axis)

            rotation = Rotation.from_rotvec(np.pi * axis)
            return rotation.as_matrix()
        raise ValueError("Cannot find a conjugator for anti-parallel axes.")
    else:
        axis = np.cross(src_axis, dst_axis)
        axis /= np.linalg.norm(axis)
        angle = np.arccos(cos_angle)
        rotation = Rotation.from_rotvec(angle * axis)
        return rotation.as_matrix()


def get_rotation_axis(rotation: NDArrayInt | NDArrayFloat) -> NDArrayFloat | bool:
    """Get the rotation axis of a rotation matrix."""
    if isinstance(rotation, np.ndarray):
        rotation = rotation.astype(float)
    proper_rotation = np.around(np.linalg.det(rotation)) * rotation
    if np.allclose(proper_rotation, np.eye(3)):
        return True

    eigvals, eigvecs = np.linalg.eig(proper_rotation)
    for i, eigval in enumerate(eigvals):
        if np.isclose(eigval, 1):
            # Found the eigenvector corresponding to the rotation axis
            axis = np.real(eigvecs[:, i])
            axis /= np.linalg.norm(axis)
            return axis

    return False
