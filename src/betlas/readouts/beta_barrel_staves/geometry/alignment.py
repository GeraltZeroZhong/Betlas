import numpy as np

from ..constants import COVARIANCE_FLOOR


class PCAAligner:
    """
    Use PCA to align the principal axis of a point cloud to the Z axis.
    """
    def __init__(self):
        self.center = None
        self.rotation_matrix = None
        self.eigenvalues = None
        self.eigenvalue_gaps = None

    @staticmethod
    def _as_coordinate_matrix(points, *, min_points: int = 3) -> np.ndarray:
        coords = np.asarray(points, dtype=float)
        if coords.ndim != 2 or coords.shape[1] != 3:
            raise ValueError("PCA alignment requires a finite coordinate array with shape (N, 3).")
        if coords.shape[0] < min_points:
            raise ValueError(f"At least {min_points} point(s) are required for PCA alignment.")
        if not np.all(np.isfinite(coords)):
            raise ValueError("PCA alignment coordinates must all be finite.")
        return coords

    @staticmethod
    def _stabilize_eigenvector_signs(eigenvectors: np.ndarray) -> np.ndarray:
        stabilized = np.array(eigenvectors, dtype=float, copy=True)
        for column_index in range(stabilized.shape[1]):
            column = stabilized[:, column_index]
            pivot = int(np.argmax(np.abs(column)))
            if column[pivot] < 0.0:
                stabilized[:, column_index] *= -1.0
        if np.linalg.det(stabilized) < 0.0:
            stabilized[:, 0] *= -1.0
        return stabilized

    def fit(self, points):
        """
        Compute the principal axes of the point cloud.

        For best results, pass only C-alpha coordinates from beta-sheet residues.

        Args:
            points (np.ndarray): Coordinate array with shape ``(N, 3)``.
        """
        coords = self._as_coordinate_matrix(points)

        # 1. Compute the centroid and center the coordinates.
        self.center = np.mean(coords, axis=0)
        centered_coords = coords - self.center

        # 2. Build the covariance matrix.
        # rowvar=False means columns are variables (x, y, z) and rows are samples.
        cov_matrix = np.cov(centered_coords, rowvar=False)

        # 3. Eigen decomposition.
        # eigh is appropriate for symmetric matrices such as covariance matrices,
        # and is usually more stable than eig here.
        # eig_vals are returned in ascending order (min -> max), and each column
        # of eig_vecs is a corresponding eigenvector.
        if not np.all(np.isfinite(cov_matrix)):
            raise ValueError("PCA covariance matrix contains non-finite values.")
        if float(np.max(np.diag(cov_matrix))) <= COVARIANCE_FLOOR:
            raise ValueError("PCA alignment requires non-degenerate coordinate variance.")

        eig_vals, eig_vecs = np.linalg.eigh(cov_matrix)

        self.eigenvalues = eig_vals
        self.eigenvalue_gaps = np.diff(eig_vals)
        self.rotation_matrix = self._stabilize_eigenvector_signs(eig_vecs)
        
        # Debug helper: variance contribution of each PCA axis.
        # variance_ratio = eig_vals / np.sum(eig_vals)
        # print(f"PCA Variance Ratios (X, Y, Z): {variance_ratio}")

    def transform(self, points):
        """
        Project the point cloud into the PCA coordinate system.

        After transformation, the Z axis corresponds to the dominant principal
        axis of the original point cloud.
        """
        if self.rotation_matrix is None:
            raise RuntimeError("Aligner is not fitted. Call fit() first.")

        coords = self._as_coordinate_matrix(points, min_points=1)

        # 1. Translate to the centered coordinate system.
        centered = coords - self.center

        # 2. Rotate by projecting onto the eigenvector basis.
        # result = centered @ eigenvectors
        # Column 0 corresponds to the smallest eigenvalue direction (new X),
        # column 1 to the middle eigenvalue direction (new Y), and column 2 to
        # the largest eigenvalue direction (new Z, the principal axis).
        transformed = np.dot(centered, self.rotation_matrix)

        return transformed

    def fit_transform(self, points):
        """Convenience wrapper that runs ``fit`` followed by ``transform``."""
        self.fit(points)
        return self.transform(points)
