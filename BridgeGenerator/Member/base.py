import numpy as np
from scipy.spatial.transform import Rotation as R


class Member:
    def __init__(self, C, t=None, quat=None):
        """

        Args:
            C: The accumulative cross-section got from an CrossSection instance
            t: Translation of the cross-sections
            quat: Rotation in quaternion of the cross-sections

        Member variables:
            n: Number of cross-sections in the instance
            npts: Vertices number of one cross-section
            t: Translation of the cross-sections
            r: Rotation instance
            v: list of three-tuples (vertices) represented the coordinates in 3D
            f: list of faces
        """
        self.n = C.shape[0]
        self.npts = C.shape[1]
        if t is None:
            t = np.zeros((self.n, 3))
            t[:, 0] = np.arange(self.n)
        self.t = t

        if quat is None:
            quat = np.zeros((self.n, 4))
            quat[:, 3] = 1.  # quaternion is (x,y,z,w)
        self.r = R.from_quat(quat)

        self.v = []
        self.f = []

        xyz = np.zeros((self.npts, 3))
        for i in range(self.n):
            xyz[:, 1:] = C[i, :, :]
            c = self.r[i].apply(xyz) + self.t[np.zeros(self.npts, dtype=int) + i, :]
            self.v = self.v + [(c[k, 0], c[k, 1], c[k, 2]) for k in range(self.npts)]
            if i > 0:
                m = self.npts * (i - 1)
                idx1 = np.arange(m, m + self.npts)
                idx2 = np.arange(m + self.npts, m + 2 * self.npts)
                self.f = self.f + [(idx1[k], idx1[np.mod(k + 1, self.npts)], idx2[np.mod(k + 1, self.npts)], idx2[k])
                                   for k in range(self.npts)]