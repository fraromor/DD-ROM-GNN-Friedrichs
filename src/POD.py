import numpy as np
from mpi4py import MPI


# TODO: fix normalization of monolithic SVD
class POD:
    def __init__(self,
                 X,
                 tolPOD=None,
                 NPOD=None,
                 normalizer=None,
                 fl_rec=True):
        assert tolPOD is not None or NPOD is not None, "`tolPOD` or `NPOD` must be passed, they are mutually exclusive."
        
        self.commMPI = MPI.COMM_WORLD
        self.size = self.commMPI.Get_size()
        self.rank = self.commMPI.Get_rank()
        self.tolPOD = tolPOD 
        self.NPOD = NPOD
        
        # normalize
        if normalizer is not None:
            assert np.max(np.abs(normalizer)) > 1e-6, "Normalizer has 0 values."
            snap = X / normalizer.reshape(-1, 1)
        else:
            snap = X
        self.V, s = np.linalg.svd(snap, full_matrices=False)[:2]
        
        tot = sum(s)
        self.eigenvalues = s / tot
        
        if self.tolPOD is not None:
            self.NPOD = 0
            while sum(self.eigenvalues[:self.NPOD]) < 1 - tolPOD:
                self.NPOD = self.NPOD + 1
        elif self.NPOD is not None:
            self.tolPOD = 1 - sum(self.eigenvalues[:self.NPOD])
            
        self.basis = self.V[:, :self.NPOD]
        
        # denormalize
        if normalizer is not None:
            self.basis *= normalizer.reshape(-1, 1)
            self.basis /= np.linalg.norm(self.basis, axis=0, keepdims=True)
            
        if fl_rec:
            print(
                self.rank, ":", "Reconstruction Error: ",
                self.eval_reconstruction_error(X))

    def set_tol(self, tolPOD):
        self.NPOD = 0
        while sum(self.eigenvalues[:self.NPOD]) < 1 - tolPOD:
            self.NPOD = self.NPOD + 1
        self.basis = self.V[:, :self.NPOD]

    def set_N(self, NPOD):
        self.NPOD = NPOD
        self.tolPOD = 1 - sum(self.eigenvalues[:self.NPOD])
        self.basis = self.V[:, :self.NPOD]
    
    def eval_reconstruction_error(self, X):
        return np.max(np.linalg.norm(X - self.basis.dot(self.basis.T.dot(X)), axis=0) / np.clip(np.linalg.norm(X, axis=0), 1e-6, None))