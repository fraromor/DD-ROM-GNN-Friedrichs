import os
from itertools import accumulate
import abc
import time

import numpy as np
from scipy.sparse import load_npz
from scipy.sparse.linalg import inv, spsolve

import sys, petsc4py
from sklearn.cluster import KMeans

petsc4py.init(sys.argv)
from petsc4py import PETSc
from mpi4py import MPI

from src.POD import POD

import logging

class FOMproblem(abc.ABC):
    def __init__(self, n_cores=1):
        self.comm = PETSc.COMM_WORLD
        self.commMPI = MPI.COMM_WORLD
        self.size = self.commMPI.Get_size()
        self.rank = self.commMPI.Get_rank()

        # variables to be initialized in child classes
        self.equation_name = None
        self.test_name = None
        self.physical_dim = None
        self.system_dim = None
        self.degree = None
        self.folder = None
        self.dim_length = None

        # basis
        self.reduced_basis = None
        self.repartitioned_basis = None

        self.solution = None
        # parameters
        self.param_range_lhs = None
        self.param_range_rhs = None
        self.mu_lhs = None
        self.mu_rhs = None

        # numerics quantities
        self.dof_each_cell = None
        self.N_dofs_global = None
        self.N_dofs_local = None
        self.N_dofs_per_variable = None
        self.N_local_cells = None
        self.N_total_cells = None
        self.localDofsGathered = None
        self.total_dofs = None
        self.up = None
        self.down = None
        self.map_local_2_variable = None
        self.map_variable_2_local = None
        
        # normalize w.r.t. dimension of the FS system
        self.dofs_1d = None
        self.local_dofs_1d = None
        self.local_dofs_1d = None
        self.down_1d = None

        # affine decomposition vars
        self.affine_lhs = None
        self.affine_rhs = None
        self.norm_matrices = None
        self.norm_residual = None

        # other variables
        self.folder_matrices = None
        self.system_time = None
        self.pos = None

        # partitioning
        self.n_partitions = 0
        self.labels = None
        self.pmask = None
        self.k_nearest = None
        self.r_dim_approx = None
        self.X_cells = None

    def setup_and_load_structures(self):
        # Loading matrices and parameter structures
        self.create_folders()
        self.setup_parameter_structures()
        self.load_FOM_matrices()
        self.set_variables()
        self.set_parameter_range(param_range_lhs=None, param_range_rhs=None)
        self.create_map_to_variables()

    def create_folders(self):
        try:
            os.makedirs(self.folder)
        except:
            if self.rank == 0: print("folder already exists")
        
        if self.folder_matrices is None:
            if self.folder == None:
                self.folder_matrices = "affine_matrices"
            else:
                self.folder_matrices = self.folder + "/affine_matrices"

    @abc.abstractmethod
    def setup_parameter_structures(self):
        pass

    @abc.abstractmethod
    def eval_coeff(self, parameter_domain_lhs, parameter_domain_rhs):
        """Override this function since the coefficients of the affine decomposition
           could not be simply the sampled parameters themselves but a function of those."""
        pass

    @abc.abstractmethod
    def pdf_constant(self, mu_lhs):
        """Override this function to compute the positive definitness lower bound constant."""
        pass

    def set_parameter_range(self, param_range_lhs=None, param_range_rhs=None):
        if param_range_lhs is None or param_range_rhs is None:
            raise ValueError("Parameter range not implemented for problem " +
                             self.test_name)
        else:
            self.param_range_lhs = param_range_lhs
            self.param_range_rhs = param_range_rhs

    def load_FOM_matrices(self):
        if self.rank == 0: print("Loading FOM affine matrices and rhs")
        self.affine_lhs = list()
        self.affine_rhs = list()
        self.norm_matrices = list()
        self.norm_residual = list()
        self.interface = list()
        rank = self.comm.Get_rank()

        # Load affine LHS matrices (sparse)
        for param_name in self.affine_parameters_lhs_names:
            if self.rank == 0: print("Loading mat_" + param_name)
            tmp_mat = load_npz(self.folder_matrices + "/" + "mat_" +
                               param_name + "_" + str(rank) + ".npz")
            print(param_name, tmp_mat.shape)
            self.affine_lhs.append(tmp_mat)

        # Load affine RHS vectors
        for param_name in self.affine_parameters_rhs_names:
            if self.rank == 0: print("Loading rhs_" + param_name)
            v = np.load(self.folder_matrices + "/" + "rhs_" + param_name +
                        "_" + str(rank) + ".npy")
            print(param_name, v.shape)
            self.affine_rhs.append(v)
        
        # Load norm matrices (sparse)
        for param_name in self.norm_matrix_names:
            if self.rank == 0: print("Loading norms_" + param_name)
            tmp_mat = load_npz(self.folder_matrices + "/" + "mat_" +
                               param_name + "_" + str(rank) + ".npz")
            print(param_name, tmp_mat.shape)
            self.norm_matrices.append(tmp_mat)
        
        # Load norm residuals (sparse)
        for param_name in self.norm_residual_names:
            if self.rank == 0: print("Loading norms_" + param_name)
            tmp_mat = load_npz(self.folder_matrices + "/" + "mat_" +
                               param_name + "_" + str(rank) + ".npz")
            self.norm_residual.append(tmp_mat)

        # Load in
        for param_name in self.interface_names:
            if self.rank == 0: print("Loading norms_" + param_name)
            try:
                tmp_mat = load_npz(self.folder_matrices + "/" + "mat_" +
                                param_name + "_" + str(rank) + ".npz")
                self.interface.append(tmp_mat)
            except:
                self.interface = None

        print("Matrices loaded.\n")
        
    def set_variables(self):
        assert self.degree is not None
        assert self.physical_dim is not None
        assert self.affine_lhs is not None
        assert self.affine_rhs is not None
        assert self.size is not None
        assert self.commMPI is not None

        self.dof_each_cell = np.int64((self.degree + 1)**self.physical_dim)
        self.N_dofs_global = self.affine_lhs[0].shape[1]
        self.N_dofs_local = len(self.affine_rhs[0])
        self.N_dofs_per_variable = self.N_dofs_local // self.system_dim
        self.N_local_cells = self.N_dofs_local // (self.system_dim *
                                                   self.dof_each_cell)
        
        N_total_cells = np.zeros(1, dtype=np.int32)
        self.commMPI.Allreduce(np.array([self.N_local_cells], dtype=np.int32), N_total_cells)
        self.N_total_cells = N_total_cells[0]
        
        split = np.array([self.N_dofs_local], dtype=np.int32)
        self.localDofsGathered = np.zeros(self.size,
                                          dtype=np.int32)  # only at root

        self.commMPI.Allgatherv(split, [
            self.localDofsGathered,
            np.ones(self.size, dtype=np.int32),
            np.arange(self.size, dtype=np.int32), MPI.INT32_T
        ])

        self.total_dofs = self.localDofsGathered.sum()
        self.up = np.array(list(accumulate(self.localDofsGathered)),
                           dtype=np.int32)
        self.down = np.zeros(self.size, dtype=np.int32)
        self.down[1:] = self.up[:-1]
        self.down[0] = 0
                
        # normalize w.r.t. dimension of the FS system
        self.dofs_1d = int(self.total_dofs / self.system_dim)
        self.local_dofs_1d = self.localDofsGathered / self.system_dim
        self.local_dofs_1d = self.local_dofs_1d.astype(np.int32)
        self.down_1d = self.down / self.system_dim
        self.down_1d = self.down_1d.astype(np.int32)

        # Load positions
        if self.pos_names is not None:
            pos = list()
            for param_name in self.pos_names:
                if self.rank == 0: print("Loading pos" + param_name)
                v = np.load(self.folder_matrices + "/" + param_name + "_" + str(self.rank) + ".npy")
                print(param_name, v.shape)
                pos.append(v)
            self.local_pos = np.vstack(pos).reshape(-1)
            self.global_pos = np.zeros((self.physical_dim*self.N_dofs_global//self.system_dim))
            
            ss = self.physical_dim*self.localDofsGathered//self.system_dim
            dd = self.physical_dim*self.down//self.system_dim            
            self.commMPI.barrier()
            self.commMPI.Allgatherv(self.local_pos, [
                self.global_pos,
                ss, 
                dd, MPI.DOUBLE
            ])
            self.global_pos = self.global_pos.reshape(self.physical_dim, self.N_dofs_global//self.system_dim)
            self.centers = np.mean(self.global_pos.reshape(self.physical_dim, -1, self.dof_each_cell), axis=2).T
            np.save(self.folder+"/centers.npy", self.centers)
        else:
            print("Load cell centers")
            self.centers = np.load(self.folder_matrices + "/"+"/centers.npy")
            
        
    @staticmethod
    def check_pos_def(x):
        return np.all(np.linalg.eigvals(x) > 0)

    @staticmethod
    def check_symmetric(a, rtol=1e-05, atol=1e-08):
        return np.allclose(a, a.T, rtol=rtol, atol=atol)

    def solve(self, mu_lhs, mu_rhs):
        tic_total = time.time()
        self.set_parameters(mu_lhs, mu_rhs)
        self.assemble_operators()
        self.solve_system()
        toc_total = time.time()
        self.total_time = toc_total - tic_total
        print("Total time: ", self.total_time)
        return self.solution

    def set_parameters(self, mu_lhs, mu_rhs):
        if len(mu_lhs) == len(self.affine_parameters_lhs_names):
            self.mu_lhs = mu_lhs
        else:
            raise ValueError("Parameter lhs of dimension " + str(len(mu_lhs)) +
                             " instead of " +
                             str(len(self.affine_parameters_lhs_names)))

        if len(mu_rhs) == len(self.affine_parameters_rhs_names):
            self.mu_rhs = mu_rhs
        else:
            raise ValueError("Parameter rhs of dimension " + str(len(mu_rhs)) +
                             " instead of " +
                             str(len(self.affine_parameters_rhs_names)))

    def assemble_operators(self):
        if self.rank == 0:
            print("Assembling operators for params: lhs {} rhs {}".format(
                self.mu_lhs, self.mu_rhs))
        tic = time.time()

        self.rhs = sum([
            self.mu_rhs[i] * self.affine_rhs[i]
            for i in range(len(self.mu_rhs))
        ])
                
        self.lhs = sum([
           self.mu_lhs[i] * self.affine_lhs[i]
           for i in range(len(self.mu_lhs))
        ])
        
        toc = time.time()
        self.assemble_time = toc - tic

    def solve_system(self):
        if self.rank == 0:
            print("Solving FOM system")
        OptDB = PETSc.Options()
        
        b = PETSc.Vec()
        b.createWithArray(array=list(self.rhs),
                          size=[self.localDofsGathered[self.rank], self.N_dofs_global],
                          comm=self.comm)
        b.assemble()
        
        A = PETSc.Mat()
        A.createAIJWithArrays(
            [[self.localDofsGathered[self.rank], self.N_dofs_global], [self.N_dofs_global, self.N_dofs_global]],
            # [self.N_dofs_global, self.N_dofs_global],
            (self.lhs.indptr, self.lhs.indices, self.lhs.data),
            bsize=None,
            comm=self.comm)
        A.assemble()
        
        x = A.getVecRight()
        x.set(0.)
        
        ksp = PETSc.KSP()
        ksp.create(self.comm)
        ksp.setOperators(A)
        ksp.setType('preonly')
        pc = ksp.getPC()
        pc.setType('lu')
        pc.setFactorSolverType('mumps')
        ksp.setFromOptions()
        
        tic = time.time()
        ksp.solve(b, x)
        toc = time.time()
        self.system_time = toc - tic
        
        solution = np.zeros(self.N_dofs_global, dtype=np.float64)
        solution_local = np.zeros(self.localDofsGathered[self.rank], dtype=np.float64)
        
        self.comm.barrier()
        if self.rank == 0:    
            Istart, Iend = x.getOwnershipRange()
            solution = np.array(x.getValues(list(np.arange(Istart, Iend, 1))), dtype=np.float64)
        
        self.commMPI.Bcast(solution)
        A.destroy()
        b.destroy()
        x.destroy()
        ksp.destroy()
        pc.destroy()
        self.solution = solution[self.down[self.rank]:self.up[self.rank]]#solution_local
        return self.solution

    def create_map_to_variables(self):
        assert self.N_dofs_local % (self.system_dim * self.dof_each_cell
                                    ) == 0, "Length of system is not correct"

        self.map_local_2_variable = np.zeros((self.N_dofs_local, 2),
                                             dtype=np.int64)
        self.map_variable_2_local = np.zeros(
            (self.system_dim, self.N_local_cells * self.dof_each_cell),
            dtype=np.int64)
        idx_cell = np.arange(self.dof_each_cell)
        idx_local = np.arange(self.dof_each_cell)
        for i_cell in range(self.N_local_cells):
            for i_var in range(self.system_dim):
                self.map_local_2_variable[idx_local, 0] = i_var
                self.map_local_2_variable[idx_local, 1] = idx_cell
                self.map_variable_2_local[i_var, idx_cell] = idx_local
                idx_local = idx_local + self.dof_each_cell
            idx_cell = idx_cell + self.dof_each_cell

    def separate_components(self, u):
        vec = np.zeros((self.system_dim, self.N_dofs_local // self.system_dim))
        for i in range(self.system_dim):
            vec[i, :] = u[self.map_variable_2_local[i, :]]
        return vec

    def unify_components(self, vec):
        u = np.zeros(self.N_dofs_local)
        for i in range(self.system_dim):
            u[self.map_variable_2_local[i, :]] = vec[i, :]
        return u

    def expand_one_component(self, vec, i_var):
        u = np.zeros(self.N_dofs_local)
        u[self.map_variable_2_local[i_var, :]] = vec
        return u

    def compute_l2_norm(self, vec):
        L2normLocal = 0
        for col_block in range(self.size):
            vecBlock = np.empty(self.localDofsGathered[col_block])
            if self.rank == col_block:
                vecBlock[:] = vec[:]

            self.commMPI.barrier()
            self.commMPI.Bcast(vecBlock, root=col_block)
            self.commMPI.barrier()

            term = vec.T @ self.norm_matrices[0][:, self.down[col_block]:self.
                                    up[col_block]] @ vecBlock

            L2normLocal += term

        L2norm = np.zeros(1)
        self.commMPI.Allreduce(np.array([L2normLocal]), L2norm)
        return np.sqrt(L2norm[0])
    
    def compute_R_norm(self, vec, cc):
        L2normLocal = 0
        R_norm_local = cc *self.norm_matrices[0] + self.norm_matrices[2]
        for col_block in range(self.size):
            vecBlock = np.empty(self.localDofsGathered[col_block])
            if self.rank == col_block:
                vecBlock[:] = vec[:]

            self.commMPI.barrier()
            self.commMPI.Bcast(vecBlock, root=col_block)
            self.commMPI.barrier()

            term = vec.T @ R_norm_local[:, self.down[col_block]:self.
                                    up[col_block]] @ vecBlock

            L2normLocal += term

        L2norm = np.zeros(1)
        self.commMPI.Allreduce(np.array([L2normLocal]), L2norm)
        return np.sqrt(L2norm[0])

    def compute_opt_norm(self, vec, cc):
        L2normLocal = 0
        R_norm_local = cc *self.norm_matrices[0] + self.norm_matrices[2] + self.norm_matrices[3]/(self.physical_dim**(3/2)*self.degree)
        for col_block in range(self.size):
            vecBlock = np.empty(self.localDofsGathered[col_block])
            if self.rank == col_block:
                vecBlock[:] = vec[:]

            self.commMPI.barrier()
            self.commMPI.Bcast(vecBlock, root=col_block)
            self.commMPI.barrier()

            term = vec.T @ R_norm_local[:, self.down[col_block]:self.
                                    up[col_block]] @ vecBlock

            L2normLocal += term

        L2norm = np.zeros(1)
        self.commMPI.Allreduce(np.array([L2normLocal]), L2norm)
        return np.sqrt(L2norm[0])
    
    def compute_R_norm_inv(self, vec, cc, mu_lhs, mu_rhs):
        A_mat = cc *self.norm_matrices[0] + self.norm_matrices[2]+0.5*self.norm_matrices[1]
        
        b = PETSc.Vec()
        b.createWithArray(
            array=list(vec),
            size=[self.localDofsGathered[self.rank], self.N_dofs_global],
            comm=self.comm)
        b.assemble()
        A = PETSc.Mat()
        A.createAIJWithArrays(
            [[self.localDofsGathered[self.rank], self.N_dofs_global], [self.N_dofs_global, self.N_dofs_global]],
            (A_mat.indptr, A_mat.indices, A_mat.data),
            bsize=None,
            comm=self.comm)
        A.assemble()
        x = A.getVecRight()
        x.set(0.)        
        ksp = PETSc.KSP()
        ksp.create(self.comm)
        ksp.setOperators(A)
        ksp.setType('preonly')
        pc = ksp.getPC()
        pc.setType('lu')
        pc.setFactorSolverType('mumps')
        ksp.setFromOptions()
        tic = time.time()
        ksp.solve(b, x)
        toc = time.time()
        self.system_time = toc - tic
        solution = np.zeros(self.N_dofs_global, dtype=np.float64)
        self.comm.barrier()
        A.destroy()
        b.destroy()
        ksp.destroy()
        pc.destroy()
        if self.rank == 0:    
            Istart, Iend = x.getOwnershipRange()
            solution = np.array(x.getValues(list(np.arange(Istart, Iend, 1))), dtype=np.float64)
        self.commMPI.Bcast(solution)
        x.destroy()
        ret = solution[self.down[self.rank]:self.up[self.rank]]
        
        L2normLocal = vec.dot(ret)
        L2norm = np.zeros(1)
        self.commMPI.Allreduce(np.array([L2normLocal]), L2norm)
        return np.sqrt(L2norm[0])
    
    def compute_l2_norm_inv(self, vec, cc, mu_lhs, mu_rhs):
        logging.info('compute_l2_norm_inv')
        A_mat = self.norm_matrices[0]
        b = PETSc.Vec()
        b.createWithArray(
            array=list(vec),
            size=[self.localDofsGathered[self.rank], self.N_dofs_global],
            comm=self.comm)
        b.assemble()
        print("b size :", b.size, b.local_size)
        A = PETSc.Mat()
        A.createAIJWithArrays(
            [[self.localDofsGathered[self.rank], self.N_dofs_global], [self.N_dofs_global, self.N_dofs_global]],
            (A_mat.indptr, A_mat.indices, A_mat.data),
            bsize=None,
            comm=self.comm)
        A.assemble()
        print("A size :", A.size, A.local_size)
        logging.info('A size {}'.format(A.getInfo()))
        x = A.getVecRight()
        x.set(0.)        
        print("x size :", x.size, x.local_size)
        
        ksp = PETSc.KSP()
        ksp.create(self.comm)
        ksp.setOperators(A)
        ksp.setType('preonly')
        pc = ksp.getPC()
        pc.setType('lu')
        pc.setFactorSolverType('mumps')
        ksp.setFromOptions()
        
        tic = time.time()
        ksp.solve(b, x)
        toc = time.time()
        logging.info('Solved l2 estimator')
        
        self.system_time = toc - tic
        solution = np.zeros(self.N_dofs_global, dtype=np.float64)
        self.comm.barrier()
        A.destroy()
        b.destroy()
        ksp.destroy()
        pc.destroy()
        if self.rank == 0:    
            Istart, Iend = x.getOwnershipRange()
            solution = np.array(x.getValues(list(np.arange(Istart, Iend, 1))), dtype=np.float64)
        self.commMPI.Bcast(solution)
        x.destroy()
        ret = solution[self.down[self.rank]:self.up[self.rank]]
        
        L2normLocal = vec.dot(ret)
        L2norm = np.zeros(1)
        self.commMPI.Allreduce(np.array([L2normLocal]), L2norm)
        return np.sqrt(L2norm[0])

    def compute_Mseminorm_squared(self, vec):
        MseminormLocal = 0
        for col_block in range(self.size):
            vecBlock = np.empty(self.localDofsGathered[col_block])
            if self.rank == col_block:
                vecBlock[:] = vec[:]

            self.commMPI.barrier()
            self.commMPI.Bcast(vecBlock, root=col_block)
            self.commMPI.barrier()

            term = vec.T @ self.norm_matrices[1][:, self.down[col_block]:self.
                                    up[col_block]] @ vecBlock

            MseminormLocal += term
            
        Mseminorm = np.zeros(1)
        self.commMPI.Allreduce(np.array([MseminormLocal]), Mseminorm)
        return Mseminorm[0]

    def compute_Sseminorm_squared(self, vec):
        SseminormLocal = 0
        for col_block in range(self.size):
            vecBlock = np.empty(self.localDofsGathered[col_block])
            if self.rank == col_block:
                vecBlock[:] = vec[:]

            self.commMPI.barrier()
            self.commMPI.Bcast(vecBlock, root=col_block)
            self.commMPI.barrier()

            term = vec.T @ self.norm_matrices[2][:, self.down[col_block]:self.
                                    up[col_block]] @ vecBlock

            SseminormLocal += term
        
        Sseminorm = np.zeros(1)
        self.commMPI.Allreduce(np.array([SseminormLocal]), Sseminorm)
        return Sseminorm[0]
            
    def compute_energynorm(self, vec, mu_lhs, mu_rhs):
        self.set_parameters(mu_lhs, mu_rhs)
        self.lhs = sum([
           self.mu_lhs[i] * self.affine_lhs[i]
           for i in range(len(self.mu_lhs))
        ])
        
        energynormLocal = 0
        for col_block in range(self.size):
            term = 0
            vecBlock = np.empty(self.localDofsGathered[col_block])
            if self.rank == col_block:
                vecBlock[:] = vec[:]

            self.commMPI.barrier()
            self.commMPI.Bcast(vecBlock, root=col_block)
            self.commMPI.barrier()

            term = vec.T @ self.lhs[:,
                                    self.down[col_block]:self.
                                    up[col_block]] @ vecBlock

            energynormLocal += term
            self.commMPI.barrier()
        
        energynorm = np.zeros(1)
        self.commMPI.Allreduce(np.array([energynormLocal]), energynorm)
        
        return np.sqrt(energynorm[0])
    
    def normalize(self, snapshots):
        # TODO provisional implementation
        N = snapshots.shape[1]
        maxs = np.zeros(snapshots.shape[0])
        cell_dofs = self.dof_each_cell * self.system_dim
        indices = np.arange(snapshots.shape[0]) % cell_dofs

        fields_length = list(np.roll(list(accumulate(self.dim_length)), 1))
        fields_length[0] = 0
        fields_length.append(sum(self.dim_length))
        for i in range(len(fields_length) - 1):
            mask = np.logical_and(
                fields_length[i] * self.dof_each_cell <= indices,
                indices < fields_length[i + 1] * self.dof_each_cell)
            value = np.max(snapshots[mask, :])
            maxs[mask] = value
        return maxs

    def partition_dofswise(self, snapshots, n_subdomains=2, tol_repartitioning=1e-3, mask=[-1], indicator='var'):
        assert indicator=='var' or indicator=='grassmannian', "Indicator is: {}".format(indicator)
        if indicator=='grassmannian':
            assert self.k_nearest is not None and self.r_dim_approx is not None
        
        self.n_partitions = n_subdomains
        n_snaps = snapshots.shape[1]
        
        # repartition keeping together the dofs of the same cell
        M = snapshots.T.reshape(
            n_snaps, self.N_local_cells, self.system_dim,
            self.dof_each_cell).transpose(0, 1, 3, 2).reshape(
                n_snaps, self.N_local_cells, self.dof_each_cell,
                self.system_dim)        
        
        # mm = np.zeros([1, 1, 1, self.system_dim])
        # MM = np.zeros([1, 1, 1, self.system_dim])
        # for j in range(self.system_dim):
        #     mm[0, 0, 0, j] = np.min(M[:, :, :, j])
        #     MM[0, 0, 0, j] = np.max(M[:, :, :, j])
        
        # M = (2/MM-mm)*(M - (MM+mm)/2)
        
        if indicator=='var':
            print("Variance indicator")
            X = np.var(M, axis=0).reshape(-1)
            XX = np.zeros(self.total_dofs)
            self.commMPI.Gatherv(
                X, [XX, self.localDofsGathered, self.down, MPI.DOUBLE])
        elif indicator=='grassmannian':
            print("Grassmannian indicator")
            X = np.mean(M, axis=2).reshape(-1)
            XX = np.zeros(n_snaps*self.total_dofs//self.dof_each_cell)
            self.commMPI.Gatherv(
                X, [XX, n_snaps*self.localDofsGathered//self.dof_each_cell, n_snaps*self.down//self.dof_each_cell, MPI.DOUBLE])

        all_labels = np.zeros(self.dofs_1d, dtype=np.int32)
        if self.rank == 0 and self.n_partitions!=1:
            if indicator=='var':
                print("Indicator var")
                X_avg_m = np.max(XX.reshape(-1, self.system_dim), axis=1, keepdims=True)
                self.X_cells = np.max(X_avg_m.reshape(-1, self.dof_each_cell), axis=1)
                # self.X_cells = np.zeros(self.N_total_cells)
                # print("DEBUG" ,self.centers[:, 0])
                # self.X_cells[self.centers[:, 0]>0] =1                
            elif indicator=='grassmannian':
                print("Indicator Grassmannian")
                XX = XX.reshape(n_snaps, self.N_total_cells, self.system_dim).transpose(1, 0, 2).reshape(self.N_total_cells, -1)
                self.X_cells = np.zeros(self.N_total_cells)
                from sklearn.neighbors import KDTree
                adjList = KDTree(self.centers, leaf_size=2)
                ind = adjList.query(self.centers, k=self.k_nearest)[1]
                for cell in range(self.N_total_cells):
                    Xtmp = XX[ind[cell], :]
                    U = np.linalg.svd(Xtmp)[0][:, :self.r_dim_approx]
                    self.X_cells[cell] = np.linalg.norm(Xtmp-U.dot(U.T.dot(Xtmp)))/np.linalg.norm(Xtmp)
            
            all_labels_ = np.ones(self.X_cells.shape[0], dtype=np.int32)
            
            ind = np.argsort(self.X_cells)
            all_labels_[ind[:int(ind.shape[0]*tol_repartitioning)]] = 0
            all_labels = np.kron(all_labels_, np.ones(self.dof_each_cell, dtype=np.int32)).reshape(-1)
            
            print("Cellwise indicator values: ", self.X_cells[ind])
            print("Repartitioning with tol: ", tol_repartitioning, int(ind.shape[0]*tol_repartitioning), ind.shape[0]*tol_repartitioning, ind.shape[0])
            print("Bins cellwise: ", np.bincount(all_labels_[all_labels_>=0]))
            
        self.labels = np.zeros(self.local_dofs_1d[self.rank])
        self.labels = self.labels.astype(np.int32)
        self.commMPI.Scatterv(
            [all_labels, self.local_dofs_1d, self.down_1d, MPI.INT32_T],
            self.labels)

        self.labels = np.kron(self.labels, np.ones(self.system_dim)).reshape(
            -1, self.dof_each_cell, self.system_dim).transpose(0, 2,
                                                               1).reshape(-1)
        np.savetxt(
            self.folder_matrices + "/labels_dofs_" + str(self.rank) + ".txt",
            self.labels.reshape(-1, self.system_dim)[:, 0])

        self.pmask = list()
        tmp = np.ones(self.labels.shape[0])
        self.plocalDofs = np.zeros(n_subdomains)
        for n in range(n_subdomains):
            self.pmask.append(self.labels == n)
            self.plocalDofs[n] = tmp[self.pmask[n]].sum()
            print("Local partition, rank: ", self.rank, ", numer: ", n, ", size: ", self.plocalDofs[n])

        self.plocalDofsGathered = np.zeros(self.size * self.n_partitions)
        self.plocalDofs = self.plocalDofs.astype(np.int32)
        self.plocalDofsGathered = self.plocalDofsGathered.astype(np.int32)

        s = np.ones(self.size) * self.n_partitions
        s = s.astype(np.int32)
        d = np.arange(self.size) * self.n_partitions
        self.commMPI.Allgatherv(self.plocalDofs,
                                [self.plocalDofsGathered, s, d, MPI.INT32_T])
        self.plocalDofsGathered = self.plocalDofsGathered.reshape(
            self.size, self.n_partitions)

        self.pdown = np.zeros((self.n_partitions, self.size), dtype=np.int32)
        for n in range(n_subdomains):
            up = np.array(list(accumulate(self.plocalDofsGathered[:, n])),
                          dtype=np.int32)
            self.pdown[n, 1:] = up[:-1]
        
class FOM_trainingSet:
    def __init__(self,
                 problem,
                 N_training=100,
                 load_snapshots=False,
                 load_reduced_space=False):
        self.comm = PETSc.COMM_WORLD
        self.commMPI = MPI.COMM_WORLD
        self.rank = self.comm.getRank()
        
        self.parameter_domain_lhs = None
        self.parameter_domain_rhs = None
        self.mu_lhs = None
        self.mu_rhs = None
        
        self.problem = problem
        folder = self.problem.folder
        if folder == "":
            self.snapshot_folder = "snapshots"
        else:
            self.snapshot_folder = folder + "/snapshots"
        try:
            os.makedirs(self.snapshot_folder)
        except:
            if self.rank == 0: print("snapshot_folder already exists")
        self.parameter_range_lhs = np.array(self.problem.param_range_lhs)
        self.N_param_lhs = self.parameter_range_lhs.shape[0]
        self.parameter_range_rhs = np.array(self.problem.param_range_rhs)
        self.N_param_rhs = self.parameter_range_rhs.shape[0]
        self.N_dofs_local = self.problem.N_dofs_local
        self.N = N_training

        self.normalizer = None

        self.snapshots_matrix = None
        
        if load_snapshots:
            self.load_snapshots()
        if load_reduced_space:
            self.load_reduced_space()

    def generate_parameters(self):
        if self.parameter_domain_lhs is None or self.parameter_domain_rhs is None:
            delta_param = np.diff(self.parameter_range_lhs)
            self.parameter_domain_lhs = np.zeros((self.N, self.N_param_lhs))
            self.parameter_domain_rhs = np.zeros((self.N, self.N_param_rhs))

            # self.parameter_domain_lhs = np.load(self.snapshot_folder +
            #                                     "/parameter_domain_lhs_" +
            #                                     str(self.rank) + ".npy")
            # self.parameter_domain_rhs = np.load(self.snapshot_folder +
            #                                     "/parameter_domain_rhs_" +
            #                                     str(self.rank) + ".npy")

            if self.rank == 0:
                self.parameter_domain_lhs = np.random.rand(
                    self.N,
                    self.N_param_lhs) * delta_param.T + self.parameter_range_lhs[:, 0]

                delta_param = np.diff(self.parameter_range_rhs)
                self.parameter_domain_rhs = np.random.rand(
                    self.N,
                    self.N_param_rhs) * delta_param.T + self.parameter_range_rhs[:,
                                                                                0]

            self.commMPI.Bcast(self.parameter_domain_lhs, root=0)
            self.commMPI.Bcast(self.parameter_domain_rhs, root=0)
            self.evaluate_domain_parameters()
        else:
            self.N = len(self.parameter_domain_lhs)

    def evaluate_domain_parameters(self):
        self.mu_lhs = []
        self.mu_rhs = []
        for i in range(self.N):
            mu_lhs, mu_rhs = self.problem.eval_coeff(
                self.parameter_domain_lhs[i, :],
                self.parameter_domain_rhs[i, :])
            self.mu_lhs.append(mu_lhs)
            self.mu_rhs.append(mu_rhs)

    def generate_snapshots(self):
        self.generate_parameters()  #Randomly selected equispaced
        self.snapshots_matrix = np.zeros((self.N_dofs_local, self.N))
        self.assemble_time = np.zeros(self.N)
        self.system_time = np.zeros(self.N)
        self.total_time = np.zeros(self.N)
        for i in range(self.N):
            if self.rank == 0:
                print("Parameter nr {} out of {}".format(i + 1, self.N))
            mu_lhs = self.mu_lhs[i]
            mu_rhs = self.mu_rhs[i]
            self.snapshots_matrix[:, i] = self.problem.solve(mu_lhs, mu_rhs)
            self.assemble_time[i] = self.problem.assemble_time
            self.system_time[i] = self.problem.system_time
            self.total_time[i] = self.problem.total_time
            np.savetxt(
                self.snapshot_folder + f"/snapshot_{i}_" + str(self.rank) +
                ".txt", self.snapshots_matrix[:, i][np.newaxis])
        self.save_snapshots()
        self.export_snapshots()

    def compress_with_POD(self,
                          snapshots_matrix=None,
                          tol=None,
                          N_max=None,
                          tol_repartitioning=1e-3,
                          indicator='var'):
        if snapshots_matrix is None:
            if self.snapshots_matrix is not None:
                snapshots_matrix = self.snapshots_matrix
            else:
                raise Exception("Empty snapshot matrix.")
        
        N_train = snapshots_matrix.shape[1]
        if self.problem.n_partitions>=1:
            self.problem.partition_dofswise(snapshots_matrix,
                                            self.problem.n_partitions, tol_repartitioning,
                                            indicator=indicator)
            self.reduced_basis_partitioned = list()
            self.N_POD_partitioned = list()
            
            for n in range(self.problem.n_partitions):
                # assemble snapshots matrix restricted to sub partition in root
                rsnap = snapshots_matrix[
                    self.problem.pmask[n], :].reshape(-1)
                snap_matrix = np.zeros(
                    N_train * self.problem.plocalDofsGathered[:, n].sum())
                self.commMPI.Gatherv(rsnap, [
                    snap_matrix, self.problem.plocalDofsGathered[:, n] *
                    N_train, self.problem.pdown[n] * N_train, MPI.DOUBLE
                ])
                snap_matrix = snap_matrix.reshape(-1, N_train)
                print("Snapshots matrix shape: ", snap_matrix.shape)
                
                rdim = np.zeros(1, dtype=np.int32)

                if self.rank == 0:
                    # if n==0:
                    #     self.POD = POD(snap_matrix, tolPOD=tol, NPOD=2)
                    # else:
                    #     self.POD = POD(snap_matrix, tolPOD=tol, NPOD=N_max)
                    self.POD = POD(snap_matrix, tolPOD=tol, NPOD=N_max)
                    rdim[0] = self.POD.basis.shape[1]
                    basis = self.POD.basis.reshape(-1)
                    print("DEBUG REC: ", self.POD.eval_reconstruction_error(snap_matrix), self.POD.basis.shape[1])
                
                self.commMPI.barrier()
                
                self.commMPI.Bcast(rdim)
                if self.rank != 0:
                    basis = np.zeros(
                        rdim[0] * self.problem.plocalDofsGathered[:, n].sum())
                
                print(
                    "partition number: ", n, " mask size: ",
                    np.ones(self.problem.localDofsGathered[self.rank])[
                        self.problem.pmask[n]].sum(), "/",
                    self.problem.localDofsGathered[self.rank])
                
                localMaskedBasis = np.zeros(
                    rdim[0] * self.problem.plocalDofsGathered[self.rank, n], dtype=np.float64)
                basis = np.ascontiguousarray(basis)
                self.commMPI.Scatterv([
                    basis, self.problem.plocalDofsGathered[:, n] * rdim[0],
                    self.problem.pdown[n] * rdim[0], MPI.DOUBLE
                ], localMaskedBasis)

                localMaskedBasis = localMaskedBasis.reshape(-1, rdim[0])
                maskedBasis = np.zeros(
                    (self.problem.localDofsGathered[self.rank], rdim[0]))                
                maskedBasis[self.problem.pmask[n]] = localMaskedBasis
                
                # scatter basis to each core, and fill with 0 outside mask
                self.reduced_basis_partitioned.append(maskedBasis)
                self.N_POD_partitioned.append(rdim[0])
        else:
            # TODO fix normalizer
            # self.normalizer = self.problem.normalize(self.snapshots_matrix)
            
            self.POD = POD(snapshots_matrix, tolPOD=tol, NPOD=N_max)
            self.reduced_basis = self.POD.basis
            self.N_POD = self.POD.NPOD
        
        self.save_reduced_space()
        self.export_reduced_space()

    def compress_for_partitioning_studies(self, recErr, snapshots_matrix=None, tol=None, N_max=None, tol_repartitioning=1e-3, indicator='var'):
        if snapshots_matrix is None:
            if self.snapshots_matrix is not None:
                snapshots_matrix = self.snapshots_matrix
            else:
                raise Exception("Empty snapshot matrix.")
        
        N_train = snapshots_matrix.shape[1]
        
        if self.problem.n_partitions>=1:
            self.problem.partition_dofswise(snapshots_matrix,
                                            self.problem.n_partitions, tol_repartitioning=tol_repartitioning, indicator=indicator)
            self.reduced_basis_partitioned = list()
            self.N_POD_partitioned = list()
            
            num = 0
            denom = 0
            for n in range(self.problem.n_partitions):
                # assemble snapshots matrix restricted to sub partition in root
                rsnap = snapshots_matrix[
                    self.problem.pmask[n], :].reshape(-1)
                snap_matrix = np.zeros(
                    N_train * self.problem.plocalDofsGathered[:, n].sum())
                self.commMPI.Gatherv(rsnap, [
                    snap_matrix, self.problem.plocalDofsGathered[:, n] *
                    N_train, self.problem.pdown[n] * N_train, MPI.DOUBLE
                ])
                snap_matrix = snap_matrix.reshape(-1, N_train)
                print("Snapshots matrix shape: ", snap_matrix.shape)
                
                rdim = np.zeros(1, dtype=np.int32)

                if self.rank == 0:
                    self.POD = POD(snap_matrix, tolPOD=tol, NPOD=N_max)
                    rdim[0] = self.POD.basis.shape[1]
                    recErr.append(self.POD.eval_reconstruction_error(snap_matrix))
                    num += np.linalg.norm(snap_matrix - self.POD.basis.dot(self.POD.basis.T.dot(snap_matrix)), axis=0)**2
                    denom +=np.linalg.norm(snap_matrix, axis=0)**2
                self.commMPI.barrier()
            if self.rank == 0:
                recErr.append(np.max(np.sqrt(num/denom)))
        return recErr
                
    def compress_with_POD_component(self, tol=None, N_max=None):
        self.separate_components_training()
        self.POD_components = []
        self.reduced_basis_components = []
        for i_var in range(self.problem.system_dim):
            self.POD_components.append(
                POD(self.snapshot_components[i_var, :, :],
                    tolPOD=tol,
                    NPOD=N_max))
            self.reduced_basis_components.append(
                self.POD_components[i_var].basis)
        self.compute_reduce_space_from_components()
        self.N_POD = self.NPOD_sum
        self.save_reduced_space()
        self.export_reduced_space()

    def compute_reduce_space_from_components(self):
        self.NPOD_sum = sum(
            [reduced_space.NPOD for reduced_space in self.POD_components])
        self.reduced_basis = np.zeros(
            (self.problem.N_dofs_local, self.NPOD_sum))
        i_basis = 0
        for i_var in range(self.problem.system_dim):
            for i in range(self.POD_components[i_var].NPOD):
                self.reduced_basis[:,
                                   i_basis] = self.problem.expand_one_component(
                                       self.reduced_basis_components[i_var][:, i],
                                       i_var)
                i_basis += 1

    def separate_components_training(self):
        self.snapshot_components = np.zeros(
            (self.problem.system_dim, self.problem.N_dofs_per_variable,
             self.N))
        for i in range(self.N):
            self.snapshot_components[:, :,
                                     i] = self.problem.separate_components(
                                         self.snapshots_matrix[:, i])

    def save_snapshots(self):
        # Only numpy format to be easily read by this class again
        file_names = [
            "snapshots_matrix", "assemble_time", "system_time", "total_time",
            "parameter_domain_lhs", "parameter_domain_rhs"
        ]
        variables = [
            self.snapshots_matrix, self.assemble_time, self.system_time,
            self.total_time, self.parameter_domain_lhs,
            self.parameter_domain_rhs
        ]

        for file_name, variable in zip(file_names, variables):
            np.save(
                self.snapshot_folder + "/" + file_name + "_" + str(self.rank) +
                ".npy", variable)

    def export_snapshots(self):
        # Export vectors in txt to be read by deal.II
        for i in range(self.snapshots_matrix.shape[1]):
            np.savetxt(
                self.snapshot_folder + f"/snapshot_{i}_" + str(self.rank) +
                ".txt", self.snapshots_matrix[:, i][np.newaxis])

            if self.rank == 0:
                np.savetxt(
                    self.snapshot_folder + f"/parameters_lhs_snapshot_{i}.txt",
                    self.parameter_domain_lhs[i, :][np.newaxis])
                np.savetxt(
                    self.snapshot_folder + f"/parameters_rhs_snapshot_{i}.txt",
                    self.parameter_domain_rhs[i, :][np.newaxis])

    def save_reduced_space(self):
        if self.problem.n_partitions==0:
            np.save(
                self.snapshot_folder + "/reduced_basis_" + str(self.rank) +
                ".npy", self.reduced_basis)
        else:
            np.save(
                self.snapshot_folder + "/reduced_basis_" + str(self.rank) +
                ".npy", np.hstack(self.reduced_basis_partitioned))

    def export_reduced_space(self):
        if self.problem.n_partitions==0:
            # Export vectors in txt to be read by deal.II
            for i in range(self.reduced_basis.shape[1]):
                np.savetxt(
                    self.snapshot_folder + f"/basis_{i}_" + str(self.rank) +
                    ".txt", self.reduced_basis[:, i][np.newaxis])
        else:
            # Export vectors in txt to be read by deal.II
            idx = 0
            for n in range(self.problem.n_partitions):
                for i in range(self.reduced_basis_partitioned[n].shape[1]):
                    np.savetxt(
                        self.snapshot_folder + f"/basis_{idx}_" +
                        str(self.rank) + ".txt",
                        self.reduced_basis_partitioned[n][:, i])
                    idx += 1

    def load_snapshots(self):
        try:
            if os.path.exists("./" + self.problem.folder + "/snapshots"):
                self.snapshots_matrix = np.load(self.snapshot_folder +
                                                "/snapshots_matrix_" +
                                                str(self.rank) + ".npy")
                self.assemble_time = np.load(self.snapshot_folder +
                                             "/assemble_time_" +
                                             str(self.rank) + ".npy")
                self.system_time = np.load(self.snapshot_folder +
                                           "/system_time_" + str(self.rank) +
                                           ".npy")
                self.total_time = np.load(self.snapshot_folder +
                                          "/total_time_" + str(self.rank) +
                                          ".npy")

                self.parameter_domain_lhs = np.load(self.snapshot_folder +
                                                    "/parameter_domain_lhs_" +
                                                    str(self.rank) + ".npy")
                self.parameter_domain_rhs = np.load(self.snapshot_folder +
                                                    "/parameter_domain_rhs_" +
                                                    str(self.rank) + ".npy")
                self.evaluate_domain_parameters()
                if self.snapshots_matrix.shape[1] != self.N:
                    print(
                        "Number of snapshots different from input snapshots, recomputing"
                    )
                    self.generate_snapshots()
                print("Snapshots loaded\n")
        except:
            if self.rank==0: print("Loading of snapshots failed")
            # self.generate_snapshots()

    def load_reduced_space(self):
        try:
            self.reduced_basis = np.load(self.snapshot_folder +
                                         "/reduced_basis_" + str(self.rank) +
                                         ".npy")
            print("Loaded reduced basis.")
        except:
            if self.rank==0: print("Failed in reading reduced space, I compute it")
            self.compress_with_POD()


class FOM_testSet:
    def __init__(self, problem, N_training=100, load_snapshots=False):
        self.comm = PETSc.COMM_WORLD
        self.commMPI = MPI.COMM_WORLD
        self.rank = self.comm.getRank()

        self.problem = problem
        folder = self.problem.folder
        if folder == "":
            self.snapshot_folder = "test_snapshots"
        else:
            self.snapshot_folder = folder + "/test_snapshots"
        try:
            os.makedirs(self.snapshot_folder)
        except:
            if self.rank == 0: print("snapshot_folder already exists")
        self.parameter_range_lhs = np.array(self.problem.param_range_lhs)
        self.N_param_lhs = self.parameter_range_lhs.shape[0]
        self.parameter_range_rhs = np.array(self.problem.param_range_rhs)
        self.N_param_rhs = self.parameter_range_rhs.shape[0]
        self.N_dofs_local = self.problem.N_dofs_local
        self.N = N_training
        
        self.mu_lhs = None
        self.mu_rhs = None
        self.parameter_domain_lhs = None
        self.parameter_domain_rhs = None

        self.snapshots_matrix = None
        
        if load_snapshots:
            self.load_snapshots()

    def generate_parameters(self):
        if self.parameter_domain_lhs is None or self.parameter_domain_rhs is None:
            delta_param = np.diff(self.parameter_range_lhs)
            self.parameter_domain_lhs = np.zeros((self.N, self.N_param_lhs))
            self.parameter_domain_rhs = np.zeros((self.N, self.N_param_rhs))

            # self.parameter_domain_lhs = np.load(self.snapshot_folder +
            #                                     "/FOM_parameter_domain_lhs_" +
            #                                     str(self.rank) + ".npy")
            # self.parameter_domain_rhs = np.load(self.snapshot_folder +
            #                                     "/FOM_parameter_domain_rhs_" +
            #                                     str(self.rank) + ".npy")

            if self.rank == 0:
                self.parameter_domain_lhs = np.random.rand(
                    self.N,
                    self.N_param_lhs) * delta_param.T + self.parameter_range_lhs[:,
                                                                                0]

                delta_param = np.diff(self.parameter_range_rhs)
                self.parameter_domain_rhs = np.random.rand(
                    self.N,
                    self.N_param_rhs) * delta_param.T + self.parameter_range_rhs[:,
                                                                                0]

            self.commMPI.Bcast(self.parameter_domain_lhs, root=0)
            self.commMPI.Bcast(self.parameter_domain_rhs, root=0)
            self.evaluate_domain_parameters()
        else:
            self.N = len(self.parameter_domain_lhs)

    def evaluate_domain_parameters(self):
        self.mu_lhs = []
        self.mu_rhs = []
        for i in range(self.N):
            mu_lhs, mu_rhs = self.problem.eval_coeff(
                self.parameter_domain_lhs[i, :],
                self.parameter_domain_rhs[i, :])
            self.mu_lhs.append(mu_lhs)
            self.mu_rhs.append(mu_rhs)

    def generate_snapshots(self):
        self.generate_parameters()  #Randomly selected equispaced
        self.snapshots_matrix = np.zeros((self.N_dofs_local, self.N))
        self.assemble_time = np.zeros(self.N)
        self.system_time = np.zeros(self.N)
        self.total_time = np.zeros(self.N)
        for i in range(self.N):
            if self.rank == 0:
                print("Parameter nr {} out of {}".format(i + 1, self.N))
            mu_lhs = self.mu_lhs[i]
            mu_rhs = self.mu_rhs[i]
            self.snapshots_matrix[:, i] = self.problem.solve(mu_lhs, mu_rhs)
            self.assemble_time[i] = self.problem.assemble_time
            self.system_time[i] = self.problem.system_time
            self.total_time[i] = self.problem.total_time
        self.save_snapshots()
        self.export_snapshots()

    def separate_components_training(self):
        self.snapshot_components = np.zeros(
            (self.problem.system_dim, self.problem.N_dofs_per_variable,
             self.N))
        for i in range(self.N):
            self.snapshot_components[:, :,
                                     i] = self.problem.separate_components(
                                         self.snapshots_matrix[:, i])

    def save_snapshots(self):
        # Only numpy format to be easily read by this class again
        file_names = [
            "snapshots_matrix", "assemble_time", "system_time", "total_time",
            "parameter_domain_lhs", "parameter_domain_rhs"
        ]
        variables = [
            self.snapshots_matrix, self.assemble_time, self.system_time,
            self.total_time, self.parameter_domain_lhs,
            self.parameter_domain_rhs
        ]

        for file_name, variable in zip(file_names, variables):
            np.save(
                self.snapshot_folder + "/FOM_" + file_name + "_" +
                str(self.rank) + ".npy", variable)

    def export_snapshots(self):
        # Export vectors in txt to be read by deal.II
        for i in range(self.snapshots_matrix.shape[1]):
            np.savetxt(
                self.snapshot_folder + f"/FOM_snapshot_{i}_" + str(self.rank) +
                ".txt", self.snapshots_matrix[:, i][np.newaxis])

            if self.rank == 0:
                np.savetxt(
                    self.snapshot_folder + f"/parameters_lhs_snapshot_{i}.txt",
                    self.parameter_domain_lhs[i, :][np.newaxis])
                np.savetxt(
                    self.snapshot_folder + f"/parameters_rhs_snapshot_{i}.txt",
                    self.parameter_domain_rhs[i, :][np.newaxis])

    def load_snapshots(self):
        try:
            if os.path.exists("./" + self.problem.folder + "/snapshots"):
                self.snapshots_matrix = np.load(self.snapshot_folder +
                                                "/FOM_snapshots_matrix_" +
                                                str(self.rank) + ".npy")
                self.assemble_time = np.load(self.snapshot_folder +
                                             "/FOM_assemble_time_" +
                                             str(self.rank) + ".npy")
                self.system_time = np.load(self.snapshot_folder +
                                           "/FOM_system_time_" +
                                           str(self.rank) + ".npy")
                self.total_time = np.load(self.snapshot_folder +
                                          "/FOM_total_time_" + str(self.rank) +
                                          ".npy")
                self.parameter_domain_lhs = np.load(
                    self.snapshot_folder + "/FOM_parameter_domain_lhs_" +
                    str(self.rank) + ".npy")
                self.parameter_domain_rhs = np.load(
                    self.snapshot_folder + "/FOM_parameter_domain_rhs_" +
                    str(self.rank) + ".npy")
                self.evaluate_domain_parameters()
                print("Snapshots loaded\n")
        except:
            print("Loading of snapshots failed.")
            # self.generate_snapshots()
