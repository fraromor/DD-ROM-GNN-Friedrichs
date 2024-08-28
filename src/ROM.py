from itertools import accumulate
import time

import numpy as np
import matplotlib.pyplot as plt
from mpi4py import MPI

from src.FOM import FOM_testSet

import logging

class ROM:
    def __init__(self,
                 problem,
                 basis,
                 normalizer=None):
        self.comm = MPI.COMM_WORLD
        self.size = self.comm.Get_size()
        self.rank = self.comm.Get_rank()

        self.pr = problem
        self.basis = basis

        self.normalizer = normalizer  #TODO normalizer

        (self.N_dofs_local, self.N_RB_local) = self.basis.shape

        self.N_dofs_global = np.zeros(1, dtype=np.int32)
        self.comm.Allreduce(np.array([self.N_dofs_local], dtype=np.int32), self.N_dofs_global)

        # set localDofsGathered, only at root
        split = np.array([self.N_dofs_local], dtype=np.int32)
        self.localDofsGathered = np.zeros(self.size,
                                          dtype=np.int32)
        self.comm.Allgatherv(split, [
            self.localDofsGathered,
            np.ones(self.size, dtype=np.int32),
            np.arange(self.size, dtype=np.int32), MPI.INT32_T
        ])

        # set lower and upper indices of blocks
        self.up = np.array(list(accumulate(self.localDofsGathered)),
                           dtype=np.int32)
        self.down = np.zeros(self.size, dtype=np.int32)
        self.down[1:] = self.up[:-1]
        self.down[0] = 0

        # set local RB dimensions, only at root
        N_RB_global = np.array([self.N_RB_local], dtype=np.int32)
        self.N_RB_globalGathered = np.zeros(self.size,
                                            dtype=np.int32)
        self.comm.Allgatherv(N_RB_global, [
            self.N_RB_globalGathered,
            np.ones(self.size, dtype=np.int32),
            np.arange(self.size, dtype=np.int32), MPI.INT32_T
        ])
        self.N_RB_global = int(self.N_RB_globalGathered.sum())
        
        # set lower and upper indices of reduced blocks
        self.up_rb = np.array(list(accumulate(self.N_RB_globalGathered)),
                              dtype=np.int32)
        self.down_rb = np.zeros(self.size, dtype=np.int32)
        self.down_rb[1:] = self.up_rb[:-1]
        self.down_rb[0] = 0

        if self.rank == 0:
            print("Local dofs: ", self.localDofsGathered)
            print("Local reduced dims: ", self.N_RB_globalGathered)
            print("Reduced dimension: ", self.N_RB_global)
            print("Down and up indices ", self.down, self.up)
            print("Down and up indices rb ", self.down_rb, self.up_rb)

        # only root vars
        self.solution_global = None
        self.affine_lhs = None
        self.affine_rhs = None

        # all processes
        self.solution_local = None
        self.solution_reconstructed_local = None
        
        # parameters
        self.mu_lhs = None
        self.mu_rhs = None
        self.mu_penalty = None
        
        # timings
        self.timings_rom = list()

        logging.info('End ROM init')
        self.assemble_reduced_operators()
        logging.info('End ROM assemble')

    def assemble_reduced_operators(self):
                
        if self.rank == 0:
            self.affine_rhs = list()
            self.affine_lhs = list()
            self.affine_penalty_lhs = list()

        for i_rhs, vec in enumerate(self.pr.affine_rhs):
            vec_tmp = self.basis.T @ vec
            vecGathered = np.zeros(self.N_RB_global)
            self.comm.Gatherv(
                vec_tmp,
                [vecGathered, self.N_RB_local, self.down_rb, MPI.DOUBLE])

            if self.rank == 0:
                self.affine_rhs.append(vecGathered)
                np.save("./"+self.pr.folder+"/ROM/rom_rhs_"+str(i_rhs)+".npy", vecGathered)

        rowBlocks = [list() for n in range(len(self.pr.affine_lhs))]

        for col_block in range(self.size):
            if self.rank == col_block:
                basisBlock = np.ravel(self.basis)
            else:
                basisBlock = np.empty(self.localDofsGathered[col_block] *
                                    self.N_RB_globalGathered[col_block])

            self.comm.barrier()
            self.comm.Bcast(basisBlock, root=col_block)

            basisBlock = basisBlock.reshape(
                self.localDofsGathered[col_block],
                self.N_RB_globalGathered[col_block])

            for i_lhs, mat in enumerate(self.pr.affine_lhs):
                rowBlocks[i_lhs].append(
                    self.basis.T @ mat[:,
                                    self.down[col_block]:self.up[col_block]]
                    @ basisBlock)

        self.comm.barrier()
        for i_lhs, mat in enumerate(self.pr.affine_lhs):
            matGathered = np.hstack(rowBlocks[i_lhs])

            matGlobal = np.zeros(self.N_RB_global**2)
            self.comm.barrier()
            self.comm.Gatherv(matGathered.reshape(-1), [
                matGlobal, self.N_RB_globalGathered * self.N_RB_global,
                self.down_rb * self.N_RB_global, MPI.DOUBLE
            ])
            if self.rank == 0:
                self.affine_lhs.append(
                    matGlobal.reshape(self.N_RB_global, self.N_RB_global))
                np.save("./"+self.pr.folder+"/ROM/rom_lhs_"+str(i_lhs)+".npy", matGlobal.reshape(self.N_RB_global, self.N_RB_global))

        self.comm.barrier()
        
        if self.pr.interface is not None:
            rowBlocks = [list() for n in range(len(self.pr.interface))]

            for col_block in range(self.size):
                if self.rank == col_block:
                    basisBlock = np.ravel(self.basis)
                else:
                    basisBlock = np.empty(self.localDofsGathered[col_block] *
                                        self.N_RB_globalGathered[col_block])

                self.comm.barrier()
                self.comm.Bcast(basisBlock, root=col_block)

                basisBlock = basisBlock.reshape(
                    self.localDofsGathered[col_block],
                    self.N_RB_globalGathered[col_block])

                for i_lhs, mat in enumerate(self.pr.interface):
                    rowBlocks[i_lhs].append(
                        self.basis.T @ mat[:,
                                        self.down[col_block]:self.up[col_block]]
                        @ basisBlock)

            self.comm.barrier()
            for i_lhs, mat in enumerate(self.pr.interface):
                matGathered = np.hstack(rowBlocks[i_lhs])

                matGlobal = np.zeros(self.N_RB_global**2)
                self.comm.barrier()
                self.comm.Gatherv(matGathered.reshape(-1), [
                    matGlobal, self.N_RB_globalGathered * self.N_RB_global,
                    self.down_rb * self.N_RB_global, MPI.DOUBLE
                ])
                if self.rank == 0:
                    self.affine_penalty_lhs.append(
                        matGlobal.reshape(self.N_RB_global, self.N_RB_global))
                    np.save("./"+self.pr.folder+"/ROM/rom_interface_"+str(i_lhs)+".npy")
            
            self.comm.barrier()
        else:
            self.affine_penalty_lhs = None            

    def solve(self, mu_lhs, mu_rhs, mu_penalty=None):
        tic_total = time.time()
        self.set_parameters(mu_lhs, mu_rhs, mu_penalty)
        self.assemble_system()
        self.solve_system()
        toc_total = time.time()
        self.total_time = toc_total - tic_total
        print("Reduced system total time: ", self.total_time)

    def set_parameters(self, mu_lhs, mu_rhs, mu_penalty):
        if self.rank == 0:
            if len(mu_lhs) == len(self.pr.affine_parameters_lhs_names):
                self.mu_lhs = mu_lhs
            else:
                raise ValueError("Parameter lhs of dimension " +
                                 str(len(mu_lhs)) + " instead of " +
                                 str(len(self.pr.affine_parameters_lhs_names)))

            if len(mu_rhs) == len(self.pr.affine_parameters_rhs_names):
                self.mu_rhs = mu_rhs
            else:
                raise ValueError("Parameter rhs of dimension " +
                                 str(len(mu_rhs)) + " instead of " +
                                 str(len(self.pr.affine_parameters_rhs_names)))
            self.mu_penalty = mu_penalty

    def assemble_system(self):
        if self.rank == 0:
            print("Assembling ROM operators for params: lhs {} rhs {}".format(
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

    def solve_system(self, fl_show_sparsity=False):
        if self.rank == 0:
            tic = time.time()
            self.solution_global = np.linalg.solve(self.lhs, self.rhs)
            toc = time.time()
            self.system_time = toc - tic
            
            if fl_show_sparsity:
                plt.spy(self.lhs)
                plt.show()
                plt.close()
                
            print("Reduced system solved in: ", self.system_time)
            self.timings_rom.append(self.system_time)
        
        self.solution_local = np.zeros(self.N_RB_local, dtype=np.float64)
        self.comm.barrier()
        self.comm.Scatterv([
            self.solution_global,
            self.N_RB_globalGathered,
            self.down_rb,
            MPI.DOUBLE
        ], self.solution_local)

    def reconstruct_solution(self):
        self.comm.barrier()
        self.solution_reconstructed_local = self.basis @ self.solution_local
        if self.normalizer is not None:
            self.solution_reconstructed_local = self.solution_reconstructed_local * self.normalizer
        return self.solution_reconstructed_local

    def evaluate_R_norm_a_posteriori(self, mu_lhs, mu_rhs):
        rhs = sum(
            [mu_rhs[i] * self.pr.affine_rhs[i] for i in range(len(mu_rhs))])

        lhs = np.zeros(self.localDofsGathered[self.rank])
        for col_block in range(self.size):
            if self.rank == col_block:
                basisBlock = np.ravel(self.basis)
                solutionBlock = self.solution_local
            else:
                basisBlock = np.empty(self.localDofsGathered[col_block] *
                                      self.N_RB_globalGathered[col_block])
                solutionBlock = np.empty(self.N_RB_globalGathered[col_block])

            self.comm.barrier()
            self.comm.Bcast(basisBlock, root=col_block)

            basisBlock = basisBlock.reshape(
                self.localDofsGathered[col_block],
                self.N_RB_globalGathered[col_block])

            self.comm.barrier()
            self.comm.Bcast(solutionBlock, root=col_block)

            lhs = lhs + sum([
                mu_lhs[i] *
                self.pr.affine_lhs[i][:,
                                      self.down[col_block]:self.up[col_block]]
                @ basisBlock @ solutionBlock for i in range(len(mu_lhs))
            ])

        res = rhs - lhs
        del rhs, lhs, basisBlock, solutionBlock
        cc = self.pr.pdf_constant(mu_lhs)
        enres = self.pr.compute_R_norm_inv(res, cc, mu_lhs, mu_rhs)
        return enres
    
    def evaluate_l2_norm_a_posteriori(self, mu_lhs, mu_rhs):
        rhs = sum(
            [mu_rhs[i] * self.pr.affine_rhs[i] for i in range(len(mu_rhs))])

        lhs = np.zeros(self.localDofsGathered[self.rank])
        for col_block in range(self.size):
            if self.rank==0:logging.info('Estimator L2 {}'.format(col_block))
            t = time.time()
        
            if self.rank == col_block:
                basisBlock = np.ravel(self.basis)
                solutionBlock = self.solution_local
            else:
                basisBlock = np.empty(self.localDofsGathered[col_block] *
                                      self.N_RB_globalGathered[col_block])
                solutionBlock = np.empty(self.N_RB_globalGathered[col_block])

            self.comm.barrier()
            self.comm.Bcast(basisBlock, root=col_block)

            basisBlock = basisBlock.reshape(
                self.localDofsGathered[col_block],
                self.N_RB_globalGathered[col_block])

            self.comm.barrier()
            self.comm.Bcast(solutionBlock, root=col_block)

            lhs = lhs + sum([
                mu_lhs[i] *
                self.pr.affine_lhs[i][:,
                                      self.down[col_block]:self.up[col_block]] @ basisBlock @ solutionBlock
                 for i in range(len(mu_lhs))
            ])
            if self.rank==0:
                logging.info('Estimator L2 {} {}'.format(col_block, time.time()-t))
                
        res = rhs - lhs
        del rhs, lhs, basisBlock, solutionBlock
        cc = self.pr.pdf_constant(mu_lhs)
        logging.info('Begin Invert L2')
        enres = self.pr.compute_l2_norm_inv(res, cc, mu_lhs, mu_rhs)
        return enres

    def solve_and_estimate(self, mu_lhs, mu_rhs, test_snap, mu_penalty=None, mask=None, folder=".", fl_plot=False):
        logging.info('ROM solve and estimate')
        n_params = len(mu_lhs)
        l2errlist = []
        l2estimatorlist = []
        Rerrlist = []
        Restimatorlist = []
        enerrlist = []
        l2enestimatorlist = []
        Renestimatorlist = []
        for i in range(n_params):
            logging.info('Online parameter number: {}'.format(i))
            print("\nOnline parameter number: ", i)
            self.comm.barrier()
            if mu_penalty is not None:
                self.solve(mu_lhs[i], mu_rhs[i], mu_penalty[i])
            else:
                self.solve(mu_lhs[i], mu_rhs[i])
            self.reconstruct_solution()
            logging.info('Solution reconstructed')
            
            ll = self.evaluate_l2_norm_a_posteriori(mu_lhs[i], mu_rhs[i])
            RR = self.evaluate_R_norm_a_posteriori(mu_lhs[i], mu_rhs[i])
            cc = self.pr.pdf_constant(mu_lhs[i])
            
            l2_estimator = ll/cc
            l2_en_estimator = ll/np.sqrt(cc)
            R_estimator = RR
            
            xx = test_snap[:, i]
            if mask is not None:
                xx[mask] = 0
                self.solution_reconstructed_local[mask] = 0
                
            np.savetxt(
                "./"+self.pr.folder+"/snapshots/rsnapshot_" + str(i) + "_" +
                str(self.rank) + ".txt",
                self.solution_reconstructed_local)
            
            denom_l2_normalizer = self.pr.compute_l2_norm(xx)
            denom_R_normalizer = self.pr.compute_R_norm(xx, cc)
            denom_en_normalizer = self.pr.compute_energynorm(xx, mu_lhs[i], mu_rhs[i])

            errl2 = self.pr.compute_l2_norm(self.solution_reconstructed_local-xx)/denom_l2_normalizer
            estl2 = l2_estimator/denom_l2_normalizer
            print("l2 err: ", errl2, estl2, denom_l2_normalizer)
            l2estimatorlist.append(estl2)
            l2errlist.append(errl2)
            
            errR = self.pr.compute_R_norm(self.solution_reconstructed_local-xx, cc)/denom_R_normalizer
            estR = R_estimator/denom_R_normalizer
            print("R err: ", errR, estR, denom_R_normalizer)
            Restimatorlist.append(estR)
            Rerrlist.append(errR)
            
            enerr = self.pr.compute_energynorm(
                    self.solution_reconstructed_local - xx, mu_lhs[i],
                    mu_rhs[i])/denom_en_normalizer
            l2enest = l2_en_estimator/denom_en_normalizer
            Renest = R_estimator/denom_en_normalizer
            print("Energy err: ", enerr, l2enest, Renest,denom_en_normalizer)
            l2enestimatorlist.append(l2enest)
            Renestimatorlist.append(Renest)
            enerrlist.append(enerr)

        if self.rank == 0:
            np.savetxt(folder+"l2err_ROM.txt", np.array(l2errlist))
            np.savetxt(folder+"Rerr_ROM.txt", np.array(Rerrlist))
            np.savetxt(folder+"enerr_ROM.txt", np.array(enerrlist))
            np.savetxt(folder+"l2estimator_ROM.txt", np.array(l2estimatorlist))
            np.savetxt(folder+"Restimator_ROM.txt", np.array(Restimatorlist))
            np.savetxt(folder+"Renestimator_ROM.txt", np.array(Renestimatorlist))
            np.savetxt(folder+"l2enestimator_ROM.txt", np.array(l2enestimatorlist))
            np.savetxt(folder+"local_dofs.txt", self.localDofsGathered)

class pROM:
    def __init__(self,
                 problem,
                 basis,
                 normalizer=None):
        self.comm = MPI.COMM_WORLD
        self.size = self.comm.Get_size()
        self.rank = self.comm.Get_rank()

        self.pr = problem
        self.n_partitions = len(basis)
        self.basis = basis
        self.normalizer = normalizer # TODO

        self.N_dofs_local = self.basis[0].shape[0]
        self.N_RB_plocal = list()
        for n in range(self.n_partitions):
            self.N_RB_plocal.append(self.basis[n].shape[1])

        self.N_dofs_global = np.zeros(1, dtype=np.int32)
        self.comm.Allreduce(np.array([self.N_dofs_local], dtype=np.int32), self.N_dofs_global)

        # set localDofsGathered, only root
        split = np.array([self.N_dofs_local], dtype=np.int32)
        self.localDofsGathered = np.zeros(self.size,
                                          dtype=np.int32)

        self.comm.Allgatherv(split, [
            self.localDofsGathered,
            np.ones(self.size, dtype=np.int32),
            np.arange(self.size, dtype=np.int32), MPI.INT32_T
        ])

        # set upper and lower indices for dofs and rb system
        self.up = np.array(list(accumulate(self.localDofsGathered)),
                           dtype=np.int32)
        self.down = np.zeros(self.size, dtype=np.int32)
        self.down[1:] = self.up[:-1]
        self.down[0] = 0

        self.N_RB_global = sum(self.N_RB_plocal)
        self.up_rb = np.array(list(accumulate(self.N_RB_plocal)),
                              dtype=np.int32)
        self.down_rb = np.zeros(self.n_partitions, dtype=np.int32)
        self.down_rb[1:] = self.up_rb[:-1]
        self.down_rb[0] = 0

        if self.rank == 0:
            print("Local dofs: ", self.localDofsGathered)
            print("Local reduced dims: ", self.N_RB_plocal)
            print("Reduced dimension: ", self.N_RB_global)
            print("Down and up indices ", self.down, self.up)
            print("Down and up indices rb ", self.down_rb, self.up_rb)

        # only root vars
        self.solution_global = None
        self.affine_lhs = None
        self.affine_rhs = None

        # all processes
        self.solution_local = None
        self.solution_reconstructed_local = None
        
        # timings
        self.timings_rom = list()

        self.assemble_reduced_operators()

    def assemble_reduced_operators(self):
        if self.rank == 0:
            self.affine_rhs = list()
            self.affine_lhs = list()

        for vec in self.pr.affine_rhs:
            vec_list = list()
            for n in range(self.n_partitions):
                vec_tmp_loc = self.basis[n].T @ vec
                vec_tmp = np.zeros(vec_tmp_loc.shape[0])
                self.comm.Reduce(vec_tmp_loc, vec_tmp, op=MPI.SUM, root=0)
                vec_list.append(vec_tmp)

            vecGathered = np.hstack(vec_list)
            if self.rank == 0:
                self.affine_rhs.append(vecGathered)

        # init reduced mat in root
        if self.rank == 0:
            for i_lhs, mat in enumerate(self.pr.affine_lhs):
                self.affine_lhs.append(
                    np.zeros((self.N_RB_global, self.N_RB_global)))

        for n in range(self.n_partitions):
            for m in range(self.n_partitions):
                blocks = [
                    np.zeros((self.N_RB_plocal[n], self.N_RB_plocal[m]))
                    for i in range(len(self.pr.affine_lhs))
                ]
                for col_block in range(self.size):
                    if self.rank == col_block:
                        basisBlock = np.ravel(self.basis[m])
                    else:
                        basisBlock = np.empty(
                            self.localDofsGathered[col_block] *
                            self.N_RB_plocal[m])

                    self.comm.barrier()
                    self.comm.Bcast(basisBlock, root=col_block)

                    basisBlock = basisBlock.reshape(
                        self.localDofsGathered[col_block],
                        self.N_RB_plocal[m])

                    for i_lhs, mat in enumerate(self.pr.affine_lhs):
                        blocks[i_lhs] += self.basis[
                            n].T @ mat[:, self.down[col_block]:self.
                                       up[col_block]] @ basisBlock

                for i_lhs, mat in enumerate(self.pr.affine_lhs):
                    rootBlock = np.zeros(
                        (self.N_RB_plocal[n], self.N_RB_plocal[m])).reshape(-1)
                    blockLocal = blocks[i_lhs].reshape(-1)
                    self.comm.Reduce(blockLocal, rootBlock, op=MPI.SUM, root=0)

                    if self.rank == 0:
                        self.affine_lhs[i_lhs][
                            self.down_rb[n]:self.up_rb[n],
                            self.down_rb[m]:self.up_rb[m]] = rootBlock.reshape(
                                self.N_RB_plocal[n], self.N_RB_plocal[m])

        self.comm.barrier()
        
    def solve(self, mu_lhs, mu_rhs):
        tic_total = time.time()
        self.set_parameters(mu_lhs, mu_rhs)
        self.assemble_system()
        self.solve_system()
        toc_total = time.time()
        self.total_time = toc_total - tic_total
        print("Reduced system total time: ", self.total_time)

    def set_parameters(self, mu_lhs, mu_rhs):
        if self.rank == 0:
            if len(mu_lhs) == len(self.pr.affine_parameters_lhs_names):
                self.mu_lhs = mu_lhs
            else:
                raise ValueError("Parameter lhs of dimension " +
                                 str(len(mu_lhs)) + " instead of " +
                                 str(len(self.pr.affine_parameters_lhs_names)))

            if len(mu_rhs) == len(self.pr.affine_parameters_rhs_names):
                self.mu_rhs = mu_rhs
            else:
                raise ValueError("Parameter rhs of dimension " +
                                 str(len(mu_rhs)) + " instead of " +
                                 str(len(self.pr.affine_parameters_rhs_names)))

    def assemble_system(self):
        if self.rank == 0:
            print("Assembling ROM operators for params: lhs {} rhs {}".format(
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
        self.solution_global = np.zeros(self.N_RB_global, dtype=np.float64)
        if self.rank == 0:
            tic = time.time()
            self.solution_global = np.linalg.solve(self.lhs, self.rhs)
            toc = time.time()
            self.system_time = toc - tic
            print("Reduced system solved in: ", self.system_time)
            self.timings_rom.append(self.system_time)
        self.comm.Bcast(self.solution_global, root=0)

    def reconstruct_solution(self):
        self.comm.barrier()
        self.solution_reconstructed_local = np.zeros(self.basis[0].shape[0])
        for n in range(self.n_partitions):
            self.solution_reconstructed_local += self.basis[
                n] @ self.solution_global[self.down_rb[n]:self.up_rb[n]]
        return self.solution_reconstructed_local

    def evaluate_R_norm_a_posteriori(self, mu_lhs, mu_rhs):
        rhs = sum(
            [mu_rhs[i] * self.pr.affine_rhs[i] for i in range(len(mu_rhs))])

        lhs = np.zeros(self.localDofsGathered[self.rank])
        for m in range(self.n_partitions):
            for col_block in range(self.size):
                if self.rank == col_block:
                    basisBlock = np.ravel(self.basis[m])
                else:
                    basisBlock = np.empty(
                        self.localDofsGathered[col_block] *
                        self.N_RB_plocal[m])

                self.comm.barrier()
                self.comm.Bcast(basisBlock, root=col_block)

                basisBlock = basisBlock.reshape(
                    self.localDofsGathered[col_block], self.N_RB_plocal[m])

                lhs += sum([
                    mu_lhs[i] * self.pr.affine_lhs[i]
                    [:, self.down[col_block]:self.up[col_block]] @ basisBlock
                    @ self.solution_global[self.down_rb[m]:self.up_rb[m]]
                    for i in range(len(mu_lhs))
                ])

        res = rhs - lhs
        cc = self.pr.pdf_constant(mu_lhs)
        enres = self.pr.compute_R_norm_inv(res, cc, mu_lhs, mu_rhs)
        return enres
    
    def evaluate_l2_norm_a_posteriori(self, mu_lhs, mu_rhs):
        rhs = sum(
            [mu_rhs[i] * self.pr.affine_rhs[i] for i in range(len(mu_rhs))])

        lhs = np.zeros(self.localDofsGathered[self.rank])
        for m in range(self.n_partitions):
            for col_block in range(self.size):
                if self.rank == col_block:
                    basisBlock = np.ravel(self.basis[m])
                else:
                    basisBlock = np.empty(
                        self.localDofsGathered[col_block] *
                        self.N_RB_plocal[m])

                self.comm.barrier()
                self.comm.Bcast(basisBlock, root=col_block)

                basisBlock = basisBlock.reshape(
                    self.localDofsGathered[col_block], self.N_RB_plocal[m])

                lhs += sum([
                    mu_lhs[i] * self.pr.affine_lhs[i]
                    [:, self.down[col_block]:self.up[col_block]] @ basisBlock
                    @ self.solution_global[self.down_rb[m]:self.up_rb[m]]
                    for i in range(len(mu_lhs))
                ])

        res = rhs - lhs
        cc = self.pr.pdf_constant(mu_lhs)
        enres = self.pr.compute_l2_norm_inv(res, cc, mu_lhs, mu_rhs)
        return enres

    def solve_and_estimate(self, mu_lhs, mu_rhs, test_snap, mask=None, folder=".", fl_plot=False):
        n_params = len(mu_lhs)
        l2errlist = []
        l2estimatorlist = []
        Rerrlist = []
        Restimatorlist = []
        enerrlist = []
        l2enestimatorlist = []
        Renestimatorlist = []
        for i in range(n_params):
            print("\nOnline parameter number: ", i)
            self.comm.barrier()
            self.solve(mu_lhs[i], mu_rhs[i])
            self.reconstruct_solution()
            
            RR = self.evaluate_R_norm_a_posteriori(mu_lhs[i], mu_rhs[i])
            ll = self.evaluate_l2_norm_a_posteriori(mu_lhs[i], mu_rhs[i])
            cc = self.pr.pdf_constant(mu_lhs[i])
            
            l2_estimator = ll/cc
            l2_en_estimator = ll/np.sqrt(cc)
            R_estimator = RR
            
            xx = test_snap[:, i]
            if mask is not None:
                print("Mask solutions")
                xx[mask] = 0
                self.solution_reconstructed_local[mask] = 0
                
            denom_l2_normalizer = self.pr.compute_l2_norm(xx)
            denom_R_normalizer = self.pr.compute_R_norm(xx, cc)
            denom_en_normalizer = self.pr.compute_energynorm(xx, mu_lhs[i], mu_rhs[i])

            np.savetxt(
                "./"+self.pr.folder+"/snapshots/rsnapshot_" + str(i) + "_" +
                str(self.rank) + ".txt",
                self.solution_reconstructed_local)

            errl2 = self.pr.compute_l2_norm(self.solution_reconstructed_local-xx)/denom_l2_normalizer
            estl2 = l2_estimator/denom_l2_normalizer
            print("l2 err: ", errl2, estl2, denom_l2_normalizer)
            l2estimatorlist.append(estl2)
            l2errlist.append(errl2)

            errR = self.pr.compute_R_norm(self.solution_reconstructed_local-xx, cc)/denom_R_normalizer
            estR = R_estimator/denom_R_normalizer
            print("R err: ", errR, estR, denom_R_normalizer)
            Restimatorlist.append(estR)
            Rerrlist.append(errR)
            
            enerr = self.pr.compute_energynorm(
                    xx-self.solution_reconstructed_local, mu_lhs[i],
                    mu_rhs[i])/denom_en_normalizer
            l2enest = l2_en_estimator/denom_en_normalizer
            Renest = R_estimator/denom_en_normalizer
            print("Energy err: ", enerr, l2enest, Renest,denom_en_normalizer, l2_en_estimator)
            l2enestimatorlist.append(l2enest)
            Renestimatorlist.append(Renest)
            enerrlist.append(enerr)

        if self.rank == 0:
            np.savetxt(folder+"l2err_ROM.txt", np.array(l2errlist))
            np.savetxt(folder+"Rerr_ROM.txt", np.array(Rerrlist))
            np.savetxt(folder+"enerr_ROM.txt", np.array(enerrlist))
            np.savetxt(folder+"l2estimator_ROM.txt", np.array(l2estimatorlist))
            np.savetxt(folder+"Restimator_ROM.txt", np.array(Restimatorlist))
            np.savetxt(folder+"Renestimator_ROM.txt", np.array(Renestimatorlist))
            np.savetxt(folder+"l2enestimator_ROM.txt", np.array(l2enestimatorlist))
            np.savetxt(folder+"local_dofs.txt", self.localDofsGathered)

class ErrorAnalysis:
    def __init__(self, offlineStage):
        self.pr = offlineStage.problem
        self.offlineStage = offlineStage
        self.N_train = self.offlineStage.N
        self.commMPI = MPI.COMM_WORLD
        self.size = self.commMPI.Get_size()
        self.rank = self.commMPI.Get_rank()

    def generate_test_set(self, N_test, load_snapshots=False):
        self.N_test = N_test
        self.test_set = FOM_testSet(self.pr,
                                    self.N_test,
                                    load_snapshots=load_snapshots)
        if self.test_set.snapshots_matrix is None:
            if self.rank==0:
                print("Compute test dataset\n.")
            self.test_set.generate_snapshots()
            self.test_set.export_snapshots()

    def save_test_set_ROM_solutions(self,
                                    tol=None,
                                    N_max=None,
                                    components=False):
        self.POD_on_components = components
        if self.POD_on_components:
            self.offlineStage.compress_with_POD_component(tol=tol, N_max=N_max)
        else:
            self.offlineStage.compress_with_POD(tol=tol, N_max=N_max)
        reduced_problem = ROM(self.pr, self.offlineStage.reduced_basis)

        self.errors_l2 = np.zeros(self.test_set.N)
        self.errors_R = np.zeros(self.test_set.N)
        self.errors_energy = np.zeros(self.test_set.N)
        self.estimators_l2 = np.zeros(self.test_set.N)
        self.estimators_R = np.zeros(self.test_set.N)
        self.estimators_energy = np.zeros(self.test_set.N)
        self.speed_ups = np.zeros(self.test_set.N)

        if self.rank == 0: print(f" ========== NRB = {N_max} ==========")

        if not hasattr(self.test_set, "snapshots_matrix"):
            raise ValueError("FOM not computed on test set")

        for j in range(self.test_set.N):
            mu_lhs_test, mu_rhs_test = self.pr.eval_coeff(
                self.test_set.parameter_domain_lhs[j, :],
                self.test_set.parameter_domain_rhs[j, :])
            reduced_problem.solve(mu_lhs_test, mu_rhs_test)
            sol_rom = reduced_problem.reconstruct_solution()
            np.savetxt(
                self.test_set.snapshot_folder + f"/ROM_snapshot_{j}.txt",
                sol_rom[np.newaxis])
            sol_fom = self.test_set.snapshots_matrix[:, j]
            #self.errors_linf[j] = np.max(np.abs(sol_rom - sol_fom))
            cc = self.pr.pdf_constant(mu_lhs_test)
            self.errors_l2[j] = self.pr.compute_l2_norm(sol_rom - sol_fom)
            self.errors_R[j] = self.pr.compute_R_norm(sol_rom - sol_fom, cc)
            self.errors_energy[j] = self.pr.compute_energynorm(
                sol_rom - sol_fom, mu_lhs_test, mu_rhs_test)
            
            estimator = reduced_problem.evaluate_l2_norm_a_posteriori(
                mu_lhs_test, mu_rhs_test)
            self.estimators_l2[j] = estimator/cc
            self.estimators_R[
                j] = reduced_problem.evaluate_R_norm_a_posteriori(
                mu_lhs_test, mu_rhs_test)
            self.estimators_energy[
                j] = estimator/np.sqrt(cc)
            self.speed_ups[
                j] = self.test_set.total_time[j] / reduced_problem.total_time
        return self.errors_l2, self.errors_R, self.errors_energy, self.estimators_l2, self.estimators_R, self.estimators_energy, self.speed_ups

    def plot_error_vs_parameter(self, lhs_rhs_param="lhs", param_index=1):
        param_index = np.int64(param_index)
        if not hasattr(self, "errors_l2"):
            if self.rank == 0: print("Computing the ROM test set")
            self.save_test_set_ROM_solutions()
        if lhs_rhs_param == "lhs":
            plt.figure()
            plt.semilogy(self.test_set.parameter_domain_lhs[:, param_index],
                         self.errors_l2,
                         "o",
                         label=r"$L^2$ error")
            plt.semilogy(self.test_set.parameter_domain_lhs[:, param_index],
                         self.estimators_l2,
                         "x",
                         label=r"$L^2$ error estimator")
            plt.semilogy(self.test_set.parameter_domain_lhs[:, param_index],
                         self.errors_R,
                         "o",
                         label="R-norm error")
            plt.semilogy(self.test_set.parameter_domain_lhs[:, param_index],
                         self.estimators_R,
                         "x",
                         label=r"R-norm error estimator")
            plt.grid(which="both")
            plt.xlabel(
                f"Parameter {self.pr.affine_parameters_lhs_names[param_index]}"
            )
            plt.ylabel(f"Error")
            plt.legend()
            plt.tight_layout()
            plt.savefig(
                self.test_set.snapshot_folder +
                f"/Error_vs_parameter_lhs_{param_index}" +
                f"_NRB_{np.shape(self.offlineStage.reduced_basis)[1]}.pdf")
            plt.show(block=False)

            plt.figure()
            plt.semilogy(self.test_set.parameter_domain_lhs[:, param_index],
                         self.errors_energy,
                         "o",
                         label="Energy error")
            plt.semilogy(self.test_set.parameter_domain_lhs[:, param_index],
                         self.estimators_energy,
                         "x",
                         label=r"$L^2$ Energy error estimator")
            plt.grid(which="both")
            plt.semilogy(self.test_set.parameter_domain_lhs[:, param_index],
                         self.estimators_R,
                         "x",
                         label="R-norm Energy error estimator")
            plt.grid(which="both")
            plt.xlabel(
                f"Parameter {self.pr.affine_parameters_lhs_names[param_index]}"
            )
            plt.ylabel(f"Error")
            plt.legend()
            plt.tight_layout()
            plt.savefig(
                self.test_set.snapshot_folder +
                f"/Error_energy_vs_parameter_lhs_{param_index}" +
                f"_NRB_{np.shape(self.offlineStage.reduced_basis)[1]}.pdf")
            plt.show(block=False)

    def compute_error_analysis(self, N_max, tol_rep=0.5, components=False, folder=None):
        if folder is None:
            folder = self.test_set.snapshot_folder
            
        self.POD_on_components = components
        self.NROMs = range(2, N_max + 1)
        NNROMs = len(self.NROMs)
        self.mean_errors_l2 = np.zeros(NNROMs)
        self.mean_errors_R = np.zeros(NNROMs)
        #self.mean_errors_linf = np.zeros(NNROMs)
        self.mean_errors_nrg = np.zeros(NNROMs)
        self.mean_estimators_l2 = np.zeros(NNROMs)
        self.mean_estimators_R = np.zeros(NNROMs)
        self.mean_estimators_energy = np.zeros(NNROMs)
        self.mean_speed_up = np.zeros(NNROMs)

        self.std_errors_l2 = np.zeros(NNROMs)
        self.std_errors_R = np.zeros(NNROMs)
        #self.std_errors_linf = np.zeros(NNROMs)
        self.std_errors_nrg = np.zeros(NNROMs)
        self.std_estimators_l2 = np.zeros(NNROMs)
        self.std_estimators_R = np.zeros(NNROMs)
        self.std_estimators_energy = np.zeros(NNROMs)
        self.std_speed_up = np.zeros(NNROMs)
        
        self.localDofsGathered = []
        for iROM, NROM in enumerate(self.NROMs):
            if self.rank == 0:
                print(f"========= Reducing with {NROM} POD bases =========")
            if self.POD_on_components:
                self.offlineStage.compress_with_POD_component(N_max=NROM, tol_repartitioning=tol_rep)
            else:
                self.offlineStage.compress_with_POD(N_max=NROM, tol_repartitioning=tol_rep)

            if self.offlineStage.problem.n_partitions>=1:
                reduced_problem = pROM(self.pr, self.offlineStage.reduced_basis_partitioned)
                self.localDofsGathered.append(self.offlineStage.problem.plocalDofsGathered.reshape(-1))
            else:
                reduced_problem = ROM(self.pr, self.offlineStage.reduced_basis)
                self.localDofsGathered.append(self.offlineStage.problem.localDofsGathered)

            errors_l2 = np.zeros(self.test_set.N)
            errors_R = np.zeros(self.test_set.N)
            errors_energy = np.zeros(self.test_set.N)
            estimators_l2 = np.zeros(self.test_set.N)
            estimators_R = np.zeros(self.test_set.N)
            estimators_energy = np.zeros(self.test_set.N)
            speed_ups = np.zeros(self.test_set.N)

            for j in range(self.test_set.N):
                mu_lhs_test = self.test_set.mu_lhs[j]
                mu_rhs_test = self.test_set.mu_rhs[j]
                reduced_problem.solve(mu_lhs_test, mu_rhs_test)
                sol_rom = reduced_problem.reconstruct_solution()
                sol_fom = self.test_set.snapshots_matrix[:, j]
                #errors_linf[j] = np.max(np.abs(sol_rom - sol_fom))
                cc = self.pr.pdf_constant(mu_lhs_test)
                errors_l2[j] = self.pr.compute_l2_norm(sol_rom - sol_fom)
                errors_R[j] = self.pr.compute_R_norm(sol_rom - sol_fom, cc)
                errors_energy[j] = self.pr.compute_energynorm(
                    sol_rom - sol_fom, mu_lhs_test, mu_rhs_test)
                estimator = reduced_problem.evaluate_l2_norm_a_posteriori(mu_lhs_test, mu_rhs_test)
                estimators_l2[j] = estimator/cc
                estimators_R[j] = reduced_problem.evaluate_R_norm_a_posteriori(mu_lhs_test, mu_rhs_test)
                estimators_energy[j] = estimator/np.sqrt(cc)
                speed_ups[j] = self.test_set.total_time[
                    j] / reduced_problem.total_time
                
            self.mean_errors_l2[iROM] = np.mean(errors_l2)
            self.mean_errors_R[iROM] = np.mean(errors_R)
            #self.mean_errors_linf[iROM] = np.mean(errors_linf)
            self.mean_errors_nrg[iROM] = np.mean(errors_energy)
            self.mean_estimators_l2[iROM] = np.nanmean(estimators_l2)
            self.mean_estimators_R[iROM] = np.mean(estimators_R)
            self.mean_estimators_energy[iROM] = np.mean(estimators_energy)
            self.mean_speed_up[iROM] = np.mean(speed_ups)

            self.std_errors_l2[iROM] = np.std(errors_l2)
            self.std_errors_R[iROM] = np.std(errors_R)
            #self.std_errors_linf[iROM] = np.std(errors_linf)
            self.std_errors_nrg[iROM] = np.std(errors_energy)
            self.std_estimators_l2[iROM] = np.std(estimators_l2)
            self.std_estimators_R[iROM] = np.std(estimators_R)
            self.std_estimators_energy[iROM] = np.std(estimators_energy)
            self.std_speed_up[iROM] = np.std(speed_ups)
        
        np.savetxt(folder+"local_dofs_nrb.txt", np.vstack(self.localDofsGathered))
        np.savetxt(folder + "/mean_errors_l2_nrb.txt",
                   self.mean_errors_l2)
        np.savetxt(folder + "/mean_errors_R_nrb.txt",
                   self.mean_errors_R)
        # np.savetxt(folder + "/mean_errors_linf_nrb.txt",
        #            self.mean_errors_linf)
        np.savetxt(folder + "/mean_errors_nrg_nrb.txt",
                   self.mean_errors_nrg)
        np.savetxt(folder + "/mean_estimators_l2_nrb.txt",
                   self.mean_estimators_l2)
        np.savetxt(folder + "/mean_estimators_R_nrb.txt",
                   self.mean_estimators_R)
        np.savetxt(folder + "/mean_estimators_nrg_nrb.txt",
                   self.mean_estimators_energy)
        np.savetxt(folder + "/mean_speed_up_nrb.txt",
                   self.mean_speed_up)
        np.savetxt(folder + "/std_errors_l2_nrb.txt",
                   self.std_errors_l2)
        np.savetxt(folder + "/std_errors_R_nrb.txt",
                   self.std_errors_R)
        #np.savetxt(folder + "/std_errors_linf_nrb.txt",
        #           self.std_errors_linf)
        np.savetxt(folder + "/std_errors_nrg_nrb.txt",
                   self.std_errors_nrg)
        np.savetxt(folder + "/std_estimators_l2_nrb.txt",
                   self.std_estimators_l2)
        np.savetxt(folder + "/std_estimators_R_nrb.txt",
                   self.std_estimators_R)
        np.savetxt(folder + "/std_estimators_nrg_nrb.txt",
                   self.std_estimators_energy)
        np.savetxt(folder + "/std_speed_up_nrb.txt",
                   self.std_speed_up)

    def compute_error_analysis_toleranace(self, N_max, tol_rep, components=False, folder=None):
        if folder is None:
            folder = self.test_set.snapshot_folder
            
        self.POD_on_components = components
        self.NROMs = range(2, N_max + 1)
        NNROMs = len(self.NROMs)
        self.mean_errors_l2 = np.zeros(NNROMs)
        self.mean_errors_R = np.zeros(NNROMs)
        #self.mean_errors_linf = np.zeros(NNROMs)
        self.mean_errors_nrg = np.zeros(NNROMs)
        self.mean_estimators_l2 = np.zeros(NNROMs)
        self.mean_estimators_R = np.zeros(NNROMs)
        self.mean_estimators_energy = np.zeros(NNROMs)
        self.mean_speed_up = np.zeros(NNROMs)

        self.std_errors_l2 = np.zeros(NNROMs)
        self.std_errors_R = np.zeros(NNROMs)
        #self.std_errors_linf = np.zeros(NNROMs)
        self.std_errors_nrg = np.zeros(NNROMs)
        self.std_estimators_l2 = np.zeros(NNROMs)
        self.std_estimators_R = np.zeros(NNROMs)
        self.std_estimators_energy = np.zeros(NNROMs)
        self.std_speed_up = np.zeros(NNROMs)
        
        self.reduced_dims = []
        self.tolerance = np.zeros(NNROMs)
        self.localDofsGathered = []
        for iROM, NROM in enumerate(self.NROMs):
            if self.rank == 0:
                print(
                    f"========= Reducing with {np.exp(-2-NROM)} tol =========")
            
            self.tolerance[iROM]= np.exp(-2-NROM)
            if self.POD_on_components:
                self.offlineStage.compress_with_POD_component(tol=np.exp(-2 -
                                                                        NROM), tol_repartitioning=tol_rep)
            else:
                self.offlineStage.compress_with_POD(tol=np.exp(-2 - NROM), tol_repartitioning=tol_rep)
            
            if self.offlineStage.problem.n_partitions>=1:
                reduced_problem = pROM(self.pr, self.offlineStage.reduced_basis_partitioned)
                self.reduced_dims.append(reduced_problem.N_RB_plocal)
                self.localDofsGathered.append(self.offlineStage.problem.plocalDofsGathered.reshape(-1))
            else:
                reduced_problem = ROM(self.pr, self.offlineStage.reduced_basis)
                self.reduced_dims.append(reduced_problem.N_RB_globalGathered)
                self.localDofsGathered.append(self.offlineStage.problem.localDofsGathered)
            
            errors_l2 = np.zeros(self.test_set.N)
            errors_R = np.zeros(self.test_set.N)
            errors_energy = np.zeros(self.test_set.N)
            estimators_l2 = np.zeros(self.test_set.N)
            estimators_R = np.zeros(self.test_set.N)
            estimators_energy = np.zeros(self.test_set.N)
            speed_ups = np.zeros(self.test_set.N)

            for j in range(self.test_set.N):
                mu_lhs_test = self.test_set.mu_lhs[j]
                mu_rhs_test = self.test_set.mu_rhs[j]
                reduced_problem.solve(mu_lhs_test, mu_rhs_test)
                sol_rom = reduced_problem.reconstruct_solution()
                sol_fom = self.test_set.snapshots_matrix[:, j]
                #errors_linf[j] = np.max(np.abs(sol_rom - sol_fom))
                cc = self.pr.pdf_constant(mu_lhs_test)
                errors_l2[j] = self.pr.compute_l2_norm(sol_rom - sol_fom)
                errors_R[j] = self.pr.compute_R_norm(sol_rom - sol_fom, cc)
                errors_energy[j] = self.pr.compute_energynorm(
                    sol_rom - sol_fom, mu_lhs_test, mu_rhs_test)
                estimator = reduced_problem.evaluate_l2_norm_a_posteriori(mu_lhs_test, mu_rhs_test)
                estimators_l2[j] = estimator/cc
                estimators_R[j] = reduced_problem.evaluate_R_norm_a_posteriori(mu_lhs_test, mu_rhs_test)
                estimators_energy[j] = estimator/np.sqrt(cc)
                speed_ups[j] = self.test_set.total_time[
                    j] / reduced_problem.total_time
            self.mean_errors_l2[iROM] = np.mean(errors_l2)
            self.mean_errors_R[iROM] = np.mean(errors_R)
            #self.mean_errors_linf[iROM] = np.mean(errors_linf)
            self.mean_errors_nrg[iROM] = np.mean(errors_energy)
            self.mean_estimators_l2[iROM] = np.mean(estimators_l2)
            self.mean_estimators_R[iROM] = np.mean(estimators_R)
            self.mean_estimators_energy[iROM] = np.mean(estimators_energy)
            self.mean_speed_up[iROM] = np.mean(speed_ups)

            self.std_errors_l2[iROM] = np.std(errors_l2)
            self.std_errors_R[iROM] = np.std(errors_R)
            #self.std_errors_linf[iROM] = np.std(errors_linf)
            self.std_errors_nrg[iROM] = np.std(errors_energy)
            self.std_estimators_l2[iROM] = np.std(estimators_l2)
            self.std_estimators_R[iROM] = np.std(estimators_R)
            self.std_estimators_energy[iROM] = np.std(estimators_energy)
            self.std_speed_up[iROM] = np.std(speed_ups)
        
        np.savetxt(folder+"local_dofs_tol.txt", np.vstack(self.localDofsGathered))
        np.savetxt(folder + "/reduced_dimensions_tol.txt",
                   np.vstack(self.reduced_dims))
        np.savetxt(folder + "/tolerance_tol.txt",
                   self.tolerance)
        np.savetxt(folder + "/mean_errors_l2_tol.txt",
                   self.mean_errors_l2)
        np.savetxt(folder + "/mean_errors_R_tol.txt",
                   self.mean_errors_R)
        # np.savetxt(folder + "/mean_errors_linf_tol.txt",
        #            self.mean_errors_linf)
        np.savetxt(folder + "/mean_errors_nrg_tol.txt",
                   self.mean_errors_nrg)
        np.savetxt(folder + "/mean_estimators_l2_tol.txt",
                   self.mean_estimators_l2)
        np.savetxt(folder + "/mean_estimators_R_tol.txt",
                   self.mean_estimators_R)
        np.savetxt(folder + "/mean_estimators_nrg_tol.txt",
                   self.mean_estimators_energy)
        np.savetxt(folder + "/mean_speed_up_tol.txt",
                   self.mean_speed_up)
        np.savetxt(folder + "/std_errors_l2_tol.txt",
                   self.std_errors_l2)
        np.savetxt(folder + "/std_errors_R_tol.txt",
                   self.std_errors_R)
        #np.savetxt(folder + "/std_errors_linf_tol.txt",
        #           self.std_errors_linf)
        np.savetxt(folder + "/std_errors_nrg_tol.txt",
                   self.std_errors_nrg)
        np.savetxt(folder + "/std_estimators_l2_tol.txt",
                   self.std_estimators_l2)
        np.savetxt(folder + "/std_estimators_R_tol.txt",
                   self.std_estimators_R)
        np.savetxt(folder + "/std_estimators_nrg_tol.txt",
                   self.std_estimators_energy)
        np.savetxt(folder + "/std_speed_up_tol.txt",
                   self.std_speed_up)

    def plot_errors(self, folder=None, fl_plot=False, title='nrb'):
        if folder is None:
            folder = self.test_set.snapshot_folder
        if self.rank == 0:
            plt.figure()
            plt.semilogy(self.NROMs, self.mean_errors_l2, label=r"$L^2$ error")
            plt.semilogy(self.NROMs, self.mean_errors_R, label="R-norm error")
            #plt.semilogy(self.NROMs, self.mean_errors_linf, label="Linf error")
            plt.semilogy(self.NROMs,
                         self.mean_errors_nrg,
                         label="Energy error")
            plt.semilogy(self.NROMs,
                         self.mean_estimators_l2,
                         "--",
                         label=r"$L^2$ Error estimator")
            plt.semilogy(self.NROMs,
                         self.mean_estimators_R,
                         "--",
                         label="R-norm Error estimator")
            plt.semilogy(self.NROMs,
                         self.mean_estimators_energy,
                         "--",
                         label="Error energy estimator")
            plt.xlabel("ROM dimension")
            plt.xticks(self.NROMs)
            plt.legend()
            plt.grid(which='both')
            plt.tight_layout()
            plt.savefig(folder + "/error_decay_"+title+".pdf")
            if fl_plot:
                plt.show()
            plt.close()

            plt.figure()
            plt.semilogy(self.NROMs, self.mean_speed_up, label="Speed up")
            plt.xlabel("ROM dimension")
            plt.xticks(self.NROMs)
            plt.legend()
            plt.grid(which='both')
            plt.tight_layout()
            plt.savefig(folder + "/speed_up_"+title+".pdf")
            if fl_plot:
                plt.show()
            plt.close()
