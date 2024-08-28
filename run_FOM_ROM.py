import numpy as np
import time
import os

from src.FOM import FOMproblem, FOM_trainingSet, FOM_testSet
from src.ROM import ROM, ErrorAnalysis, pROM
from mpi4py import MPI

from pathlib import Path

import logging
logging.basicConfig(filename='run.log', encoding='utf-8', level=logging.DEBUG)

   
class MaxwellProblem(FOMproblem):
    def __init__(self, n_partitions=0):
        FOMproblem.__init__(self)
        self.equation_name = "maxwell"
        self.test_name = "torus"
        self.physical_dim = 3
        self.system_dim = 6
        self.folder = "maxwellParallel"
        self.degree = 2
        self.n_partitions = n_partitions
        self.folder_results = "plots/"+self.test_name+"/partition_"+str(self.n_partitions)+"/"
        Path(self.folder_results).mkdir(parents=True, exist_ok=True)

    def setup_parameter_structures(self):
        self.affine_parameters_lhs_names = ["no_par", "sigma", "mu"]
        self.affine_parameters_rhs_names = ["no_par", "sigma", "mu"]
        self.norm_matrix_names = ["L", "M", "S", "A", "D"]
        self.norm_residual_names = ["sigma", "mu"]
        self.interface_names = ["interface"]
        self.pos_names = None #["pos_x_0", "pos_y_0", "pos_z_0"]

    def set_parameter_range(self, param_range_lhs=None, param_range_rhs=None):
        if param_range_lhs is None or param_range_rhs is None:
            self.param_range_lhs = [[1., 1.], [0.5, 2.], [0.5, 3.]]  # 1.,1.,1.
            self.param_range_rhs = [[1., 1.], [0.5, 2.], [0.5, 3.]]
            self.min_mu = 0.5
            self.max_mu = 3
        else:
            self.param_range_lhs = param_range_lhs
            self.param_range_rhs = param_range_rhs
            self.min_mu = min(param_range_lhs[1][0], param_range_lhs[2][0])
            self.max_mu = min(param_range_lhs[1][1], param_range_lhs[2][1])
        
    # override eval_coeff() of FOMproblem class
    def eval_coeff(self, parameter_domain_lhs, parameter_domain_rhs):
        new_rhs = [1, 1, 1]
        new_rhs[1] = parameter_domain_lhs[1]
        new_rhs[2] = 1 / parameter_domain_lhs[2]
        return parameter_domain_lhs, new_rhs

    def pdf_constant(self, mu_lhs):
        return min(mu_lhs[1], mu_lhs[2])


def single_study(test_problem, monolithic=True, tol=1e-5):
    if RANK == 0:
        print("========= load matrices ============")
    test_problem.setup_and_load_structures()
    
    if RANK == 0:
        print("========= snapshots evaluation ============")
    N_training = 100
    tol_POD = 1e-3
    
    load_reduced_space = False
    if os.path.isfile(test_problem.folder+"/snapshots/basis_0_0.txt") is True:
        load_reduced_space = True
        
    load_snapshots = False
    if os.path.isfile(test_problem.folder+"/snapshots/snapshot_0_0.txt"):
        load_snapshots = True
        
    offlineStage = FOM_trainingSet(test_problem,
                            N_training,
                            load_snapshots=load_snapshots,
                            load_reduced_space=load_reduced_space)

    logging.info('Start Loading')
    if offlineStage.snapshots_matrix is None:
        offlineStage.generate_parameters()
        offlineStage.generate_snapshots()
        offlineStage.export_snapshots()
    logging.info('End Loading')

    offlineStage.commMPI.barrier()
    logging.info('Start POD')
    if monolithic:
        if RANK == 0: print("========= monolithic POD ============")
        offlineStage.compress_with_POD(snapshots_matrix=offlineStage.snapshots_matrix[:, ::5], N_max=3, tol_repartitioning=tol, indicator=test_problem.indicator)#tol=tol_POD, tol_repartitioning=tol_repartitioning)
    else:
        if RANK == 0: print("========= partitioned POD ==== C 1========")
        offlineStage.compress_with_POD_component(tol=tol_POD)
    logging.info('End POD')
    
    logging.info('Start ROM studies')
    
    if test_problem.n_partitions==0:
        reduced_problem = ROM(test_problem,
                              offlineStage.reduced_basis)
        logging.info('Loaded ROM')
        reduced_problem.solve_and_estimate(
            offlineStage.mu_lhs,
            offlineStage.mu_rhs,
            offlineStage.snapshots_matrix,
            folder=test_problem.folder_results)
    else:
        reduced_problem = pROM(test_problem,
                               offlineStage.reduced_basis_partitioned)
        reduced_problem.solve_and_estimate(
            offlineStage.mu_lhs,
            offlineStage.mu_rhs,
            offlineStage.snapshots_matrix,
            folder=test_problem.folder_results)
    
    if RANK==0:
        np.save(test_problem.folder_results+"/timings_rom.npy", np.array(reduced_problem.timings_rom))

def partitioning_studies(test_problem, step):
    if RANK == 0:
        print("========= load matrices ============")
    test_problem.setup_and_load_structures()
    
    if RANK == 0:
        print("========= snapshots evaluation ============")
    N_training = 100
    tol_POD = 1e-3        
    offlineStage = FOM_trainingSet(test_problem,
                            N_training,
                            load_snapshots=True,
                            load_reduced_space=True)
    if offlineStage.snapshots_matrix is None:
        offlineStage.generate_snapshots()
        offlineStage.export_snapshots()
    
    offlineStage.commMPI.barrier()
    recErr = []
    for indicator in ['var', 'grassmannian']:
        for percentage in range(0, test_problem.N_total_cells+step, step):
            if RANK == 0: print("Percentage: ", percentage, "/", test_problem.N_total_cells, " step: ", step)
            start = time.time()
            offlineStage.compress_for_partitioning_studies(recErr, snapshots_matrix=offlineStage.snapshots_matrix[:, ::5], N_max=3, tol_repartitioning=percentage/test_problem.N_total_cells, indicator=indicator)
            if RANK == 0: print("time: ", time.time()-start)
        offlineStage.commMPI.barrier()
    
    if RANK == 0:
        plot_rec = np.array(recErr).reshape(-1, test_problem.N_total_cells//step +1, 3)
        np.save(test_problem.folder_results+"/partition_studies.npy", np.array(plot_rec))
        import matplotlib.pyplot as plt
        plt.semilogy(plot_rec[0, :, 0], label="low variance", marker='D', c='r')
        plt.semilogy(plot_rec[0, :, 1], label="high variance", marker='o', c='c')
        plt.semilogy(plot_rec[0, :, 2], label="whole", marker='+', c='k')
        plt.semilogy(plot_rec[1, :, 0], label="low Grassmannian rec error", marker='^', c='tab:orange')
        plt.semilogy(plot_rec[1, :, 1], label="high Grassmannian rec error", marker='v', c='b')
        plt.grid(which='both')
        plt.title("Max reconstruction error")
        plt.xticks([i for i in range(1+test_problem.N_total_cells//step)], [str(int(i*(100/test_problem.N_total_cells))) for i in range(0, test_problem.N_total_cells+step, step)])
        plt.xlabel("Percentage of cells with fast decaying KnW")
        plt.tight_layout()
        plt.legend()
        plt.savefig("plots/"+test_problem.test_name+"/partitioning_studies.png")
        plt.close()
                
def convergence_study(test_problem, monolithic=True, tol=1e-5):
    test_problem.setup_and_load_structures()
    
    N_training = 100
    tol_POD = 1e-3
    offlineStage = FOM_trainingSet(test_problem,
                            N_training,
                            load_snapshots=True,
                            load_reduced_space=False)
    
    if offlineStage.snapshots_matrix is None:
        offlineStage.generate_snapshots()
        offlineStage.export_snapshots()
    
    offlineStage.commMPI.barrier()
    if monolithic:
        if RANK == 0: print("========= monolithic POD ============")
        offlineStage.compress_with_POD(snapshots_matrix=offlineStage.snapshots_matrix[:, ::5], N_max=5,
            tol_repartitioning=tol, indicator=test_problem.indicator)
    else:
        if RANK == 0: print("========= partitioned POD ============")
        offlineStage.compress_with_POD_component(tol=tol_POD)
    
    N_max_reduced_dim = 10
    N_test_samples = 50
    error_analysis = ErrorAnalysis(offlineStage)
    error_analysis.generate_test_set(
        N_test_samples,
        load_snapshots=True)
    error_analysis.test_set.save_snapshots()
    
    # local RB dim uniform
    error_analysis.compute_error_analysis(
        N_max_reduced_dim,
        components=False,
        folder="plots/"+test_problem.test_name+"/", tol_rep=tol)
    error_analysis.plot_errors(folder="plots/"+test_problem.test_name+"/", title='nrb')
    
    # local RB dim not uniform, based on residual energy
    error_analysis.compute_error_analysis_toleranace(N_max_reduced_dim,
    components=False,
    folder="plots/"+test_problem.test_name+"/", tol_rep=tol)
    error_analysis.plot_errors(folder="plots/"+test_problem.test_name+"/", title='tol')
    
if __name__=="__main__":
    global RANK
    RANK = MPI.COMM_WORLD.Get_rank()
    
    # MaxwellProblem
    # n_partitions==0 : corresponds to all the partitions
    # n_partitions==1 : corresponds to none
    # n_partitions==2 : corresponds to two partitions
    for n_partitions in range(3):
        test_problem = MaxwellProblem(n_partitions)
        
        # indicator for partitioning in case n_partitions==2
        test_problem.indicator = 'var' # or 'grassmannian'
        
        # percentage value of the indicator used to split the domain: 0.5 means that the 'domain' will be split in two equal parts in terms of number of cells
        tol_repartitioning = 0.5
        
        # Grassmannian indicator parameters
        test_problem.k_nearest = 1 # >=1
        test_problem.r_dim_approx = 1 # >=1
        
        if n_partitions==2:
            # study which indicator among 'var' and 'grassmannian' is the best for the problem at hand
            partitioning_studies(test_problem, step=2)
            
            # can take more than half an hour
            convergence_study(test_problem, tol=tol_repartitioning)
        
        # run offline and online stages with n_partitions
        single_study(test_problem, tol=tol_repartitioning)
