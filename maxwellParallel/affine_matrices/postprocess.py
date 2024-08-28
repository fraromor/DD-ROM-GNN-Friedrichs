import numpy as np
from scipy.sparse import csr_matrix, save_npz
import matplotlib.pyplot as plt
import time
import sys

DOFS = 6480  # needed
system_dim = 6
dim = 3

sigma = 1
mu = 1

file_names = [
    "petsc_mat_whole_system", "petsc_mat_no_par", "petsc_mat_sigma",
    "petsc_mat_mu", "petsc_mat_L", "petsc_mat_M", "petsc_mat_S", "petsc_mat_A", "petsc_mat_D"
    ]

# file_names 7
rhs_names = ["petsc_rhs_whole_system",
            "petsc_rhs_no_par",
            "petsc_rhs_mu",
            "petsc_rhs_sigma"]

def save_system_not_partitioned(fl_reconstruct=True):
    mats = []
    for name in file_names:
        fname = name + ".txt"
        print("Load " + fname)
        with open(fname, 'r') as file:
            with open(fname[6:], 'w') as outfile:
                data = file.read()
                data = data.replace(",", " ")
                data = data.replace("(", " ")
                data = data.replace(")", " ")
                outfile.write(data)

        [row, column, data] = np.loadtxt(fname[6:]).transpose()
        mats.append(csr_matrix((data, (row, column)), [DOFS, DOFS]))
        np.save(name[6:] + '.npy', np.vstack((row, column, data)))

    for i, mat in enumerate(mats):
        print(file_names[i], " ", mat.shape)

    if fl_reconstruct:
        reconstruct = mats[1] + sigma * mats[2] + mu * mats[3]
        print(reconstruct[:3, :3], "\n", mats[0][:3, :3])
        err = np.max(np.abs(reconstruct - mats[0]))
        rel = err / np.max(np.abs(mats[0]))
        print("Check consistency: ", err, rel)

    rhs_names = ["petsc_rhs_whole_system"]
    rhs_list = []
    for name in rhs_names:
        fname = name + ".txt"
        with open(fname, 'r') as file:
            with open(fname[6:], 'w') as outfile:
                data = file.read()
                data = data.lstrip('[Proc0 0-' + str(DOFS - 1) + ']')
                outfile.write(data)
        rhs = np.loadtxt(fname[6:])
        rhs_list.append(rhs)
        np.save(fname[6:] + '.npy', np.array(rhs))

    print("rhs shape: ", rhs.shape)


def save_system_partitioned(cores=4, fl_reconstruct=True, fl_spy=False):
    mats = [list() for core in range(cores)]
    for name in file_names:
        for core in range(cores):
            fname = name + "_" + str(core) + ".txt"
            print("Load " + fname)
            with open(fname, 'r') as file:
                with open(fname[6:], 'w') as outfile:
                    data = file.read()
                    data = data.replace(",", " ")
                    data = data.replace("(", " ")
                    data = data.replace(")", " ")
                    outfile.write(data)

            [row, column, data] = np.loadtxt(fname[6:]).transpose()
            LOCALROWS = int(row[-1] + 1 - row[0])
            mat = csr_matrix((data, (row - row[0], column)), [LOCALROWS, DOFS])
            mats[core].append(mat)

            save_npz(name[6:] + "_" + str(core) + '.npz', mat)

            if fl_spy:
                plt.spy(mat)
                plt.show()
                plt.close()

    for core in range(cores):
        for i, mat in enumerate(mats[core]):
            print(file_names[i], " ", mat.shape)

        if fl_reconstruct:
            reconstruct = mats[core][
                1] + sigma * mats[core][2] + mu * mats[core][3]
            err = np.max(np.abs(reconstruct - mats[core][0]))
            rel = err / np.max(np.abs(mats[core][0]))
            print("Check consistency: ", err, rel)

    rhs_list = [list() for core in range(cores)]
    for name in rhs_names:
        print(name)
        for core in range(cores):
            fname = name + "_" + str(core) + ".txt"
            with open(fname, 'r') as file:
                with open(fname[6:], 'w') as outfile:
                    data = file.read()
                    data = data.split(']')[1]
                    outfile.write(data)
            rhs = np.loadtxt(fname[6:])
            rhs_list[core].append(rhs)

            np.save(name[6:] + "_" + str(core) + '.npy', np.array(rhs))

            print("rhs shape: ", rhs_list[core][0].shape)
    
    for core in range(cores):
        if fl_reconstruct:
            reconstruct = rhs_list[core][
                1] + sigma * rhs_list[core][2] + mu * rhs_list[core][3]
            err = np.max(np.abs(reconstruct - rhs_list[core][0]))
            rel = err / np.max(np.abs(rhs_list[core][0]))
            print("Check rhs consistency: ", err, rel)

def save_pos(cores = 4, fl_reconstruct=True, fl_spy=False):
    rhs_names = ["petsc_pos_x_0", "petsc_pos_y_0", "petsc_pos_z_0"]
    rhs_list = [list() for core in range(cores)]
    for name in rhs_names:
        for core in range(cores):
            fname = name + "_"+str(core)+".txt"
            with open(fname, 'r') as file:
                with open(fname[6:], 'w') as outfile:
                    data = file.read()
                    data = data.lstrip('[Proc0 0-269]')
                    data = data.lstrip('[Proc1 270-539]')
                    data = data.lstrip('[Proc2 540-809]')
                    data = data.lstrip('[Proc3 810-1079]')
                    outfile.write(data)
            rhs = np.loadtxt(fname[6:])
            rhs_list[core].append(rhs)

            np.save(name[6:]+ "_"+str(core)+'.npy', np.array(rhs))
            print("rhs shape: ", rhs_list[core][0].shape)

def save_cells(cores = 4, fl_reconstruct=True, fl_spy=False):
    lis = list()
    for core in range(cores):
        lis.append(np.loadtxt("petsc_pos_cell_0_"+str(core)+".txt"))
        print(lis[-1].shape)
    
    cells = np.vstack(lis)
    np.save("centers.npy", cells)
            
if len(sys.argv) < 2:
    print("Pass number of cores.")
else:
    print("Number of cores: ", sys.argv[1])
    save_system_partitioned(eval(sys.argv[1]), True, False)
    save_pos()
    save_cells()