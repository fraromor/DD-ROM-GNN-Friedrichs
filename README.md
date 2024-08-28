## Table of contents
- [Table of contents](#table-of-contents)
- [Description](#description)
- [Dependencies and installation](#dependencies-and-installation)
- [Documentation DD-ROMs for Friedrichs' systems](#documentation-dd-roms-for-friedrichs-systems)
  - [Procedure for DD-ROMs with repartitioning strategies](#procedure-for-dd-roms-with-repartitioning-strategies)
  - [Procedure for vanishing viscosity solutions inference with GNNs](#procedure-for-vanishing-viscosity-solutions-inference-with-gnns)
- [How to cite](#how-to-cite)
- [Authors and contributors](#authors-and-contributors)
- [License](#license)

## Description
This repository contains the code used to generate the results in the pre-print "**Friedrichs’ systems discretized with the Discontinuous Galerkin method:
domain decomposable model order reduction and Graph Neural Networks
approximating vanishing viscosity solutions**", https://arxiv.org/pdf/2308.03378.

## Dependencies and installation
The following packages are required `deal.II`, `PETSc`, `petsc4py`, `torch`, `torch_geometric`, `pytorch_lightning` among others.

## Documentation DD-ROMs for Friedrichs' systems
### Procedure for DD-ROMs with repartitioning strategies

Compile the deal.II script. As first step we need to save the matrices needed for the affine decomposition:
```bash
> cd maxwellParallel
> cmake .
> make release
> make 
> mpirun -np 4 ./maxwell
```
we also need the positions of the support points of the dofs for our repartitioning strategies. Inside "parameters.prm" set "task to perform" from "affine" to "save_pos". Then run:
```bash
> mpirun -np 4 ./maxwell
> cd affine_matrices
> python3 postprocess.py 4
> cd ..
> cd ..
```
run offline and online stages in parallel with:
```bash
> mpirun -np 4 python3 run_FOM_ROM.py
```

**Additionally**: to see the partitioning of the computational domain do the following:
```bash
> cd maxwellParallel
```
inside "parameters.prm" set "task to perform" from "affine" to "labels"
```bash
> mpirun -np 4 ./maxwell
```
and open in paraview:
```bash
> paraview partition_0.pvtu
```

**Additionally**: to plot the FOM and ROM solutions do the following:
```bash
> cd maxwellParallel
```
inside "parameters.prm" set "task to perform" from "affine" to "plot"
```bash
> mpirun -np 4 ./maxwell
```
and open the fields in paraview: the FOM fields with
```bash
> paraview snapshots/reconstructed_[0-9].pvtu
```
and the DD-ROM predicted fields
```bash
> paraview snapshots/rreconstructed_[0-9].pvtu
```

### Procedure for vanishing viscosity solutions inference with GNNs
The following procedure is used to infer the vanishing viscosity solutions through our multi-fidelity multi-resolution strategy that makes use of DD-ROMs and GNNs. Run the following to train a GNN with data already obtained from an advection-diffusion-reaction test case: download first the dataset with
```bash
> cd vv
> wget https://zenodo.org/records/13946510/files/vv.tar.gz
> tar -zxfv vv.tar.gz
```
then run the training with:
```bash
> python3 vvgraph_all.py
```
move the file that starts with "tb_logs/vv/version_0/checkpoints/epoch*" in "tb_logs/vv.ckpt" so that it can be loaded by the script "predict.py"
```bash
> python3 predict.py
```

## How to cite
If you use this package in your publications please cite the package as follows:

Romor, F., Torlo, D. and Rozza, G., 2023. Friedrichs' systems discretized with the Discontinuous Galerkin method: domain decomposable model order reduction and Graph Neural Networks approximating vanishing viscosity solutions. arXiv preprint arXiv:2308.03378.

Or if you use LaTeX:
```tex
@article{romor2023friedrichs,
  title={Friedrichs' systems discretized with the Discontinuous Galerkin method: domain decomposable model order reduction and Graph Neural Networks approximating vanishing viscosity solutions},
  author={Romor, Francesco and Torlo, Davide and Rozza, Gianluigi},
  journal={arXiv preprint arXiv:2308.03378},
  year={2023}
}
```

## Authors and contributors
The authors of the reported results are
* [Francesco Romor](mailto:francesco.romor@gmail.com)
* [Davide Torlo](davidetorlo@gmail.com)

under the supervision of [Prof. Gianluigi Rozza](mailto:gianluigi.rozza@sissa.it).

Contact us by email for further information or questions, or suggest pull requests. Contributions improving either the code or the documentation are welcome!

## License

See the [LICENSE](LICENSE.rst) file for license rights and limitations (MIT).
