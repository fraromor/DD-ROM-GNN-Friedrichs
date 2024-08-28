#include <deal.II/base/function.h>
#include <deal.II/base/function_parser.h>
#include <deal.II/base/parameter_acceptor.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/tensor.h>
#include <deal.II/base/multithread_info.h>
#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/utilities.h>
#include <deal.II/base/index_set.h>
#include <deal.II/base/timer.h>
#include <deal.II/base/convergence_table.h>

#include <deal.II/lac/generic_linear_algebra.h>
#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/vector.h>
#include <deal.II/lac/precondition_block.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/solver_gmres.h>
#include <deal.II/lac/solver_richardson.h>
#include <deal.II/lac/sparse_direct.h>
#include <deal.II/lac/petsc_solver.h>
#include <deal.II/lac/sparsity_tools.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_refinement.h>
#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_out.h>
#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_iterator.h>
#include <deal.II/grid/manifold_lib.h>

#include <deal.II/dofs/dof_accessor.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/error_estimator.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/derivative_approximation.h>
#include <deal.II/numerics/fe_field_function.h>

#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/fe_dgq.h>
#include <deal.II/fe/fe_interface_values.h>
#include <deal.II/fe/mapping_q1.h>
#include <deal.II/fe/mapping_q_cache.h>
#include <deal.II/fe/fe_system.h>

#include <deal.II/meshworker/dof_info.h>
#include <deal.II/meshworker/mesh_loop.h>

#include <deal.II/distributed/tria.h>
#include <deal.II/distributed/grid_refinement.h>

#include <fstream>
#include <iostream>
#include <iomanip>

namespace LA
{
#if defined(DEAL_II_WITH_PETSC)
  using namespace dealii::LinearAlgebraPETSc;
#define USE_PETSC_LA
#else
#error DEAL_II_WITH_PETSC or DEAL_II_WITH_TRILINOS required
#endif
} // namespace LA

using namespace dealii;

namespace MaxwellDG
{
  template <int dim>
  class MaxwellProblem : ParameterAcceptor
  {
  public:
    MaxwellProblem();
    void run();
    void run_fom();
    void run_affine();
    void run_plot();
    void run_plot_labels();
    void save_pos();

  private:
    void setup_system();
    void assemble_system();
    void assemble_system_affine();
    void solve();
    void refine_grid();
    void output_results(const unsigned int cycle);

    // utilities
    void save_sparsity_pattern(DynamicSparsityPattern dsp);

    MPI_Comm mpi_communicator;

    parallel::distributed::Triangulation<dim> triangulation;

    FESystem<dim> fe;
    DoFHandler<dim> dof_handler;

    const double mapping_degree{2};
    MappingQ<dim> mapping;

    IndexSet locally_owned_dofs;
    IndexSet locally_relevant_dofs;

    AffineConstraints<double> constraints;
    bool is_on_boundary(Point<dim> p);

    LA::MPI::SparseMatrix system_matrix;
    LA::MPI::Vector locally_relevant_solution;
    LA::MPI::Vector system_rhs;

    LA::MPI::Vector error_square_per_cell;
    LA::MPI::Vector exact_sol;

    ConditionalOStream pcout;
    TimerOutput computing_timer;
    ConvergenceTable convergence_table;

    // Friedrichs' system matrices
    static const int system_dim{dim * 2};
    std::vector<FullMatrix<double>> A{dim + 1};
    std::vector<FullMatrix<double>> T{dim};

    // Affine decomposition
    std::vector<FullMatrix<double>> H{9};

    // parameters
    std::string task = "fom";
    double mu = 1;
    double sigma = 1;
    unsigned int n_snapshots = 1;
    unsigned int n_refinements = 0;

    std::map<std::string, double> function_constants;
    std::string forcing_term_expression = "1";
    std::string forcing_mu_term_expression = "1";
    std::string forcing_sigma_term_expression = "1";
    std::string grid_generator_function = "torus";
    std::string grid_generator_arguments = "2: 0.5: 8";

    bool save_sparsity = false;

    FunctionParser<dim> forcing_term;
    FunctionParser<dim> forcing_mu_term;
    FunctionParser<dim> forcing_sigma_term;

    bool fl_L = true;
    bool fl_M = true;
    bool fl_S = true;
    bool fl_A = true;
    bool fl_no_par = true;
    bool fl_sigma = true;
    bool fl_mu = true;
    bool fl_AdFS = true;
    bool fl_A0 = true;
    bool fl_D = true;
  };

  template <int dim>
  struct ScratchData
  {
    ScratchData(const Mapping<dim> &mapping, const FiniteElement<dim> &fe,
                const unsigned int quadrature_degree,
                const UpdateFlags update_flags = update_values |
                                                  update_gradients |
                                                  update_quadrature_points |
                                                  update_JxW_values,
                const UpdateFlags interface_update_flags =
                    update_values |
                    update_gradients |
                    update_quadrature_points |
                    update_JxW_values |
                    update_normal_vectors)
        : fe_values(mapping, fe, QGauss<dim>(quadrature_degree), update_flags),
          fe_interface_values(mapping, fe, QGauss<dim - 1>(quadrature_degree),
                              interface_update_flags)
    {
    }

    ScratchData(const ScratchData<dim> &scratch_data)
        : fe_values(scratch_data.fe_values.get_mapping(),
                    scratch_data.fe_values.get_fe(),
                    scratch_data.fe_values.get_quadrature(),
                    scratch_data.fe_values.get_update_flags()),
          fe_interface_values(
              scratch_data.fe_values
                  .get_mapping(), // TODO: implement for fe_interface_values
              scratch_data.fe_values.get_fe(),
              scratch_data.fe_interface_values.get_quadrature(),
              scratch_data.fe_interface_values.get_update_flags())
    {
    }

    FEValues<dim> fe_values;
    FEInterfaceValues<dim> fe_interface_values;
  };

  struct CopyDataFace
  {
    FullMatrix<double> cell_matrix;
    std::vector<types::global_dof_index> joint_dof_indices;
  };

  struct CopyData
  {
    FullMatrix<double> cell_matrix;
    Vector<double> cell_rhs;
    std::vector<types::global_dof_index> local_dof_indices;
    std::vector<CopyDataFace> face_data;
    unsigned int cell_index;
    // error evaluation
    double value;
    Vector<double> cell_error, cell_exact;

    template <class Iterator>
    void reinit(const Iterator &cell, unsigned int dofs_per_cell)
    {
      cell_matrix.reinit(dofs_per_cell, dofs_per_cell);
      cell_rhs.reinit(dofs_per_cell);
      cell_error.reinit(dofs_per_cell);
      cell_exact.reinit(dofs_per_cell);

      local_dof_indices.resize(dofs_per_cell);
      cell->get_dof_indices(local_dof_indices);
    }
  };
}