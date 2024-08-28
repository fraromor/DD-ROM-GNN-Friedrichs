/* 
 * Author: Francesco Romor, WIAS, 2024
 */

#include "maxwell.h"

using namespace dealii;

namespace MaxwellDG
{
  template <int dim>
  MaxwellProblem<dim>::MaxwellProblem()
      : ParameterAcceptor("MaxwellProblem"),
        mpi_communicator(MPI_COMM_WORLD),
        triangulation(mpi_communicator,
                      typename Triangulation<dim>::MeshSmoothing(
                          Triangulation<dim>::smoothing_on_refinement |
                          Triangulation<dim>::smoothing_on_coarsening)),
        fe(FE_DGQ<dim>(2), system_dim),
        dof_handler(triangulation),
        mapping(mapping_degree),
        pcout(std::cout,
              (Utilities::MPI::this_mpi_process(mpi_communicator) == 0)),
        computing_timer(mpi_communicator,
                        pcout,
                        TimerOutput::never,
                        TimerOutput::wall_times),
        forcing_term(dim),
        forcing_mu_term(dim),
        forcing_sigma_term(dim)
  {
    add_parameter("task to perform", task);
    add_parameter("mu", mu);
    add_parameter("sigma", sigma);
    add_parameter("number of refinements", n_refinements);
    add_parameter("number of snapshots to reconstruct from ROM", n_snapshots);
    add_parameter("grid generator arguments", grid_generator_arguments);
    add_parameter("grid generator function", grid_generator_function);
    add_parameter("forcing term expression", forcing_term_expression);
    add_parameter("forcing mu term expression", forcing_mu_term_expression);
    add_parameter("forcing sigma term expression", forcing_sigma_term_expression);
    add_parameter("function constants", function_constants);
    add_parameter("save sparsity pattern", save_sparsity);

    add_parameter("fl_L", fl_L ); 
    add_parameter("fl_M",   fl_M ); 
    add_parameter("fl_S",   fl_S ); 
    add_parameter("fl_A",   fl_A ); 
    add_parameter("fl_no_par",   fl_no_par ); 
    add_parameter("fl_sigma",   fl_sigma ); 
    add_parameter("fl_mu",   fl_mu ); 
    add_parameter("fl_AdFS",   fl_AdFS); 
    add_parameter("fl_A0",   fl_A0); 
    add_parameter("fl_D",   fl_D); 

    // A_0
    A[0].reinit(system_dim, system_dim);
    A[0](0, 0) = 1;
    A[0](1, 1) = 1;
    A[0](2, 2) = 1;
    A[0](3, 3) = 1;
    A[0](4, 4) = 1;
    A[0](5, 5) = 1;

    // A_1
    A[1].reinit(system_dim, system_dim);
    A[1](1, 5) = -2;
    A[1](2, 4) = 2;
    A[1].symmetrize();

    // A_2
    A[2].reinit(system_dim, system_dim);
    A[2](0, 5) = 2;
    A[2](2, 3) = -2;
    A[2].symmetrize();

    // A_3
    A[3].reinit(system_dim, system_dim);
    A[3](0, 4) = -2;
    A[3](1, 3) = 2;
    A[3].symmetrize();

    // T
    T[0].reinit(system_dim, system_dim);
    T[0](1, 2) = -1;
    T[0](2, 1) = 1;

    T[1].reinit(system_dim, system_dim);
    T[1](0, 2) = 1;
    T[1](2, 0) = -1;

    T[2].reinit(system_dim, system_dim);
    T[2](0, 1) = -1;
    T[2](1, 0) = 1;

    // evalute norms for convergence
    H[0].reinit(system_dim, system_dim);
    H[1].reinit(system_dim, system_dim);
    H[2].reinit(system_dim, system_dim);
    H[3].reinit(system_dim, system_dim);
    H[4].reinit(system_dim, system_dim);
    H[5].reinit(system_dim, system_dim);
    H[6].reinit(system_dim, system_dim);
    H[7].reinit(system_dim, system_dim);
    H[8].reinit(system_dim, system_dim);
    A[1].Tmmult(H[0], A[1]);
    A[1].Tmmult(H[1], A[2]);
    A[1].Tmmult(H[2], A[3]);
    A[2].Tmmult(H[3], A[1]);
    A[2].Tmmult(H[4], A[2]);
    A[2].Tmmult(H[5], A[3]);
    A[3].Tmmult(H[6], A[1]);
    A[3].Tmmult(H[7], A[2]);
    A[3].Tmmult(H[8], A[3]);
  }

  template <int dim>
  void MaxwellProblem<dim>::setup_system()
  {
    TimerOutput::Scope t(computing_timer, "setup");

    dof_handler.distribute_dofs(fe);

    locally_owned_dofs = dof_handler.locally_owned_dofs();
    locally_relevant_dofs =
        DoFTools::extract_locally_relevant_dofs(dof_handler);

    constraints.clear();
    constraints.reinit(locally_relevant_dofs);
    constraints.close();

    DynamicSparsityPattern dsp(locally_relevant_dofs);

    DoFTools::make_flux_sparsity_pattern(dof_handler, dsp);

    SparsityTools::distribute_sparsity_pattern(dsp,
                                               dof_handler.locally_owned_dofs(),
                                               mpi_communicator,
                                               locally_relevant_dofs);

    locally_relevant_solution.reinit(locally_owned_dofs,
                                     locally_relevant_dofs,
                                     mpi_communicator);
    system_matrix.reinit(locally_owned_dofs,
                         locally_owned_dofs,
                         dsp,
                         mpi_communicator);
    system_rhs.reinit(locally_owned_dofs, mpi_communicator);

    // error evaluation
    error_square_per_cell.reinit(locally_owned_dofs, mpi_communicator);
    exact_sol.reinit(locally_owned_dofs, mpi_communicator);

    if (save_sparsity)
    {
      save_sparsity_pattern(dsp);
    }

    forcing_term.initialize("x, y, z", forcing_term_expression, function_constants);
    forcing_mu_term.initialize("x, y, z", forcing_mu_term_expression, function_constants);
    forcing_sigma_term.initialize("x, y, z", forcing_sigma_term_expression, function_constants);
  }

  template <int dim>
  void MaxwellProblem<dim>::save_sparsity_pattern(DynamicSparsityPattern dsp)
  {
    std::ofstream out("sparsity_pattern" + std::to_string(Utilities::MPI::this_mpi_process(mpi_communicator)) + ".svg");
    SparsityPattern sparsity_pattern;
    sparsity_pattern.copy_from(dsp);
    sparsity_pattern.print_svg(out);
  }

  template <int dim>
  void MaxwellProblem<dim>::assemble_system()
  {
    TimerOutput::Scope t(computing_timer, "assembly");

    using Iterator = typename DoFHandler<dim>::active_cell_iterator;

    auto cell_worker = [&](const Iterator &cell, ScratchData<dim> &scratch_data,
                           CopyData &copy_data)
    {
      // pcout << "cellworker\n";
      const unsigned int n_dofs = scratch_data.fe_values.get_fe().dofs_per_cell;

      copy_data.reinit(cell, n_dofs);
      scratch_data.fe_values.reinit(cell);
      const FEValues<dim> &fe_v = scratch_data.fe_values;

      for (unsigned int i = 0; i < n_dofs; ++i)
      {
        const unsigned int component_i = fe.system_to_component_index(i).first;
        for (unsigned int j = 0; j < n_dofs; ++j)
        {
          const unsigned int component_j = fe.system_to_component_index(j).first;
          for (const unsigned int q_point : fe_v.quadrature_point_indices())
          {
            copy_data.cell_matrix(i, j) +=
                (((component_j == component_i) ? ((component_i >= 3) ?
                (fe_v.shape_value_component(j, q_point, component_j) *
                sigma *
                A[0](component_j, component_i) *
                fe_v.shape_value_component(i, q_point, component_i))
              : (fe_v.shape_value_component(j, q_point, component_j) *
                mu * A[0](component_j, component_i) *
                fe_v.shape_value_component(i, q_point, component_i)))
                                               : 0) -
                 fe_v.shape_value_component(j, q_point, component_j) *
                     A[1](component_j, component_i) *
                     fe_v.shape_grad_component(i, q_point, component_i)[0] -
                 fe_v.shape_value_component(j, q_point, component_j) *
                     A[2](component_j, component_i) *
                     fe_v.shape_grad_component(i, q_point, component_i)[1] -
                 fe_v.shape_value_component(j, q_point, component_j) *
                     A[3](component_j, component_i) *
                     fe_v.shape_grad_component(i, q_point, component_i)[2]) *
                fe_v.JxW(q_point);
          }
        }
      }

      for (unsigned int i = 0; i < n_dofs; ++i)
      {
        for (const unsigned int q_point : fe_v.quadrature_point_indices())
        {
          copy_data.cell_rhs(i) += (fe_v.shape_value_component(i, q_point, 3) *
                                        forcing_term.value(fe_v.quadrature_point(q_point), 0) +
                                    fe_v.shape_value_component(i, q_point, 4) *
                                        forcing_term.value(fe_v.quadrature_point(q_point), 1) +
                                    fe_v.shape_value_component(i, q_point, 5) *
                                        forcing_term.value(fe_v.quadrature_point(q_point), 2)) *
                                   fe_v.JxW(q_point);
        }
      }
    };

    auto boundary_worker = [&](const Iterator &cell, const unsigned int &face_no,
                               ScratchData<dim> &scratch_data,
                               CopyData &copy_data)
    {
      scratch_data.fe_interface_values.reinit(cell, face_no);
      const FEFaceValuesBase<dim> &fe_face =
          scratch_data.fe_interface_values.get_fe_face_values(0);

      const auto &q_points = fe_face.get_quadrature_points();

      const unsigned int n_facet_dofs = fe_face.get_fe().n_dofs_per_cell();

      const std::vector<Tensor<1, dim>> &normals = fe_face.get_normal_vectors();

      const unsigned int degree = fe.degree;
      const double eta = 1 * 2 * degree * (degree + 1);
      const double penalty =
          eta * cell->face(face_no)->measure() / cell->measure();

      Tensor<1, dim> T_penalty_i, T_penalty_j, NHD;
      int id_1_i, id_2_i, id_1_j, id_2_j;

      for (unsigned int q_point = 0; q_point < q_points.size(); ++q_point)
      {
        for (unsigned int i = 0; i < n_facet_dofs; ++i)
        {
          const unsigned int component_i = fe.system_to_component_index(i).first;

          T_penalty_i.clear();

          if ((component_i > 2))
          {
            id_1_i = (component_i + 1) % 3;
            id_2_i = (component_i + 2) % 3;

            T_penalty_i[id_1_i] =
                T[id_2_i](id_1_i, component_i % 3) * normals[q_point][id_2_i];
            T_penalty_i[id_2_i] =
                T[id_1_i](id_2_i, component_i % 3) * normals[q_point][id_1_i];
          }

          for (unsigned int j = 0; j < n_facet_dofs; ++j)
          {
            const unsigned int component_j =
                fe.system_to_component_index(j).first;

            T_penalty_j.clear();

            if ((component_j > 2))
            {
              id_1_j = (component_j + 1) % 3;
              id_2_j = (component_j + 2) % 3;

              T_penalty_j[id_1_j] =
                  T[id_2_j](id_1_j, component_j % 3) * normals[q_point][id_2_j];
              T_penalty_j[id_2_j] =
                  T[id_1_j](id_2_j, component_j % 3) * normals[q_point][id_1_j];
            }

            copy_data.cell_matrix(i, j) +=
                (((component_j < 3 && component_i > 2)
                      ? (fe_face.shape_value_component(j, q_point, component_j) *
                         (A[1](component_i, component_j) * normals[q_point][0] +
                          A[2](component_i, component_j) * normals[q_point][1] +
                          A[3](component_i, component_j) * normals[q_point][2]) *
                         fe_face.shape_value_component(i, q_point, component_i))
                      : 0) +
                 ((component_j > 2 && component_i > 2) ? (penalty * fe_face.shape_value_component(i, q_point, component_i) *
                  fe_face.shape_value_component(j, q_point, component_j) *
                  T_penalty_i * T_penalty_j)
                : 0)) *
                fe_face.JxW(q_point);
          }
        }
      }
    };

    auto face_worker = [&](const Iterator &cell, const unsigned int &f,
                           const unsigned int &sf, const Iterator &ncell,
                           const unsigned int &nf, const unsigned int &nsf,
                           ScratchData<dim> &scratch_data, CopyData &copy_data)
    {
      FEInterfaceValues<dim> &fe_iv = scratch_data.fe_interface_values;
      fe_iv.reinit(cell, f, sf, ncell, nf, nsf);
      const auto &q_points = fe_iv.get_quadrature_points();

      copy_data.face_data.emplace_back();
      CopyDataFace &copy_data_face = copy_data.face_data.back();

      const unsigned int n_dofs = fe_iv.n_current_interface_dofs();

      copy_data_face.joint_dof_indices = fe_iv.get_interface_dof_indices();

      copy_data_face.cell_matrix.reinit(n_dofs, n_dofs);

      const std::vector<Tensor<1, dim>> &normals = fe_iv.get_normal_vectors();

      const unsigned int degree = fe.degree;
      const double eta = 1 * 2. * degree * (degree + 1);
      const double penalty1 = cell->face(f)->measure() / cell->measure();
      const double penalty2 = ncell->face(f)->measure() / ncell->measure();
      const double penalty = eta * (penalty1 + penalty2);

      Tensor<1, dim> T_penalty_i, T_penalty_j;
      int id_1_i, id_2_i, id_1_j, id_2_j;

      for (unsigned int q_point = 0; q_point < q_points.size(); ++q_point)
      {
        for (unsigned int i = 0; i < n_dofs; ++i)
        {
          const unsigned int component_i =
              fe.system_to_component_index(i % int(n_dofs / 2)).first;
          for (unsigned int j = 0; j < n_dofs; ++j)
          {
            const unsigned int component_j =
                fe.system_to_component_index(j % int(n_dofs / 2)).first;

            T_penalty_j.clear();
            T_penalty_i.clear();

            if ((component_i < 3 && component_j < 3) || (component_i > 2 && component_j > 2))
            {
              id_1_j = (component_j + 1) % 3;
              id_2_j = (component_j + 2) % 3;

              T_penalty_j[id_1_j] =
                  T[id_2_j](id_1_j, component_j % 3) * normals[q_point][id_2_j];
              T_penalty_j[id_2_j] =
                  T[id_1_j](id_2_j, component_j % 3) * normals[q_point][id_1_j];

              id_1_i = (component_i + 1) % 3;
              id_2_i = (component_i + 2) % 3;

              T_penalty_i[id_1_i] =
                  T[id_2_i](id_1_i, component_i % 3) * normals[q_point][id_2_i];
              T_penalty_i[id_2_i] =
                  T[id_1_i](id_2_i, component_i % 3) * normals[q_point][id_1_i];
            }

            copy_data_face.cell_matrix(i, j) +=
                (fe_iv.average_of_shape_values(j, q_point, component_j) *
                     (A[1](component_i, component_j) * normals[q_point][0] +
                      A[2](component_i, component_j) * normals[q_point][1] +
                      A[3](component_i, component_j) * normals[q_point][2]) *
                     fe_iv.jump_in_shape_values(i, q_point, component_i) +
                 // \eta \T  \dot \T
                 +penalty * fe_iv.jump_in_shape_values(i, q_point, component_i) *
                     fe_iv.jump_in_shape_values(j, q_point, component_j) *
                     T_penalty_i * T_penalty_j) * //
                fe_iv.JxW(q_point);
          }
        }
      }
    };

    auto copier = [&](const CopyData &c)
    {
      constraints.distribute_local_to_global(c.cell_matrix, c.cell_rhs,
                                             c.local_dof_indices, system_matrix,
                                             system_rhs);
      for (auto &cdf : c.face_data)
      {
        constraints.distribute_local_to_global(
            cdf.cell_matrix, cdf.joint_dof_indices, system_matrix);
      }
    };

    const unsigned int n_gauss_points = std::max(
      static_cast<unsigned int>(std::ceil(1. * (mapping.get_degree() + 1) / 2)),
      dof_handler.get_fe().degree + 1) + 1;

    ScratchData<dim> scratch_data(mapping, fe, n_gauss_points);
    CopyData copy_data;

    MeshWorker::mesh_loop(dof_handler.begin_active(), dof_handler.end(),
                          cell_worker,
                          copier,
                          scratch_data,
                          copy_data,
                          MeshWorker::assemble_own_cells |
                              MeshWorker::assemble_boundary_faces |
                              MeshWorker::assemble_own_interior_faces_once |
                              MeshWorker::assemble_ghost_faces_once,
                          boundary_worker,
                          face_worker);

    system_matrix.compress(VectorOperation::add);
    system_rhs.compress(VectorOperation::add);
  }

  template<int dim>
  bool MaxwellProblem<dim>::is_on_boundary(Point<dim> p){
    double x = p[0];
    double y = p[1];
    double z = p[2];
    double x_2 = std::pow(x, 2);
    double y_2 = std::pow(y, 2);
    double z_2 = std::pow(z, 2);
    double RR = std::sqrt(x_2 + z_2);
    double dist =  std::pow((x-2 * x/RR), 2)+y_2+std::pow((z - 2 * z/RR), 2);
    if (0.23 - dist < 2e-2){
      return true;}
    else {return false;}
  }

  template <int dim>
  void MaxwellProblem<dim>::assemble_system_affine()
  {
    TimerOutput::Scope t(computing_timer, "assembly_affine");

    // matrices flags for affine decomposition
    enum class ROM_mat
    {
      pb,
      no_par,
      sigma,
      mu,
      L,
      M,
      S,
      A,
      AdFS,
      A0,
      D
    };

    std::string rom_mat_labels[] = {"whole_system", "no_par", "sigma", "mu", "L", "M", "S", "A", "AdFS", "A0", "D"};
    ROM_mat rom_flag;

    using Iterator = typename DoFHandler<dim>::active_cell_iterator;

    // This is the function that will be executed for each cell.
    auto cell_worker = [&](const Iterator &cell, ScratchData<dim> &scratch_data,
                           CopyData &copy_data)
    {
      const unsigned int n_dofs = scratch_data.fe_values.get_fe().dofs_per_cell;
      copy_data.reinit(cell, n_dofs);
      scratch_data.fe_values.reinit(cell);
      const FEValues<dim> &fe_v = scratch_data.fe_values;

      for (unsigned int i = 0; i < n_dofs; ++i)
      {
        const unsigned int component_i = fe.system_to_component_index(i).first;
        for (unsigned int j = 0; j < n_dofs; ++j)
        {
          const unsigned int component_j = fe.system_to_component_index(j).first;
          for (const unsigned int q_point : fe_v.quadrature_point_indices())
          {
            switch (rom_flag)
            {
            case ROM_mat::pb:
              copy_data.cell_matrix(i, j) +=
                  (((component_j == component_i) ? ((component_i >= 3) ? (fe_v.shape_value_component(j, q_point, component_j) * sigma * A[0](component_j, component_i) * fe_v.shape_value_component(i, q_point, component_i))
                : (fe_v.shape_value_component(j, q_point, component_j) * mu * A[0](component_j, component_i) *
                  fe_v.shape_value_component(i, q_point, component_i)))
                                                 : 0) -
                   fe_v.shape_value_component(j, q_point, component_j) *
                       A[1](component_j, component_i) *
                       fe_v.shape_grad_component(i, q_point, component_i)[0] -
                   fe_v.shape_value_component(j, q_point, component_j) *
                       A[2](component_j, component_i) *
                       fe_v.shape_grad_component(i, q_point, component_i)[1] -
                   fe_v.shape_value_component(j, q_point, component_j) *
                       A[3](component_j, component_i) *
                       fe_v.shape_grad_component(i, q_point, component_i)[2]) *
                  fe_v.JxW(q_point);
              break;
            case ROM_mat::no_par:
              copy_data.cell_matrix(i, j) +=
                  (-fe_v.shape_value_component(j, q_point, component_j) *A[1](component_j, component_i) *
                       fe_v.shape_grad_component(i, q_point, component_i)[0] -
                   fe_v.shape_value_component(j, q_point, component_j) *
                       A[2](component_j, component_i) *
                       fe_v.shape_grad_component(i, q_point, component_i)[1] -
                   fe_v.shape_value_component(j, q_point, component_j) *
                       A[3](component_j, component_i) *
                       fe_v.shape_grad_component(i, q_point, component_i)[2]) *
                  fe_v.JxW(q_point);
              break;
            case ROM_mat::sigma:
              copy_data.cell_matrix(i, j) +=
                  ((component_j == component_i) ? ((component_i >= 3) ? fe_v.shape_value_component(j, q_point, component_j)* A[0](component_j, component_i) * fe_v.shape_value_component(i, q_point, component_i):0): 0) *
                  fe_v.JxW(q_point);
              break;
            case ROM_mat::mu:
              copy_data.cell_matrix(i, j) +=
                  ((component_j == component_i) ? ((component_i >= 3) ? 0: (fe_v.shape_value_component(j, q_point, component_j) * A[0](component_j, component_i) *
                  fe_v.shape_value_component(i, q_point, component_i)))
                                                 : 0) *
                  fe_v.JxW(q_point);
              break;
            case ROM_mat::L:
              copy_data.cell_matrix(i, j) +=
                  (component_j == component_i) ? (fe_v.shape_value_component(j, q_point, component_j) *
                   fe_v.shape_value_component(i, q_point, component_i)) *
                  fe_v.JxW(q_point):0;
              break;
            case ROM_mat::A:
              copy_data.cell_matrix(i, j) += cell->measure()*cell->measure()*
            (fe_v.shape_grad_component(j, q_point, component_j)[0] *
                H[0](component_j, component_i) *
                fe_v.shape_grad_component(i, q_point, component_i)[0] +
            fe_v.shape_grad_component(j, q_point, component_j)[0] *
                H[1](component_j, component_i) *
                fe_v.shape_grad_component(i, q_point, component_i)[1] +
            fe_v.shape_grad_component(j, q_point, component_j)[0] *
                H[2](component_j, component_i) *
                fe_v.shape_grad_component(i, q_point, component_i)[2] +
            fe_v.shape_grad_component(j, q_point, component_j)[1] *
                H[3](component_j, component_i) *
                fe_v.shape_grad_component(i, q_point, component_i)[0] +
            fe_v.shape_grad_component(j, q_point, component_j)[1] *
                H[4](component_j, component_i) *
                fe_v.shape_grad_component(i, q_point, component_i)[1] +
            fe_v.shape_grad_component(j, q_point, component_j)[1] *
                H[5](component_j, component_i) *
                fe_v.shape_grad_component(i, q_point, component_i)[2] +
            fe_v.shape_grad_component(j, q_point, component_j)[2] *
                H[6](component_j, component_i) *
                fe_v.shape_grad_component(i, q_point, component_i)[0] +
            fe_v.shape_grad_component(j, q_point, component_j)[2] *
                H[7](component_j, component_i) *
                fe_v.shape_grad_component(i, q_point, component_i)[1] +
            fe_v.shape_grad_component(j, q_point, component_j)[2] *
                H[8](component_j, component_i) *
                fe_v.shape_grad_component(i, q_point, component_i)[2]) *
            fe_v.JxW(q_point);
              break;
            case ROM_mat::AdFS:
              copy_data.cell_matrix(i, j) +=
                    (((component_j == component_i) ? ((component_i >= 3) ?
                    (fe_v.shape_value_component(j, q_point, component_j) *
                      sigma *
                      A[0](component_j, component_i) *
                      fe_v.shape_value_component(i, q_point, component_i))
                    : (fe_v.shape_value_component(j, q_point, component_j) *
                      mu * A[0](component_j, component_i) *
                      fe_v.shape_value_component(i, q_point, component_i)))
                                                  : 0) -
                    fe_v.shape_value_component(j, q_point, component_j) *
                        A[1](component_j, component_i) *
                        fe_v.shape_grad_component(i, q_point, component_i)[0] -
                    fe_v.shape_value_component(j, q_point, component_j) *
                        A[2](component_j, component_i) *
                        fe_v.shape_grad_component(i, q_point, component_i)[1] -
                    fe_v.shape_value_component(j, q_point, component_j) *
                        A[3](component_j, component_i) *
                        fe_v.shape_grad_component(i, q_point, component_i)[2]) *
                    fe_v.JxW(q_point);
              break;
            case ROM_mat::A0:
              copy_data.cell_matrix(i, j) +=
                  (-fe_v.shape_value_component(j, q_point, component_j) *
                       A[1](component_j, component_i) *
                       fe_v.shape_grad_component(i, q_point, component_i)[0] -
                   fe_v.shape_value_component(j, q_point, component_j) *
                       A[2](component_j, component_i) *
                       fe_v.shape_grad_component(i, q_point, component_i)[1] -
                   fe_v.shape_value_component(j, q_point, component_j) *
                       A[3](component_j, component_i) *
                       fe_v.shape_grad_component(i, q_point, component_i)[2]) *
                  fe_v.JxW(q_point);
              break;
            default:
              break;
            }
          }
        }
      }

      // Assembling the right hand side is also just as discussed in the
      // introduction:
      for (unsigned int i = 0; i < n_dofs; ++i)
      {
        for (const unsigned int q_point : fe_v.quadrature_point_indices())
        {
          switch (rom_flag)
          {
          case ROM_mat::pb:
            copy_data.cell_rhs(i) += (fe_v.shape_value_component(i, q_point, 3) *
                                        forcing_term.value(fe_v.quadrature_point(q_point), 0) +
                                    fe_v.shape_value_component(i, q_point, 4) *
                                        forcing_term.value(fe_v.quadrature_point(q_point), 1) +
                                    fe_v.shape_value_component(i, q_point, 5) *
                                        forcing_term.value(fe_v.quadrature_point(q_point), 2)) *
                                   fe_v.JxW(q_point);
            break;
          case ROM_mat::sigma:
              copy_data.cell_rhs(i) += (fe_v.shape_value_component(i, q_point, 3) *
                                          forcing_mu_term.value(fe_v.quadrature_point(q_point), 0) +
                                      fe_v.shape_value_component(i, q_point, 4) *
                                          forcing_mu_term.value(fe_v.quadrature_point(q_point), 1) +
                                      fe_v.shape_value_component(i, q_point, 5) *
                                          forcing_mu_term.value(fe_v.quadrature_point(q_point), 2)) *
                                    fe_v.JxW(q_point);
            break;
          case ROM_mat::mu:
              copy_data.cell_rhs(i) += (fe_v.shape_value_component(i, q_point, 3) *
                                          forcing_sigma_term.value(fe_v.quadrature_point(q_point), 0) +
                                      fe_v.shape_value_component(i, q_point, 4) *
                                          forcing_sigma_term.value(fe_v.quadrature_point(q_point), 1) +
                                      fe_v.shape_value_component(i, q_point, 5) *
                                          forcing_sigma_term.value(fe_v.quadrature_point(q_point), 2)) *
                                    fe_v.JxW(q_point);
            break;
          default:
            break;
          }
        }
      }
    };

    auto boundary_worker = [&](const Iterator &cell, const unsigned int &face_no,
                               ScratchData<dim> &scratch_data,
                               CopyData &copy_data)
    {
      scratch_data.fe_interface_values.reinit(cell, face_no);
      const FEFaceValuesBase<dim> &fe_face =
          scratch_data.fe_interface_values.get_fe_face_values(0);

      const auto &q_points = fe_face.get_quadrature_points();

      const unsigned int n_facet_dofs = fe_face.get_fe().n_dofs_per_cell();

      const std::vector<Tensor<1, dim>> &normals = fe_face.get_normal_vectors();

      const unsigned int degree = fe.degree;
      const double eta = 1 * 2 * degree * (degree + 1);
      const double penalty =
          eta * cell->face(face_no)->measure() / cell->measure();

      Tensor<1, dim> T_penalty_i, T_penalty_j;
      int id_1, id_2;

      for (unsigned int q_point = 0; q_point < q_points.size(); ++q_point)
      {
        for (unsigned int i = 0; i < n_facet_dofs; ++i)
        {
          const unsigned int component_i = fe.system_to_component_index(i).first;
          T_penalty_i.clear();

          if ((component_i > 2))
          {
            id_1 = (component_i + 1) % 3;
            id_2 = (component_i + 2) % 3;

            T_penalty_i[id_1] =
                T[id_2](id_1, component_i % 3) * normals[q_point][id_2];
            T_penalty_i[id_2] =
                T[id_1](id_2, component_i % 3) * normals[q_point][id_1];
          }

          for (unsigned int j = 0; j < n_facet_dofs; ++j)
          {
            const unsigned int component_j =
                fe.system_to_component_index(j).first;

            switch (rom_flag)
            {
            case ROM_mat::pb:
            case ROM_mat::no_par:
              T_penalty_j.clear();

              if ((component_j > 2))
              {
                id_1 = (component_j + 1) % 3;
                id_2 = (component_j + 2) % 3;

                T_penalty_j[id_1] =
                    T[id_2](id_1, component_j % 3) * normals[q_point][id_2];
                T_penalty_j[id_2] =
                    T[id_1](id_2, component_j % 3) * normals[q_point][id_1];
              }

              copy_data.cell_matrix(i, j) +=
                  ((component_j < 3 && component_i > 2)
                       ? (fe_face.shape_value_component(j, q_point, component_j) *
                          (A[1](component_i, component_j) * normals[q_point][0] +
                           A[2](component_i, component_j) * normals[q_point][1] +
                           A[3](component_i, component_j) * normals[q_point][2]) *
                          fe_face.shape_value_component(i, q_point, component_i))
                       : 0 + ((component_j > 2 && component_i > 2) ? (penalty * fe_face.shape_value_component(i, q_point, component_i) *
                       fe_face.shape_value_component(j, q_point, component_j) *
                      T_penalty_i * T_penalty_j)
                    : 0)) *
                  fe_face.JxW(q_point);
              break;
            case ROM_mat::M:
              copy_data.cell_matrix(i, j) +=
                  (-((component_j > 2 && component_i < 3)
                       ? (fe_face.shape_value_component(j, q_point, component_j) *
                          (A[1](component_i, component_j) * normals[q_point][0] +
                           A[2](component_i, component_j) * normals[q_point][1] +
                           A[3](component_i, component_j) * normals[q_point][2]) *
                          fe_face.shape_value_component(i, q_point, component_i))
                       : 0 )+ ((component_j < 3 && component_i > 2)
                        ? (fe_face.shape_value_component(j, q_point, component_j) *
                            (A[1](component_i, component_j) * normals[q_point][0] +
                            A[2](component_i, component_j) * normals[q_point][1] +
                            A[3](component_i, component_j) * normals[q_point][2]) *
                            fe_face.shape_value_component(i, q_point, component_i))
                                  : 0)) *
                  fe_face.JxW(q_point);
              break;
            case ROM_mat::D:
              copy_data.cell_matrix(i, j) +=
                  (((component_j > 2 && component_i < 3)
                       ? (fe_face.shape_value_component(j, q_point, component_j) *
                          (A[1](component_i, component_j) * normals[q_point][0] +
                           A[2](component_i, component_j) * normals[q_point][1] +
                           A[3](component_i, component_j) * normals[q_point][2]) *
                          fe_face.shape_value_component(i, q_point, component_i))
                       : 0 )+ ((component_j < 3 && component_i > 2)
                  ? (fe_face.shape_value_component(j, q_point, component_j) *
                      (A[1](component_i, component_j) * normals[q_point][0] +
                      A[2](component_i, component_j) * normals[q_point][1] +
                      A[3](component_i, component_j) * normals[q_point][2]) *
                      fe_face.shape_value_component(i, q_point, component_i))
                  : 0)) *
                  fe_face.JxW(q_point);
              break;
            case ROM_mat::S:
              T_penalty_i.clear();
              T_penalty_j.clear();

              if ((component_j > 2))
              {
                id_1 = (component_j + 1) % 3;
                id_2 = (component_j + 2) % 3;

                T_penalty_j[id_1] =
                    T[id_2](id_1, component_j % 3) * normals[q_point][id_2];
                T_penalty_j[id_2] =
                    T[id_1](id_2, component_j % 3) * normals[q_point][id_1];
              }

              copy_data.cell_matrix(i, j) +=
                  ((component_j > 2 && component_i > 2) ? (penalty *
                                                           fe_face.shape_value_component(i, q_point, component_i) *
                                                           fe_face.shape_value_component(j, q_point, component_j) *
                                                           T_penalty_i * T_penalty_j)
                                                        : 0) *
                  fe_face.JxW(q_point);
              break;
            case ROM_mat::A0:
              T_penalty_j.clear();

              if ((component_j > 2))
              {
                id_1 = (component_j + 1) % 3;
                id_2 = (component_j + 2) % 3;

                T_penalty_j[id_1] =
                    T[id_2](id_1, component_j % 3) * normals[q_point][id_2];
                T_penalty_j[id_2] =
                    T[id_1](id_2, component_j % 3) * normals[q_point][id_1];
              }
              copy_data.cell_matrix(i, j) +=
                  ((component_j < 3 && component_i > 2)
                       ? (fe_face.shape_value_component(j, q_point, component_j) *
                          (A[1](component_i, component_j) * normals[q_point][0] +
                           A[2](component_i, component_j) * normals[q_point][1] +
                           A[3](component_i, component_j) * normals[q_point][2]) *
                          fe_face.shape_value_component(i, q_point, component_i))
                       : 0 + ((component_j > 2 && component_i > 2) ? (penalty * fe_face.shape_value_component(i, q_point, component_i) *
                                                                      fe_face.shape_value_component(j, q_point, component_j) *
                                                                      T_penalty_i * T_penalty_j)
                                                                   : 0)) *
                  fe_face.JxW(q_point);
              break;
            default:
              break;
            }
          }
        }
      }
    };

    auto face_worker = [&](const Iterator &cell, const unsigned int &f,
                           const unsigned int &sf, const Iterator &ncell,
                           const unsigned int &nf, const unsigned int &nsf,
                           ScratchData<dim> &scratch_data, CopyData &copy_data)
    {
      FEInterfaceValues<dim> &fe_iv = scratch_data.fe_interface_values;
      fe_iv.reinit(cell, f, sf, ncell, nf, nsf);
      const auto &q_points = fe_iv.get_quadrature_points();

      copy_data.face_data.emplace_back();
      CopyDataFace &copy_data_face = copy_data.face_data.back();

      const unsigned int n_dofs = fe_iv.n_current_interface_dofs();

      copy_data_face.joint_dof_indices = fe_iv.get_interface_dof_indices();

      copy_data_face.cell_matrix.reinit(n_dofs, n_dofs);

      const std::vector<Tensor<1, dim>> &normals = fe_iv.get_normal_vectors();

      const unsigned int degree = fe.degree;
      const double eta = 1 * 2. * degree * (degree + 1);
      const double penalty1 = cell->face(f)->measure() / cell->measure();
      const double penalty2 = ncell->face(f)->measure() / ncell->measure();
      const double penalty = eta * (penalty1 + penalty2);

      Tensor<1, dim> T_penalty_i, T_penalty_j;
      int id_1, id_2;

      for (unsigned int q_point = 0; q_point < q_points.size(); ++q_point)
      {
        for (unsigned int i = 0; i < n_dofs; ++i)
        {
          const unsigned int component_i =
              fe.system_to_component_index(i % int(n_dofs / 2)).first;
          for (unsigned int j = 0; j < n_dofs; ++j)
          {
            const unsigned int component_j =
                fe.system_to_component_index(j % int(n_dofs / 2)).first;

            switch (rom_flag)
            {
            case ROM_mat::pb:
            case ROM_mat::no_par:
              T_penalty_j.clear();
              T_penalty_i.clear();

              if ((component_i < 3 && component_j < 3) || (component_i > 2 && component_j > 2))
              {
                id_1 = (component_j + 1) % 3;
                id_2 = (component_j + 2) % 3;

                T_penalty_j[id_1] =
                    T[id_2](id_1, component_j % 3) * normals[q_point][id_2];
                T_penalty_j[id_2] =
                    T[id_1](id_2, component_j % 3) * normals[q_point][id_1];

                id_1 = (component_i + 1) % 3;
                id_2 = (component_i + 2) % 3;

                T_penalty_i[id_1] =
                    T[id_2](id_1, component_i % 3) * normals[q_point][id_2];
                T_penalty_i[id_2] =
                    T[id_1](id_2, component_i % 3) * normals[q_point][id_1];
              }

              copy_data_face.cell_matrix(i, j) +=
                  (fe_iv.average_of_shape_values(j, q_point, component_j) *
                       (A[1](component_i, component_j) * normals[q_point][0] +
                        A[2](component_i, component_j) * normals[q_point][1] +
                        A[3](component_i, component_j) * normals[q_point][2]) *
                       fe_iv.jump_in_shape_values(i, q_point, component_i) +
                   // \eta \T  \dot \T
                   +penalty * fe_iv.jump_in_shape_values(i, q_point, component_i) *
                       fe_iv.jump_in_shape_values(j, q_point, component_j) *
                       T_penalty_i * T_penalty_j) * //
                  fe_iv.JxW(q_point);

              break;
            case ROM_mat::S:
              T_penalty_j.clear();
              T_penalty_i.clear();

              if ((component_i < 3 && component_j < 3) || (component_i > 2 && component_j > 2))
              {
                id_1 = (component_j + 1) % 3;
                id_2 = (component_j + 2) % 3;

                T_penalty_j[id_1] =
                    T[id_2](id_1, component_j % 3) * normals[q_point][id_2];
                T_penalty_j[id_2] =
                    T[id_1](id_2, component_j % 3) * normals[q_point][id_1];

                id_1 = (component_i + 1) % 3;
                id_2 = (component_i + 2) % 3;

                T_penalty_i[id_1] =
                    T[id_2](id_1, component_i % 3) * normals[q_point][id_2];
                T_penalty_i[id_2] =
                    T[id_1](id_2, component_i % 3) * normals[q_point][id_1];
              }

              copy_data_face.cell_matrix(i, j) +=
                  (penalty * fe_iv.jump_in_shape_values(i, q_point, component_i) *
                   fe_iv.jump_in_shape_values(j, q_point, component_j) *
                   T_penalty_i * T_penalty_j) * //
                  fe_iv.JxW(q_point);
              break;
            case ROM_mat::D:
              copy_data_face.cell_matrix(i, j) +=
                  (fe_iv.average_of_shape_values(j, q_point, component_j) *
                       (A[1](component_i, component_j) * normals[q_point][0] +
                        A[2](component_i, component_j) * normals[q_point][1] +
                        A[3](component_i, component_j) * normals[q_point][2]) *
                       fe_iv.jump_in_shape_values(i, q_point, component_i)) * //
                  fe_iv.JxW(q_point);

              break;
            case ROM_mat::A0:
              T_penalty_j.clear();
              T_penalty_i.clear();

              if ((component_i < 3 && component_j < 3) || (component_i > 2 && component_j > 2))
              {
                id_1 = (component_j + 1) % 3;
                id_2 = (component_j + 2) % 3;

                T_penalty_j[id_1] =
                    T[id_2](id_1, component_j % 3) * normals[q_point][id_2];
                T_penalty_j[id_2] =
                    T[id_1](id_2, component_j % 3) * normals[q_point][id_1];

                id_1 = (component_i + 1) % 3;
                id_2 = (component_i + 2) % 3;

                T_penalty_i[id_1] =
                    T[id_2](id_1, component_i % 3) * normals[q_point][id_2];
                T_penalty_i[id_2] =
                    T[id_1](id_2, component_i % 3) * normals[q_point][id_1];
              }
              
              copy_data_face.cell_matrix(i, j) +=
                  (fe_iv.average_of_shape_values(j, q_point, component_j) *
                       (A[1](component_i, component_j) * normals[q_point][0] +
                        A[2](component_i, component_j) * normals[q_point][1] +
                        A[3](component_i, component_j) * normals[q_point][2]) *
                       fe_iv.jump_in_shape_values(i, q_point, component_i) +
                   // \eta \T  \dot \T
                   +penalty * fe_iv.jump_in_shape_values(i, q_point, component_i) *
                       fe_iv.jump_in_shape_values(j, q_point, component_j) *
                       T_penalty_i * T_penalty_j) * //
                  fe_iv.JxW(q_point);

              break;
            default:
              break;
            }
          }
        }
      }
    };

    const auto filtered_iterator_range =
        filter_iterators(dof_handler.active_cell_iterators(),
                         IteratorFilters::LocallyOwnedCell());

    auto copier = [&](const CopyData &c)
    {
      constraints.distribute_local_to_global(c.cell_matrix, c.cell_rhs,
                                             c.local_dof_indices, system_matrix,
                                             system_rhs);

      for (auto &cdf : c.face_data)
      {
        constraints.distribute_local_to_global(
            cdf.cell_matrix, cdf.joint_dof_indices, system_matrix);
      }
    };

    const unsigned int n_gauss_points = std::max(
                                            static_cast<unsigned int>(std::ceil(1. * (mapping.get_degree() + 1) / 2)),
                                            dof_handler.get_fe().degree + 1) +
                                        1;

    pcout << "   Gauss points: " << n_gauss_points << std::endl;

    // Affine decomposition
    // files specs
    const std::string matrix_folder = "affine_matrices";
    const std::string stanpshot_prefix = "snapshot";

    // available decompositions
    std::vector<ROM_mat> rom_mat_vec;
    if (fl_L)
    {
      rom_mat_vec.push_back(ROM_mat::L);
    }
    if (fl_M)
    {
      rom_mat_vec.push_back(ROM_mat::M);
    }
    if (fl_S)
    {
      rom_mat_vec.push_back(ROM_mat::S);
    }
    if (fl_A)
    {
      rom_mat_vec.push_back(ROM_mat::A);
    }
    if (fl_no_par)
    {
      rom_mat_vec.push_back(ROM_mat::no_par);
    }
    if (fl_sigma)
    {
      rom_mat_vec.push_back(ROM_mat::sigma);
    }
    if (fl_mu)
    {
      rom_mat_vec.push_back(ROM_mat::mu);
    }
    if (fl_AdFS)
    {
      rom_mat_vec.push_back(ROM_mat::AdFS);
    }
    if (fl_A0)
    {
      rom_mat_vec.push_back(ROM_mat::A0);
    }
    if (fl_D)
    {
      rom_mat_vec.push_back(ROM_mat::D);
    }

    for (auto flag : rom_mat_vec)
    {
      rom_flag = flag;

      ScratchData<dim> scratch_data(mapping, fe, n_gauss_points);
      CopyData copy_data;

      auto t1 = std::chrono::high_resolution_clock::now();
      MeshWorker::mesh_loop(filtered_iterator_range,
                            cell_worker,
                            copier,
                            scratch_data,
                            copy_data,
                            MeshWorker::assemble_own_cells |
                              MeshWorker::assemble_boundary_faces |
                              MeshWorker::assemble_own_interior_faces_once |
                              MeshWorker::assemble_ghost_faces_once,
                            boundary_worker,
                            face_worker);

      system_matrix.compress(VectorOperation::add);
      system_rhs.compress(VectorOperation::add);
      auto t2 = std::chrono::high_resolution_clock::now();

      pcout << "Assembled: " << rom_mat_labels[int(rom_flag)] << " Elapsed time: " << std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count() << " ms" << std::endl;
      std::ofstream output("./" + matrix_folder + "/petsc_mat_" + rom_mat_labels[int(rom_flag)] + "_" + std::to_string(Utilities::MPI::this_mpi_process(mpi_communicator)) + ".txt");
      output << std::setprecision(17);
      system_matrix.print(output);

      if (flag == ROM_mat::mu || flag == ROM_mat::sigma || flag == ROM_mat::no_par)
      {
        std::ofstream output_rhs("./" + matrix_folder + "/petsc_rhs_" + rom_mat_labels[int(rom_flag)] + "_" + std::to_string(Utilities::MPI::this_mpi_process(mpi_communicator)) + ".txt");
        system_rhs.print(output_rhs, 17);
      }
      setup_system();
    }

    auto t1 = std::chrono::high_resolution_clock::now();
    rom_flag = ROM_mat::pb;
    ScratchData<dim> scratch_data(mapping, fe, n_gauss_points);
    CopyData copy_data;
    MeshWorker::mesh_loop(filtered_iterator_range,
                          cell_worker,
                          copier,
                          scratch_data,
                          copy_data,
                          MeshWorker::assemble_own_cells |
                              MeshWorker::assemble_boundary_faces |
                              MeshWorker::assemble_own_interior_faces_once |
                              MeshWorker::assemble_ghost_faces_once,
                          boundary_worker,
                          face_worker);
    system_matrix.compress(VectorOperation::add);
    system_rhs.compress(VectorOperation::add);

    auto t2 = std::chrono::high_resolution_clock::now();

    pcout << "Assembled: " << rom_mat_labels[int(rom_flag)] << " .Elapsed time: " << std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count()
          << " ms" << std::endl;
    std::ofstream output("./" + matrix_folder + "/petsc_mat_whole_system_" + std::to_string(Utilities::MPI::this_mpi_process(mpi_communicator)) + ".txt");
    output << std::setprecision(17);
    system_matrix.print(output);

    std::ofstream output_rhs("./" + matrix_folder + "/petsc_rhs_whole_system_" + std::to_string(Utilities::MPI::this_mpi_process(mpi_communicator)) + ".txt");
    system_rhs.print(output_rhs, 17);
    pcout << "Finished saving matrices\n";
  }

  template <int dim>
  void MaxwellProblem<dim>::solve()
  {
    TimerOutput::Scope t(computing_timer, "solve");

    LA::MPI::Vector completely_distributed_solution(
        locally_owned_dofs, mpi_communicator);

    // MUMPS
    SolverControl solver_control;
    PETScWrappers::SparseDirectMUMPS solver(solver_control, mpi_communicator);
    // solver.set_symmetric_mode(true);
    solver.solve(system_matrix, completely_distributed_solution, system_rhs);

    pcout << "   Solved in " << solver_control.last_step() << " iterations."
          << std::endl;
    constraints.distribute(completely_distributed_solution);
    locally_relevant_solution = completely_distributed_solution;
  }

  template <int dim>
  void MaxwellProblem<dim>::refine_grid()
  {
    TimerOutput::Scope t(computing_timer, "refine");

    Vector<float> estimated_error_per_cell(triangulation.n_active_cells());

    KellyErrorEstimator<dim>::estimate(dof_handler,
                                       QGauss<dim - 1>(fe.degree + 1), {},
                                       locally_relevant_solution,
                                       estimated_error_per_cell);

    parallel::distributed::GridRefinement::refine_and_coarsen_fixed_number(
        triangulation, estimated_error_per_cell, 0.3, 0.03);

    triangulation.execute_coarsening_and_refinement();
  }

  template <int dim>
  void MaxwellProblem<dim>::output_results(const unsigned int cycle)
  {
    TimerOutput::Scope t(computing_timer, "results");

    DataOut<dim> data_out;

    DataOutBase::VtkFlags flags;
    flags.write_higher_order_cells = true;
    data_out.set_flags(flags);

    std::vector<std::string> solution_names(dim, "H");
    solution_names.insert(solution_names.end(), {"E", "E", "E"});

    std::vector<DataComponentInterpretation::DataComponentInterpretation>
        data_component_interpretation(dim * 2, DataComponentInterpretation::component_is_part_of_vector);

    data_out.attach_dof_handler(dof_handler);

    data_out.add_data_vector(locally_relevant_solution, solution_names,
                             DataOut<dim>::type_dof_data,
                             data_component_interpretation);

    // std::vector<std::string> solution_names;
    // solution_names.emplace_back("H0");
    // solution_names.emplace_back("H1");
    // solution_names.emplace_back("H2");
    // solution_names.emplace_back("E0");
    // solution_names.emplace_back("E1");
    // solution_names.emplace_back("E2");
    // data_out.add_data_vector(locally_relevant_solution, solution_names,
    //                           DataOut<dim>::type_dof_data);

    Vector<float> subdomain(triangulation.n_active_cells());
    for (unsigned int i = 0; i < subdomain.size(); ++i)
      subdomain(i) = triangulation.locally_owned_subdomain();
    data_out.add_data_vector(subdomain, "subdomain");

    data_out.build_patches(mapping, mapping.get_degree(),
                           DataOut<dim>::curved_inner_cells);

    // if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0 )
    //   {
    //     std::vector<std::string>    filenames;
    //     for (unsigned int i=0;
    //           i < Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD);
    //           i++)
    //       {
    //         filenames.push_back("solution." +
    //                             Utilities::int_to_string(i,4) +
    //                             ".vtu");
    //       }
    //     std::ofstream master_output("solution.pvtu");
    //     data_out.write_pvtu_record(master_output, filenames);
    //   }

    data_out.write_vtu_with_pvtu_record(
        "./", "solution", cycle, mpi_communicator);

    // std::ofstream output("solution-" + std::to_string(cycle) + ".vtk");
    // std::string output = "solution-" + std::to_string(cycle) + ".vtk";
    // data_out.write_vtu_in_parallel(output, mpi_communicator);
    // data_out.write_vtk(output);
  }

  template <int dim>
  void MaxwellProblem<dim>::run()
  {
    pcout << "Running with PETSc on "
          << Utilities::MPI::n_mpi_processes(mpi_communicator)
          << " MPI rank(s)..." << std::endl;
    pcout << "n threads: " << dealii::MultithreadInfo::n_threads()
          << std::endl;

    if (task == "fom")
      run_fom();
    else if (task == "affine")
      run_affine();
    else if (task == "plot")
      run_plot();
    else if (task == "labels")
      run_plot_labels();
    else if (task == "save_pos")
      save_pos();
    else
      pcout << "Task must be fom|affine|plot|labels|save_pos\n";
  }

  template <int dim>
  void MaxwellProblem<dim>::run_fom()
  {
    unsigned int last_cycle = 2;
    for (unsigned int cycle = 0; cycle < last_cycle; ++cycle)
    {
      pcout << "Cycle " << cycle << ':' << std::endl;

      if (cycle == 0)
      {
        GridGenerator::generate_from_name_and_arguments(triangulation,
                                                        grid_generator_function,
                                                        grid_generator_arguments);
        
        std::ofstream out("mesh_"+std::to_string(Utilities::MPI::this_mpi_process(MPI_COMM_WORLD))+".vtu");
        GridOut       grid_out;
        grid_out.write_vtu(triangulation, out);
        pcout << " written to " << "mesh" << std::endl << std::endl;
      }
      else
        triangulation.refine_global(1);

      setup_system();

      pcout << "   Number of active cells:       "
            << triangulation.n_global_active_cells() << std::endl
            << "   Number of degrees of freedom: " << dof_handler.n_dofs()
            << std::endl;

      assemble_system();
      solve();

      computing_timer.print_summary();
      computing_timer.reset();

      pcout << std::endl;
    }
  }

  template <int dim>
  void MaxwellProblem<dim>::run_affine()
  {
    GridGenerator::generate_from_name_and_arguments(triangulation,
                                                    grid_generator_function,
                                                    grid_generator_arguments);
    triangulation.refine_global(n_refinements);
    setup_system();

    pcout << "   Number of active cells:       "
          << triangulation.n_global_active_cells() << std::endl
          << "   Number of degrees of freedom: " << dof_handler.n_dofs()
          << std::endl;

    assemble_system_affine();
    solve();
    output_results(0);

    // print timings
    computing_timer.print_summary();
    computing_timer.reset();
    pcout << std::endl;
  }

  template <int dim>
  void MaxwellProblem<dim>::run_plot()
  {
    GridGenerator::generate_from_name_and_arguments(triangulation,
                                                    grid_generator_function,
                                                    grid_generator_arguments);
    setup_system();
    pcout << "   Number of active cells:       "
          << triangulation.n_global_active_cells() << std::endl
          << "   Number of degrees of freedom: " << dof_handler.n_dofs()
          << std::endl;

    for (unsigned int i = 0; i < n_snapshots; i++)
    {
      std::ifstream is("./snapshots/snapshot_" + std::to_string(i) + "_" + std::to_string(Utilities::MPI::this_mpi_process(mpi_communicator)) + ".txt");

      std::istream_iterator<double> start(is), end;
      std::vector<double> numbers(start, end);
      Vector<double> vec_loaded(numbers.size());
      std::copy(numbers.begin(), numbers.end(), vec_loaded.begin());

      std::vector<unsigned int> indices;
      locally_owned_dofs.fill_index_vector(indices);
      locally_relevant_solution.reinit(locally_owned_dofs,
                                     locally_relevant_dofs,
                                     mpi_communicator);
      locally_relevant_solution.add(indices, vec_loaded);
      locally_relevant_solution.compress(VectorOperation::add);

      DataOut<dim> data_out;

      DataOutBase::VtkFlags flags;
      flags.write_higher_order_cells = true;
      data_out.set_flags(flags);

      data_out.attach_dof_handler(dof_handler);

      std::vector<std::string> solution_names(dim, "H");
      solution_names.insert(solution_names.end(), {"E", "E", "E"});

      std::vector<DataComponentInterpretation::DataComponentInterpretation>
          data_component_interpretation(dim * 2,
                                        DataComponentInterpretation::component_is_part_of_vector);

      data_out.add_data_vector(locally_relevant_solution, solution_names,
                               DataOut<dim>::type_dof_data,
                               data_component_interpretation);

      Vector<float> subdomain(triangulation.n_active_cells());
      for (unsigned int i = 0; i < subdomain.size(); ++i)
        subdomain(i) = triangulation.locally_owned_subdomain();
      data_out.add_data_vector(subdomain, "subdomain");

      data_out.build_patches(mapping, mapping.get_degree(),
                             DataOut<dim>::curved_inner_cells);

      data_out.write_vtu_with_pvtu_record(
          "./snapshots/", "reconstructed", i, mpi_communicator);
    }

    for (unsigned int i = 0; i < n_snapshots; i++)
    {
      std::ifstream is("./snapshots/rsnapshot_" + std::to_string(i) + "_" + std::to_string(Utilities::MPI::this_mpi_process(mpi_communicator)) + ".txt");

      std::istream_iterator<double> start(is), end;
      std::vector<double> numbers(start, end);
      Vector<double> vec_loaded(numbers.size());
      std::copy(numbers.begin(), numbers.end(), vec_loaded.begin());

      std::vector<unsigned int> indices;
      locally_owned_dofs.fill_index_vector(indices);
      locally_relevant_solution.reinit(locally_owned_dofs,
                                     locally_relevant_dofs,
                                     mpi_communicator);
      locally_relevant_solution.add(indices, vec_loaded);
      locally_relevant_solution.compress(VectorOperation::add);

      DataOut<dim> data_out;

      DataOutBase::VtkFlags flags;
      flags.write_higher_order_cells = true;
      data_out.set_flags(flags);

      data_out.attach_dof_handler(dof_handler);

      std::vector<std::string> solution_names(dim, "H");
      solution_names.insert(solution_names.end(), {"E", "E", "E"});

      std::vector<DataComponentInterpretation::DataComponentInterpretation>
          data_component_interpretation(dim * 2,
                                        DataComponentInterpretation::component_is_part_of_vector);

      data_out.add_data_vector(locally_relevant_solution, solution_names,
                               DataOut<dim>::type_dof_data,
                               data_component_interpretation);

      Vector<float> subdomain(triangulation.n_active_cells());
      for (unsigned int i = 0; i < subdomain.size(); ++i)
        subdomain(i) = triangulation.locally_owned_subdomain();
      data_out.add_data_vector(subdomain, "subdomain");

      data_out.build_patches(mapping, mapping.get_degree(),
                             DataOut<dim>::curved_inner_cells);

      data_out.write_vtu_with_pvtu_record(
          "./snapshots/", "rreconstructed", i, mpi_communicator);
    }
  }

  template <int dim>
  void MaxwellProblem<dim>::run_plot_labels()
  {
    GridGenerator::generate_from_name_and_arguments(triangulation,
                                                    grid_generator_function,
                                                    grid_generator_arguments);
    FESystem<dim> fe_p(FE_DGQ<dim>(2), 1);
    DoFHandler<dim> dof_handler_p(triangulation);
    dof_handler_p.distribute_dofs(fe_p);

    pcout << "   Number of active cells:       "
          << triangulation.n_global_active_cells() << std::endl
          << "   Number of degrees of freedom: " << dof_handler_p.n_dofs()
          << std::endl;

    IndexSet locally_owned_dofs_p = dof_handler_p.locally_owned_dofs();
    IndexSet locally_relevant_dofs_p =
        DoFTools::extract_locally_relevant_dofs(dof_handler_p);

    AffineConstraints<double> constraints_p;
    constraints_p.clear();
    constraints_p.reinit(locally_relevant_dofs);
    constraints_p.close();

    DynamicSparsityPattern dsp_p(locally_relevant_dofs_p);

    DoFTools::make_flux_sparsity_pattern(dof_handler_p, dsp_p);

    SparsityTools::distribute_sparsity_pattern(dsp_p,
                                               dof_handler_p.locally_owned_dofs(),
                                               mpi_communicator,
                                               locally_relevant_dofs_p);

    LA::MPI::Vector locally_relevant_solution_p;

    locally_relevant_solution_p.reinit(locally_owned_dofs_p,
                                       locally_relevant_dofs_p,
                                       mpi_communicator);

    std::ifstream is_("./affine_matrices/labels_dofs_" + std::to_string(Utilities::MPI::this_mpi_process(mpi_communicator)) + ".txt");
    std::istream_iterator<double> start_(is_), end_;
    std::vector<double> numbers_(start_, end_);
    Vector<double> partition_(numbers_.size());
    std::copy(numbers_.begin(), numbers_.end(), partition_.begin());
    std::cout  << "rank: " << std::to_string(Utilities::MPI::this_mpi_process(mpi_communicator));
    for (size_t i = 0; i < partition_.size(); i++)
    {
      std::cout << partition_[i] << " ";
    }

    pcout << std::endl << "Dofs partition field size: " << partition_.size() << std::endl;

    std::vector<unsigned int> indices;
    locally_owned_dofs_p.fill_index_vector(indices);
    locally_relevant_solution_p.add(indices, partition_);
    locally_relevant_solution_p.compress(VectorOperation::add);

    DataOut<dim> data_out;

    DataOutBase::VtkFlags flags;
    flags.write_higher_order_cells = true;
    data_out.set_flags(flags);
    data_out.attach_dof_handler(dof_handler_p);

    std::vector<std::string> solution_names(1, "partition_dofs");

    std::vector<DataComponentInterpretation::DataComponentInterpretation>
        data_component_interpretation(1,
                                      DataComponentInterpretation::component_is_scalar);

    data_out.add_data_vector(locally_relevant_solution_p, solution_names,
                             DataOut<dim>::type_dof_data,
                             data_component_interpretation);

    Vector<float> subdomain(triangulation.n_active_cells());
    for (unsigned int i = 0; i < subdomain.size(); ++i)
      subdomain(i) = triangulation.locally_owned_subdomain();
    data_out.add_data_vector(subdomain, "subdomain");
    
    data_out.build_patches(mapping, mapping.get_degree(),
                           DataOut<dim>::curved_inner_cells);

    data_out.write_vtu_with_pvtu_record(
        "./", "partition", 0, mpi_communicator);
  }

  template <int dim>
  void MaxwellProblem<dim>::save_pos()
  {
    pcout << "Save pos\n";
    GridGenerator::generate_from_name_and_arguments(triangulation,
                                                        grid_generator_function,
                                                        grid_generator_arguments);
    FESystem<dim> fe_pdo(FE_DGQ<dim>(2), 1);
    DoFHandler<dim> dof_handler_pdo(triangulation);
    dof_handler_pdo.distribute_dofs(fe_pdo);

    std::map<types::global_dof_index, Point<dim>> support_points;
    DoFTools::map_dofs_to_support_points(mapping, dof_handler_pdo, support_points);

    pcout << "   Number of active cells:       "
          << triangulation.n_global_active_cells() << std::endl
          << "   Number of degrees of freedom: " << dof_handler_pdo.n_dofs()
          << std::endl;

    IndexSet locally_owned_dofs_pdo = dof_handler_pdo.locally_owned_dofs();
    IndexSet locally_relevant_dofs_pdo =
        DoFTools::extract_locally_relevant_dofs(dof_handler_pdo);

    AffineConstraints<double> constraints_pdo;
    constraints_pdo.clear();
    constraints_pdo.reinit(locally_relevant_dofs);
    constraints_pdo.close();

    DynamicSparsityPattern dsp_pdo(locally_relevant_dofs_pdo);

    DoFTools::make_flux_sparsity_pattern(dof_handler_pdo, dsp_pdo);

    SparsityTools::distribute_sparsity_pattern(dsp_pdo,
                                               dof_handler_pdo.locally_owned_dofs(),
                                               mpi_communicator,
                                               locally_relevant_dofs_pdo);

    LA::MPI::Vector pos_x, pos_y, pos_z;
    FullMatrix<double> pos_cells;
    pos_cells.reinit(triangulation.n_locally_owned_active_cells(), 3);
    pos_x.reinit(locally_owned_dofs_pdo, mpi_communicator);
    pos_y.reinit(locally_owned_dofs_pdo, mpi_communicator);
    pos_z.reinit(locally_owned_dofs_pdo, mpi_communicator);

    size_t cell_id{0};
    const auto filtered_iterators_range = filter_iterators(dof_handler_pdo.active_cell_iterators(), IteratorFilters::LocallyOwnedCell());
    for (const auto &cell : filtered_iterators_range)
    {
      pos_cells(cell_id, 0) = cell->center()[0];
      pos_cells(cell_id, 1) = cell->center()[1];
      pos_cells(cell_id, 2) = cell->center()[2];
      cell_id++;
    }
    
    for (const auto& [key, value] : support_points)
    {
      pos_x(key) = value[0];
      pos_y(key) = value[1];
      pos_z(key) = value[2];
    }

    const std::string matrix_folder = "affine_matrices";
    std::ofstream output_cell("./" + matrix_folder + "/petsc_pos_cell_" +std::to_string(n_refinements)+"_"+ std::to_string(Utilities::MPI::this_mpi_process(mpi_communicator)) + ".txt");
    output_cell << std::setprecision(17);
    pos_cells.print_formatted(output_cell, 3, true, 0, "0");

    std::ofstream output_x("./" + matrix_folder + "/petsc_pos_x_" +std::to_string(n_refinements)+"_"+ std::to_string(Utilities::MPI::this_mpi_process(mpi_communicator)) + ".txt");
    output_x << std::setprecision(17);
    pos_x.print(output_x);
    std::ofstream output_y("./" + matrix_folder + "/petsc_pos_y_" +std::to_string(n_refinements)+"_"+ std::to_string(Utilities::MPI::this_mpi_process(mpi_communicator)) + ".txt");
    output_y << std::setprecision(17);
    pos_y.print(output_y);
    std::ofstream output_z("./" + matrix_folder + "/petsc_pos_z_" +std::to_string(n_refinements)+"_"+ std::to_string(Utilities::MPI::this_mpi_process(mpi_communicator)) + ".txt");
    output_z << std::setprecision(17);
    pos_z.print(output_z);
  }
}

int main(int argc, char *argv[])
{
  try
  {
    // deallog.depth_console(2);
    dealii::Utilities::MPI::MPI_InitFinalize
        mpi_initialization(argc, argv, 1);//dealii::numbers::invalid_unsigned_int);

    MaxwellDG::MaxwellProblem<3> maxwell_problem;
    ParameterAcceptor::initialize("parameters.prm");
    maxwell_problem.run();
  }
  catch (std::exception &exc)
  {
    std::cerr << std::endl
              << std::endl
              << "----------------------------------------------------"
              << std::endl;
    std::cerr << "Exception on processing: " << std::endl
              << exc.what() << std::endl
              << "Aborting!" << std::endl
              << "----------------------------------------------------"
              << std::endl;

    return 1;
  }
  catch (...)
  {
    std::cerr << std::endl
              << std::endl
              << "----------------------------------------------------"
              << std::endl;
    std::cerr << "Unknown exception!" << std::endl
              << "Aborting!" << std::endl
              << "----------------------------------------------------"
              << std::endl;
    return 1;
  }
  return 0;
}
