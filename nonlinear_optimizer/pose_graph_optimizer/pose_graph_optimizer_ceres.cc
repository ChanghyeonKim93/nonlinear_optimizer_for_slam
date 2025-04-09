#include "nonlinear_optimizer/pose_graph_optimizer/pose_graph_optimizer_ceres.h"

#include "ceres/ceres.h"

#include "nonlinear_optimizer/ceres_loss_function.h"
#include "nonlinear_optimizer/pose_graph_optimizer/ceres_cost_functor.h"

namespace nonlinear_optimizer {
namespace pose_graph_optimizer {

PoseGraphOptimizerCeres::PoseGraphOptimizerCeres() {}

PoseGraphOptimizerCeres::~PoseGraphOptimizerCeres() {}

bool PoseGraphOptimizerCeres::Solve(const Options& options) {
  ceres::Problem problem;
  for (const auto& constraint : constraints_) {
    auto& query_pose = optimized_pose_map_.at(constraint.query_pose_index);
    auto& reference_pose =
        optimized_pose_map_.at(constraint.reference_pose_index);

    auto& query_position = query_pose.position;
    auto& query_orientation = query_pose.orientation;
    auto& reference_position = reference_pose.position;
    auto& reference_orientation = reference_pose.orientation;

    ceres::CostFunction* cost_function =
        RelativePoseCostFunctor::Create(constraint);
    ceres::LossFunction* loss_function = nullptr;
    problem.AddResidualBlock(cost_function, loss_function, reference_position,
                             reference_orientation, query_position,
                             query_orientation);
  }
  for (const auto& fixed_pose_index : fixed_pose_index_set_) {
    auto& fixed_pose = optimized_pose_map_.at(fixed_pose_index);
    problem.SetParameterBlockConstant(fixed_pose.position);
    problem.SetParameterBlockConstant(fixed_pose.orientation);
  }

  ceres::Solver::Options ceres_options;
  ceres_options.max_num_iterations = options.max_iterations;
  ceres_options.gradient_tolerance =
      options.convergence_handle.gradient_tolerance;
  ceres_options.parameter_tolerance =
      options.convergence_handle.parameter_tolerance;
  ceres_options.function_tolerance =
      options.convergence_handle.function_tolerance;
  ceres::Solver::Summary ceres_summary;
  ceres::Solver solver;
  solver.Solve(ceres_options, &problem, &ceres_summary);

  std::cerr << ceres_summary.BriefReport() << std::endl;
  std::cerr << ceres_summary.FullReport() << std::endl;

  if (!ceres_summary.IsSolutionUsable()) return false;

  for (const auto& [index, optimized_pose] : optimized_pose_map_) {
    auto& original_pose_ptr = index_to_pose_ptr_bimap_.GetValue(index);
    original_pose_ptr->translation().x() = optimized_pose.position[0];
    original_pose_ptr->translation().y() = optimized_pose.position[1];
    original_pose_ptr->translation().z() = optimized_pose.position[2];
    Orientation optimized_orientation(
        optimized_pose.orientation[0], optimized_pose.orientation[1],
        optimized_pose.orientation[2], optimized_pose.orientation[3]);
    optimized_orientation.normalize();
    original_pose_ptr->linear() = optimized_orientation.toRotationMatrix();
  };

  return true;
}

}  // namespace pose_graph_optimizer
}  // namespace nonlinear_optimizer