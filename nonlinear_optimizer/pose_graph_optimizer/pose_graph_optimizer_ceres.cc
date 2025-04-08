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
  ceres::Solver::Summary ceres_summary;
  ceres::Solver solver;
  solver.Solve(ceres_options, &problem, &ceres_summary);

  std::cerr << ceres_summary.FullReport() << std::endl;

  return true;
}

}  // namespace pose_graph_optimizer
}  // namespace nonlinear_optimizer