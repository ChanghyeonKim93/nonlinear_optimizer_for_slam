#include "nonlinear_optimizer/pose_graph_optimizer/pose_graph_optimizer_analytic.h"

#include "nonlinear_optimizer/loss_function.h"

namespace nonlinear_optimizer {
namespace pose_graph_optimizer {

PoseGraphOptimizerAnalytic::PoseGraphOptimizerAnalytic() {}

PoseGraphOptimizerAnalytic::~PoseGraphOptimizerAnalytic() {}

bool PoseGraphOptimizerAnalytic::Solve(const Options& options) {
  bool success = true;

  size_t iteration = 0;
  double cost = 0.0;
  double previous_cost = 1e30;
  for (; iteration < options.max_iterations; ++iteration) {
    // TODO(@ChanghyeonKim): Implement the optimization algorithm

    for (auto& constraint : constraints_) {
      auto& query_pose = optimized_pose_map_.at(constraint.query_pose_index);
      auto& reference_pose =
          optimized_pose_map_.at(constraint.reference_pose_index);

      auto& query_position = query_pose.position;
      auto& query_orientation = query_pose.orientation;
      auto& reference_position = reference_pose.position;
      auto& reference_orientation = reference_pose.orientation;
    }
    for (const auto& fixed_pose_index : fixed_pose_index_set_) {
      auto& fixed_pose = optimized_pose_map_.at(fixed_pose_index);
      // TODO(@ChanghyeonKim): Set fixed pose
    }

    // Make sparse Hessian matrix (eigen)

    // Solve normal equation using Sparse Cholesky decomposition

    // Update poses

    // Check convergence
  }

  // Check success
  if (!success) return false;

  UpdateOptimizedPose();

  return true;
}

}  // namespace pose_graph_optimizer
}  // namespace nonlinear_optimizer