#include "nonlinear_optimizer/pose_graph_optimizer/pose_graph_optimizer_ceres.h"

#include "ceres/ceres.h"

#include "nonlinear_optimizer/ceres_loss_function.h"
#include "nonlinear_optimizer/pose_graph_optimizer/ceres_cost_functor.h"

namespace nonlinear_optimizer {
namespace pose_graph_optimizer {

PoseGraphOptimizerCeres::PoseGraphOptimizerCeres() {}

PoseGraphOptimizerCeres::~PoseGraphOptimizerCeres() {}

bool PoseGraphOptimizerCeres::Solve(const Options& options) {
  std::set<int> active_pose_set;
  for (const auto& [index, pose_ptr] : index_to_pose_ptr_bimap_) {
    if (fixed_pose_index_set_.find(index) == fixed_pose_index_set_.end())
      continue;
    active_pose_set.insert(index);
  }

  return true;
}

}  // namespace pose_graph_optimizer
}  // namespace nonlinear_optimizer