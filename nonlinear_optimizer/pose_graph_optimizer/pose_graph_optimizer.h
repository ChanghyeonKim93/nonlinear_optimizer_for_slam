#ifndef NONLINEAR_OPTIMIZER_POSE_GRAPH_OPTIMIZER_POSE_GRAPH_OPTIMIZER_H_
#define NONLINEAR_OPTIMIZER_POSE_GRAPH_OPTIMIZER_POSE_GRAPH_OPTIMIZER_H_

#include <memory>

#include "nonlinear_optimizer/loss_function.h"
#include "nonlinear_optimizer/options.h"
#include "nonlinear_optimizer/pose_graph_optimizer/types.h"
#include "types.h"

namespace nonlinear_optimizer {
namespace pose_graph_optimizer {

class PoseGraphOptimizer {
 public:
  PoseGraphOptimizer() {}

  ~PoseGraphOptimizer() {}

  void SetLossFunction(const std::shared_ptr<LossFunction>& loss_function) {
    loss_function_ = loss_function;
  }

  void SetConstraint(const Constraint& constraint) {
    constraints_.push_back(constraint);
  }

  void SetPose(const int pose_index, Pose* pose_ptr) {
    pose_ptr_map_.insert({pose_index, pose_ptr});
    optimized_pose_map_.insert({pose_index, *pose_ptr});
  }

  /// @brief Solve the pose graph optimization problem
  /// @param options Optimization options
  /// @return Success or not
  virtual bool Solve(const Options& options) = 0;

 protected:
  std::shared_ptr<LossFunction> loss_function_{nullptr};
  std::unordered_map<int, Pose*> pose_ptr_map_;
  std::unordered_map<int, Pose> optimized_pose_map_;
  std::vector<Constraint> constraints_;
};

}  // namespace pose_graph_optimizer
}  // namespace nonlinear_optimizer

#endif  // NONLINEAR_OPTIMIZER_POSE_GRAPH_OPTIMIZER_POSE_GRAPH_OPTIMIZER_H_
