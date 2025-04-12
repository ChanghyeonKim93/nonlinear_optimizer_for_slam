#ifndef NONLINEAR_OPTIMIZER_POSE_GRAPH_OPTIMIZER_POSE_GRAPH_OPTIMIZER_H_
#define NONLINEAR_OPTIMIZER_POSE_GRAPH_OPTIMIZER_POSE_GRAPH_OPTIMIZER_H_

#include <memory>
#include <set>

#include "nonlinear_optimizer/loss_function.h"
#include "nonlinear_optimizer/options.h"
#include "nonlinear_optimizer/pose_graph_optimizer/types.h"
#include "nonlinear_optimizer/unordered_bimap.h"
#include "types.h"

namespace nonlinear_optimizer {
namespace pose_graph_optimizer {

struct PoseParameter {
  double position[3];
  double orientation[4];
};

class PoseGraphOptimizer {
 public:
  PoseGraphOptimizer() {}

  ~PoseGraphOptimizer() {}

  void SetLossFunction(const std::shared_ptr<LossFunction>& loss_function) {
    loss_function_ = loss_function;
  }

  void SetConstraint(const Constraint& constraint) {
    if (!index_to_pose_ptr_bimap_.IsKeyExist(constraint.query_pose_index) ||
        !index_to_pose_ptr_bimap_.IsKeyExist(constraint.reference_pose_index)) {
      std::cerr << "Constraint is invalid.\n";
      return;
    }
    constraints_.push_back(constraint);
  }

  void SetPose(const int pose_index, Pose* pose_ptr) {
    index_to_pose_ptr_bimap_.Insert(pose_index, pose_ptr);
    PoseParameter pose_parameter;
    pose_parameter.position[0] = pose_ptr->translation()(0);
    pose_parameter.position[1] = pose_ptr->translation()(1);
    pose_parameter.position[2] = pose_ptr->translation()(2);
    Orientation quaternion(pose_ptr->rotation());
    pose_parameter.orientation[0] = quaternion.w();
    pose_parameter.orientation[1] = quaternion.x();
    pose_parameter.orientation[2] = quaternion.y();
    pose_parameter.orientation[3] = quaternion.z();
    optimized_pose_map_.insert({pose_index, pose_parameter});
  }

  void SetPoseConstant(const int pose_index) {
    if (!index_to_pose_ptr_bimap_.IsKeyExist(pose_index)) {
      std::cerr << "Queried pose index is never registered into the solver.\n";
      return;
    }
    fixed_pose_index_set_.insert(pose_index);
  }

  /// @brief Solve the pose graph optimization problem. In case of success,
  /// registered poses are changed to the optimized poses. If not, the poses are
  /// not changed.
  /// @param options Optimization options
  /// @return Success or not
  virtual bool Solve(const Options& options) = 0;

 protected:
  Orientation ComputeQuaternion(const Vec3& w) {
    Orientation orientation{Orientation::Identity()};
    const double theta = w.norm();
    if (theta < 1e-6) {
      orientation.w() = 1.0;
      orientation.x() = 0.5 * w.x();
      orientation.y() = 0.5 * w.y();
      orientation.z() = 0.5 * w.z();
    } else {
      const double half_theta = theta * 0.5;
      const double sin_half_theta_divided_theta = std::sin(half_theta) / theta;
      orientation.w() = std::cos(half_theta);
      orientation.x() = sin_half_theta_divided_theta * w.x();
      orientation.y() = sin_half_theta_divided_theta * w.y();
      orientation.z() = sin_half_theta_divided_theta * w.z();
    }
    return orientation;
  }

  void UpdateOptimizedPose() {
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
    }
  }

  std::shared_ptr<LossFunction> loss_function_{nullptr};
  UnorderedBimap<int, Pose*> index_to_pose_ptr_bimap_;
  std::unordered_map<int, PoseParameter> optimized_pose_map_;
  std::set<int> fixed_pose_index_set_;
  std::vector<Constraint> constraints_;
};

}  // namespace pose_graph_optimizer
}  // namespace nonlinear_optimizer

#endif  // NONLINEAR_OPTIMIZER_POSE_GRAPH_OPTIMIZER_POSE_GRAPH_OPTIMIZER_H_
