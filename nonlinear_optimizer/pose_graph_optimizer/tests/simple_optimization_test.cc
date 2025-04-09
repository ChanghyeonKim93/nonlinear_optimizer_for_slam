#include <iostream>
#include <random>

#include "Eigen/Dense"

#include "ceres/ceres.h"

#include "nonlinear_optimizer/pose_graph_optimizer/ceres_cost_functor.h"
#include "nonlinear_optimizer/pose_graph_optimizer/pose_graph_optimizer.h"
#include "nonlinear_optimizer/pose_graph_optimizer/pose_graph_optimizer_ceres.h"
#include "nonlinear_optimizer/time_checker.h"
#include "nonlinear_optimizer/types.h"

nonlinear_optimizer::Options options;

namespace nonlinear_optimizer {
namespace pose_graph_optimizer {

std::vector<Pose> GenerateTruePoses() {
  std::vector<Pose> poses(80, Pose::Identity());

  constexpr double kPositionStep{0.2};
  double x = 0.0, y = 0.0, z = 0.0;
  for (int index = 0; index < 20; ++index) {
    poses.at(index).translation().x() = x;
    poses.at(index).translation().y() = y;
    poses.at(index).translation().z() = z;
    x += kPositionStep;
    z += kPositionStep;
  }

  for (int index = 20; index < 40; ++index) {
    y += kPositionStep;
    z += kPositionStep;
    poses.at(index).translation().x() = x;
    poses.at(index).translation().y() = y;
    poses.at(index).translation().z() = z;
  }

  for (int index = 40; index < 60; ++index) {
    x -= kPositionStep;
    z -= kPositionStep;
    poses.at(index).translation().x() = x;
    poses.at(index).translation().y() = y;
    poses.at(index).translation().z() = z;
  }

  for (int index = 60; index < 80; ++index) {
    y -= kPositionStep;
    z -= kPositionStep;
    poses.at(index).translation().x() = x;
    poses.at(index).translation().y() = y;
    poses.at(index).translation().z() = z;
  }

  return poses;
}

std::vector<Pose> ApplyNoiseOnPoses(const std::vector<Pose>& poses) {
  constexpr double kPoseNoise{0.08};
  std::vector<Pose> noisy_poses;
  noisy_poses.push_back(Pose::Identity());
  for (size_t index = 1; index < poses.size(); ++index) {
    Pose noisy_pose = poses.at(index);
    const int k = index % 3;
    noisy_pose.translation()(k) += (index % 2 ? 1 : -1) * kPoseNoise;
    noisy_poses.push_back(noisy_pose);
  }
  return noisy_poses;
}

}  // namespace pose_graph_optimizer
}  // namespace nonlinear_optimizer

using namespace nonlinear_optimizer;
using namespace nonlinear_optimizer::pose_graph_optimizer;

int main(int, char**) {
  const auto true_poses = GenerateTruePoses();
  auto poses = ApplyNoiseOnPoses(true_poses);

  std::vector<std::pair<int, int>> odometry_pairs;
  for (int index = 0; index < 79; ++index)
    odometry_pairs.push_back({index, index + 1});

  std::vector<std::pair<int, int>> loop_pairs{
      {18, 21}, {38, 42}, {59, 61}, {77, 1}};

  std::cerr << "# odometry pairs: " << odometry_pairs.size() << std::endl;
  std::cerr << "# loop pairs: " << loop_pairs.size() << std::endl;

  // Make constraints
  std::vector<Constraint> constraints;
  for (const auto& [i0, i1] : odometry_pairs) {
    const auto& reference_pose = true_poses.at(i0);
    const auto& query_pose = true_poses.at(i1);
    const auto relative_pose_from_reference_to_query =
        reference_pose.inverse() * query_pose;
    Constraint constraint;
    constraint.reference_pose_index = i0;
    constraint.query_pose_index = i1;
    constraint.relative_pose_from_reference_to_query =
        relative_pose_from_reference_to_query;
    constraint.type = ConstraintType::kOdometry;
    constraints.push_back(constraint);
  }
  for (const auto& [i0, i1] : loop_pairs) {
    const auto& reference_pose = true_poses.at(i0);
    const auto& query_pose = true_poses.at(i1);
    const auto relative_pose_from_reference_to_query =
        reference_pose.inverse() * query_pose;
    Constraint constraint;
    constraint.reference_pose_index = i0;
    constraint.query_pose_index = i1;
    constraint.relative_pose_from_reference_to_query =
        relative_pose_from_reference_to_query;
    constraint.type = ConstraintType::kLoop;
    constraints.push_back(constraint);
  }

  std::unique_ptr<PoseGraphOptimizer> optimizer =
      std::make_unique<PoseGraphOptimizerCeres>();
  for (size_t index = 0; index < poses.size(); ++index)
    optimizer->SetPose(index, &poses.at(index));

  optimizer->SetPoseConstant(0);

  for (const auto& constraint : constraints)
    optimizer->SetConstraint(constraint);

  Options options;
  options.convergence_handle.function_tolerance = 1e-12;
  options.convergence_handle.gradient_tolerance = 1e-12;
  options.convergence_handle.parameter_tolerance = 1e-12;
  const bool success = optimizer->Solve(options);
  std::cerr << "Optimization is succeeded? : " << (success ? "yes" : "no")
            << std::endl;

  // Check results
  for (size_t index = 0; index < poses.size(); ++index) {
    const auto& true_pose = true_poses.at(index);
    const auto& est_pose = poses.at(index);
    const Pose pose_diff = true_pose.inverse() * est_pose;

    std::cerr << index << "-th pose diff:\n"
              << pose_diff.translation().transpose() << "\n"
              << pose_diff.linear() << std::endl;
  }

  return 0;
}