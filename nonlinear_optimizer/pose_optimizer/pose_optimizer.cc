#include "pose_optimizer.h"

namespace pose_optimizer {

PoseOptimizer::PoseOptimizer() {}

PoseOptimizer::~PoseOptimizer() {}

bool PoseOptimizer::Solve(Pose* pose) {
  constexpr double kMaxLambda = 2.0;
  constexpr double kMinLambda = 0.1;

  if (pose == nullptr) return false;

  Pose optimized_pose = *pose;

  double lambda = 1e-3;

  Mat6x6 H_prev = Mat6x6::Zero();
  Vec6 g_prev = Vec6::Zero();
  double cost_prev = 0.0;

  bool is_previous_step_valid = true;

  double previous_cost = std::numeric_limits<double>::max();
  for (int iteration = 0; iteration < parameters_.max_iteration; ++iteration) {
    double cost = 0.0;
    Mat6x6 H{Mat6x6::Zero()};
    Vec6 g{Vec6::Zero()};

    if (is_previous_step_valid) {
      // Compute the residuals and Jacobians here.
      const auto& reproj_error_result =
          ComputeReprojectionError(optimized_pose, false);
      ApplyLocalResult(reproj_error_result, &H, &g, &cost);

      const auto& imu_preint_error_result =
          ComputeImuPreintegrationError(optimized_pose, false);
      ApplyLocalResult(imu_preint_error_result, &H, &g, &cost);
    } else {
      // Reuse the previous Hessian and gradient to save computation.
      H = H_prev;
      g = g_prev;
      cost = cost_prev;
    }

    // Compute update step
    Vec6 dx = -(H + lambda * H.diagonal()).ldlt().solve(g);

    // Compute new cost
    const Pose pose_after_update = optimized_pose;
    const auto& new_reproj_err_result =
        ComputeReprojectionError(pose_after_update, true);
    const auto& new_imu_preint_err_result =
        ComputeImuPreintegrationError(pose_after_update, true);

    const double cost_after_update =
        new_reproj_err_result.cost + new_imu_preint_err_result.cost;
    const double real_cost_change = cost - cost_after_update;

    // Check whether the optimization step decreases the cost function
    const double estimated_cost_change = -0.5 * dx.dot(lambda * dx - g);

    const double rho = real_cost_change / estimated_cost_change;
    if (rho > 0 && real_cost_change > 0 && std::isfinite(real_cost_change)) {
      // optimized_pose = optimized_pose + dx;
      previous_cost = cost + 0.5 * dx.dot(H * dx);
      lambda = std::max(lambda * 0.5, kMinLambda);
    } else {
      lambda = std::min(lambda * 2.0, kMaxLambda);
    }

    // Update the previous step information
    H_prev = H;
    g_prev = g;
    cost_prev = cost;
    is_previous_step_valid = true;
  }

  return true;
}

LocalResult PoseOptimizer::ComputeReprojectionError(const Pose& pose,
                                                    bool compute_jacobians) {
  LocalResult local_result;
  (void)pose;
  if (compute_jacobians) {
    // Compute Jacobians here.
  } else {
    // Only compute residuals here.
  }
  return local_result;
}

LocalResult PoseOptimizer::ComputeImuPreintegrationError(
    const Pose& pose, bool compute_jacobians) {
  LocalResult local_result;
  (void)pose;
  if (compute_jacobians) {
    // Compute Jacobians here.
  } else {
    // Only compute residuals here.
  }
  return local_result;
}

void PoseOptimizer::ApplyLocalResult(const LocalResult& local_result, Mat6x6* H,
                                     Vec6* g, double* cost) {
  if (H != nullptr) {
    *H += local_result.H;
  }
  if (g != nullptr) {
    *g += local_result.g;
  }
  if (cost != nullptr) {
    *cost += local_result.cost;
  }
}

}  // namespace pose_optimizer