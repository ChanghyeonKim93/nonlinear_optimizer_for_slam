#include "nonlinear_optimizer/mahalanobis_distance_minimizer/mahalanobis_distance_minimizer.h"

namespace nonlinear_optimizer {
namespace mahalanobis_distance_minimizer {

MahalanobisDistanceMinimizer::MahalanobisDistanceMinimizer() {}

MahalanobisDistanceMinimizer::~MahalanobisDistanceMinimizer() {}

void MahalanobisDistanceMinimizer::SetMultiThreadExecutor(
    const std::shared_ptr<MultiThreadExecutor>& multi_thread_executor) {
  multi_thread_executor_ = multi_thread_executor;
}

void MahalanobisDistanceMinimizer::SetLossFunction(
    const std::shared_ptr<LossFunction>& loss_function) {
  loss_function_ = loss_function;
}

Orientation MahalanobisDistanceMinimizer::ComputeQuaternion(const Vec3& w) {
  Orientation orientation{Orientation::Identity()};
  const double theta = w.norm();
  if (theta < 1e-6) {
    orientation.w() = 1.0;
    orientation.vec() = 0.5 * w;
  } else {
    const double half_theta = theta * 0.5;
    const double sin_half_theta_divided_theta = std::sin(half_theta) / theta;
    orientation.w() = std::cos(half_theta);
    orientation.vec() = sin_half_theta_divided_theta * w;
  }
  return orientation;
}

}  // namespace mahalanobis_distance_minimizer
}  // namespace nonlinear_optimizer