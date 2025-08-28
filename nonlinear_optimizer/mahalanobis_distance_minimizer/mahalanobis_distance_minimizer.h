#ifndef NONLINEAR_OPTIMIZER_MAHALANOBIS_DISTANCE_MINIMIZER_MAHALANOBIS_DISTANCE_MINIMIZER_H_
#define NONLINEAR_OPTIMIZER_MAHALANOBIS_DISTANCE_MINIMIZER_MAHALANOBIS_DISTANCE_MINIMIZER_H_

#include <memory>

#include "nonlinear_optimizer/loss_function.h"
#include "nonlinear_optimizer/multi_thread_executor.h"
#include "nonlinear_optimizer/options.h"
#include "types.h"

namespace nonlinear_optimizer {
namespace mahalanobis_distance_minimizer {

struct PartialResult {
  Vec6 gradient{Vec6::Zero()};
  Mat6x6 hessian{Mat6x6::Zero()};
  double cost{0.0};
};
class MahalanobisDistanceMinimizer {
 public:
  MahalanobisDistanceMinimizer() {}

  ~MahalanobisDistanceMinimizer() {}

  void SetMultiThreadExecutor(
      const std::shared_ptr<MultiThreadExecutor>& multi_thread_executor) {
    multi_thread_executor_ = multi_thread_executor;
  }

  void SetLossFunction(const std::shared_ptr<LossFunction>& loss_function) {
    loss_function_ = loss_function;
  }

  virtual bool Solve(const Options& options,
                     const std::vector<Correspondence>& correspondences,
                     Pose* pose) = 0;

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

  std::shared_ptr<LossFunction> loss_function_{nullptr};

  std::shared_ptr<MultiThreadExecutor> multi_thread_executor_{nullptr};
};

}  // namespace mahalanobis_distance_minimizer
}  // namespace nonlinear_optimizer

#endif  // NONLINEAR_OPTIMIZER_MAHALANOBIS_DISTANCE_MINIMIZER_MAHALANOBIS_DISTANCE_MINIMIZER_H_