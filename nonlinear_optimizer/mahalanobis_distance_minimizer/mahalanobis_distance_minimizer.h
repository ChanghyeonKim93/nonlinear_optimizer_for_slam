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
  MahalanobisDistanceMinimizer();

  ~MahalanobisDistanceMinimizer();

  void SetMultiThreadExecutor(
      const std::shared_ptr<MultiThreadExecutor>& multi_thread_executor);

  void SetLossFunction(const std::shared_ptr<LossFunction>& loss_function);

  virtual bool Solve(const Options& options,
                     const std::vector<Correspondence>& correspondences,
                     Pose* pose) = 0;

 protected:
  Orientation ComputeQuaternion(const Vec3& w);

  std::optional<Eigen::Vector3d> translation_prior_constraint_{std::nullopt};
  std::optional<Eigen::Quaterniond> rotation_prior_constraint_{std::nullopt};
  std::shared_ptr<LossFunction> loss_function_{nullptr};
  std::shared_ptr<MultiThreadExecutor> multi_thread_executor_{nullptr};
};

}  // namespace mahalanobis_distance_minimizer
}  // namespace nonlinear_optimizer

#endif  // NONLINEAR_OPTIMIZER_MAHALANOBIS_DISTANCE_MINIMIZER_MAHALANOBIS_DISTANCE_MINIMIZER_H_