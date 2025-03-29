#ifndef NONLINEAR_OPTIMIZER_MAHALANOBIS_DISTANCE_MINIMIZER_MAHALANOBIS_DISTANCE_MINIMIZER_H_
#define NONLINEAR_OPTIMIZER_MAHALANOBIS_DISTANCE_MINIMIZER_MAHALANOBIS_DISTANCE_MINIMIZER_H_

#include <memory>

#include "nonlinear_optimizer/loss_function.h"
#include "types.h"

namespace nonlinear_optimizer {
namespace mahalanobis_distance_minimizer {

class MahalanobisDistanceMinimizer {
 public:
  MahalanobisDistanceMinimizer() {}

  ~MahalanobisDistanceMinimizer() {}

  void SetLossFunction(const std::shared_ptr<LossFunction>& loss_function) {
    loss_function_ = loss_function;
  }

  virtual bool Solve(const std::vector<Correspondence>& correspondences,
                     Pose* pose) = 0;

 protected:
  std::shared_ptr<LossFunction> loss_function_{nullptr};
};

}  // namespace mahalanobis_distance_minimizer
}  // namespace nonlinear_optimizer

#endif  // NONLINEAR_OPTIMIZER_MAHALANOBIS_DISTANCE_MINIMIZER_MAHALANOBIS_DISTANCE_MINIMIZER_H_