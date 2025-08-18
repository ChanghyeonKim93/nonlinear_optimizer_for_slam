#ifndef NONLINEAR_OPTIMIZER_MAHALANOBIS_DISTANCE_MINIMIZER_MAHALANOBIS_DISTANCE_MINIMIZER_ANALYTIC_3DOF_SIMD_H_
#define NONLINEAR_OPTIMIZER_MAHALANOBIS_DISTANCE_MINIMIZER_MAHALANOBIS_DISTANCE_MINIMIZER_ANALYTIC_3DOF_SIMD_H_

#include <vector>

#include "nonlinear_optimizer/mahalanobis_distance_minimizer/mahalanobis_distance_minimizer.h"
#include "nonlinear_optimizer/mahalanobis_distance_minimizer/types.h"
#include "nonlinear_optimizer/types.h"

#include "simd_helper/simd_helper.h"

namespace nonlinear_optimizer {
namespace mahalanobis_distance_minimizer {

class MahalanobisDistanceMinimizerAnalytic3DOFSIMD
    : public MahalanobisDistanceMinimizer {
 public:
  MahalanobisDistanceMinimizerAnalytic3DOFSIMD();

  ~MahalanobisDistanceMinimizerAnalytic3DOFSIMD();

  bool Solve(const Options& options,
             const std::vector<Correspondence>& correspondences,
             Pose* pose) final;

 private:
  void ReflectHessian(Mat3x3* hessian);
};

}  // namespace mahalanobis_distance_minimizer
}  // namespace nonlinear_optimizer

#endif  // NONLINEAR_OPTIMIZER_MAHALANOBIS_DISTANCE_MINIMIZER_MAHALANOBIS_DISTANCE_MINIMIZER_ANALYTIC_3DOF_H_