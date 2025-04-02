#ifndef NONLINEAR_OPTIMIZER_MAHALANOBIS_DISTANCE_MINIMIZER_MAHALANOBIS_DISTANCE_MINIMIZER_ANALYTIC_SIMD_H_
#define NONLINEAR_OPTIMIZER_MAHALANOBIS_DISTANCE_MINIMIZER_MAHALANOBIS_DISTANCE_MINIMIZER_ANALYTIC_SIMD_H_

#include <vector>

#include "nonlinear_optimizer/mahalanobis_distance_minimizer/mahalanobis_distance_minimizer.h"
#include "nonlinear_optimizer/mahalanobis_distance_minimizer/types.h"
#include "nonlinear_optimizer/types.h"

namespace nonlinear_optimizer {
namespace mahalanobis_distance_minimizer {

class MahalanobisDistanceMinimizerAnalyticSIMD
    : public MahalanobisDistanceMinimizer {
 public:
  MahalanobisDistanceMinimizerAnalyticSIMD();

  ~MahalanobisDistanceMinimizerAnalyticSIMD();

  bool Solve(const std::vector<Correspondence>& correspondences,
             Pose* pose) final;
  bool SolveFloat(const std::vector<Correspondence>& correspondences,
                  Pose* pose);
  bool SolveUsingHelper(const std::vector<Correspondence>& correspondences,
                        Pose* pose);
  bool SolveUsingHelperFloat(const std::vector<Correspondence>& correspondences,
                             Pose* pose);

 private:
  void ReflectHessian(Mat6x6* hessian);
  Orientation ComputeQuaternion(const Vec3& w);
};

}  // namespace mahalanobis_distance_minimizer
}  // namespace nonlinear_optimizer

#endif  // NONLINEAR_OPTIMIZER_MAHALANOBIS_DISTANCE_MINIMIZER_MAHALANOBIS_DISTANCE_MINIMIZER_ANALYTIC_SIMD_H_