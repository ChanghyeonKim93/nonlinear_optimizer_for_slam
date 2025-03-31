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

 private:
  void ComputeJacobianAndResidual(const Mat3x3& rotation,
                                  const Vec3& translation,
                                  const Correspondence& corr, Mat3x6* jacobian,
                                  Vec3* residual);
  void ComputeHessianOnlyUpperTriangle(const Mat3x6& jacobian,
                                       Mat6x6* local_hessian);
  void MultiplyWeightOnlyUpperTriangle(const double weight,
                                       Mat6x6* local_hessian);
  void AddHessianOnlyUpperTriangle(const Mat6x6& local_hessian,
                                   Mat6x6* global_hessian);
  void ReflectHessian(Mat6x6* hessian);
  Orientation ComputeQuaternion(const Vec3& w);
};

}  // namespace mahalanobis_distance_minimizer
}  // namespace nonlinear_optimizer

#endif  // NONLINEAR_OPTIMIZER_MAHALANOBIS_DISTANCE_MINIMIZER_MAHALANOBIS_DISTANCE_MINIMIZER_ANALYTIC_SIMD_H_