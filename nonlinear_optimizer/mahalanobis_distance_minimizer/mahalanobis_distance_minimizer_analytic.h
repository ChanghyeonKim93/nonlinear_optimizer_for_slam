#ifndef NONLINEAR_OPTIMIZER_MAHALANOBIS_DISTANCE_MINIMIZER_MAHALANOBIS_DISTANCE_MINIMIZER_ANALYTIC_H_
#define NONLINEAR_OPTIMIZER_MAHALANOBIS_DISTANCE_MINIMIZER_MAHALANOBIS_DISTANCE_MINIMIZER_ANALYTIC_H_

#include <vector>

#include "nonlinear_optimizer/mahalanobis_distance_minimizer/mahalanobis_distance_minimizer.h"
#include "nonlinear_optimizer/mahalanobis_distance_minimizer/types.h"
#include "nonlinear_optimizer/types.h"

namespace nonlinear_optimizer {
namespace mahalanobis_distance_minimizer {

class MahalanobisDistanceMinimizerAnalytic
    : public MahalanobisDistanceMinimizer {
 public:
  MahalanobisDistanceMinimizerAnalytic();

  ~MahalanobisDistanceMinimizerAnalytic();

  bool Solve(const Options& options,
             const std::vector<Correspondence>& correspondences,
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
};

}  // namespace mahalanobis_distance_minimizer
}  // namespace nonlinear_optimizer

#endif  // OPTIMIZER_MAHALANOBIS_DISTANCE_MINIMIZER_MAHALANOBIS_DISTANCE_MINIMIZER_ANALYTIC_H_