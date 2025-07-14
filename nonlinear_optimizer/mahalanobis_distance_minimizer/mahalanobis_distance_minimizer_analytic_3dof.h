#ifndef NONLINEAR_OPTIMIZER_MAHALANOBIS_DISTANCE_MINIMIZER_MAHALANOBIS_DISTANCE_MINIMIZER_ANALYTIC_3DOF_H_
#define NONLINEAR_OPTIMIZER_MAHALANOBIS_DISTANCE_MINIMIZER_MAHALANOBIS_DISTANCE_MINIMIZER_ANALYTIC_3DOF_H_

#include <vector>

#include "nonlinear_optimizer/mahalanobis_distance_minimizer/mahalanobis_distance_minimizer.h"
#include "nonlinear_optimizer/mahalanobis_distance_minimizer/types.h"
#include "nonlinear_optimizer/types.h"

namespace nonlinear_optimizer {
namespace mahalanobis_distance_minimizer {

class MahalanobisDistanceMinimizerAnalytic3DOF
    : public MahalanobisDistanceMinimizer {
 public:
  MahalanobisDistanceMinimizerAnalytic3DOF();

  ~MahalanobisDistanceMinimizerAnalytic3DOF();

  bool Solve(const Options& options,
             const std::vector<Correspondence>& correspondences,
             Pose* pose) final;

 private:
  void ComputeJacobianAndResidual(const Mat2x2& rotation,
                                  const Vec2& translation,
                                  const Correspondence& corr, Mat3x3* jacobian,
                                  Vec3* residual);
  void ComputeHessianOnlyUpperTriangle(const Mat3x3& jacobian,
                                       Mat3x3* local_hessian);
  void MultiplyWeightOnlyUpperTriangle(const double weight,
                                       Mat3x3* local_hessian);
  void AddHessianOnlyUpperTriangle(const Mat3x3& local_hessian,
                                   Mat3x3* global_hessian);
  void ReflectHessian(Mat3x3* hessian);
};

}  // namespace mahalanobis_distance_minimizer
}  // namespace nonlinear_optimizer

#endif  // NONLINEAR_OPTIMIZER_MAHALANOBIS_DISTANCE_MINIMIZER_MAHALANOBIS_DISTANCE_MINIMIZER_ANALYTIC_3DOF_H_