#ifndef NONLINEAR_OPTIMIZER_REPROJECTION_ERROR_MINIMIZER_REPROJECTION_ERROR_MINIMIZER_ANALYTIC_H_
#define NONLINEAR_OPTIMIZER_REPROJECTION_ERROR_MINIMIZER_REPROJECTION_ERROR_MINIMIZER_ANALYTIC_H_

#include <vector>

#include "nonlinear_optimizer/reprojection_error_minimizer/reprojection_error_minimizer.h"
#include "nonlinear_optimizer/reprojection_error_minimizer/types.h"
#include "nonlinear_optimizer/types.h"

namespace nonlinear_optimizer {
namespace reprojection_error_minimizer {

class ReprojectionErrorMinimizerAnalytic : public ReprojectionErrorMinimizer {
 public:
  ReprojectionErrorMinimizerAnalytic();

  ~ReprojectionErrorMinimizerAnalytic();

  bool Solve(const Options& options,
             const std::vector<Correspondence>& correspondences,
             const CameraIntrinsics& camera_intrinsics, Pose* pose) final;

 private:
  void ComputeJacobianAndResidual(const Mat3x3& rotation,
                                  const Vec3& translation,
                                  const Correspondence& corr,
                                  const CameraIntrinsics& camera_intrinsics,
                                  Mat2x6* jacobian, Vec2* residual);
  void ComputeHessianOnlyUpperTriangle(const Mat2x6& jacobian,
                                       Mat6x6* local_hessian);
  void MultiplyWeightOnlyUpperTriangle(const double weight,
                                       Mat6x6* local_hessian);
  void AddHessianOnlyUpperTriangle(const Mat6x6& local_hessian,
                                   Mat6x6* global_hessian);
  void ReflectHessian(Mat6x6* hessian);
};

}  // namespace reprojection_error_minimizer
}  // namespace nonlinear_optimizer

#endif  // NONLINEAR_OPTIMIZER_REPROJECTION_ERROR_MINIMIZER_REPROJECTION_ERROR_MINIMIZER_ANALYTIC_H_