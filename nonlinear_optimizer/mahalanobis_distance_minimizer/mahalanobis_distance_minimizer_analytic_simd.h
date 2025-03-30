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
  void AllocateSIMDBuffer();
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

  // For SIMD
  // double* x__ = nullptr;
  // double* y__ = nullptr;
  // double* z__ = nullptr;
  // double* mx__ = nullptr;
  // double* my__ = nullptr;
  // double* mz__ = nullptr;
  // double* x_warp__ = nullptr;
  // double* y_warp__ = nullptr;
  // double* z_warp__ = nullptr;
  // double* sqrt_info_xx__ = nullptr;
  // double* sqrt_info_xy__ = nullptr;
  // double* sqrt_info_xz__ = nullptr;
  // double* sqrt_info_yy__ = nullptr;
  // double* sqrt_info_yz__ = nullptr;
  // double* sqrt_info_zz__ = nullptr;
  // double* ex__ = nullptr;
  // double* ey__ = nullptr;
  // double* ez__ = nullptr;
  // double* hessian__ = nullptr;   // 15 elements for upper diagonal
  // double* gradient__ = nullptr;  // 6 elements
};

}  // namespace mahalanobis_distance_minimizer
}  // namespace nonlinear_optimizer

#endif  // NONLINEAR_OPTIMIZER_MAHALANOBIS_DISTANCE_MINIMIZER_MAHALANOBIS_DISTANCE_MINIMIZER_ANALYTIC_SIMD_H_