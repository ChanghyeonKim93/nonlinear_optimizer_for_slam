#ifndef NONLINEAR_OPTIMIZER_MAHALANOBIS_DISTANCE_MINIMIZER_MAHALANOBIS_DISTANCE_MINIMIZER_ANALYTIC_SIMD_H_
#define NONLINEAR_OPTIMIZER_MAHALANOBIS_DISTANCE_MINIMIZER_MAHALANOBIS_DISTANCE_MINIMIZER_ANALYTIC_SIMD_H_

#include <vector>

#include "nonlinear_optimizer/mahalanobis_distance_minimizer/mahalanobis_distance_minimizer.h"
#include "nonlinear_optimizer/mahalanobis_distance_minimizer/types.h"
#include "nonlinear_optimizer/simd_helper.h"
#include "nonlinear_optimizer/types.h"

namespace nonlinear_optimizer {
namespace mahalanobis_distance_minimizer {

struct AlignedBuffer {
  float* sqrt_info[3][3] = {nullptr};
  float* x = {nullptr};
  float* y = {nullptr};
  float* z = {nullptr};
  float* mx = {nullptr};
  float* my = {nullptr};
  float* mz = {nullptr};
  AlignedBuffer(const size_t num_data) {
    for (int row = 0; row < 3; ++row)
      for (int col = 0; col < 3; ++col)
        sqrt_info[row][col] = simd::GetAlignedMemory<float>(num_data);
    x = simd::GetAlignedMemory<float>(num_data);
    y = simd::GetAlignedMemory<float>(num_data);
    z = simd::GetAlignedMemory<float>(num_data);
    mx = simd::GetAlignedMemory<float>(num_data);
    my = simd::GetAlignedMemory<float>(num_data);
    mz = simd::GetAlignedMemory<float>(num_data);
  }
  ~AlignedBuffer() {
    for (int row = 0; row < 3; ++row)
      for (int col = 0; col < 3; ++col)
        simd::FreeAlignedMemory<float>(sqrt_info[row][col]);
    simd::FreeAlignedMemory<float>(x);
    simd::FreeAlignedMemory<float>(y);
    simd::FreeAlignedMemory<float>(z);
    simd::FreeAlignedMemory<float>(mx);
    simd::FreeAlignedMemory<float>(my);
    simd::FreeAlignedMemory<float>(mz);
  }
};

class MahalanobisDistanceMinimizerAnalyticSIMD
    : public MahalanobisDistanceMinimizer {
 public:
  MahalanobisDistanceMinimizerAnalyticSIMD();

  ~MahalanobisDistanceMinimizerAnalyticSIMD();

  bool Solve(const std::vector<Correspondence>& correspondences,
             Pose* pose) final;
  bool SolveFloat(const std::vector<Correspondence>& correspondences,
                  Pose* pose);
  bool SolveFloat_FAST1(const std::vector<Correspondence>& correspondences,
                        Pose* pose);
  bool SolveFloat_FAST2(const std::vector<Correspondence>& correspondences,
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