#ifndef NONLINEAR_OPTIMIZER_REPROJECTION_ERROR_MINIMIZER_REPROJECTION_ERROR_MINIMIZER_ANALYTIC_SIMD_H_
#define NONLINEAR_OPTIMIZER_REPROJECTION_ERROR_MINIMIZER_REPROJECTION_ERROR_MINIMIZER_ANALYTIC_SIMD_H_

#include <vector>

#include "nonlinear_optimizer/reprojection_error_minimizer/reprojection_error_minimizer.h"
#include "nonlinear_optimizer/reprojection_error_minimizer/types.h"
#include "nonlinear_optimizer/simd_helper.h"
#include "nonlinear_optimizer/types.h"

namespace nonlinear_optimizer {
namespace reprojection_error_minimizer {

struct AlignedBuffer {
  float* x = {nullptr};
  float* y = {nullptr};
  float* z = {nullptr};
  float* px = {nullptr};
  float* py = {nullptr};
  AlignedBuffer(const size_t num_data) {
    x = simd::GetAlignedMemory<float>(num_data);
    y = simd::GetAlignedMemory<float>(num_data);
    z = simd::GetAlignedMemory<float>(num_data);
    px = simd::GetAlignedMemory<float>(num_data);
    py = simd::GetAlignedMemory<float>(num_data);
  }
  ~AlignedBuffer() {
    simd::FreeAlignedMemory<float>(x);
    simd::FreeAlignedMemory<float>(y);
    simd::FreeAlignedMemory<float>(z);
    simd::FreeAlignedMemory<float>(px);
    simd::FreeAlignedMemory<float>(py);
  }
};

class ReprojectionErrorMinimizerAnalyticSIMD
    : public ReprojectionErrorMinimizer {
 public:
  ReprojectionErrorMinimizerAnalyticSIMD();

  ~ReprojectionErrorMinimizerAnalyticSIMD();

  bool Solve(const Options& options,
             const std::vector<Correspondence>& correspondences,
             const CameraIntrinsics& camera_intrinsics, Pose* pose) final;

 private:
  void ReflectHessian(Mat6x6* hessian);
};

}  // namespace reprojection_error_minimizer
}  // namespace nonlinear_optimizer

#endif  // NONLINEAR_OPTIMIZER_REPROJECTION_ERROR_MINIMIZER_REPROJECTION_ERROR_MINIMIZER_ANALYTIC_SIMD_H_