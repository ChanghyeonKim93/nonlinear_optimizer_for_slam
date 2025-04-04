#ifndef NONLINEAR_OPTIMIZER_REPROJECTION_ERROR_MINIMIZER_REPROJECTION_ERROR_MINIMIZER_CERES_H_
#define NONLINEAR_OPTIMIZER_REPROJECTION_ERROR_MINIMIZER_REPROJECTION_ERROR_MINIMIZER_CERES_H_

#include "nonlinear_optimizer/reprojection_error_minimizer/reprojection_error_minimizer.h"
#include "nonlinear_optimizer/reprojection_error_minimizer/types.h"

namespace nonlinear_optimizer {
namespace reprojection_error_minimizer {

class ReprojectionErrorMinimizerCeres : public ReprojectionErrorMinimizer {
 public:
  ReprojectionErrorMinimizerCeres();

  ~ReprojectionErrorMinimizerCeres();

  /// @brief Solve the reprojection error minimization problem w.r.t. pose
  /// @param options Optimization options
  /// @param correspondences Feature correspondences
  /// @param pose Pose from query frame to reference frame (`T_qr`)
  /// @return
  bool Solve(const Options& options,
             const std::vector<Correspondence>& correspondences,
             const CameraIntrinsics& camera_intrinsics, Pose* pose) final;
};

}  // namespace reprojection_error_minimizer
}  // namespace nonlinear_optimizer

#endif  // NONLINEAR_OPTIMIZER_REPROJECTION_ERROR_MINIMIZER_REPROJECTION_ERROR_MINIMIZER_CERES_H_
