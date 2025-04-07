#ifndef NONLINEAR_OPTIMIZER_REPROJECTION_ERROR_MINIMIZER_REPROJECTION_ERROR_MINIMIZER_H_
#define NONLINEAR_OPTIMIZER_REPROJECTION_ERROR_MINIMIZER_REPROJECTION_ERROR_MINIMIZER_H_

#include <memory>

#include "nonlinear_optimizer/loss_function.h"
#include "nonlinear_optimizer/options.h"
#include "nonlinear_optimizer/reprojection_error_minimizer/types.h"
#include "types.h"

namespace nonlinear_optimizer {
namespace reprojection_error_minimizer {

class ReprojectionErrorMinimizer {
 public:
  ReprojectionErrorMinimizer() {}

  ~ReprojectionErrorMinimizer() {}

  void SetLossFunction(const std::shared_ptr<LossFunction>& loss_function) {
    loss_function_ = loss_function;
  }

  /// @brief Solve the reprojection error minimization problem w.r.t. pose
  /// @param options Optimization options
  /// @param correspondences Feature correspondences
  /// @param camera_intrinsics Camera intrinsics
  /// @param pose Pose from query frame to reference frame (`T_qr`)
  /// @return Success or not
  virtual bool Solve(const Options& options,
                     const std::vector<Correspondence>& correspondences,
                     const CameraIntrinsics& camera_intrinsics, Pose* pose) = 0;

 protected:
  Orientation ComputeQuaternion(const Vec3& w) {
    Orientation orientation{Orientation::Identity()};
    const double theta = w.norm();
    if (theta < 1e-6) {
      orientation.w() = 1.0;
      orientation.x() = 0.5 * w.x();
      orientation.y() = 0.5 * w.y();
      orientation.z() = 0.5 * w.z();
    } else {
      const double half_theta = theta * 0.5;
      const double sin_half_theta_divided_theta = std::sin(half_theta) / theta;
      orientation.w() = std::cos(half_theta);
      orientation.x() = sin_half_theta_divided_theta * w.x();
      orientation.y() = sin_half_theta_divided_theta * w.y();
      orientation.z() = sin_half_theta_divided_theta * w.z();
    }
    return orientation;
  }

  std::shared_ptr<LossFunction> loss_function_{nullptr};
};

}  // namespace reprojection_error_minimizer
}  // namespace nonlinear_optimizer

#endif  // NONLINEAR_OPTIMIZER_REPROJECTION_ERROR_MINIMIZER_REPROJECTION_ERROR_MINIMIZER_H_
