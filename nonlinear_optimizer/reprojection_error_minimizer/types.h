#ifndef NONLINEAR_OPTIMIZER_REPROJECTION_ERROR_MINIMIZER_TYPES_H_
#define NONLINEAR_OPTIMIZER_REPROJECTION_ERROR_MINIMIZER_TYPES_H_

#include <unordered_map>

#include "nonlinear_optimizer/types.h"

namespace nonlinear_optimizer {
namespace reprojection_error_minimizer {

/// @brief Camera intrinsic parameters including focal lengths, image width, and
/// height. Assuming undistorted (and stereo-rectified) image with pinhole
/// model.
struct CameraIntrinsics {
  double fx{0.0};
  double fy{0.0};
  double cx{0.0};
  double cy{0.0};
  int width{0};
  int height{0};
};

struct Correspondence {
  Vec3 local_point{Vec3::Zero()};    // represented in reference frame
  Vec2 matched_pixel{Vec2::Zero()};  // represented in query frame
};

}  // namespace reprojection_error_minimizer
}  // namespace nonlinear_optimizer

#endif  // NONLINEAR_OPTIMIZER_REPROJECTION_ERROR_MINIMIZER_TYPES_H_
