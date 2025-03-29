#ifndef NONLINEAR_OPTIMIZER_MAHALANOBIS_DISTANCE_MINIMIZER_TYPES_H_
#define NONLINEAR_OPTIMIZER_MAHALANOBIS_DISTANCE_MINIMIZER_TYPES_H_

#include <unordered_map>

#include "nonlinear_optimizer/types.h"

namespace nonlinear_optimizer {
namespace mahalanobis_distance_minimizer {

struct NDT {
  int count{0};
  Vec3 sum{Vec3::Zero()};
  Mat3x3 moment{Mat3x3::Identity()};

  Vec3 mean{Vec3::Zero()};
  Mat3x3 information{Mat3x3::Identity()};
  Mat3x3 sqrt_information{Mat3x3::Identity()};
  bool is_valid{false};
  bool is_planar{false};
};

struct Correspondence {
  Vec3 point{Vec3::Zero()};
  NDT ndt;
};

}  // namespace mahalanobis_distance_minimizer
}  // namespace nonlinear_optimizer

#endif  // NONLINEAR_OPTIMIZER_MAHALANOBIS_DISTANCE_MINIMIZER_TYPES_H_
