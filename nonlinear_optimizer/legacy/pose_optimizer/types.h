#ifndef NONLINEAR_OPTIMIZER_POSE_OPTIMIZER_TYPES_H_
#define NONLINEAR_OPTIMIZER_POSE_OPTIMIZER_TYPES_H_

#include "Eigen/Dense"

namespace nonlinear_optimizer {
namespace pose_optimizer {

using Vec3 = Eigen::Vector3d;
using Mat3x3 = Eigen::Matrix3d;

struct Correspondence {
  Eigen::Vector3d point{Eigen::Vector3d::Zero()};
  Eigen::Vector3d mean{Eigen::Vector3d::Zero()};
  Eigen::Matrix3d sqrt_information{Eigen::Matrix3d::Zero()};
  Eigen::Vector3d plane_normal_vector{Eigen::Vector3d::Zero()};
  bool is_planar;
};

}  // namespace pose_optimizer
}  // namespace nonlinear_optimizer

#endif  // NONLINEAR_OPTIMIZER_POSE_OPTIMIZER_TYPES_H_