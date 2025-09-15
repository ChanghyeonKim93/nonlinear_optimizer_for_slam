#ifndef POSE_OPTIMIZER_TYPES_H_
#define POSE_OPTIMIZER_TYPES_H_

#include <vector>

#include "Eigen/Dense"

namespace pose_optimizer {

using Mat3x6 = Eigen::Matrix<double, 3, 6>;
using Mat3x3 = Eigen::Matrix<double, 3, 3>;
using Mat6x6 = Eigen::Matrix<double, 6, 6>;
using Vec2 = Eigen::Vector2d;
using Vec3 = Eigen::Vector3d;
using Vec6 = Eigen::Matrix<double, 6, 1>;
using Pose = Eigen::Isometry3d;

}  // namespace pose_optimizer

#endif  // POSE_OPTIMIZER_TYPES_H_
