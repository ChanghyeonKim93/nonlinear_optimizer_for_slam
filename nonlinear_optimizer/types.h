#ifndef NONLINEAR_OPTIMIZER_TYPES_H_
#define NONLINEAR_OPTIMIZER_TYPES_H_

#include "Eigen/Dense"

namespace nonlinear_optimizer {

using Vec2 = Eigen::Matrix<double, 2, 1>;
using Vec3 = Eigen::Matrix<double, 3, 1>;
using Vec4 = Eigen::Matrix<double, 4, 1>;
using Vec5 = Eigen::Matrix<double, 5, 1>;
using Vec6 = Eigen::Matrix<double, 6, 1>;

using Mat1x2 = Eigen::Matrix<double, 1, 2>;
using Mat2x1 = Eigen::Matrix<double, 2, 1>;

using Mat2x2 = Eigen::Matrix<double, 2, 2>;
using Mat2x3 = Eigen::Matrix<double, 2, 3>;
using Mat3x2 = Eigen::Matrix<double, 3, 2>;
using Mat3x3 = Eigen::Matrix<double, 3, 3>;
using Mat4x4 = Eigen::Matrix<double, 4, 4>;
using Mat5x5 = Eigen::Matrix<double, 5, 5>;
using Mat6x6 = Eigen::Matrix<double, 6, 6>;

using Mat1x6 = Eigen::Matrix<double, 1, 6>;
using Mat2x6 = Eigen::Matrix<double, 2, 6>;
using Mat3x6 = Eigen::Matrix<double, 3, 6>;
using Mat6x3 = Eigen::Matrix<double, 6, 3>;
using Mat6x2 = Eigen::Matrix<double, 6, 2>;
using Mat6x1 = Eigen::Matrix<double, 6, 1>;

using Orientation = Eigen::Quaterniond;
using Pose2 = Eigen::Isometry2d;
using Pose = Eigen::Isometry3d;

using Vec2f = Eigen::Matrix<float, 2, 1>;
using Vec3f = Eigen::Matrix<float, 3, 1>;
using Vec4f = Eigen::Matrix<float, 4, 1>;
using Vec5f = Eigen::Matrix<float, 5, 1>;
using Vec6f = Eigen::Matrix<float, 6, 1>;

using Mat2x2f = Eigen::Matrix<float, 2, 2>;
using Mat2x3f = Eigen::Matrix<float, 2, 3>;
using Mat3x2f = Eigen::Matrix<float, 3, 2>;
using Mat3x3f = Eigen::Matrix<float, 3, 3>;
using Mat4x4f = Eigen::Matrix<float, 4, 4>;
using Mat5x5f = Eigen::Matrix<float, 5, 5>;
using Mat6x6f = Eigen::Matrix<float, 6, 6>;

using Mat1x6f = Eigen::Matrix<float, 1, 6>;
using Mat2x6f = Eigen::Matrix<float, 2, 6>;
using Mat3x6f = Eigen::Matrix<float, 3, 6>;
using Mat6x3f = Eigen::Matrix<float, 6, 3>;
using Mat6x2f = Eigen::Matrix<float, 6, 2>;
using Mat6x1f = Eigen::Matrix<float, 6, 1>;

using Orientationf = Eigen::Quaternionf;
using Posef = Eigen::Isometry3f;

}  // namespace nonlinear_optimizer

#endif  // NONLINEAR_OPTIMIZER_TYPES_H_