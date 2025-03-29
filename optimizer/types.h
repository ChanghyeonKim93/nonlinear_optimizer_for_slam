#ifndef OPTIMIZER_TYPES_H_
#define OPTIMIZER_TYPES_H_

#include "Eigen/Dense"

namespace optimizer {

using Vec3 = Eigen::Matrix<double, 3, 1>;
using Vec4 = Eigen::Matrix<double, 4, 1>;
using Vec5 = Eigen::Matrix<double, 5, 1>;
using Vec6 = Eigen::Matrix<double, 6, 1>;

using Mat2x2 = Eigen::Matrix<double, 2, 2>;
using Mat3x3 = Eigen::Matrix<double, 3, 3>;
using Mat4x4 = Eigen::Matrix<double, 4, 4>;
using Mat5x5 = Eigen::Matrix<double, 5, 5>;
using Mat6x6 = Eigen::Matrix<double, 6, 6>;

}  // namespace optimizer

#endif  // OPTIMIZER_TYPES_H_