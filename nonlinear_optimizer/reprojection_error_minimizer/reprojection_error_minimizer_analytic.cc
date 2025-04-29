#include "nonlinear_optimizer/reprojection_error_minimizer/reprojection_error_minimizer_analytic.h"

#include <iostream>

namespace nonlinear_optimizer {
namespace reprojection_error_minimizer {

ReprojectionErrorMinimizerAnalytic::ReprojectionErrorMinimizerAnalytic() {}

ReprojectionErrorMinimizerAnalytic::~ReprojectionErrorMinimizerAnalytic() {}

bool ReprojectionErrorMinimizerAnalytic::Solve(
    const Options& options, const std::vector<Correspondence>& correspondences,
    const CameraIntrinsics& camera_intrinsics, Pose* pose) {
  constexpr double min_lambda = 1e-6;
  constexpr double max_lambda = 1e-2;

  const Pose initial_pose = *pose;

  Vec3 optimized_translation{initial_pose.translation()};
  Orientation optimized_orientation(initial_pose.rotation());

  double lambda = 0.001;
  double previous_cost = std::numeric_limits<double>::max();
  int iteration = 0;
  for (; iteration < options.max_iterations; ++iteration) {
    Mat6x6 hessian{Mat6x6::Zero()};
    Vec6 gradient{Vec6::Zero()};

    double cost = 0.0;
    for (size_t i = 0; i < correspondences.size(); ++i) {
      const auto& corr = correspondences.at(i);

      Mat2x6 jacobian{Mat2x6::Zero()};
      Vec2 residual{Vec2::Zero()};
      ComputeJacobianAndResidual(optimized_orientation.toRotationMatrix(),
                                 optimized_translation, corr, camera_intrinsics,
                                 &jacobian, &residual);

      // Compute the local gradient
      Vec6 local_gradient{Vec6::Zero()};
      local_gradient = jacobian.transpose() * residual;

      // Compute the local hessian
      Mat6x6 local_hessian{Mat6x6::Zero()};
      ComputeHessianOnlyUpperTriangle(jacobian, &local_hessian);

      // Compute loss and weight,
      // and add the local gradient and hessian to the global ones
      const double squared_residual = residual.transpose() * residual;
      if (loss_function_ != nullptr) {
        double loss_output[3] = {0.0, 0.0, 0.0};
        loss_function_->Evaluate(squared_residual, loss_output);
        const double weight = loss_output[1];
        gradient += weight * local_gradient;
        MultiplyWeightOnlyUpperTriangle(weight, &local_hessian);
        AddHessianOnlyUpperTriangle(local_hessian, &hessian);
        cost += loss_output[0];
      } else {
        gradient += local_gradient;
        AddHessianOnlyUpperTriangle(local_hessian, &hessian);
        cost += squared_residual;
      }
    }
    // Reflect the hessian
    ReflectHessian(&hessian);

    // Damping hessian
    for (int k = 0; k < 6; k++) hessian(k, k) *= 1.0 + lambda;

    // Compute the step
    // const Vec6 update_step = hessian.ldlt().solve(-gradient);
    const Vec6 update_step = hessian.inverse() * (-gradient);

    // Update the pose
    const Vec3 delta_t = update_step.block<3, 1>(0, 0);
    const Vec3 delta_R = update_step.block<3, 1>(3, 0);
    optimized_translation += delta_t;
    optimized_orientation *= ComputeQuaternion(delta_R);
    optimized_orientation.normalize();

    // Check convergence
    if (update_step.norm() < options.convergence_handle.parameter_tolerance) {
      break;
    }
    if (gradient.norm() < options.convergence_handle.gradient_tolerance) {
      break;
    }

    if (cost > previous_cost) {
      lambda *= 2.0;
    } else {
      lambda *= 0.6;
    }
    lambda = std::clamp(lambda, min_lambda, max_lambda);
    previous_cost = cost;
  }
  std::cerr << "COST: " << previous_cost << ", iter: " << iteration
            << std::endl;

  pose->translation() = optimized_translation;
  pose->linear() = optimized_orientation.toRotationMatrix();

  return true;
}

void ReprojectionErrorMinimizerAnalytic::ComputeJacobianAndResidual(
    const Mat3x3& rotation, const Vec3& translation, const Correspondence& corr,
    const CameraIntrinsics& camera_intrinsics, Mat2x6* jacobian,
    Vec2* residual) {
  constexpr double kMinDepth = 0.03;

  const auto& R = rotation;
  const auto& t = translation;
  const auto& X = corr.local_point;
  const auto& p = corr.matched_pixel;

  Vec3 Xw = R * X + t;
  if (Xw(2) < kMinDepth) {
    jacobian->setZero();
    residual->setZero();
    return;
  }
  const double inverse_zw = 1.0 / Xw(2);
  Vec2 projected_image_coordinate(Xw(0) * inverse_zw, Xw(1) * inverse_zw);
  Vec2 matched_image_coordinate(
      camera_intrinsics.inv_fx * (p(0) - camera_intrinsics.cx),
      camera_intrinsics.inv_fy * (p(1) - camera_intrinsics.cy));

  // Compute the residual
  *residual = projected_image_coordinate - matched_image_coordinate;

  static auto skew = [](const Vec3& v) {
    Mat3x3 skew{Mat3x3::Zero()};
    skew << 0.0, -v.z(), v.y(),  //
        v.z(), 0.0, -v.x(),      //
        -v.y(), v.x(), 0.0;
    return skew;
  };

  const double squared_inverse_zw = inverse_zw * inverse_zw;
  Mat2x3 dK_dXw{Mat2x3::Zero()};
  dK_dXw(0, 0) = inverse_zw;
  dK_dXw(0, 2) = -Xw(0) * squared_inverse_zw;
  dK_dXw(1, 1) = inverse_zw;
  dK_dXw(1, 2) = -Xw(1) * squared_inverse_zw;

  // Compute the Jacobian
  Mat3x3 R_skew_p = R * skew(X);
  //   Mat3x3 R_skew_p{Mat3x3::Zero()};
  //   R_skew_p(0, 0) = R(0, 1) * p(2) - R(0, 2) * p(1);
  //   R_skew_p(0, 1) = R(0, 2) * p(0) - R(0, 0) * p(2);
  //   R_skew_p(0, 2) = R(0, 0) * p(1) - R(0, 1) * p(0);
  //   R_skew_p(1, 0) = R(1, 1) * p(2) - R(1, 2) * p(1);
  //   R_skew_p(1, 1) = R(1, 2) * p(0) - R(1, 0) * p(2);
  //   R_skew_p(1, 2) = R(1, 0) * p(1) - R(1, 1) * p(0);
  //   R_skew_p(2, 0) = R(2, 1) * p(2) - R(2, 2) * p(1);
  //   R_skew_p(2, 1) = R(2, 2) * p(0) - R(2, 0) * p(2);
  //   R_skew_p(2, 2) = R(2, 0) * p(1) - R(2, 1) * p(0);
  (*jacobian).block<2, 3>(0, 0) = dK_dXw;
  (*jacobian).block<2, 3>(0, 3) = -dK_dXw * R_skew_p;
}

void ReprojectionErrorMinimizerAnalytic::ComputeHessianOnlyUpperTriangle(
    const Mat2x6& jacobian, Mat6x6* local_hessian) {
  auto& H = *local_hessian;
  auto& J = jacobian;
  H.setZero();
  for (int row = 0; row < 6; ++row)
    for (int col = row; col < 6; ++col)
      H(row, col) += J(0, row) * J(0, col) + J(1, row) * J(1, col);
}

void ReprojectionErrorMinimizerAnalytic::MultiplyWeightOnlyUpperTriangle(
    const double weight, Mat6x6* local_hessian) {
  for (int row = 0; row < 6; ++row)
    for (int col = row; col < 6; ++col) (*local_hessian)(row, col) *= weight;
}

void ReprojectionErrorMinimizerAnalytic::AddHessianOnlyUpperTriangle(
    const Mat6x6& local_hessian, Mat6x6* global_hessian) {
  auto& H = *global_hessian;
  for (int row = 0; row < 6; ++row)
    for (int col = row; col < 6; ++col) H(row, col) += local_hessian(row, col);
}

void ReprojectionErrorMinimizerAnalytic::ReflectHessian(Mat6x6* hessian) {
  auto& H = *hessian;
  for (int row = 0; row < 6; ++row)
    for (int col = row + 1; col < 6; ++col) H(col, row) = H(row, col);
}

}  // namespace reprojection_error_minimizer
}  // namespace nonlinear_optimizer