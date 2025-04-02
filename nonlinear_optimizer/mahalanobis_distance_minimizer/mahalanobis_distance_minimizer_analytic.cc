#include "nonlinear_optimizer/mahalanobis_distance_minimizer/mahalanobis_distance_minimizer_analytic.h"

#include <iostream>

namespace nonlinear_optimizer {
namespace mahalanobis_distance_minimizer {

MahalanobisDistanceMinimizerAnalytic::MahalanobisDistanceMinimizerAnalytic() {}

MahalanobisDistanceMinimizerAnalytic::~MahalanobisDistanceMinimizerAnalytic() {}

bool MahalanobisDistanceMinimizerAnalytic::Solve(
    const std::vector<Correspondence>& correspondences, Pose* pose) {
  constexpr int max_iteration = 30;
  constexpr double min_lambda = 1e-6;
  constexpr double max_lambda = 1e-2;

  const Pose initial_pose = *pose;

  Vec3 optimized_translation{initial_pose.translation()};
  Orientation optimized_orientation(initial_pose.rotation());

  double lambda = 0.001;
  double previous_cost = std::numeric_limits<double>::max();
  int iteration = 0;
  for (; iteration < max_iteration; ++iteration) {
    Mat6x6 hessian{Mat6x6::Zero()};
    Vec6 gradient{Vec6::Zero()};

    const size_t stride = 4;
    const int num_stride = correspondences.size() / stride;
    double cost = 0.0;
    for (size_t i = 0; i < stride * num_stride; ++i) {
      const auto& corr = correspondences.at(i);

      Mat3x6 jacobian{Mat3x6::Zero()};
      Vec3 residual{Vec3::Zero()};
      ComputeJacobianAndResidual(optimized_orientation.toRotationMatrix(),
                                 optimized_translation, corr, &jacobian,
                                 &residual);

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
    if (update_step.norm() < 1e-7) {
      break;
    }
    if (gradient.norm() < 1e-7) {
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

void MahalanobisDistanceMinimizerAnalytic::ComputeJacobianAndResidual(
    const Mat3x3& rotation, const Vec3& translation, const Correspondence& corr,
    Mat3x6* jacobian, Vec3* residual) {
  const auto& R = rotation;
  const auto& t = translation;
  const auto& p = corr.point;
  const auto& sqrt_information = corr.ndt.sqrt_information;

  Vec3 p_warped = R * p + t;
  Vec3 e_i = p_warped - corr.ndt.mean;

  // Compute the residual
  *residual = sqrt_information * e_i;  // residual

  static auto skew = [](const Vec3& v) {
    Mat3x3 skew{Mat3x3::Zero()};
    skew << 0.0, -v.z(), v.y(),  //
        v.z(), 0.0, -v.x(),      //
        -v.y(), v.x(), 0.0;
    return skew;
  };

  // Compute the Jacobian
  Mat3x3 R_skew_p = R * skew(p);
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
  (*jacobian).block<3, 3>(0, 0) = sqrt_information;
  (*jacobian).block<3, 3>(0, 3) = -sqrt_information * R_skew_p;
}

void MahalanobisDistanceMinimizerAnalytic::ComputeHessianOnlyUpperTriangle(
    const Mat3x6& jacobian, Mat6x6* local_hessian) {
  auto& H = *local_hessian;
  auto& J = jacobian;
  H.setZero();
  for (int row = 0; row < 6; ++row) {
    for (int col = row; col < 6; ++col) {
      for (int k = 0; k < 3; ++k) {
        H(row, col) += J(k, row) * J(k, col);
      }
    }
  }
}

void MahalanobisDistanceMinimizerAnalytic::MultiplyWeightOnlyUpperTriangle(
    const double weight, Mat6x6* local_hessian) {
  for (int row = 0; row < 6; ++row) {
    for (int col = row; col < 6; ++col) {
      (*local_hessian)(row, col) *= weight;
    }
  }
}

void MahalanobisDistanceMinimizerAnalytic::AddHessianOnlyUpperTriangle(
    const Mat6x6& local_hessian, Mat6x6* global_hessian) {
  auto& H = *global_hessian;
  for (int row = 0; row < 6; ++row) {
    for (int col = row; col < 6; ++col) {
      H(row, col) += local_hessian(row, col);
    }
  }
}

void MahalanobisDistanceMinimizerAnalytic::ReflectHessian(Mat6x6* hessian) {
  auto& H = *hessian;
  for (int row = 0; row < 6; ++row) {
    for (int col = row + 1; col < 6; ++col) {
      H(col, row) = H(row, col);
    }
  }
}

Orientation MahalanobisDistanceMinimizerAnalytic::ComputeQuaternion(
    const Vec3& w) {
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

}  // namespace mahalanobis_distance_minimizer
}  // namespace nonlinear_optimizer