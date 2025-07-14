#include "nonlinear_optimizer/mahalanobis_distance_minimizer/mahalanobis_distance_minimizer_analytic_3dof.h"

#include <iostream>

namespace nonlinear_optimizer {
namespace mahalanobis_distance_minimizer {

MahalanobisDistanceMinimizerAnalytic3DOF::
    MahalanobisDistanceMinimizerAnalytic3DOF() {}

MahalanobisDistanceMinimizerAnalytic3DOF::
    ~MahalanobisDistanceMinimizerAnalytic3DOF() {}

bool MahalanobisDistanceMinimizerAnalytic3DOF::Solve(
    const Options& options, const std::vector<Correspondence>& correspondences,
    Pose* pose) {
  constexpr double min_lambda = 1e-6;
  constexpr double max_lambda = 1e-2;

  const Pose initial_pose = *pose;

  Pose2 optimized_pose{Pose2::Identity()};
  optimized_pose.linear() = initial_pose.linear().block<2, 2>(0, 0);
  optimized_pose.translation() = initial_pose.translation().block<2, 1>(0, 0);

  double lambda = 0.001;
  double previous_cost = std::numeric_limits<double>::max();
  int iteration = 0;
  for (; iteration < options.max_iterations; ++iteration) {
    Mat3x3 hessian{Mat3x3::Zero()};
    Vec3 gradient{Vec3::Zero()};

    const size_t stride = 4;
    const int num_stride = correspondences.size() / stride;
    double cost = 0.0;
    for (size_t i = 0; i < stride * num_stride; ++i) {
      const auto& corr = correspondences.at(i);

      Mat3x3 jacobian{Mat3x3::Zero()};
      Vec3 residual{Vec3::Zero()};
      ComputeJacobianAndResidual(optimized_pose.linear(),
                                 optimized_pose.translation(), corr, &jacobian,
                                 &residual);

      // Compute the local gradient
      Vec3 local_gradient{Vec3::Zero()};
      local_gradient = jacobian.transpose() * residual;

      // Compute the local hessian
      Mat3x3 local_hessian{Mat3x3::Zero()};
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
    for (int k = 0; k < 3; k++) hessian(k, k) *= 1.0 + lambda;

    // Compute the step
    const Vec3 update_step = hessian.inverse() * (-gradient);

    // Update the pose
    const Vec2 delta_t = update_step.block<2, 1>(0, 0);
    const double delta_R = update_step(2);
    optimized_pose.translation() += delta_t;
    optimized_pose.rotate(delta_R);

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

  pose->translation().block<2, 1>(0, 0) = optimized_pose.translation();
  pose->linear().block<2, 2>(0, 0) = optimized_pose.rotation();

  return true;
}

void MahalanobisDistanceMinimizerAnalytic3DOF::ComputeJacobianAndResidual(
    const Mat2x2& rotation, const Vec2& translation, const Correspondence& corr,
    Mat3x3* jacobian, Vec3* residual) {
  const auto& R = rotation;
  const auto& t = translation;
  const auto& p = corr.point;
  const auto& sqrt_information = corr.ndt.sqrt_information;
  const Mat2x2& A = sqrt_information.block<2, 2>(0, 0);
  const Mat1x2& c = sqrt_information.block<1, 2>(2, 0);

  const Vec2& u = p.block<2, 1>(0, 0);
  const Vec2 u_warped = R * u + t;
  Vec3 p_warped{Vec3::Zero()};
  p_warped(0) = u_warped(0);
  p_warped(1) = u_warped(1);
  p_warped(2) = p(2);
  Vec3 e_i = p_warped - corr.ndt.mean;

  // Compute the residual
  *residual = sqrt_information * e_i;

  // Compute the Jacobian
  Vec2 R_skew_p{Vec2::Zero()};
  R_skew_p(0) = -R(0, 0) * u(1) + R(0, 1) * u(0);
  R_skew_p(1) = -R(1, 0) * u(1) + R(1, 1) * u(0);
  (*jacobian).block<2, 2>(0, 0) = A;
  (*jacobian).block<2, 1>(0, 2) = A * R_skew_p;
  (*jacobian).block<1, 2>(2, 0) = c;
  (*jacobian).block<1, 1>(2, 2) = c * R_skew_p;
}

void MahalanobisDistanceMinimizerAnalytic3DOF::ComputeHessianOnlyUpperTriangle(
    const Mat3x3& jacobian, Mat3x3* local_hessian) {
  auto& H = *local_hessian;
  auto& J = jacobian;
  H.setZero();
  for (int row = 0; row < 3; ++row) {
    for (int col = row; col < 3; ++col) {
      for (int k = 0; k < 3; ++k) {
        H(row, col) += J(k, row) * J(k, col);
      }
    }
  }
}

void MahalanobisDistanceMinimizerAnalytic3DOF::MultiplyWeightOnlyUpperTriangle(
    const double weight, Mat3x3* local_hessian) {
  for (int row = 0; row < 3; ++row) {
    for (int col = row; col < 3; ++col) {
      (*local_hessian)(row, col) *= weight;
    }
  }
}

void MahalanobisDistanceMinimizerAnalytic3DOF::AddHessianOnlyUpperTriangle(
    const Mat3x3& local_hessian, Mat3x3* global_hessian) {
  auto& H = *global_hessian;
  for (int row = 0; row < 3; ++row) {
    for (int col = row; col < 3; ++col) {
      H(row, col) += local_hessian(row, col);
    }
  }
}

void MahalanobisDistanceMinimizerAnalytic3DOF::ReflectHessian(Mat3x3* hessian) {
  auto& H = *hessian;
  for (int row = 0; row < 3; ++row) {
    for (int col = row + 1; col < 3; ++col) {
      H(col, row) = H(row, col);
    }
  }
}

}  // namespace mahalanobis_distance_minimizer
}  // namespace nonlinear_optimizer