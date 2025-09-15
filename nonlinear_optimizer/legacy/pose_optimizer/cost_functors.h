#ifndef NONLINEAR_OPTIMIZER_POSE_OPTIMIZER_COST_FUNCTORS_H_
#define NONLINEAR_OPTIMIZER_POSE_OPTIMIZER_COST_FUNCTORS_H_

#include "cost_function.h"
#include "types.h"

namespace nonlinear_optimizer {
namespace pose_optimizer {

class ExponentialLossFunction : public LossFunction {
 public:
  ExponentialLossFunction(const double c1, const double c2)
      : c1_{c1}, c2_{c2}, two_c1c2_{2.0 * c1_ * c2_} {
    if (c1_ < 0.0) throw std::out_of_range("`c1_` should be positive number.");
    if (c2_ < 0.0) throw std::out_of_range("`c2_` should be positive number.");
  }

  void Evaluate(const double& squared_residual, double output[3]) final {
    const double exp_term = exp(-c2_ * squared_residual);
    output[0] = c1_ - c1_ * exp_term;
    output[1] = two_c1c2_ * exp_term;
    output[2] = -2.0 * c2_ * output[1];
  }

 private:
  double c1_{0.0};
  double c2_{0.0};
  double two_c1c2_{0.0};
};

class HuberLossFunction : public LossFunction {
 public:
  HuberLossFunction(const double threshold)
      : threshold_{threshold}, squared_threshold_{threshold_ * threshold_} {
    if (threshold_ <= 0.0)
      throw std::out_of_range("threshold value should be larger than zero.");
  }

  void Evaluate(const double& squared_residual, double output[3]) final {
    if (squared_residual > squared_threshold_) {
      const double residual = std::sqrt(squared_residual);
      output[0] = 2.0 * threshold_ * residual - squared_threshold_;
      output[1] = threshold_ / residual;
    } else {
      output[0] = squared_residual;
      output[1] = 1.0;
    }
  }

 private:
  double threshold_{0.0};
  const double squared_threshold_;
};

class MahalanobisDistanceCostFunctor final : public SizedCostFunction<3, 3, 3> {
 public:
  MahalanobisDistanceCostFunctor(const Correspondence& correspondence)
      : correspondence_(correspondence) {}

  bool Evaluate(const RotationMatrix& rotation_matrix,
                const TranslationVector& translation_vector,
                double* jacobian_matrix_data_ptr,
                double* residual_vector_data_ptr) final {
    if (jacobian_matrix_data_ptr == nullptr ||
        residual_vector_data_ptr == nullptr)
      return false;

    Eigen::Map<Eigen::Matrix<double, 3, 6>> jacobian_matrix(
        jacobian_matrix_data_ptr);
    Eigen::Map<Eigen::Matrix<double, 3, 1>> residual_vector(
        residual_vector_data_ptr);
    jacobian_matrix.setZero();
    residual_vector.setZero();

    const auto& p = correspondence_.point;
    const auto& mean = correspondence_.mean;
    const auto& sqrt_information = correspondence_.sqrt_information;
    const auto& R = rotation_matrix;
    const auto& t = translation_vector;

    const Vec3 warped_p = R * p + t;
    const Vec3 e = warped_p - mean;
    residual_vector = sqrt_information * e;

    // == Mat3x3 R_skew_p = R * skew(p);
    Eigen::Matrix3d R_skew_p{Eigen::Matrix3d::Zero()};
    R_skew_p(0, 0) = R(0, 1) * p(2) - R(0, 2) * p(1);
    R_skew_p(0, 1) = R(0, 2) * p(0) - R(0, 0) * p(2);
    R_skew_p(0, 2) = R(0, 0) * p(1) - R(0, 1) * p(0);
    R_skew_p(1, 0) = R(1, 1) * p(2) - R(1, 2) * p(1);
    R_skew_p(1, 1) = R(1, 2) * p(0) - R(1, 0) * p(2);
    R_skew_p(1, 2) = R(1, 0) * p(1) - R(1, 1) * p(0);
    R_skew_p(2, 0) = R(2, 1) * p(2) - R(2, 2) * p(1);
    R_skew_p(2, 1) = R(2, 2) * p(0) - R(2, 0) * p(2);
    R_skew_p(2, 2) = R(2, 0) * p(1) - R(2, 1) * p(0);
    jacobian_matrix.block<3, 3>(0, 0) = sqrt_information;
    jacobian_matrix.block<3, 3>(0, 3) = -sqrt_information * R_skew_p;

    return true;
  }

 private:
  const Correspondence correspondence_;
};

class PointToPlaneCostFunctor final : public SizedCostFunction<1, 3, 3> {
 public:
  PointToPlaneCostFunctor(const Correspondence& correspondence)
      : correspondence_(correspondence) {}

  bool Evaluate(const RotationMatrix& rotation_matrix,
                const TranslationVector& translation_vector,
                double* jacobian_matrix_data_ptr,
                double* residual_vector_data_ptr) const {
    if (jacobian_matrix_data_ptr == nullptr ||
        residual_vector_data_ptr == nullptr)
      return false;

    Eigen::Map<Eigen::Matrix<double, 1, 6>> jacobian_matrix(
        jacobian_matrix_data_ptr);
    jacobian_matrix.setZero();

    const auto& p = correspondence_.point;
    const auto& mean = correspondence_.mean;
    const auto& plane_normal_vector = correspondence_.plane_normal_vector;
    const auto& R = rotation_matrix;
    const auto& t = translation_vector;

    const Vec3 warped_p = R * p + t;
    const Vec3 e = warped_p - mean;
    const double r = plane_normal_vector.dot(e);

    residual_vector_data_ptr[0] = r;

    // == Mat3x3 R_skew_p = R * skew(p);
    Eigen::Matrix3d R_skew_p{Eigen::Matrix3d::Zero()};
    R_skew_p(0, 0) = R(0, 1) * p(2) - R(0, 2) * p(1);
    R_skew_p(0, 1) = R(0, 2) * p(0) - R(0, 0) * p(2);
    R_skew_p(0, 2) = R(0, 0) * p(1) - R(0, 1) * p(0);
    R_skew_p(1, 0) = R(1, 1) * p(2) - R(1, 2) * p(1);
    R_skew_p(1, 1) = R(1, 2) * p(0) - R(1, 0) * p(2);
    R_skew_p(1, 2) = R(1, 0) * p(1) - R(1, 1) * p(0);
    R_skew_p(2, 0) = R(2, 1) * p(2) - R(2, 2) * p(1);
    R_skew_p(2, 1) = R(2, 2) * p(0) - R(2, 0) * p(2);
    R_skew_p(2, 2) = R(2, 0) * p(1) - R(2, 1) * p(0);
    const Eigen::Matrix<double, 1, 3> transposed_plane_normal_vector =
        plane_normal_vector.transpose();
    jacobian_matrix.block<1, 3>(0, 0) = transposed_plane_normal_vector;
    jacobian_matrix.block<1, 3>(0, 3) =
        -transposed_plane_normal_vector * R_skew_p;

    return true;
  }

 private:
  const Correspondence correspondence_;
};

class TranslationDeltaCostFunctor final : public SizedCostFunction<3, 3, 3> {
 public:
  TranslationDeltaCostFunctor(const Eigen::Vector3d& translation_prior)
      : translation_prior_(translation_prior) {}

  bool Evaluate(const RotationMatrix& rotation_matrix,
                const TranslationVector& translation_vector,
                double* jacobian_matrix_data_ptr,
                double* residual_vector_data_ptr) const {
    if (jacobian_matrix_data_ptr == nullptr ||
        residual_vector_data_ptr == nullptr)
      return false;

    Eigen::Map<Eigen::Matrix<double, 3, 6>> jacobian_matrix(
        jacobian_matrix_data_ptr);
    Eigen::Map<Eigen::Matrix<double, 3, 1>> residual_vector(
        residual_vector_data_ptr);
    jacobian_matrix.setZero();
    residual_vector.setZero();

    const auto& t = translation_vector;
    residual_vector = translation_vector - translation_prior_;

    jacobian_matrix.block<3, 3>(0, 0).setIdentity();
    jacobian_matrix.block<3, 3>(0, 3).setZero();

    return true;
  }

 private:
  const Eigen::Vector3d translation_prior_;
};

class RotationDeltaCostFunctor final : public SizedCostFunction<3, 3, 3> {
 public:
  RotationDeltaCostFunctor(const Eigen::Quaterniond& rotation_prior)
      : rotation_prior_(rotation_prior) {}

  bool Evaluate(const RotationMatrix& rotation_matrix,
                const TranslationVector& translation_vector,
                double* jacobian_matrix_data_ptr,
                double* residual_vector_data_ptr) const {
    if (jacobian_matrix_data_ptr == nullptr ||
        residual_vector_data_ptr == nullptr)
      return false;

    Eigen::Map<Eigen::Matrix<double, 3, 6>> jacobian_matrix(
        jacobian_matrix_data_ptr);
    Eigen::Map<Eigen::Matrix<double, 3, 1>> residual_vector(
        residual_vector_data_ptr);
    jacobian_matrix.setZero();
    residual_vector.setZero();

    const auto& R = rotation_matrix;
    // residual_vector = translation_vector - translation_prior_;

    // jacobian_matrix.block<3, 3>(0, 0).setIdentity();
    // jacobian_matrix.block<3, 3>(0, 3).setZero();

    return true;
  }

 private:
  const Eigen::Quaterniond rotation_prior_;
};

}  // namespace pose_optimizer
}  // namespace nonlinear_optimizer

#endif  // NONLINEAR_OPTIMIZER_POSE_OPTIMIZER_COST_FUNCTORS_H_