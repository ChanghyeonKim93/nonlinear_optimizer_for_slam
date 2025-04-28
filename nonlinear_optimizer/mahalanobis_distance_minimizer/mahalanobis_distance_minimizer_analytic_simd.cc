#include "nonlinear_optimizer/mahalanobis_distance_minimizer/mahalanobis_distance_minimizer_analytic_simd.h"

#include <iostream>

#include "nonlinear_optimizer/simd_helper/simd_helper.h"

namespace nonlinear_optimizer {
namespace mahalanobis_distance_minimizer {

MahalanobisDistanceMinimizerAnalyticSIMD::
    MahalanobisDistanceMinimizerAnalyticSIMD() {}

MahalanobisDistanceMinimizerAnalyticSIMD::
    ~MahalanobisDistanceMinimizerAnalyticSIMD() {}

bool MahalanobisDistanceMinimizerAnalyticSIMD::SolveDoubleMatrix(
    const Options& options, const std::vector<Correspondence>& correspondences,
    Pose* pose) {
  constexpr double min_lambda = 1e-6;
  constexpr double max_lambda = 1e-2;

  const Pose initial_pose = *pose;

  Vec3 optimized_translation{initial_pose.translation()};
  Orientation optimized_orientation(initial_pose.rotation());

  double lambda = 0.001;
  double previous_cost = std::numeric_limits<double>::max();
  int iteration = 0;
  for (; iteration < options.max_iterations; ++iteration) {
    const Mat3x3 opt_R = optimized_orientation.toRotationMatrix();
    const Vec3 opt_t = optimized_translation;
    simd::Matrix<3, 3> R__(opt_R);
    simd::Vector<3> t__(opt_t);

    simd::Vector<6> gradient__(Eigen::Matrix<double, 6, 1>::Zero());
    simd::Matrix<6, 6> hessian__(Eigen::Matrix<double, 6, 6>::Zero());
    simd::Scalar cost__(0.0);
    const size_t stride = simd::Scalar::GetDataStep();
    const int num_stride = correspondences.size() / stride;
    for (size_t point_idx = 0; point_idx < num_stride * stride;
         point_idx += stride) {
      simd::Vector<3> p__;
      simd::Vector<3> mu__;
      simd::Matrix<3, 3> sqrt_info__;

      const auto& corr1 = correspondences.at(point_idx);
      const auto& corr2 = correspondences.at(point_idx + 1);
      const auto& corr3 = correspondences.at(point_idx + 2);
      const auto& corr4 = correspondences.at(point_idx + 3);
      p__ =
          simd::Vector<3>({corr1.point, corr2.point, corr3.point, corr4.point});
      mu__ = simd::Vector<3>(
          {corr1.ndt.mean, corr2.ndt.mean, corr3.ndt.mean, corr4.ndt.mean});
      sqrt_info__ = simd::Matrix<3, 3>(
          {corr1.ndt.sqrt_information, corr2.ndt.sqrt_information,
           corr3.ndt.sqrt_information, corr4.ndt.sqrt_information});

      // clang-format off
      // pw = R*p + t
      const simd::Vector<3> pw__ = R__ * p__ + t__;

      // e_i = pw - mean
      const simd::Vector<3> e__ = pw__ - mu__;

      // r = sqrt_info * e
      const simd::Vector<3> r__ = sqrt_info__ * e__;

      simd::Matrix<3,3> minus_R_skewp__;
      minus_R_skewp__(0, 0) =  R__(0, 2) * p__(1) - R__(0, 1) * p__(2);
      minus_R_skewp__(1, 0) =  R__(1, 2) * p__(1) - R__(1, 1) * p__(2);
      minus_R_skewp__(2, 0) =  R__(2, 2) * p__(1) - R__(2, 1) * p__(2);
      minus_R_skewp__(0, 1) =  R__(0, 0) * p__(2) - R__(0, 2) * p__(0);
      minus_R_skewp__(1, 1) =  R__(1, 0) * p__(2) - R__(1, 2) * p__(0);
      minus_R_skewp__(2, 1) =  R__(2, 0) * p__(2) - R__(2, 2) * p__(0);
      minus_R_skewp__(0, 2) =  R__(0, 1) * p__(0) - R__(0, 0) * p__(1);
      minus_R_skewp__(1, 2) =  R__(1, 1) * p__(0) - R__(1, 0) * p__(1);
      minus_R_skewp__(2, 2) =  R__(2, 1) * p__(0) - R__(2, 0) * p__(1);
      const simd::Matrix<3,3> sqrt_info_minus_R_skewp_ = sqrt_info__ * minus_R_skewp__;

      // Direct calculation is far faster than helper operator.
      simd::Matrix<3,6> J__;
      // J__(0,3) = sqrt_info__(0,0) * minus_R_skewp__(0,0) + sqrt_info__(0,1) * minus_R_skewp__(1,0) + sqrt_info__(0,2) * minus_R_skewp__(2,0);
      // J__(0,4) = sqrt_info__(0,0) * minus_R_skewp__(0,1) + sqrt_info__(0,1) * minus_R_skewp__(1,1) + sqrt_info__(0,2) * minus_R_skewp__(2,1);
      // J__(0,5) = sqrt_info__(0,0) * minus_R_skewp__(0,2) + sqrt_info__(0,1) * minus_R_skewp__(1,2) + sqrt_info__(0,2) * minus_R_skewp__(2,2);
      // J__(1,3) = sqrt_info__(1,0) * minus_R_skewp__(0,0) + sqrt_info__(1,1) * minus_R_skewp__(1,0) + sqrt_info__(1,2) * minus_R_skewp__(2,0);
      // J__(1,4) = sqrt_info__(1,0) * minus_R_skewp__(0,1) + sqrt_info__(1,1) * minus_R_skewp__(1,1) + sqrt_info__(1,2) * minus_R_skewp__(2,1);
      // J__(1,5) = sqrt_info__(1,0) * minus_R_skewp__(0,2) + sqrt_info__(1,1) * minus_R_skewp__(1,2) + sqrt_info__(1,2) * minus_R_skewp__(2,2);
      // J__(2,3) = sqrt_info__(2,0) * minus_R_skewp__(0,0) + sqrt_info__(2,1) * minus_R_skewp__(1,0) + sqrt_info__(2,2) * minus_R_skewp__(2,0);
      // J__(2,4) = sqrt_info__(2,0) * minus_R_skewp__(0,1) + sqrt_info__(2,1) * minus_R_skewp__(1,1) + sqrt_info__(2,2) * minus_R_skewp__(2,1);
      // J__(2,5) = sqrt_info__(2,0) * minus_R_skewp__(0,2) + sqrt_info__(2,1) * minus_R_skewp__(1,2) + sqrt_info__(2,2) * minus_R_skewp__(2,2);
      J__(0, 0) = sqrt_info__(0, 0); J__(0, 1) = sqrt_info__(0, 1); J__(0, 2) = sqrt_info__(0, 2);
      J__(1, 0) = sqrt_info__(1, 0); J__(1, 1) = sqrt_info__(1, 1); J__(1, 2) = sqrt_info__(1, 2);
      J__(2, 0) = sqrt_info__(2, 0); J__(2, 1) = sqrt_info__(2, 1); J__(2, 2) = sqrt_info__(2, 2);
      J__(0, 3) = sqrt_info_minus_R_skewp_(0, 0); J__(0, 4) = sqrt_info_minus_R_skewp_(0, 1); J__(0, 5) = sqrt_info_minus_R_skewp_(0, 2);
      J__(1, 3) = sqrt_info_minus_R_skewp_(1, 0); J__(1, 4) = sqrt_info_minus_R_skewp_(1, 1); J__(1, 5) = sqrt_info_minus_R_skewp_(1, 2);
      J__(2, 3) = sqrt_info_minus_R_skewp_(2, 0); J__(2, 4) = sqrt_info_minus_R_skewp_(2, 1); J__(2, 5) = sqrt_info_minus_R_skewp_(2, 2);
      // clang-format on

      // Compute loss and weight,
      // and add the local gradient and hessian to the global ones
      simd::Scalar sq_r__ = r__.GetNorm();
      simd::Scalar loss__(sq_r__);
      simd::Scalar weight__(1.0);
      if (loss_function_ != nullptr) {
        double sq_r_buf[4];
        sq_r__.StoreData(sq_r_buf);
        double loss_buf[4];
        double weight_buf[4];
        for (int k = 0; k < 4; ++k) {
          double loss_output[3] = {0.0, 0.0, 0.0};
          loss_function_->Evaluate(sq_r_buf[k], loss_output);
          loss_buf[k] = loss_output[0];
          weight_buf[k] = loss_output[1];
        }
        loss__ = simd::Scalar(loss_buf);
        weight__ = simd::Scalar(weight_buf);
      }

      // g(i) += (J(0,i)*r(0) + J(1,i)*r(1) + J(2,i)*r(2))
      gradient__ += (J__.transpose() * r__) * weight__;

      // H(i,j) = sum_{k} w * J(k,i) * J(k,j)
      for (int ii = 0; ii < 6; ++ii) {
        for (int jj = ii; jj < 6; ++jj) {
          hessian__(ii, jj) +=
              (weight__ * (J__(0, ii) * J__(0, jj) + J__(1, ii) * J__(1, jj) +
                           J__(2, ii) * J__(2, jj)));
        }
      }

      cost__ += loss__;
    }
    double buf[4];
    cost__.StoreData(buf);
    double cost = 0.0;
    cost += (buf[0] + buf[1] + buf[2] + buf[3]);

    Mat6x6 hessian{Mat6x6::Zero()};
    Vec6 gradient{Vec6::Zero()};
    for (int ii = 0; ii < 6; ++ii) {
      gradient__(ii).StoreData(buf);
      gradient(ii) += (buf[0] + buf[1] + buf[2] + buf[3]);
      for (int jj = ii; jj < 6; ++jj) {
        hessian__(ii, jj).StoreData(buf);
        hessian(ii, jj) += (buf[0] + buf[1] + buf[2] + buf[3]);
      }
    }

    // Reflect the hessian
    ReflectHessian(&hessian);

    // Damping hessian
    for (int k = 0; k < 6; k++) hessian(k, k) *= 1.0 + lambda;

    // Compute the step
    const Vec6 update_step = hessian.ldlt().solve(-gradient);

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

bool MahalanobisDistanceMinimizerAnalyticSIMD::SolveFloatMatrix(
    const Options& options, const std::vector<Correspondence>& correspondences,
    Pose* pose) {
  using Vec3f = Eigen::Vector3f;
  using Mat3x3f = Eigen::Matrix3f;
  using Orientationf = Eigen::Quaternionf;

  constexpr float min_lambda = 1e-6;
  constexpr float max_lambda = 1e-2;

  const Pose initial_pose = *pose;

  Vec3f optimized_translation{initial_pose.translation().cast<float>()};
  Orientationf optimized_orientation(initial_pose.rotation().cast<float>());

  float lambda = 0.001f;
  float previous_cost = std::numeric_limits<float>::max();
  int iteration = 0;
  for (; iteration < options.max_iterations; ++iteration) {
    const Mat3x3f opt_R = optimized_orientation.toRotationMatrix();
    const Vec3f opt_t = optimized_translation;
    simd::MatrixF<3, 3> R__(opt_R);
    simd::VectorF<3> t__(opt_t);

    simd::VectorF<6> gradient__(Eigen::Matrix<float, 6, 1>::Zero());
    simd::MatrixF<6, 6> hessian__(Eigen::Matrix<float, 6, 6>::Zero());
    simd::ScalarF cost__(0.0);
    const size_t stride = simd::ScalarF::GetDataStep();
    const int num_stride = correspondences.size() / stride;
    for (size_t point_idx = 0; point_idx < num_stride * stride;
         point_idx += stride) {
      simd::VectorF<3> p__;
      simd::VectorF<3> mu__;
      simd::MatrixF<3, 3> sqrt_info__;
      const auto& corr1 = correspondences.at(point_idx);
      const auto& corr2 = correspondences.at(point_idx + 1);
      const auto& corr3 = correspondences.at(point_idx + 2);
      const auto& corr4 = correspondences.at(point_idx + 3);
      const auto& corr5 = correspondences.at(point_idx + 4);
      const auto& corr6 = correspondences.at(point_idx + 5);
      const auto& corr7 = correspondences.at(point_idx + 6);
      const auto& corr8 = correspondences.at(point_idx + 7);
      p__ = simd::VectorF<3>(
          {corr1.point.cast<float>(), corr2.point.cast<float>(),
           corr3.point.cast<float>(), corr4.point.cast<float>(),
           corr5.point.cast<float>(), corr6.point.cast<float>(),
           corr7.point.cast<float>(), corr8.point.cast<float>()});
      mu__ = simd::VectorF<3>(
          {corr1.ndt.mean.cast<float>(), corr2.ndt.mean.cast<float>(),
           corr3.ndt.mean.cast<float>(), corr4.ndt.mean.cast<float>(),
           corr5.ndt.mean.cast<float>(), corr6.ndt.mean.cast<float>(),
           corr7.ndt.mean.cast<float>(), corr8.ndt.mean.cast<float>()});
      sqrt_info__ =
          simd::MatrixF<3, 3>({corr1.ndt.sqrt_information.cast<float>(),
                               corr2.ndt.sqrt_information.cast<float>(),
                               corr3.ndt.sqrt_information.cast<float>(),
                               corr4.ndt.sqrt_information.cast<float>(),
                               corr5.ndt.sqrt_information.cast<float>(),
                               corr6.ndt.sqrt_information.cast<float>(),
                               corr7.ndt.sqrt_information.cast<float>(),
                               corr8.ndt.sqrt_information.cast<float>()});

      // clang-format off
    // pw = R*p + t
    const simd::VectorF<3> pw__ = R__ * p__ + t__;

    // e_i = pw - mean
    const simd::VectorF<3> e__ = pw__ - mu__;

    // r = sqrt_info * e
    const simd::VectorF<3> r__ = sqrt_info__ * e__;

    simd::MatrixF<3,3> minus_R_skewp__;
    minus_R_skewp__(0, 0) =  R__(0, 2) * p__(1) - R__(0, 1) * p__(2);
    minus_R_skewp__(1, 0) =  R__(1, 2) * p__(1) - R__(1, 1) * p__(2);
    minus_R_skewp__(2, 0) =  R__(2, 2) * p__(1) - R__(2, 1) * p__(2);
    minus_R_skewp__(0, 1) =  R__(0, 0) * p__(2) - R__(0, 2) * p__(0);
    minus_R_skewp__(1, 1) =  R__(1, 0) * p__(2) - R__(1, 2) * p__(0);
    minus_R_skewp__(2, 1) =  R__(2, 0) * p__(2) - R__(2, 2) * p__(0);
    minus_R_skewp__(0, 2) =  R__(0, 1) * p__(0) - R__(0, 0) * p__(1);
    minus_R_skewp__(1, 2) =  R__(1, 1) * p__(0) - R__(1, 0) * p__(1);
    minus_R_skewp__(2, 2) =  R__(2, 1) * p__(0) - R__(2, 0) * p__(1);
    const simd::MatrixF<3,3> sqrt_info_minus_R_skewp_ = sqrt_info__ * minus_R_skewp__;

    // Direct calculation is far faster than helper operator.
    simd::MatrixF<3,6> J__;
    // J__(0,3) = sqrt_info__(0,0) * minus_R_skewp__(0,0) + sqrt_info__(0,1) * minus_R_skewp__(1,0) + sqrt_info__(0,2) * minus_R_skewp__(2,0);
    // J__(0,4) = sqrt_info__(0,0) * minus_R_skewp__(0,1) + sqrt_info__(0,1) * minus_R_skewp__(1,1) + sqrt_info__(0,2) * minus_R_skewp__(2,1);
    // J__(0,5) = sqrt_info__(0,0) * minus_R_skewp__(0,2) + sqrt_info__(0,1) * minus_R_skewp__(1,2) + sqrt_info__(0,2) * minus_R_skewp__(2,2);
    // J__(1,3) = sqrt_info__(1,0) * minus_R_skewp__(0,0) + sqrt_info__(1,1) * minus_R_skewp__(1,0) + sqrt_info__(1,2) * minus_R_skewp__(2,0);
    // J__(1,4) = sqrt_info__(1,0) * minus_R_skewp__(0,1) + sqrt_info__(1,1) * minus_R_skewp__(1,1) + sqrt_info__(1,2) * minus_R_skewp__(2,1);
    // J__(1,5) = sqrt_info__(1,0) * minus_R_skewp__(0,2) + sqrt_info__(1,1) * minus_R_skewp__(1,2) + sqrt_info__(1,2) * minus_R_skewp__(2,2);
    // J__(2,3) = sqrt_info__(2,0) * minus_R_skewp__(0,0) + sqrt_info__(2,1) * minus_R_skewp__(1,0) + sqrt_info__(2,2) * minus_R_skewp__(2,0);
    // J__(2,4) = sqrt_info__(2,0) * minus_R_skewp__(0,1) + sqrt_info__(2,1) * minus_R_skewp__(1,1) + sqrt_info__(2,2) * minus_R_skewp__(2,1);
    // J__(2,5) = sqrt_info__(2,0) * minus_R_skewp__(0,2) + sqrt_info__(2,1) * minus_R_skewp__(1,2) + sqrt_info__(2,2) * minus_R_skewp__(2,2);
    J__(0, 0) = sqrt_info__(0, 0); J__(0, 1) = sqrt_info__(0, 1); J__(0, 2) = sqrt_info__(0, 2);
    J__(1, 0) = sqrt_info__(1, 0); J__(1, 1) = sqrt_info__(1, 1); J__(1, 2) = sqrt_info__(1, 2);
    J__(2, 0) = sqrt_info__(2, 0); J__(2, 1) = sqrt_info__(2, 1); J__(2, 2) = sqrt_info__(2, 2);
    J__(0, 3) = sqrt_info_minus_R_skewp_(0, 0); J__(0, 4) = sqrt_info_minus_R_skewp_(0, 1); J__(0, 5) = sqrt_info_minus_R_skewp_(0, 2);
    J__(1, 3) = sqrt_info_minus_R_skewp_(1, 0); J__(1, 4) = sqrt_info_minus_R_skewp_(1, 1); J__(1, 5) = sqrt_info_minus_R_skewp_(1, 2);
    J__(2, 3) = sqrt_info_minus_R_skewp_(2, 0); J__(2, 4) = sqrt_info_minus_R_skewp_(2, 1); J__(2, 5) = sqrt_info_minus_R_skewp_(2, 2);
      // clang-format on

      // Compute loss and weight,
      // and add the local gradient and hessian to the global ones
      simd::ScalarF sq_r__ = r__.GetNorm();
      simd::ScalarF loss__(sq_r__);
      simd::ScalarF weight__(1.0);
      if (loss_function_ != nullptr) {
        float sq_r_buf[8];
        sq_r__.StoreData(sq_r_buf);
        float loss_buf[8];
        float weight_buf[8];
        for (int k = 0; k < 8; ++k) {
          double loss_output[3] = {0.0, 0.0, 0.0};
          loss_function_->Evaluate(sq_r_buf[k], loss_output);
          loss_buf[k] = loss_output[0];
          weight_buf[k] = loss_output[1];
        }
        loss__ = simd::ScalarF(loss_buf);
        weight__ = simd::ScalarF(weight_buf);
      }

      // g(i) += (J(0,i)*r(0) + J(1,i)*r(1) + J(2,i)*r(2))
      gradient__ += (J__.transpose() * r__) * weight__;

      // H(i,j) = sum_{k} w * J(k,i) * J(k,j)
      for (int ii = 0; ii < 6; ++ii) {
        for (int jj = ii; jj < 6; ++jj) {
          hessian__(ii, jj) +=
              (weight__ * (J__(0, ii) * J__(0, jj) + J__(1, ii) * J__(1, jj) +
                           J__(2, ii) * J__(2, jj)));
        }
      }

      cost__ += loss__;
    }
    float buf[8];
    cost__.StoreData(buf);
    float cost = 0.0;
    cost +=
        (buf[0] + buf[1] + buf[2] + buf[3] + buf[4] + buf[5] + buf[6] + buf[7]);

    Mat6x6 hessian{Mat6x6::Zero()};
    Vec6 gradient{Vec6::Zero()};
    for (int ii = 0; ii < 6; ++ii) {
      gradient__(ii).StoreData(buf);
      gradient(ii) += (buf[0] + buf[1] + buf[2] + buf[3] + buf[4] + buf[5] +
                       buf[6] + buf[7]);
      for (int jj = ii; jj < 6; ++jj) {
        hessian__(ii, jj).StoreData(buf);
        hessian(ii, jj) += (buf[0] + buf[1] + buf[2] + buf[3] + buf[4] +
                            buf[5] + buf[6] + buf[7]);
      }
    }

    // Reflect the hessian
    ReflectHessian(&hessian);

    // Damping hessian
    for (int k = 0; k < 6; k++) hessian(k, k) *= 1.0 + lambda;

    // Compute the step
    const Vec6 update_step = hessian.ldlt().solve(-gradient);

    // Update the pose
    const Vec3 delta_t = update_step.block<3, 1>(0, 0);
    const Vec3 delta_R = update_step.block<3, 1>(3, 0);
    optimized_translation += delta_t.cast<float>();
    optimized_orientation *= ComputeQuaternion(delta_R).cast<float>();
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

  pose->translation() = optimized_translation.cast<double>();
  pose->linear() = optimized_orientation.toRotationMatrix().cast<double>();

  return true;
}

bool MahalanobisDistanceMinimizerAnalyticSIMD::SolveFloatMatrixAligned(
    const Options& options, const std::vector<Correspondence>& correspondences,
    Pose* pose) {
  using Vec3f = Eigen::Vector3f;
  using Mat3x3f = Eigen::Matrix3f;
  using Orientationf = Eigen::Quaternionf;

  AlignedBuffer abuf(correspondences.size());
  for (size_t index = 0; index < correspondences.size(); ++index) {
    const auto& corr = correspondences.at(index);
    abuf.x[index] = corr.point.x();
    abuf.y[index] = corr.point.y();
    abuf.z[index] = corr.point.z();
    abuf.mx[index] = corr.ndt.mean.x();
    abuf.my[index] = corr.ndt.mean.y();
    abuf.mz[index] = corr.ndt.mean.z();
    for (int i = 0; i < 3; ++i) {
      for (int j = 0; j < 3; ++j) {
        abuf.sqrt_info[i][j][index] = corr.ndt.sqrt_information(i, j);
      }
    }
  }

  constexpr float min_lambda = 1e-6;
  constexpr float max_lambda = 1e-2;

  const Pose initial_pose = *pose;

  Vec3f optimized_translation{initial_pose.translation().cast<float>()};
  Orientationf optimized_orientation(initial_pose.rotation().cast<float>());

  float lambda = 0.001f;
  float previous_cost = std::numeric_limits<float>::max();
  int iteration = 0;
  for (; iteration < options.max_iterations; ++iteration) {
    const Mat3x3f opt_R = optimized_orientation.toRotationMatrix();
    const Vec3f opt_t = optimized_translation;
    simd::MatrixF<3, 3> R__(opt_R);
    simd::VectorF<3> t__(opt_t);

    simd::VectorF<6> gradient__(Eigen::Matrix<float, 6, 1>::Zero());
    simd::MatrixF<6, 6> hessian__(Eigen::Matrix<float, 6, 6>::Zero());
    simd::ScalarF cost__(0.0);
    const size_t stride = simd::ScalarF::GetDataStep();
    const int num_stride = correspondences.size() / stride;
    for (size_t point_idx = 0; point_idx < num_stride * stride;
         point_idx += stride) {
      simd::VectorF<3> p__(
          {abuf.x + point_idx, abuf.y + point_idx, abuf.z + point_idx});
      simd::VectorF<3> mu__(
          {abuf.mx + point_idx, abuf.my + point_idx, abuf.mz + point_idx});
      simd::MatrixF<3, 3> sqrt_info__(
          {abuf.sqrt_info[0][0] + point_idx, abuf.sqrt_info[0][1] + point_idx,
           abuf.sqrt_info[0][2] + point_idx, abuf.sqrt_info[1][0] + point_idx,
           abuf.sqrt_info[1][1] + point_idx, abuf.sqrt_info[1][2] + point_idx,
           abuf.sqrt_info[2][0] + point_idx, abuf.sqrt_info[2][1] + point_idx,
           abuf.sqrt_info[2][2] + point_idx});

      // clang-format off
      // pw = R*p + t
      const simd::VectorF<3> pw__ = R__ * p__ + t__;

      // e_i = pw - mean
      const simd::VectorF<3> e__ = pw__ - mu__;

      // r = sqrt_info * e
      const simd::VectorF<3> r__ = sqrt_info__ * e__;

      simd::MatrixF<3,3> minus_R_skewp__;
      minus_R_skewp__(0, 0) =  R__(0, 2) * p__(1) - R__(0, 1) * p__(2);
      minus_R_skewp__(1, 0) =  R__(1, 2) * p__(1) - R__(1, 1) * p__(2);
      minus_R_skewp__(2, 0) =  R__(2, 2) * p__(1) - R__(2, 1) * p__(2);
      minus_R_skewp__(0, 1) =  R__(0, 0) * p__(2) - R__(0, 2) * p__(0);
      minus_R_skewp__(1, 1) =  R__(1, 0) * p__(2) - R__(1, 2) * p__(0);
      minus_R_skewp__(2, 1) =  R__(2, 0) * p__(2) - R__(2, 2) * p__(0);
      minus_R_skewp__(0, 2) =  R__(0, 1) * p__(0) - R__(0, 0) * p__(1);
      minus_R_skewp__(1, 2) =  R__(1, 1) * p__(0) - R__(1, 0) * p__(1);
      minus_R_skewp__(2, 2) =  R__(2, 1) * p__(0) - R__(2, 0) * p__(1);
      const simd::MatrixF<3,3> sqrt_info_minus_R_skewp_ = sqrt_info__ * minus_R_skewp__;

      // Direct calculation is far faster than helper operator.
      simd::MatrixF<3,6> J__;
      J__(0, 0) = sqrt_info__(0, 0); J__(0, 1) = sqrt_info__(0, 1); J__(0, 2) = sqrt_info__(0, 2);
      J__(1, 0) = sqrt_info__(1, 0); J__(1, 1) = sqrt_info__(1, 1); J__(1, 2) = sqrt_info__(1, 2);
      J__(2, 0) = sqrt_info__(2, 0); J__(2, 1) = sqrt_info__(2, 1); J__(2, 2) = sqrt_info__(2, 2);
      J__(0, 3) = sqrt_info_minus_R_skewp_(0, 0); J__(0, 4) = sqrt_info_minus_R_skewp_(0, 1); J__(0, 5) = sqrt_info_minus_R_skewp_(0, 2);
      J__(1, 3) = sqrt_info_minus_R_skewp_(1, 0); J__(1, 4) = sqrt_info_minus_R_skewp_(1, 1); J__(1, 5) = sqrt_info_minus_R_skewp_(1, 2);
      J__(2, 3) = sqrt_info_minus_R_skewp_(2, 0); J__(2, 4) = sqrt_info_minus_R_skewp_(2, 1); J__(2, 5) = sqrt_info_minus_R_skewp_(2, 2);
      // clang-format on

      // Compute loss and weight,
      // and add the local gradient and hessian to the global ones
      simd::ScalarF sq_r__ = r__.GetNorm();
      simd::ScalarF loss__(sq_r__);
      simd::ScalarF weight__(1.0);
      if (loss_function_ != nullptr) {
        float sq_r_buf[8];
        sq_r__.StoreData(sq_r_buf);
        float loss_buf[8];
        float weight_buf[8];
        for (int k = 0; k < 8; ++k) {
          double loss_output[3] = {0.0, 0.0, 0.0};
          loss_function_->Evaluate(sq_r_buf[k], loss_output);
          loss_buf[k] = loss_output[0];
          weight_buf[k] = loss_output[1];
        }
        loss__ = simd::ScalarF(loss_buf);
        weight__ = simd::ScalarF(weight_buf);
      }

      // g(i) += (J(0,i)*r(0) + J(1,i)*r(1) + J(2,i)*r(2))
      gradient__ += (J__.transpose() * r__) * weight__;

      // H(i,j) = sum_{k} w * J(k,i) * J(k,j)
      for (int ii = 0; ii < 6; ++ii) {
        for (int jj = ii; jj < 6; ++jj) {
          hessian__(ii, jj) +=
              (weight__ * (J__(0, ii) * J__(0, jj) + J__(1, ii) * J__(1, jj) +
                           J__(2, ii) * J__(2, jj)));
        }
      }

      cost__ += loss__;
    }
    float buf[8];
    cost__.StoreData(buf);
    float cost = 0.0;
    cost +=
        (buf[0] + buf[1] + buf[2] + buf[3] + buf[4] + buf[5] + buf[6] + buf[7]);

    Mat6x6 hessian{Mat6x6::Zero()};
    Vec6 gradient{Vec6::Zero()};
    for (int ii = 0; ii < 6; ++ii) {
      gradient__(ii).StoreData(buf);
      gradient(ii) += (buf[0] + buf[1] + buf[2] + buf[3] + buf[4] + buf[5] +
                       buf[6] + buf[7]);
      for (int jj = ii; jj < 6; ++jj) {
        hessian__(ii, jj).StoreData(buf);
        hessian(ii, jj) += (buf[0] + buf[1] + buf[2] + buf[3] + buf[4] +
                            buf[5] + buf[6] + buf[7]);
      }
    }

    // Reflect the hessian
    ReflectHessian(&hessian);

    // Damping hessian
    for (int k = 0; k < 6; k++) hessian(k, k) *= 1.0 + lambda;

    // Compute the step
    const Vec6 update_step = hessian.ldlt().solve(-gradient);

    // Update the pose
    const Vec3 delta_t = update_step.block<3, 1>(0, 0);
    const Vec3 delta_R = update_step.block<3, 1>(3, 0);
    optimized_translation += delta_t.cast<float>();
    optimized_orientation *= ComputeQuaternion(delta_R).cast<float>();
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

  pose->translation() = optimized_translation.cast<double>();
  pose->linear() = optimized_orientation.toRotationMatrix().cast<double>();

  return true;
}

bool MahalanobisDistanceMinimizerAnalyticSIMD::Solve(
    const Options& options, const std::vector<Correspondence>& correspondences,
    Pose* pose) {
  constexpr double min_lambda = 1e-6;
  constexpr double max_lambda = 1e-2;

  const Pose initial_pose = *pose;

  Vec3 optimized_translation{initial_pose.translation()};
  Orientation optimized_orientation(initial_pose.rotation());

  double lambda = 0.001;
  double previous_cost = std::numeric_limits<double>::max();
  int iteration = 0;
  for (; iteration < options.max_iterations; ++iteration) {
    const Mat3x3 opt_R = optimized_orientation.toRotationMatrix();
    const Vec3 opt_t = optimized_translation;

    simd::Scalar R__[3][3];
    simd::Scalar t__[3];
    for (int row = 0; row < 3; ++row) {
      t__[row] = simd::Scalar(opt_t(row));
      for (int col = 0; col < 3; ++col)
        R__[row][col] = simd::Scalar(opt_R(row, col));
    }

    simd::Scalar gradient__[6];
    simd::Scalar hessian__[6][6];
    simd::Scalar cost__;
    const size_t stride = simd::Scalar::GetDataStep();
    const int num_stride = correspondences.size() / stride;
    for (size_t point_idx = 0; point_idx < num_stride * stride;
         point_idx += stride) {
      double px_buf[stride];
      double py_buf[stride];
      double pz_buf[stride];
      double mx_buf[stride];
      double my_buf[stride];
      double mz_buf[stride];
      double sqrt_info_buf[3][3][stride];
      for (size_t k = 0; k < stride; ++k) {
        const auto& corr = correspondences.at(point_idx + k);
        px_buf[k] = corr.point(0);
        py_buf[k] = corr.point(1);
        pz_buf[k] = corr.point(2);
        mx_buf[k] = corr.ndt.mean(0);
        my_buf[k] = corr.ndt.mean(1);
        mz_buf[k] = corr.ndt.mean(2);
        sqrt_info_buf[0][0][k] = corr.ndt.sqrt_information(0, 0);
        sqrt_info_buf[0][1][k] = corr.ndt.sqrt_information(0, 1);
        sqrt_info_buf[0][2][k] = corr.ndt.sqrt_information(0, 2);
        sqrt_info_buf[1][0][k] = corr.ndt.sqrt_information(1, 0);
        sqrt_info_buf[1][1][k] = corr.ndt.sqrt_information(1, 1);
        sqrt_info_buf[1][2][k] = corr.ndt.sqrt_information(1, 2);
        sqrt_info_buf[2][0][k] = corr.ndt.sqrt_information(2, 0);
        sqrt_info_buf[2][1][k] = corr.ndt.sqrt_information(2, 1);
        sqrt_info_buf[2][2][k] = corr.ndt.sqrt_information(2, 2);
      }

      simd::Scalar p__[3];
      p__[0] = simd::Scalar(px_buf);
      p__[1] = simd::Scalar(py_buf);
      p__[2] = simd::Scalar(pz_buf);

      simd::Scalar mu__[3];
      mu__[0] = simd::Scalar(mx_buf);
      mu__[1] = simd::Scalar(my_buf);
      mu__[2] = simd::Scalar(mz_buf);

      simd::Scalar sqrt_info__[3][3];
      sqrt_info__[0][0] = simd::Scalar(sqrt_info_buf[0][0]);
      sqrt_info__[0][1] = simd::Scalar(sqrt_info_buf[0][1]);
      sqrt_info__[0][2] = simd::Scalar(sqrt_info_buf[0][2]);
      sqrt_info__[1][0] = simd::Scalar(sqrt_info_buf[1][0]);
      sqrt_info__[1][1] = simd::Scalar(sqrt_info_buf[1][1]);
      sqrt_info__[1][2] = simd::Scalar(sqrt_info_buf[1][2]);
      sqrt_info__[2][0] = simd::Scalar(sqrt_info_buf[2][0]);
      sqrt_info__[2][1] = simd::Scalar(sqrt_info_buf[2][1]);
      sqrt_info__[2][2] = simd::Scalar(sqrt_info_buf[2][2]);

      // clang-format off
      // pw = R*p + t
      simd::Scalar pw__[3];
      pw__[0] = R__[0][0] * p__[0] + R__[0][1] * p__[1] + R__[0][2] * p__[2] + t__[0];
      pw__[1] = R__[1][0] * p__[0] + R__[1][1] * p__[1] + R__[1][2] * p__[2] + t__[1];
      pw__[2] = R__[2][0] * p__[0] + R__[2][1] * p__[1] + R__[2][2] * p__[2] + t__[2];

      // e_i = pw - mean
      simd::Scalar e__[3];
      e__[0] = pw__[0] - mu__[0];
      e__[1] = pw__[1] - mu__[1];
      e__[2] = pw__[2] - mu__[2];
      // r = sqrt_info * e
      simd::Scalar r__[3];
      r__[0] = sqrt_info__[0][0] * e__[0] + sqrt_info__[0][1] * e__[1] + sqrt_info__[0][2] * e__[2];
      r__[1] = sqrt_info__[1][0] * e__[0] + sqrt_info__[1][1] * e__[1] + sqrt_info__[1][2] * e__[2];
      r__[2] = sqrt_info__[2][0] * e__[0] + sqrt_info__[2][1] * e__[1] + sqrt_info__[2][2] * e__[2];

      simd::Scalar J__[3][6];
      J__[0][0] = sqrt_info__[0][0]; J__[0][1] = sqrt_info__[0][1]; J__[0][2] = sqrt_info__[0][2];
      J__[1][0] = sqrt_info__[1][0]; J__[1][1] = sqrt_info__[1][1]; J__[1][2] = sqrt_info__[1][2];
      J__[2][0] = sqrt_info__[2][0]; J__[2][1] = sqrt_info__[2][1]; J__[2][2] = sqrt_info__[2][2];

      simd::Scalar minus_R_skewp__[3][3];
      minus_R_skewp__[0][0] =  R__[0][2] * p__[1] - R__[0][1] * p__[2];
      minus_R_skewp__[1][0] =  R__[1][2] * p__[1] - R__[1][1] * p__[2];
      minus_R_skewp__[2][0] =  R__[2][2] * p__[1] - R__[2][1] * p__[2];

      minus_R_skewp__[0][1] =  R__[0][0] * p__[2] - R__[0][2] * p__[0];
      minus_R_skewp__[1][1] =  R__[1][0] * p__[2] - R__[1][2] * p__[0];
      minus_R_skewp__[2][1] =  R__[2][0] * p__[2] - R__[2][2] * p__[0];
      
      minus_R_skewp__[0][2] =  R__[0][1] * p__[0] - R__[0][0] * p__[1];
      minus_R_skewp__[1][2] =  R__[1][1] * p__[0] - R__[1][0] * p__[1];
      minus_R_skewp__[2][2] =  R__[2][1] * p__[0] - R__[2][0] * p__[1];

      J__[0][3] = sqrt_info__[0][0] * minus_R_skewp__[0][0] + sqrt_info__[0][1] * minus_R_skewp__[1][0] + sqrt_info__[0][2] * minus_R_skewp__[2][0];
      J__[0][4] = sqrt_info__[0][0] * minus_R_skewp__[0][1] + sqrt_info__[0][1] * minus_R_skewp__[1][1] + sqrt_info__[0][2] * minus_R_skewp__[2][1];
      J__[0][5] = sqrt_info__[0][0] * minus_R_skewp__[0][2] + sqrt_info__[0][1] * minus_R_skewp__[1][2] + sqrt_info__[0][2] * minus_R_skewp__[2][2];
      J__[1][3] = sqrt_info__[1][0] * minus_R_skewp__[0][0] + sqrt_info__[1][1] * minus_R_skewp__[1][0] + sqrt_info__[1][2] * minus_R_skewp__[2][0];
      J__[1][4] = sqrt_info__[1][0] * minus_R_skewp__[0][1] + sqrt_info__[1][1] * minus_R_skewp__[1][1] + sqrt_info__[1][2] * minus_R_skewp__[2][1];
      J__[1][5] = sqrt_info__[1][0] * minus_R_skewp__[0][2] + sqrt_info__[1][1] * minus_R_skewp__[1][2] + sqrt_info__[1][2] * minus_R_skewp__[2][2];
      J__[2][3] = sqrt_info__[2][0] * minus_R_skewp__[0][0] + sqrt_info__[2][1] * minus_R_skewp__[1][0] + sqrt_info__[2][2] * minus_R_skewp__[2][0];
      J__[2][4] = sqrt_info__[2][0] * minus_R_skewp__[0][1] + sqrt_info__[2][1] * minus_R_skewp__[1][1] + sqrt_info__[2][2] * minus_R_skewp__[2][1];
      J__[2][5] = sqrt_info__[2][0] * minus_R_skewp__[0][2] + sqrt_info__[2][1] * minus_R_skewp__[1][2] + sqrt_info__[2][2] * minus_R_skewp__[2][2];
      // clang-format on

      // Compute loss and weight,
      // and add the local gradient and hessian to the global ones
      simd::Scalar sq_r__ = r__[0] * r__[0] + r__[1] * r__[1] + r__[2] * r__[2];
      simd::Scalar loss__(sq_r__);
      simd::Scalar weight__(1.0);
      if (loss_function_ != nullptr) {
        double sq_r_buf[4];
        sq_r__.StoreData(sq_r_buf);
        double loss_buf[4];
        double weight_buf[4];
        for (int k = 0; k < 4; ++k) {
          double loss_output[3] = {0.0, 0.0, 0.0};
          loss_function_->Evaluate(sq_r_buf[k], loss_output);
          loss_buf[k] = loss_output[0];
          weight_buf[k] = loss_output[1];
        }
        loss__ = simd::Scalar(loss_buf);
        weight__ = simd::Scalar(weight_buf);
      }

      // g(i) += (J(0,i)*r(0) + J(1,i)*r(1) + J(2,i)*r(2))
      for (int k = 0; k < 6; ++k)
        gradient__[k] += (weight__ * (J__[0][k] * r__[0] + J__[1][k] * r__[1] +
                                      J__[2][k] * r__[2]));

      // H(i,j) = sum_{k} w * J(k,i) * J(k,j)
      for (int ii = 0; ii < 6; ++ii) {
        for (int jj = ii; jj < 6; ++jj) {
          hessian__[ii][jj] +=
              (weight__ * (J__[0][ii] * J__[0][jj] + J__[1][ii] * J__[1][jj] +
                           J__[2][ii] * J__[2][jj]));
        }
      }

      cost__ += loss__;
    }
    double buf[4];
    cost__.StoreData(buf);
    double cost = 0.0;
    cost += (buf[0] + buf[1] + buf[2] + buf[3]);

    Mat6x6 hessian{Mat6x6::Zero()};
    Vec6 gradient{Vec6::Zero()};
    for (int ii = 0; ii < 6; ++ii) {
      gradient__[ii].StoreData(buf);
      gradient(ii) += (buf[0] + buf[1] + buf[2] + buf[3]);
      for (int jj = ii; jj < 6; ++jj) {
        hessian__[ii][jj].StoreData(buf);
        hessian(ii, jj) += (buf[0] + buf[1] + buf[2] + buf[3]);
      }
    }

    // Reflect the hessian
    ReflectHessian(&hessian);

    // Damping hessian
    for (int k = 0; k < 6; k++) hessian(k, k) *= 1.0 + lambda;

    // Compute the step
    const Vec6 update_step = hessian.ldlt().solve(-gradient);

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

bool MahalanobisDistanceMinimizerAnalyticSIMD::SolveFloat(
    const Options& options, const std::vector<Correspondence>& correspondences,
    Pose* pose) {
  using Vec3f = Eigen::Vector3f;
  using Mat3x3f = Eigen::Matrix3f;
  using Orientationf = Eigen::Quaternionf;

  constexpr float min_lambda = 1e-6f;
  constexpr float max_lambda = 1e-2f;

  const Pose initial_pose = *pose;

  Vec3f optimized_translation{initial_pose.translation().cast<float>()};
  Orientationf optimized_orientation(initial_pose.rotation().cast<float>());

  float lambda = 0.001;
  float previous_cost = std::numeric_limits<float>::max();
  int iteration = 0;
  for (; iteration < options.max_iterations; ++iteration) {
    const Mat3x3f opt_R = optimized_orientation.toRotationMatrix();
    const Vec3f opt_t = optimized_translation;

    simd::ScalarF R__[3][3];
    simd::ScalarF t__[3];
    for (int row = 0; row < 3; ++row) {
      t__[row] = simd::ScalarF(opt_t(row));
      for (int col = 0; col < 3; ++col)
        R__[row][col] = simd::ScalarF(opt_R(row, col));
    }

    simd::ScalarF gradient__[6];
    simd::ScalarF hessian__[6][6];
    simd::ScalarF cost__;
    const size_t stride = simd::ScalarF::GetDataStep();
    const int num_stride = correspondences.size() / stride;
    for (size_t point_idx = 0; point_idx < num_stride * stride;
         point_idx += stride) {
      float px_buf[stride];
      float py_buf[stride];
      float pz_buf[stride];
      float mx_buf[stride];
      float my_buf[stride];
      float mz_buf[stride];
      float sqrt_info_buf[3][3][stride];
      for (size_t k = 0; k < stride; ++k) {
        const auto& corr = correspondences.at(point_idx + k);
        px_buf[k] = corr.point(0);
        py_buf[k] = corr.point(1);
        pz_buf[k] = corr.point(2);
        mx_buf[k] = corr.ndt.mean(0);
        my_buf[k] = corr.ndt.mean(1);
        mz_buf[k] = corr.ndt.mean(2);
        sqrt_info_buf[0][0][k] = corr.ndt.sqrt_information(0, 0);
        sqrt_info_buf[0][1][k] = corr.ndt.sqrt_information(0, 1);
        sqrt_info_buf[0][2][k] = corr.ndt.sqrt_information(0, 2);
        sqrt_info_buf[1][0][k] = corr.ndt.sqrt_information(1, 0);
        sqrt_info_buf[1][1][k] = corr.ndt.sqrt_information(1, 1);
        sqrt_info_buf[1][2][k] = corr.ndt.sqrt_information(1, 2);
        sqrt_info_buf[2][0][k] = corr.ndt.sqrt_information(2, 0);
        sqrt_info_buf[2][1][k] = corr.ndt.sqrt_information(2, 1);
        sqrt_info_buf[2][2][k] = corr.ndt.sqrt_information(2, 2);
      }

      simd::ScalarF p__[3];
      p__[0] = simd::ScalarF(px_buf);
      p__[1] = simd::ScalarF(py_buf);
      p__[2] = simd::ScalarF(pz_buf);

      simd::ScalarF mu__[3];
      mu__[0] = simd::ScalarF(mx_buf);
      mu__[1] = simd::ScalarF(my_buf);
      mu__[2] = simd::ScalarF(mz_buf);

      simd::ScalarF sqrt_info__[3][3];
      sqrt_info__[0][0] = simd::ScalarF(sqrt_info_buf[0][0]);
      sqrt_info__[0][1] = simd::ScalarF(sqrt_info_buf[0][1]);
      sqrt_info__[0][2] = simd::ScalarF(sqrt_info_buf[0][2]);
      sqrt_info__[1][0] = simd::ScalarF(sqrt_info_buf[1][0]);
      sqrt_info__[1][1] = simd::ScalarF(sqrt_info_buf[1][1]);
      sqrt_info__[1][2] = simd::ScalarF(sqrt_info_buf[1][2]);
      sqrt_info__[2][0] = simd::ScalarF(sqrt_info_buf[2][0]);
      sqrt_info__[2][1] = simd::ScalarF(sqrt_info_buf[2][1]);
      sqrt_info__[2][2] = simd::ScalarF(sqrt_info_buf[2][2]);

      // clang-format off
      // pw = R*p + t
      simd::ScalarF pw__[3];
      pw__[0] = R__[0][0] * p__[0] + R__[0][1] * p__[1] + R__[0][2] * p__[2] + t__[0];
      pw__[1] = R__[1][0] * p__[0] + R__[1][1] * p__[1] + R__[1][2] * p__[2] + t__[1];
      pw__[2] = R__[2][0] * p__[0] + R__[2][1] * p__[1] + R__[2][2] * p__[2] + t__[2];

      // e_i = pw - mean
      simd::ScalarF e__[3];
      e__[0] = pw__[0] - mu__[0];
      e__[1] = pw__[1] - mu__[1];
      e__[2] = pw__[2] - mu__[2];
      // r = sqrt_info * e
      simd::ScalarF r__[3];
      r__[0] = sqrt_info__[0][0] * e__[0] + sqrt_info__[0][1] * e__[1] + sqrt_info__[0][2] * e__[2];
      r__[1] = sqrt_info__[1][0] * e__[0] + sqrt_info__[1][1] * e__[1] + sqrt_info__[1][2] * e__[2];
      r__[2] = sqrt_info__[2][0] * e__[0] + sqrt_info__[2][1] * e__[1] + sqrt_info__[2][2] * e__[2];

      simd::ScalarF J__[3][6];
      J__[0][0] = sqrt_info__[0][0]; J__[0][1] = sqrt_info__[0][1]; J__[0][2] = sqrt_info__[0][2];
      J__[1][0] = sqrt_info__[1][0]; J__[1][1] = sqrt_info__[1][1]; J__[1][2] = sqrt_info__[1][2];
      J__[2][0] = sqrt_info__[2][0]; J__[2][1] = sqrt_info__[2][1]; J__[2][2] = sqrt_info__[2][2];

      simd::ScalarF minus_R_skewp__[3][3];
      minus_R_skewp__[0][0] =  R__[0][2] * p__[1] - R__[0][1] * p__[2];
      minus_R_skewp__[1][0] =  R__[1][2] * p__[1] - R__[1][1] * p__[2];
      minus_R_skewp__[2][0] =  R__[2][2] * p__[1] - R__[2][1] * p__[2];

      minus_R_skewp__[0][1] =  R__[0][0] * p__[2] - R__[0][2] * p__[0];
      minus_R_skewp__[1][1] =  R__[1][0] * p__[2] - R__[1][2] * p__[0];
      minus_R_skewp__[2][1] =  R__[2][0] * p__[2] - R__[2][2] * p__[0];
      
      minus_R_skewp__[0][2] =  R__[0][1] * p__[0] - R__[0][0] * p__[1];
      minus_R_skewp__[1][2] =  R__[1][1] * p__[0] - R__[1][0] * p__[1];
      minus_R_skewp__[2][2] =  R__[2][1] * p__[0] - R__[2][0] * p__[1];

      J__[0][3] = sqrt_info__[0][0] * minus_R_skewp__[0][0] + sqrt_info__[0][1] * minus_R_skewp__[1][0] + sqrt_info__[0][2] * minus_R_skewp__[2][0];
      J__[0][4] = sqrt_info__[0][0] * minus_R_skewp__[0][1] + sqrt_info__[0][1] * minus_R_skewp__[1][1] + sqrt_info__[0][2] * minus_R_skewp__[2][1];
      J__[0][5] = sqrt_info__[0][0] * minus_R_skewp__[0][2] + sqrt_info__[0][1] * minus_R_skewp__[1][2] + sqrt_info__[0][2] * minus_R_skewp__[2][2];
      J__[1][3] = sqrt_info__[1][0] * minus_R_skewp__[0][0] + sqrt_info__[1][1] * minus_R_skewp__[1][0] + sqrt_info__[1][2] * minus_R_skewp__[2][0];
      J__[1][4] = sqrt_info__[1][0] * minus_R_skewp__[0][1] + sqrt_info__[1][1] * minus_R_skewp__[1][1] + sqrt_info__[1][2] * minus_R_skewp__[2][1];
      J__[1][5] = sqrt_info__[1][0] * minus_R_skewp__[0][2] + sqrt_info__[1][1] * minus_R_skewp__[1][2] + sqrt_info__[1][2] * minus_R_skewp__[2][2];
      J__[2][3] = sqrt_info__[2][0] * minus_R_skewp__[0][0] + sqrt_info__[2][1] * minus_R_skewp__[1][0] + sqrt_info__[2][2] * minus_R_skewp__[2][0];
      J__[2][4] = sqrt_info__[2][0] * minus_R_skewp__[0][1] + sqrt_info__[2][1] * minus_R_skewp__[1][1] + sqrt_info__[2][2] * minus_R_skewp__[2][1];
      J__[2][5] = sqrt_info__[2][0] * minus_R_skewp__[0][2] + sqrt_info__[2][1] * minus_R_skewp__[1][2] + sqrt_info__[2][2] * minus_R_skewp__[2][2];
      // clang-format on

      // Compute loss and weight,
      // and add the local gradient and hessian to the global ones
      simd::ScalarF sq_r__ =
          r__[0] * r__[0] + r__[1] * r__[1] + r__[2] * r__[2];
      simd::ScalarF loss__(sq_r__);
      simd::ScalarF weight__(1.0);
      if (loss_function_ != nullptr) {
        float sq_r_buf[8];
        sq_r__.StoreData(sq_r_buf);
        float loss_buf[8];
        float weight_buf[8];
        for (int k = 0; k < 8; ++k) {
          double loss_output[3] = {0.0, 0.0, 0.0};
          loss_function_->Evaluate(sq_r_buf[k], loss_output);
          loss_buf[k] = loss_output[0];
          weight_buf[k] = loss_output[1];
        }
        loss__ = simd::ScalarF(loss_buf);
        weight__ = simd::ScalarF(weight_buf);
      }

      // g(i) += (J(0,i)*r(0) + J(1,i)*r(1) + J(2,i)*r(2))
      for (int k = 0; k < 6; ++k)
        gradient__[k] += (weight__ * (J__[0][k] * r__[0] + J__[1][k] * r__[1] +
                                      J__[2][k] * r__[2]));

      // H(i,j) = sum_{k} w * J(k,i) * J(k,j)
      for (int ii = 0; ii < 6; ++ii) {
        for (int jj = ii; jj < 6; ++jj) {
          hessian__[ii][jj] +=
              (weight__ * (J__[0][ii] * J__[0][jj] + J__[1][ii] * J__[1][jj] +
                           J__[2][ii] * J__[2][jj]));
        }
      }

      cost__ += loss__;
    }
    float buf[8];
    cost__.StoreData(buf);
    float cost = 0.0;
    cost +=
        (buf[0] + buf[1] + buf[2] + buf[3] + buf[4] + buf[5] + buf[6] + buf[7]);

    Mat6x6 hessian{Mat6x6::Zero()};
    Vec6 gradient{Vec6::Zero()};
    for (int ii = 0; ii < 6; ++ii) {
      gradient__[ii].StoreData(buf);
      gradient(ii) += (buf[0] + buf[1] + buf[2] + buf[3] + buf[4] + buf[5] +
                       buf[6] + buf[7]);
      for (int jj = ii; jj < 6; ++jj) {
        hessian__[ii][jj].StoreData(buf);
        hessian(ii, jj) += (buf[0] + buf[1] + buf[2] + buf[3] + buf[4] +
                            buf[5] + buf[6] + buf[7]);
      }
    }

    // Reflect the hessian
    ReflectHessian(&hessian);

    // Damping hessian
    for (int k = 0; k < 6; k++) hessian(k, k) *= 1.0 + lambda;

    // Compute the step
    const Vec6 update_step = hessian.ldlt().solve(-gradient);

    // Update the pose
    const Vec3 delta_t = update_step.block<3, 1>(0, 0);
    const Vec3 delta_R = update_step.block<3, 1>(3, 0);
    optimized_translation += delta_t.cast<float>();
    optimized_orientation *= ComputeQuaternion(delta_R).cast<float>();
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

  pose->translation() = optimized_translation.cast<double>();
  pose->linear() = optimized_orientation.toRotationMatrix().cast<double>();

  return true;
}

bool MahalanobisDistanceMinimizerAnalyticSIMD::SolveFloatAligned(
    const Options& options, const std::vector<Correspondence>& correspondences,
    Pose* pose) {
  using Vec3f = Eigen::Vector3f;
  using Mat3x3f = Eigen::Matrix3f;
  using Orientationf = Eigen::Quaternionf;

  AlignedBuffer buf(correspondences.size());
  for (size_t index = 0; index < correspondences.size(); ++index) {
    const auto& corr = correspondences.at(index);
    buf.x[index] = corr.point.x();
    buf.y[index] = corr.point.y();
    buf.z[index] = corr.point.z();
    buf.mx[index] = corr.ndt.mean.x();
    buf.my[index] = corr.ndt.mean.y();
    buf.mz[index] = corr.ndt.mean.z();
    for (int i = 0; i < 3; ++i) {
      for (int j = 0; j < 3; ++j) {
        buf.sqrt_info[i][j][index] = corr.ndt.sqrt_information(i, j);
      }
    }
  }

  constexpr float min_lambda = 1e-6f;
  constexpr float max_lambda = 1e-2f;

  const Pose initial_pose = *pose;

  Vec3f optimized_translation{initial_pose.translation().cast<float>()};
  Orientationf optimized_orientation(initial_pose.rotation().cast<float>());

  float lambda = 0.001;
  float previous_cost = std::numeric_limits<float>::max();
  int iteration = 0;
  for (; iteration < options.max_iterations; ++iteration) {
    const Mat3x3f opt_R = optimized_orientation.toRotationMatrix();
    const Vec3f opt_t = optimized_translation;

    simd::ScalarF R__[3][3];
    simd::ScalarF t__[3];
    for (int row = 0; row < 3; ++row) {
      t__[row] = simd::ScalarF(opt_t(row));
      for (int col = 0; col < 3; ++col)
        R__[row][col] = simd::ScalarF(opt_R(row, col));
    }

    simd::ScalarF gradient__[6];
    simd::ScalarF hessian__[6][6];
    simd::ScalarF cost__;
    const size_t stride = simd::ScalarF::GetDataStep();
    const int num_stride = correspondences.size() / stride;
    for (size_t point_idx = 0; point_idx < num_stride * stride;
         point_idx += stride) {
      simd::ScalarF p__[3];
      p__[0] = simd::ScalarF(buf.x + point_idx);
      p__[1] = simd::ScalarF(buf.y + point_idx);
      p__[2] = simd::ScalarF(buf.z + point_idx);

      simd::ScalarF mu__[3];
      mu__[0] = simd::ScalarF(buf.mx + point_idx);
      mu__[1] = simd::ScalarF(buf.my + point_idx);
      mu__[2] = simd::ScalarF(buf.mz + point_idx);

      simd::ScalarF sqrt_info__[3][3];
      sqrt_info__[0][0] = simd::ScalarF(buf.sqrt_info[0][0] + point_idx);
      sqrt_info__[0][1] = simd::ScalarF(buf.sqrt_info[0][1] + point_idx);
      sqrt_info__[0][2] = simd::ScalarF(buf.sqrt_info[0][2] + point_idx);
      sqrt_info__[1][0] = simd::ScalarF(buf.sqrt_info[1][0] + point_idx);
      sqrt_info__[1][1] = simd::ScalarF(buf.sqrt_info[1][1] + point_idx);
      sqrt_info__[1][2] = simd::ScalarF(buf.sqrt_info[1][2] + point_idx);
      sqrt_info__[2][0] = simd::ScalarF(buf.sqrt_info[2][0] + point_idx);
      sqrt_info__[2][1] = simd::ScalarF(buf.sqrt_info[2][1] + point_idx);
      sqrt_info__[2][2] = simd::ScalarF(buf.sqrt_info[2][2] + point_idx);

      // clang-format off
    // pw = R*p + t
    simd::ScalarF pw__[3];
    pw__[0] = R__[0][0] * p__[0] + R__[0][1] * p__[1] + R__[0][2] * p__[2] + t__[0];
    pw__[1] = R__[1][0] * p__[0] + R__[1][1] * p__[1] + R__[1][2] * p__[2] + t__[1];
    pw__[2] = R__[2][0] * p__[0] + R__[2][1] * p__[1] + R__[2][2] * p__[2] + t__[2];

    // e_i = pw - mean
    simd::ScalarF e__[3];
    e__[0] = pw__[0] - mu__[0];
    e__[1] = pw__[1] - mu__[1];
    e__[2] = pw__[2] - mu__[2];

    // r = sqrt_info * e
    simd::ScalarF r__[3];
    r__[0] = sqrt_info__[0][0] * e__[0] + sqrt_info__[0][1] * e__[1] + sqrt_info__[0][2] * e__[2];
    r__[1] = sqrt_info__[1][0] * e__[0] + sqrt_info__[1][1] * e__[1] + sqrt_info__[1][2] * e__[2];
    r__[2] = sqrt_info__[2][0] * e__[0] + sqrt_info__[2][1] * e__[1] + sqrt_info__[2][2] * e__[2];

    simd::ScalarF J__[3][6];
    J__[0][0] = sqrt_info__[0][0]; J__[0][1] = sqrt_info__[0][1]; J__[0][2] = sqrt_info__[0][2];
    J__[1][0] = sqrt_info__[1][0]; J__[1][1] = sqrt_info__[1][1]; J__[1][2] = sqrt_info__[1][2];
    J__[2][0] = sqrt_info__[2][0]; J__[2][1] = sqrt_info__[2][1]; J__[2][2] = sqrt_info__[2][2];

    simd::ScalarF minus_R_skewp__[3][3];
    minus_R_skewp__[0][0] =  R__[0][2] * p__[1] - R__[0][1] * p__[2];
    minus_R_skewp__[1][0] =  R__[1][2] * p__[1] - R__[1][1] * p__[2];
    minus_R_skewp__[2][0] =  R__[2][2] * p__[1] - R__[2][1] * p__[2];

    minus_R_skewp__[0][1] =  R__[0][0] * p__[2] - R__[0][2] * p__[0];
    minus_R_skewp__[1][1] =  R__[1][0] * p__[2] - R__[1][2] * p__[0];
    minus_R_skewp__[2][1] =  R__[2][0] * p__[2] - R__[2][2] * p__[0];
    
    minus_R_skewp__[0][2] =  R__[0][1] * p__[0] - R__[0][0] * p__[1];
    minus_R_skewp__[1][2] =  R__[1][1] * p__[0] - R__[1][0] * p__[1];
    minus_R_skewp__[2][2] =  R__[2][1] * p__[0] - R__[2][0] * p__[1];

    J__[0][3] = sqrt_info__[0][0] * minus_R_skewp__[0][0] + sqrt_info__[0][1] * minus_R_skewp__[1][0] + sqrt_info__[0][2] * minus_R_skewp__[2][0];
    J__[0][4] = sqrt_info__[0][0] * minus_R_skewp__[0][1] + sqrt_info__[0][1] * minus_R_skewp__[1][1] + sqrt_info__[0][2] * minus_R_skewp__[2][1];
    J__[0][5] = sqrt_info__[0][0] * minus_R_skewp__[0][2] + sqrt_info__[0][1] * minus_R_skewp__[1][2] + sqrt_info__[0][2] * minus_R_skewp__[2][2];
    J__[1][3] = sqrt_info__[1][0] * minus_R_skewp__[0][0] + sqrt_info__[1][1] * minus_R_skewp__[1][0] + sqrt_info__[1][2] * minus_R_skewp__[2][0];
    J__[1][4] = sqrt_info__[1][0] * minus_R_skewp__[0][1] + sqrt_info__[1][1] * minus_R_skewp__[1][1] + sqrt_info__[1][2] * minus_R_skewp__[2][1];
    J__[1][5] = sqrt_info__[1][0] * minus_R_skewp__[0][2] + sqrt_info__[1][1] * minus_R_skewp__[1][2] + sqrt_info__[1][2] * minus_R_skewp__[2][2];
    J__[2][3] = sqrt_info__[2][0] * minus_R_skewp__[0][0] + sqrt_info__[2][1] * minus_R_skewp__[1][0] + sqrt_info__[2][2] * minus_R_skewp__[2][0];
    J__[2][4] = sqrt_info__[2][0] * minus_R_skewp__[0][1] + sqrt_info__[2][1] * minus_R_skewp__[1][1] + sqrt_info__[2][2] * minus_R_skewp__[2][1];
    J__[2][5] = sqrt_info__[2][0] * minus_R_skewp__[0][2] + sqrt_info__[2][1] * minus_R_skewp__[1][2] + sqrt_info__[2][2] * minus_R_skewp__[2][2];
      // clang-format on

      // Compute loss and weight,
      // and add the local gradient and hessian to the global ones
      simd::ScalarF sq_r__ =
          r__[0] * r__[0] + r__[1] * r__[1] + r__[2] * r__[2];
      simd::ScalarF loss__(sq_r__);
      simd::ScalarF weight__(1.0);
      if (loss_function_ != nullptr) {
        float sq_r_buf[8];
        sq_r__.StoreData(sq_r_buf);
        float loss_buf[8];
        float weight_buf[8];
        for (int k = 0; k < 8; ++k) {
          double loss_output[3] = {0.0, 0.0, 0.0};
          loss_function_->Evaluate(sq_r_buf[k], loss_output);
          loss_buf[k] = loss_output[0];
          weight_buf[k] = loss_output[1];
        }
        loss__ = simd::ScalarF(loss_buf);
        weight__ = simd::ScalarF(weight_buf);
      }

      // g(i) += (J(0,i)*r(0) + J(1,i)*r(1) + J(2,i)*r(2))
      for (int k = 0; k < 6; ++k)
        gradient__[k] += (weight__ * (J__[0][k] * r__[0] + J__[1][k] * r__[1] +
                                      J__[2][k] * r__[2]));

      // H(i,j) = sum_{k} w * J(k,i) * J(k,j)
      for (int ii = 0; ii < 6; ++ii) {
        for (int jj = ii; jj < 6; ++jj) {
          hessian__[ii][jj] +=
              (weight__ * (J__[0][ii] * J__[0][jj] + J__[1][ii] * J__[1][jj] +
                           J__[2][ii] * J__[2][jj]));
        }
      }

      cost__ += loss__;
    }
    float buf[8];
    cost__.StoreData(buf);
    float cost = 0.0;
    cost +=
        (buf[0] + buf[1] + buf[2] + buf[3] + buf[4] + buf[5] + buf[6] + buf[7]);

    Mat6x6 hessian{Mat6x6::Zero()};
    Vec6 gradient{Vec6::Zero()};
    for (int ii = 0; ii < 6; ++ii) {
      gradient__[ii].StoreData(buf);
      gradient(ii) += (buf[0] + buf[1] + buf[2] + buf[3] + buf[4] + buf[5] +
                       buf[6] + buf[7]);
      for (int jj = ii; jj < 6; ++jj) {
        hessian__[ii][jj].StoreData(buf);
        hessian(ii, jj) += (buf[0] + buf[1] + buf[2] + buf[3] + buf[4] +
                            buf[5] + buf[6] + buf[7]);
      }
    }

    // Reflect the hessian
    ReflectHessian(&hessian);

    // Damping hessian
    for (int k = 0; k < 6; k++) hessian(k, k) *= 1.0 + lambda;

    // Compute the step
    const Vec6 update_step = hessian.ldlt().solve(-gradient);

    // Update the pose
    const Vec3 delta_t = update_step.block<3, 1>(0, 0);
    const Vec3 delta_R = update_step.block<3, 1>(3, 0);
    optimized_translation += delta_t.cast<float>();
    optimized_orientation *= ComputeQuaternion(delta_R).cast<float>();
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

  pose->translation() = optimized_translation.cast<double>();
  pose->linear() = optimized_orientation.toRotationMatrix().cast<double>();

  return true;
}

bool MahalanobisDistanceMinimizerAnalyticSIMD::SolveFloatIntrinsicAligned(
    const Options& options, const std::vector<Correspondence>& correspondences,
    Pose* pose) {
  using Vec3f = Eigen::Vector3f;
  using Mat3x3f = Eigen::Matrix3f;
  using Orientationf = Eigen::Quaternionf;

  AlignedBuffer aligned_buf(correspondences.size());
  for (size_t index = 0; index < correspondences.size(); ++index) {
    const auto& corr = correspondences.at(index);
    aligned_buf.x[index] = corr.point.x();
    aligned_buf.y[index] = corr.point.y();
    aligned_buf.z[index] = corr.point.z();
    aligned_buf.mx[index] = corr.ndt.mean.x();
    aligned_buf.my[index] = corr.ndt.mean.y();
    aligned_buf.mz[index] = corr.ndt.mean.z();
    for (int i = 0; i < 3; ++i) {
      for (int j = 0; j < 3; ++j) {
        aligned_buf.sqrt_info[i][j][index] = corr.ndt.sqrt_information(i, j);
      }
    }
  }

  constexpr float min_lambda = 1e-6f;
  constexpr float max_lambda = 1e-2f;

  const Pose initial_pose = *pose;

  Vec3f optimized_translation{initial_pose.translation().cast<float>()};
  Orientationf optimized_orientation(initial_pose.rotation().cast<float>());

  float lambda = 0.001;
  float previous_cost = std::numeric_limits<float>::max();
  int iteration = 0;
  for (; iteration < options.max_iterations; ++iteration) {
    const Mat3x3f opt_R = optimized_orientation.toRotationMatrix();
    const Vec3f opt_t = optimized_translation;

    __m256 R__[3][3];
    __m256 t__[3];
    for (int row = 0; row < 3; ++row) {
      t__[row] = _mm256_set1_ps(opt_t(row));
      for (int col = 0; col < 3; ++col)
        R__[row][col] = _mm256_set1_ps(opt_R(row, col));
    }

    __m256 gradient__[6];
    __m256 hessian__[6][6];
    __m256 cost__ = _mm256_set1_ps(0.0);
    for (int ii = 0; ii < 6; ++ii) {
      gradient__[ii] = _mm256_set1_ps(0.0);
      for (int jj = 0; jj < 6; ++jj) hessian__[ii][jj] = _mm256_set1_ps(0.0);
    }
    const size_t stride = 8;
    const int num_stride = correspondences.size() / stride;
    for (size_t point_idx = 0; point_idx < num_stride * stride;
         point_idx += stride) {
      __m256 p__[3];
      p__[0] = _mm256_load_ps(aligned_buf.x + point_idx);
      p__[1] = _mm256_load_ps(aligned_buf.y + point_idx);
      p__[2] = _mm256_load_ps(aligned_buf.z + point_idx);

      __m256 mu__[3];
      mu__[0] = _mm256_load_ps(aligned_buf.mx + point_idx);
      mu__[1] = _mm256_load_ps(aligned_buf.my + point_idx);
      mu__[2] = _mm256_load_ps(aligned_buf.mz + point_idx);

      __m256 sqrt_info__[3][3];
      sqrt_info__[0][0] =
          _mm256_load_ps(aligned_buf.sqrt_info[0][0] + point_idx);
      sqrt_info__[0][1] =
          _mm256_load_ps(aligned_buf.sqrt_info[0][1] + point_idx);
      sqrt_info__[0][2] =
          _mm256_load_ps(aligned_buf.sqrt_info[0][2] + point_idx);
      sqrt_info__[1][0] =
          _mm256_load_ps(aligned_buf.sqrt_info[1][0] + point_idx);
      sqrt_info__[1][1] =
          _mm256_load_ps(aligned_buf.sqrt_info[1][1] + point_idx);
      sqrt_info__[1][2] =
          _mm256_load_ps(aligned_buf.sqrt_info[1][2] + point_idx);
      sqrt_info__[2][0] =
          _mm256_load_ps(aligned_buf.sqrt_info[2][0] + point_idx);
      sqrt_info__[2][1] =
          _mm256_load_ps(aligned_buf.sqrt_info[2][1] + point_idx);
      sqrt_info__[2][2] =
          _mm256_load_ps(aligned_buf.sqrt_info[2][2] + point_idx);

      // clang-format off
      // pw = R*p + t
      __m256 pw__[3];
      pw__[0] = _mm256_fmadd_ps(R__[0][0], p__[0], _mm256_fmadd_ps(R__[0][1], p__[1], _mm256_fmadd_ps(R__[0][2],p__[2], t__[0])));
      pw__[1] = _mm256_fmadd_ps(R__[1][0], p__[0], _mm256_fmadd_ps(R__[1][1], p__[1], _mm256_fmadd_ps(R__[1][2],p__[2], t__[1])));
      pw__[2] = _mm256_fmadd_ps(R__[2][0], p__[0], _mm256_fmadd_ps(R__[2][1], p__[1], _mm256_fmadd_ps(R__[2][2],p__[2], t__[2])));

      // e_i = pw - mean
      __m256 e__[3];
      e__[0] = _mm256_sub_ps(pw__[0], mu__[0]);
      e__[1] = _mm256_sub_ps(pw__[1], mu__[1]);
      e__[2] = _mm256_sub_ps(pw__[2], mu__[2]);

      // r = sqrt_info * e
      __m256 r__[3];
      r__[0] = _mm256_fmadd_ps(sqrt_info__[0][0], e__[0], _mm256_fmadd_ps(sqrt_info__[0][1], e__[1], _mm256_mul_ps(sqrt_info__[0][2],e__[2])));
      r__[1] = _mm256_fmadd_ps(sqrt_info__[1][0], e__[0], _mm256_fmadd_ps(sqrt_info__[1][1], e__[1], _mm256_mul_ps(sqrt_info__[1][2],e__[2])));
      r__[2] = _mm256_fmadd_ps(sqrt_info__[2][0], e__[0], _mm256_fmadd_ps(sqrt_info__[2][1], e__[1], _mm256_mul_ps(sqrt_info__[2][2],e__[2])));

      __m256 J__[3][6];
      J__[0][0] = sqrt_info__[0][0]; J__[0][1] = sqrt_info__[0][1]; J__[0][2] = sqrt_info__[0][2];
      J__[1][0] = sqrt_info__[1][0]; J__[1][1] = sqrt_info__[1][1]; J__[1][2] = sqrt_info__[1][2];
      J__[2][0] = sqrt_info__[2][0]; J__[2][1] = sqrt_info__[2][1]; J__[2][2] = sqrt_info__[2][2];

      __m256 minus_R_skewp__[3][3];
      minus_R_skewp__[0][0] = _mm256_fmsub_ps(R__[0][2], p__[1], _mm256_mul_ps(R__[0][1], p__[2]));
      minus_R_skewp__[1][0] = _mm256_fmsub_ps(R__[1][2], p__[1], _mm256_mul_ps(R__[1][1], p__[2]));
      minus_R_skewp__[2][0] = _mm256_fmsub_ps(R__[2][2], p__[1], _mm256_mul_ps(R__[2][1], p__[2]));

      minus_R_skewp__[0][1] = _mm256_fmsub_ps(R__[0][0], p__[2], _mm256_mul_ps(R__[0][2], p__[0]));
      minus_R_skewp__[1][1] = _mm256_fmsub_ps(R__[1][0], p__[2], _mm256_mul_ps(R__[1][2], p__[0]));
      minus_R_skewp__[2][1] = _mm256_fmsub_ps(R__[2][0], p__[2], _mm256_mul_ps(R__[2][2], p__[0]));

      minus_R_skewp__[0][2] = _mm256_fmsub_ps(R__[0][1], p__[0], _mm256_mul_ps(R__[0][0], p__[1]));
      minus_R_skewp__[1][2] = _mm256_fmsub_ps(R__[1][1], p__[0], _mm256_mul_ps(R__[1][0], p__[1]));
      minus_R_skewp__[2][2] = _mm256_fmsub_ps(R__[2][1], p__[0], _mm256_mul_ps(R__[2][0], p__[1]));

      J__[0][3] = _mm256_fmadd_ps(sqrt_info__[0][0], minus_R_skewp__[0][0], _mm256_fmadd_ps(sqrt_info__[0][1],minus_R_skewp__[1][0], _mm256_mul_ps(sqrt_info__[0][2],minus_R_skewp__[2][0])));
      J__[0][4] = _mm256_fmadd_ps(sqrt_info__[0][0], minus_R_skewp__[0][1], _mm256_fmadd_ps(sqrt_info__[0][1],minus_R_skewp__[1][1], _mm256_mul_ps(sqrt_info__[0][2],minus_R_skewp__[2][1])));
      J__[0][5] = _mm256_fmadd_ps(sqrt_info__[0][0], minus_R_skewp__[0][2], _mm256_fmadd_ps(sqrt_info__[0][1],minus_R_skewp__[1][2], _mm256_mul_ps(sqrt_info__[0][2],minus_R_skewp__[2][2])));
      J__[1][3] = _mm256_fmadd_ps(sqrt_info__[1][0], minus_R_skewp__[0][0], _mm256_fmadd_ps(sqrt_info__[1][1],minus_R_skewp__[1][0], _mm256_mul_ps(sqrt_info__[1][2],minus_R_skewp__[2][0])));
      J__[1][4] = _mm256_fmadd_ps(sqrt_info__[1][0], minus_R_skewp__[0][1], _mm256_fmadd_ps(sqrt_info__[1][1],minus_R_skewp__[1][1], _mm256_mul_ps(sqrt_info__[1][2],minus_R_skewp__[2][1])));
      J__[1][5] = _mm256_fmadd_ps(sqrt_info__[1][0], minus_R_skewp__[0][2], _mm256_fmadd_ps(sqrt_info__[1][1],minus_R_skewp__[1][2], _mm256_mul_ps(sqrt_info__[1][2],minus_R_skewp__[2][2])));
      J__[2][3] = _mm256_fmadd_ps(sqrt_info__[2][0], minus_R_skewp__[0][0], _mm256_fmadd_ps(sqrt_info__[2][1],minus_R_skewp__[1][0], _mm256_mul_ps(sqrt_info__[2][2],minus_R_skewp__[2][0])));
      J__[2][4] = _mm256_fmadd_ps(sqrt_info__[2][0], minus_R_skewp__[0][1], _mm256_fmadd_ps(sqrt_info__[2][1],minus_R_skewp__[1][1], _mm256_mul_ps(sqrt_info__[2][2],minus_R_skewp__[2][1])));
      J__[2][5] = _mm256_fmadd_ps(sqrt_info__[2][0], minus_R_skewp__[0][2], _mm256_fmadd_ps(sqrt_info__[2][1],minus_R_skewp__[1][2], _mm256_mul_ps(sqrt_info__[2][2],minus_R_skewp__[2][2])));
      // clang-format on

      // Compute loss and weight,
      // and add the local gradient and hessian to the global ones
      __m256 sq_r__ = _mm256_fmadd_ps(
          r__[0], r__[0],
          _mm256_fmadd_ps(r__[1], r__[1], _mm256_mul_ps(r__[2], r__[2])));
      __m256 loss__ = sq_r__;
      __m256 weight__ = _mm256_set1_ps(1.0);
      if (loss_function_ != nullptr) {
        float sq_r_buf[8];
        _mm256_store_ps(sq_r_buf, sq_r__);
        float loss_buf[8];
        float weight_buf[8];
        for (int k = 0; k < 8; ++k) {
          double loss_output[3] = {0.0, 0.0, 0.0};
          loss_function_->Evaluate(sq_r_buf[k], loss_output);
          loss_buf[k] = loss_output[0];
          weight_buf[k] = loss_output[1];
        }
        loss__ = _mm256_load_ps(loss_buf);
        weight__ = _mm256_load_ps(weight_buf);
      }

      // g(i) += (J(0,i)*r(0) + J(1,i)*r(1) + J(2,i)*r(2))
      for (int k = 0; k < 6; ++k) {
        gradient__[k] = _mm256_add_ps(
            gradient__[k],
            _mm256_mul_ps(
                weight__,
                _mm256_fmadd_ps(
                    J__[0][k], r__[0],
                    _mm256_fmadd_ps(J__[1][k], r__[1],
                                    _mm256_mul_ps(J__[2][k], r__[2])))));
      }

      // H(i,j) = sum_{k} w * J(k,i) * J(k,j)
      for (int ii = 0; ii < 6; ++ii) {
        for (int jj = ii; jj < 6; ++jj) {
          hessian__[ii][jj] = _mm256_add_ps(
              hessian__[ii][jj],
              _mm256_mul_ps(
                  weight__,
                  _mm256_fmadd_ps(
                      J__[0][ii], J__[0][jj],
                      _mm256_fmadd_ps(J__[1][ii], J__[1][jj],
                                      _mm256_mul_ps(J__[2][ii], J__[2][jj])))));
        }
      }

      cost__ = _mm256_add_ps(cost__, loss__);
    }
    float buf[8];
    _mm256_store_ps(buf, cost__);
    float cost = 0.0;
    cost +=
        (buf[0] + buf[1] + buf[2] + buf[3] + buf[4] + buf[5] + buf[6] + buf[7]);

    Mat6x6 hessian{Mat6x6::Zero()};
    Vec6 gradient{Vec6::Zero()};
    for (int ii = 0; ii < 6; ++ii) {
      _mm256_store_ps(buf, gradient__[ii]);
      gradient(ii) += (buf[0] + buf[1] + buf[2] + buf[3] + buf[4] + buf[5] +
                       buf[6] + buf[7]);
      for (int jj = ii; jj < 6; ++jj) {
        _mm256_store_ps(buf, hessian__[ii][jj]);
        hessian(ii, jj) += (buf[0] + buf[1] + buf[2] + buf[3] + buf[4] +
                            buf[5] + buf[6] + buf[7]);
      }
    }

    // Reflect the hessian
    ReflectHessian(&hessian);

    // Damping hessian
    for (int k = 0; k < 6; k++) hessian(k, k) *= 1.0 + lambda;

    // Compute the step
    const Vec6 update_step = hessian.ldlt().solve(-gradient);

    // Update the pose
    const Vec3 delta_t = update_step.block<3, 1>(0, 0);
    const Vec3 delta_R = update_step.block<3, 1>(3, 0);
    optimized_translation += delta_t.cast<float>();
    optimized_orientation *= ComputeQuaternion(delta_R).cast<float>();
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

  pose->translation() = optimized_translation.cast<double>();
  pose->linear() = optimized_orientation.toRotationMatrix().cast<double>();

  return true;
}

void MahalanobisDistanceMinimizerAnalyticSIMD::ReflectHessian(Mat6x6* hessian) {
  auto& H = *hessian;
  for (int row = 0; row < 6; ++row) {
    for (int col = row + 1; col < 6; ++col) {
      H(col, row) = H(row, col);
    }
  }
}

}  // namespace mahalanobis_distance_minimizer
}  // namespace nonlinear_optimizer