#include "nonlinear_optimizer/mahalanobis_distance_minimizer/mahalanobis_distance_minimizer_analytic_simd.h"

#include <iostream>

#include "simd_helper/simd_helper.h"

namespace nonlinear_optimizer {
namespace mahalanobis_distance_minimizer {

MahalanobisDistanceMinimizerAnalyticSIMD::
    MahalanobisDistanceMinimizerAnalyticSIMD() {}

MahalanobisDistanceMinimizerAnalyticSIMD::
    ~MahalanobisDistanceMinimizerAnalyticSIMD() {}

bool MahalanobisDistanceMinimizerAnalyticSIMD::Solve(
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
    simd::Matrix<3, 3> R__(opt_R);
    simd::Vector<3> t__(opt_t);

    simd::Vector<6> gradient__(Eigen::Matrix<float, 6, 1>::Zero());
    simd::Matrix<6, 6> hessian__(Eigen::Matrix<float, 6, 6>::Zero());
    simd::Scalar cost__(0.0);
    const size_t stride = simd::Scalar::data_stride;
    const int num_stride = correspondences.size() / stride;
    for (size_t point_idx = 0; point_idx < num_stride * stride;
         point_idx += stride) {
      simd::Vector<3> p__(
          {abuf.x + point_idx, abuf.y + point_idx, abuf.z + point_idx});
      simd::Vector<3> mu__(
          {abuf.mx + point_idx, abuf.my + point_idx, abuf.mz + point_idx});
      simd::Matrix<3, 3> sqrt_info__(
          {abuf.sqrt_info[0][0] + point_idx, abuf.sqrt_info[0][1] + point_idx,
           abuf.sqrt_info[0][2] + point_idx, abuf.sqrt_info[1][0] + point_idx,
           abuf.sqrt_info[1][1] + point_idx, abuf.sqrt_info[1][2] + point_idx,
           abuf.sqrt_info[2][0] + point_idx, abuf.sqrt_info[2][1] + point_idx,
           abuf.sqrt_info[2][2] + point_idx});
      // simd::Vector<3> p__(
      //     {correspondences.at(point_idx).point.cast<float>(),
      //      correspondences.at(point_idx + 1).point.cast<float>(),
      //      correspondences.at(point_idx + 2).point.cast<float>(),
      //      correspondences.at(point_idx + 3).point.cast<float>(),
      //      correspondences.at(point_idx + 4).point.cast<float>(),
      //      correspondences.at(point_idx + 5).point.cast<float>(),
      //      correspondences.at(point_idx + 6).point.cast<float>(),
      //      correspondences.at(point_idx + 7).point.cast<float>()});
      // simd::Vector<3> mu__(
      //     {correspondences.at(point_idx).ndt.mean.cast<float>(),
      //      correspondences.at(point_idx + 1).ndt.mean.cast<float>(),
      //      correspondences.at(point_idx + 2).ndt.mean.cast<float>(),
      //      correspondences.at(point_idx + 3).ndt.mean.cast<float>(),
      //      correspondences.at(point_idx + 4).ndt.mean.cast<float>(),
      //      correspondences.at(point_idx + 5).ndt.mean.cast<float>(),
      //      correspondences.at(point_idx + 6).ndt.mean.cast<float>(),
      //      correspondences.at(point_idx + 7).ndt.mean.cast<float>()});
      // simd::Matrix<3, 3> sqrt_info__(
      //     {correspondences.at(point_idx).ndt.sqrt_information.cast<float>(),
      //      correspondences.at(point_idx +
      //      1).ndt.sqrt_information.cast<float>(),
      //      correspondences.at(point_idx +
      //      2).ndt.sqrt_information.cast<float>(),
      //      correspondences.at(point_idx +
      //      3).ndt.sqrt_information.cast<float>(),
      //      correspondences.at(point_idx +
      //      4).ndt.sqrt_information.cast<float>(),
      //      correspondences.at(point_idx +
      //      5).ndt.sqrt_information.cast<float>(),
      //      correspondences.at(point_idx +
      //      6).ndt.sqrt_information.cast<float>(),
      //      correspondences.at(point_idx + 7)
      //          .ndt.sqrt_information.cast<float>()});

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
      J__(0, 0) = sqrt_info__(0, 0); J__(0, 1) = sqrt_info__(0, 1); J__(0, 2) = sqrt_info__(0, 2);
      J__(1, 0) = sqrt_info__(1, 0); J__(1, 1) = sqrt_info__(1, 1); J__(1, 2) = sqrt_info__(1, 2);
      J__(2, 0) = sqrt_info__(2, 0); J__(2, 1) = sqrt_info__(2, 1); J__(2, 2) = sqrt_info__(2, 2);
      J__(0, 3) = sqrt_info_minus_R_skewp_(0, 0); J__(0, 4) = sqrt_info_minus_R_skewp_(0, 1); J__(0, 5) = sqrt_info_minus_R_skewp_(0, 2);
      J__(1, 3) = sqrt_info_minus_R_skewp_(1, 0); J__(1, 4) = sqrt_info_minus_R_skewp_(1, 1); J__(1, 5) = sqrt_info_minus_R_skewp_(1, 2);
      J__(2, 3) = sqrt_info_minus_R_skewp_(2, 0); J__(2, 4) = sqrt_info_minus_R_skewp_(2, 1); J__(2, 5) = sqrt_info_minus_R_skewp_(2, 2);
      // clang-format on

      // Compute loss and weight,
      // and add the local gradient and hessian to the global ones
      simd::Scalar sq_r__ = r__.squaredNorm();
      simd::Scalar loss__(sq_r__);
      simd::Scalar weight__(1.0);
      if (loss_function_ != nullptr) {
        float sq_r_buf[8];
        sq_r__.StoreData(sq_r_buf);
        float loss_buf[8];
        float weight_buf[8];
        for (size_t k = 0; k < simd::Scalar::data_stride; ++k) {
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
    float buf[8];
    cost__.StoreData(buf);
    float cost = 0.0;
    for (size_t kk = 0; kk < simd::Scalar::data_stride; ++kk) cost += buf[kk];

    Mat6x6 hessian{Mat6x6::Zero()};
    Vec6 gradient{Vec6::Zero()};
    for (int ii = 0; ii < 6; ++ii) {
      gradient__(ii).StoreData(buf);
      for (size_t kk = 0; kk < simd::Scalar::data_stride; ++kk)
        gradient(ii) += buf[kk];
      for (int jj = ii; jj < 6; ++jj) {
        hessian__(ii, jj).StoreData(buf);
        for (size_t kk = 0; kk < simd::Scalar::data_stride; ++kk)
          hessian(ii, jj) += buf[kk];
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