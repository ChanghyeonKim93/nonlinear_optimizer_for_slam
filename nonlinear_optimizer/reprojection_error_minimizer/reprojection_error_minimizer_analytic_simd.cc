#include "nonlinear_optimizer/reprojection_error_minimizer/reprojection_error_minimizer_analytic_simd.h"

#include <iostream>

namespace nonlinear_optimizer {
namespace reprojection_error_minimizer {

ReprojectionErrorMinimizerAnalyticSIMD::
    ReprojectionErrorMinimizerAnalyticSIMD() {}

ReprojectionErrorMinimizerAnalyticSIMD::
    ~ReprojectionErrorMinimizerAnalyticSIMD() {}

bool ReprojectionErrorMinimizerAnalyticSIMD::Solve(
    const Options& options, const std::vector<Correspondence>& correspondences,
    const CameraIntrinsics& camera_intrinsics, Pose* pose) {
  using Mat3x3f = Eigen::Matrix3f;

  AlignedBuffer abuf(correspondences.size());
  for (size_t index = 0; index < correspondences.size(); ++index) {
    const auto& corr = correspondences.at(index);
    abuf.x[index] = corr.local_point.x();
    abuf.y[index] = corr.local_point.y();
    abuf.z[index] = corr.local_point.z();
    abuf.px[index] = corr.matched_pixel.x();
    abuf.py[index] = corr.matched_pixel.y();
  }

  simd::ScalarF inv_fx__(1.0f / camera_intrinsics.fx);
  simd::ScalarF inv_fy__(1.0f / camera_intrinsics.fy);
  simd::ScalarF cx__(camera_intrinsics.cx);
  simd::ScalarF cy__(camera_intrinsics.cy);

  constexpr double min_lambda = 1e-6;
  constexpr double max_lambda = 1e-2;

  const Pose initial_pose = *pose;

  Vec3 optimized_translation{initial_pose.translation()};
  Orientation optimized_orientation(initial_pose.rotation());

  double lambda = 0.001;
  double previous_cost = std::numeric_limits<double>::max();
  int iteration = 0;
  for (; iteration < options.max_iterations; ++iteration) {
    simd::MatrixF<3, 3> R__(
        optimized_orientation.toRotationMatrix().cast<float>());
    simd::VectorF<3> t__(optimized_translation.cast<float>());

    simd::VectorF<6> gradient__(Eigen::Matrix<float, 6, 1>::Zero());
    simd::MatrixF<6, 6> hessian__(Eigen::Matrix<float, 6, 6>::Zero());
    simd::ScalarF cost__(0.0);
    const size_t stride = simd::ScalarF::GetDataStep();
    const int num_stride = correspondences.size() / stride;
    for (size_t point_idx = 0; point_idx < num_stride * stride;
         point_idx += stride) {
      simd::VectorF<3> X__(
          {abuf.x + point_idx, abuf.y + point_idx, abuf.z + point_idx});
      simd::VectorF<2> matched_pixel__(
          std::vector<float*>{abuf.px + point_idx, abuf.py + point_idx});

      // Xw = R*X + t
      const simd::VectorF<3> Xw__ = R__ * X__ + t__;

      // Check if any element in Xw__(2) is less than 0
      simd::ScalarF is_nonzero__ = Xw__(2) > 0.0f;

      simd::VectorF<2> r__;
      simd::ScalarF inv_zw__ = simd::ScalarF(1.0f) / Xw__(2);
      r__(0) = Xw__(0) * inv_zw__ - inv_fx__ * (matched_pixel__(0) - cx__);
      r__(1) = Xw__(1) * inv_zw__ - inv_fy__ * (matched_pixel__(1) - cy__);

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
        for (int k = 0; k < _SIMD_DATA_STEP_FLOAT; ++k) {
          double loss_output[3] = {0.0, 0.0, 0.0};
          loss_function_->Evaluate(sq_r_buf[k], loss_output);
          loss_buf[k] = loss_output[0];
          weight_buf[k] = loss_output[1];
        }
        loss__ = simd::ScalarF(loss_buf);
        weight__ = simd::ScalarF(weight_buf);
      }
      weight__ *= is_nonzero__;

      simd::MatrixF<3, 3> m_R_skewX__(Mat3x3f::Zero());
      m_R_skewX__(0, 0) = R__(0, 2) * X__(1) - R__(0, 1) * X__(2);
      m_R_skewX__(1, 0) = R__(1, 2) * X__(1) - R__(1, 1) * X__(2);
      m_R_skewX__(2, 0) = R__(2, 2) * X__(1) - R__(2, 1) * X__(2);
      m_R_skewX__(0, 1) = R__(0, 0) * X__(2) - R__(0, 2) * X__(0);
      m_R_skewX__(1, 1) = R__(1, 0) * X__(2) - R__(1, 2) * X__(0);
      m_R_skewX__(2, 1) = R__(2, 0) * X__(2) - R__(2, 2) * X__(0);
      m_R_skewX__(0, 2) = R__(0, 1) * X__(0) - R__(0, 0) * X__(1);
      m_R_skewX__(1, 2) = R__(1, 1) * X__(0) - R__(1, 0) * X__(1);
      m_R_skewX__(2, 2) = R__(2, 1) * X__(0) - R__(2, 0) * X__(1);

      simd::MatrixF<2, 6> J__(Eigen::Matrix<float, 2, 6>::Zero());
      const simd::ScalarF inv_zwzw__ = inv_zw__ * inv_zw__;
      const simd::ScalarF xw_inv_zwzw__ = Xw__(0) * inv_zwzw__;
      const simd::ScalarF yw_inv_zwzw__ = Xw__(1) * inv_zwzw__;
      J__(0, 0) = inv_zw__;
      J__(0, 2) = -xw_inv_zwzw__;
      J__(1, 1) = inv_zw__;
      J__(1, 2) = -yw_inv_zwzw__;
      J__(0, 3) =
          inv_zw__ * m_R_skewX__(0, 0) - xw_inv_zwzw__ * m_R_skewX__(2, 0);
      J__(0, 4) =
          inv_zw__ * m_R_skewX__(0, 1) - xw_inv_zwzw__ * m_R_skewX__(2, 1);
      J__(0, 5) =
          inv_zw__ * m_R_skewX__(0, 2) - xw_inv_zwzw__ * m_R_skewX__(2, 2);
      J__(1, 3) =
          inv_zw__ * m_R_skewX__(1, 0) - yw_inv_zwzw__ * m_R_skewX__(2, 0);
      J__(1, 4) =
          inv_zw__ * m_R_skewX__(1, 1) - yw_inv_zwzw__ * m_R_skewX__(2, 1);
      J__(1, 5) =
          inv_zw__ * m_R_skewX__(1, 2) - yw_inv_zwzw__ * m_R_skewX__(2, 2);

      gradient__ += (J__.transpose() * r__) * weight__;

      // H(i,j) = sum_{k} w * J(k,i) * J(k,j)
      for (int ii = 0; ii < 6; ++ii) {
        for (int jj = ii; jj < 6; ++jj) {
          for (int kk = 0; kk < 2; ++kk) {
            hessian__(ii, jj) += (J__(kk, ii) * J__(kk, jj)) * weight__;
          }
        }
      }

      cost__ += loss__;
    }

    float buf[_SIMD_DATA_STEP_FLOAT];
    cost__.StoreData(buf);
    float cost = 0.0;
    for (int kk = 0; kk < _SIMD_DATA_STEP_FLOAT; ++kk) cost += buf[kk];

    Mat6x6 hessian{Mat6x6::Zero()};
    Vec6 gradient{Vec6::Zero()};
    for (int ii = 0; ii < 6; ++ii) {
      gradient__(ii).StoreData(buf);
      for (int kk = 0; kk < _SIMD_DATA_STEP_FLOAT; ++kk)
        gradient(ii) += buf[kk];
      for (int jj = ii; jj < 6; ++jj) {
        hessian__(ii, jj).StoreData(buf);
        for (int kk = 0; kk < _SIMD_DATA_STEP_FLOAT; ++kk)
          hessian(ii, jj) += buf[kk];
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

void ReprojectionErrorMinimizerAnalyticSIMD::ReflectHessian(Mat6x6* hessian) {
  auto& H = *hessian;
  for (int row = 0; row < 6; ++row)
    for (int col = row + 1; col < 6; ++col) H(col, row) = H(row, col);
}

}  // namespace reprojection_error_minimizer
}  // namespace nonlinear_optimizer