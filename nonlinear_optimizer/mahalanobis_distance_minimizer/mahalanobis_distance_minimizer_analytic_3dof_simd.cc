#include "nonlinear_optimizer/mahalanobis_distance_minimizer/mahalanobis_distance_minimizer_analytic_3dof_simd.h"

#include <iostream>

namespace nonlinear_optimizer {
namespace mahalanobis_distance_minimizer {

MahalanobisDistanceMinimizerAnalytic3DOFSIMD::
    MahalanobisDistanceMinimizerAnalytic3DOFSIMD() {}

MahalanobisDistanceMinimizerAnalytic3DOFSIMD::
    ~MahalanobisDistanceMinimizerAnalytic3DOFSIMD() {}

bool MahalanobisDistanceMinimizerAnalytic3DOFSIMD::Solve(
    const Options& options, const std::vector<Correspondence>& correspondences,
    Pose* pose) {
  struct AlignedBuffer {
    float* sqrt_info[3][3] = {nullptr};
    float* x = {nullptr};
    float* y = {nullptr};
    float* z = {nullptr};
    float* mx = {nullptr};
    float* my = {nullptr};
    float* mz = {nullptr};
    AlignedBuffer(const size_t num_data) {
      for (int row = 0; row < 3; ++row)
        for (int col = 0; col < 3; ++col)
          sqrt_info[row][col] = simd::GetAlignedMemory<float>(num_data);
      x = simd::GetAlignedMemory<float>(num_data);
      y = simd::GetAlignedMemory<float>(num_data);
      z = simd::GetAlignedMemory<float>(num_data);
      mx = simd::GetAlignedMemory<float>(num_data);
      my = simd::GetAlignedMemory<float>(num_data);
      mz = simd::GetAlignedMemory<float>(num_data);
    }
    ~AlignedBuffer() {
      for (int row = 0; row < 3; ++row)
        for (int col = 0; col < 3; ++col)
          simd::FreeAlignedMemory<float>(sqrt_info[row][col]);
      simd::FreeAlignedMemory<float>(x);
      simd::FreeAlignedMemory<float>(y);
      simd::FreeAlignedMemory<float>(z);
      simd::FreeAlignedMemory<float>(mx);
      simd::FreeAlignedMemory<float>(my);
      simd::FreeAlignedMemory<float>(mz);
    }
  };

  AlignedBuffer abuf(correspondences.size());
  for (size_t index = 0; index < correspondences.size(); ++index) {
    const auto& corr = correspondences.at(index);
    abuf.x[index] = corr.point.x();
    abuf.y[index] = corr.point.y();
    abuf.z[index] = corr.point.z();
    abuf.mx[index] = corr.ndt.mean.x();
    abuf.my[index] = corr.ndt.mean.y();
    abuf.mz[index] = corr.ndt.mean.z();
    for (int i = 0; i < 3; ++i)
      for (int j = 0; j < 3; ++j)
        abuf.sqrt_info[i][j][index] = corr.ndt.sqrt_information(i, j);
  }

  constexpr float min_lambda = 1e-6;
  constexpr float max_lambda = 1e-2;

  const Pose initial_pose = *pose;

  // Reduce the 3D pose to 2D
  Pose2 optimized_pose{Pose2::Identity()};
  optimized_pose.translation() = initial_pose.translation().block<2, 1>(0, 0);
  optimized_pose.linear() = initial_pose.linear().block<2, 2>(0, 0);

  float lambda = 0.001f;
  float previous_cost = std::numeric_limits<float>::max();
  int iteration = 0;
  for (; iteration < options.max_iterations; ++iteration) {
    simd::Matrix<2, 2> R__(optimized_pose.rotation().cast<float>());
    simd::Vector<2> t__(optimized_pose.translation().cast<float>());

    simd::Vector<3> gradient__(Eigen::Matrix<float, 3, 1>::Zero());
    simd::Matrix<3, 3> hessian__(Eigen::Matrix<float, 3, 3>::Zero());
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

      // pw = R*p + t
      const simd::Matrix<2, 2>& A__ = sqrt_info__.block<2, 2>(0, 0);
      const simd::Matrix<1, 2>& c__ = sqrt_info__.block<1, 2>(2, 0);
      const simd::Vector<2>& u__ = p__.block<2, 1>(0, 0);
      const simd::Vector<2> u_warped__ = R__ * u__ + t__;

      simd::Vector<3> p_warped__{simd::Vector<3>::Zeros()};
      p_warped__(0) = u_warped__(0);
      p_warped__(1) = u_warped__(1);
      p_warped__(2) = p__(2);
      simd::Vector<3> e__ = p_warped__ - mu__;

      // r = sqrt_info * e
      const simd::Vector<3> r__ = sqrt_info__ * e__;

      // Compute Jacobian
      simd::Vector<2> R_skew_p__(simd::Vector<2>::Zeros());

      R_skew_p__(0) = -R__(0, 0) * u__(1) + R__(0, 1) * u__(0);
      R_skew_p__(1) = -R__(1, 0) * u__(1) + R__(1, 1) * u__(0);

      simd::Matrix<3, 3> J__;
      J__.block<2, 2>(0, 0) = A__;
      J__.block<2, 1>(0, 2) = A__ * R_skew_p__;
      J__.block<1, 2>(2, 0) = c__;
      J__.block<1, 1>(2, 2) = c__ * R_skew_p__;

      // Compute loss and weight,
      // and add the local gradient and hessian to the global ones
      simd::Scalar sq_r__ = r__.squaredNorm();
      simd::Scalar loss__(sq_r__);
      simd::Scalar weight__(1.0);
      if (loss_function_ != nullptr) {
        float sq_r_buf[simd::Scalar::data_stride];
        sq_r__.StoreData(sq_r_buf);
        float loss_buf[simd::Scalar::data_stride];
        float weight_buf[simd::Scalar::data_stride];
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
      for (int ii = 0; ii < 3; ++ii) {
        for (int jj = ii; jj < 3; ++jj) {
          hessian__(ii, jj) +=
              (weight__ * (J__(0, ii) * J__(0, jj) + J__(1, ii) * J__(1, jj) +
                           J__(2, ii) * J__(2, jj)));
        }
      }

      cost__ += loss__;
    }

    float buf[simd::Scalar::data_stride];
    cost__.StoreData(buf);
    double cost = 0.0;
    for (size_t i = 0; i < simd::Scalar::data_stride; ++i) cost += buf[i];

    Mat3x3 hessian{Mat3x3::Zero()};
    Vec3 gradient{Vec3::Zero()};
    for (int ii = 0; ii < 3; ++ii) {
      gradient__(ii).StoreData(buf);
      gradient(ii) += (buf[0] + buf[1] + buf[2] + buf[3] + buf[4] + buf[5] +
                       buf[6] + buf[7]);
      for (int jj = ii; jj < 3; ++jj) {
        hessian__(ii, jj).StoreData(buf);
        hessian(ii, jj) += (buf[0] + buf[1] + buf[2] + buf[3] + buf[4] +
                            buf[5] + buf[6] + buf[7]);
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

void MahalanobisDistanceMinimizerAnalytic3DOFSIMD::ReflectHessian(
    Mat3x3* hessian) {
  auto& H = *hessian;
  for (int row = 0; row < 3; ++row) {
    for (int col = row + 1; col < 3; ++col) {
      H(col, row) = H(row, col);
    }
  }
}

}  // namespace mahalanobis_distance_minimizer
}  // namespace nonlinear_optimizer