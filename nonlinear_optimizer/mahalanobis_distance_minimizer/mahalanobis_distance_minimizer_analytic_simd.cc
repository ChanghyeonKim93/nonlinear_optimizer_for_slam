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

  Vec3 optimized_translation{initial_pose.translation()};
  Orientation optimized_orientation(initial_pose.rotation());

  float lambda = 0.001f;
  float previous_cost = std::numeric_limits<float>::max();
  int iteration = 0;
  for (; iteration < options.max_iterations; ++iteration) {
    Mat6x6 hessian{Mat6x6::Zero()};
    Vec6 gradient{Vec6::Zero()};
    double cost = 0.0;

    const size_t stride = simd::Scalar::data_stride;
    const int num_stride = correspondences.size() / stride;
    if (multi_thread_executor_ == nullptr) {
      auto result = ComputeCostAndDerivatives(
          optimized_orientation.toRotationMatrix(), optimized_translation,
          &abuf, 0, num_stride * stride);
      gradient = result.gradient;
      hessian = result.hessian;
      cost = result.cost;
    } else {
      const auto data_stride = simd::Scalar::data_stride;  // 8
      const int num_threads =
          multi_thread_executor_->GetNumOfTotalThreads();  // 4
      const int num_batch = std::floor(num_stride / num_threads) * data_stride;
      std::vector<std::future<PartialResult>> partial_results;
      for (int i = 0; i < num_threads; ++i) {
        partial_results.emplace_back(multi_thread_executor_->Execute(
            &MahalanobisDistanceMinimizerAnalyticSIMD::
                ComputeCostAndDerivatives,
            this, optimized_orientation.toRotationMatrix(),
            optimized_translation, &abuf, i * num_batch,
            std::min((i + 1) * num_batch,
                     static_cast<int>(num_stride * data_stride))));
      }
      for (auto& future : partial_results) {
        const auto& result = future.get();
        gradient += result.gradient;
        AddHessianOnlyUpperTriangle(result.hessian, &hessian);
        cost += result.cost;
      }
    }

    // Reflect the hessian
    ReflectHessian(&hessian);

    // Damping hessian
    for (int k = 0; k < 6; k++) hessian(k, k) *= 1.0 + lambda;

    // Update pose
    const Vec6 update_step = hessian.ldlt().solve(-gradient);
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

    lambda *= (cost > previous_cost ? 2.0 : 0.6);
    lambda = std::clamp(lambda, min_lambda, max_lambda);
    previous_cost = cost;
  }
  std::cerr << "COST: " << previous_cost << ", iter: " << iteration
            << std::endl;

  pose->translation() = optimized_translation;
  pose->linear() = optimized_orientation.toRotationMatrix();

  return true;
}

PartialResult
MahalanobisDistanceMinimizerAnalyticSIMD::ComputeCostAndDerivatives(
    const Mat3x3& rotation, const Vec3& translation, const AlignedBuffer* abuf,
    const size_t start_index, const size_t end_index) {
  simd::Matrix<3, 3> R__(rotation.cast<float>());
  simd::Vector<3> t__(translation.cast<float>());

  simd::Vector<6> gradient__(Eigen::Matrix<float, 6, 1>::Zero());
  simd::Matrix<6, 6> hessian__(Eigen::Matrix<float, 6, 6>::Zero());
  simd::Scalar cost__(0.0);
  for (size_t point_idx = start_index; point_idx < end_index;
       point_idx += simd::Scalar::data_stride) {
    simd::Vector<3> p__(
        {abuf->x + point_idx, abuf->y + point_idx, abuf->z + point_idx});
    simd::Vector<3> mu__(
        {abuf->mx + point_idx, abuf->my + point_idx, abuf->mz + point_idx});
    simd::Matrix<3, 3> sqrt_info__(
        {abuf->sqrt_info[0][0] + point_idx, abuf->sqrt_info[0][1] + point_idx,
         abuf->sqrt_info[0][2] + point_idx, abuf->sqrt_info[1][0] + point_idx,
         abuf->sqrt_info[1][1] + point_idx, abuf->sqrt_info[1][2] + point_idx,
         abuf->sqrt_info[2][0] + point_idx, abuf->sqrt_info[2][1] + point_idx,
         abuf->sqrt_info[2][2] + point_idx});

    const simd::Vector<3> pw__ = R__ * p__ + t__;
    const simd::Vector<3> e__ = pw__ - mu__;
    const simd::Vector<3> r__ = sqrt_info__ * e__;

    // Direct calculation is far faster than helper operator.
    simd::Matrix<3, 6> J__;
    J__.block<3, 3>(0, 0) = sqrt_info__;
    J__.block<3, 3>(0, 3) = -sqrt_info__ * R__ * p__.hat();

    // Compute loss and weight,
    // and add the local gradient and hessian to the global ones
    simd::Scalar sq_r__ = r__.squaredNorm();
    simd::Scalar loss__(sq_r__);
    simd::Scalar weight__(1.0);
    if (loss_function_ != nullptr) {
      simd::Scalar loss_output__[3];
      loss_function_->Evaluate(sq_r__, loss_output__);
      loss__ = loss_output__[0];
      weight__ = loss_output__[1];
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

  PartialResult partial_result;
  for (size_t kk = 0; kk < simd::Scalar::data_stride; ++kk)
    partial_result.cost += buf[kk];

  for (int ii = 0; ii < 6; ++ii) {
    gradient__(ii).StoreData(buf);
    for (size_t kk = 0; kk < simd::Scalar::data_stride; ++kk)
      partial_result.gradient(ii) += buf[kk];
    for (int jj = ii; jj < 6; ++jj) {
      hessian__(ii, jj).StoreData(buf);
      for (size_t kk = 0; kk < simd::Scalar::data_stride; ++kk)
        partial_result.hessian(ii, jj) += buf[kk];
    }
  }

  return partial_result;
}

void MahalanobisDistanceMinimizerAnalyticSIMD::ReflectHessian(Mat6x6* hessian) {
  auto& H = *hessian;
  for (int row = 0; row < 6; ++row) {
    for (int col = row + 1; col < 6; ++col) {
      H(col, row) = H(row, col);
    }
  }
}

void MahalanobisDistanceMinimizerAnalyticSIMD::AddHessianOnlyUpperTriangle(
    const Mat6x6& local_hessian, Mat6x6* global_hessian) {
  auto& H = *global_hessian;
  for (int row = 0; row < 6; ++row) {
    for (int col = row; col < 6; ++col) {
      H(row, col) += local_hessian(row, col);
    }
  }
}

}  // namespace mahalanobis_distance_minimizer
}  // namespace nonlinear_optimizer