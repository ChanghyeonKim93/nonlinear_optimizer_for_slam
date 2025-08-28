#include "nonlinear_optimizer/mahalanobis_distance_minimizer/mahalanobis_distance_minimizer_analytic.h"

#include <iostream>

namespace nonlinear_optimizer {
namespace mahalanobis_distance_minimizer {

MahalanobisDistanceMinimizerAnalytic::MahalanobisDistanceMinimizerAnalytic() {}

MahalanobisDistanceMinimizerAnalytic::~MahalanobisDistanceMinimizerAnalytic() {}

PartialResult MahalanobisDistanceMinimizerAnalytic::ComputeCostAndDerivatives(
    const Mat3x3& rotation, const Vec3& translation,
    const std::vector<Correspondence>* correspondences,
    const size_t start_index, const size_t end_index) {
  PartialResult partial_result;
  Mat3x6 jacobian{Mat3x6::Zero()};
  Vec3 residual{Vec3::Zero()};
  Vec6 local_gradient{Vec6::Zero()};
  Mat6x6 local_hessian{Mat6x6::Zero()};
  for (size_t i = start_index; i < end_index; ++i) {
    const auto& corr = correspondences->at(i);

    ComputeJacobianAndResidual(rotation, translation, corr, &jacobian,
                               &residual);

    // Compute the local gradient
    local_gradient = jacobian.transpose() * residual;

    // Compute the local hessian
    ComputeHessianOnlyUpperTriangle(jacobian, &local_hessian);

    // Compute loss and weight,
    // and add the local gradient and hessian to the global ones
    const double squared_residual = residual.transpose() * residual;
    if (loss_function_ != nullptr) {
      double loss_output[3] = {0.0, 0.0, 0.0};
      loss_function_->Evaluate(squared_residual, loss_output);
      const double weight = loss_output[1];
      partial_result.gradient += weight * local_gradient;
      MultiplyWeightOnlyUpperTriangle(weight, &local_hessian);
      AddHessianOnlyUpperTriangle(local_hessian, &partial_result.hessian);
      partial_result.cost += loss_output[0];
    } else {
      partial_result.gradient += local_gradient;
      AddHessianOnlyUpperTriangle(local_hessian, &partial_result.hessian);
      partial_result.cost += squared_residual;
    }
  }
  // std::cerr << "END PARTIAL" << std::endl;
  return partial_result;
}

bool MahalanobisDistanceMinimizerAnalytic::Solve(
    const Options& options, const std::vector<Correspondence>& correspondences,
    Pose* pose) {
  std::chrono::steady_clock::time_point begin =
      std::chrono::steady_clock::now();
  std::vector<std::vector<Correspondence>> partial_correspondences_list;
  if (multi_thread_executor_ != nullptr) {
    const int num_threads = multi_thread_executor_->GetNumOfTotalThreads();
    const int num_correspondences = static_cast<int>(correspondences.size());
    const int num_batch = static_cast<int>(
        std::max(1.0, static_cast<double>(num_correspondences) / num_threads));
    for (int idx = 0; idx < num_threads; ++idx) {
      partial_correspondences_list.push_back(std::vector<Correspondence>());
      auto& corrs = partial_correspondences_list.at(idx);
      corrs.reserve(std::max(num_batch, num_correspondences - idx * num_batch));
      for (int j = idx * num_batch;
           j < std::min((idx + 1) * num_batch, num_correspondences); ++j)
        corrs.emplace_back(correspondences[j]);
    }
  }
  std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
  std::cerr << "SPLIT TIME: "
            << std::chrono::duration_cast<std::chrono::milliseconds>(end -
                                                                     begin)
                   .count()
            << "[ms]" << std::endl;

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

    if (multi_thread_executor_ == nullptr) {
      auto result = ComputeCostAndDerivatives(
          optimized_orientation.toRotationMatrix(), optimized_translation,
          &correspondences, 0, correspondences.size());
      gradient = result.gradient;
      hessian = result.hessian;
      cost = result.cost;
    } else {
      const int num_threads = multi_thread_executor_->GetNumOfTotalThreads();
      std::vector<std::future<PartialResult>> partial_results;
      for (int i = 0; i < num_threads; ++i) {
        partial_results.emplace_back(multi_thread_executor_->Execute(
            &MahalanobisDistanceMinimizerAnalytic::ComputeCostAndDerivatives,
            this, optimized_orientation.toRotationMatrix(),
            optimized_translation, &partial_correspondences_list.at(i), 0,
            partial_correspondences_list.at(i).size()));
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

    // Compute the step
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

void MahalanobisDistanceMinimizerAnalytic::ComputeJacobianAndResidual(
    const Mat3x3& rotation, const Vec3& translation, const Correspondence& corr,
    Mat3x6* jacobian, Vec3* residual) {
  const auto& R = rotation;
  const auto& t = translation;
  const auto& p = corr.point;
  const auto& sqrt_information = corr.ndt.sqrt_information;

  const Vec3& p_warped = R * p + t;
  const Vec3& e_i = p_warped - corr.ndt.mean;

  // Compute the residual
  *residual = sqrt_information * e_i;  // residual

  auto skew = [](const Vec3& v) {
    Mat3x3 skew{Mat3x3::Zero()};
    skew << 0.0, -v.z(), v.y(),  //
        v.z(), 0.0, -v.x(),      //
        -v.y(), v.x(), 0.0;
    return skew;
  };

  // Compute the Jacobian
  Mat3x3 R_skew_p = R * skew(p);
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

}  // namespace mahalanobis_distance_minimizer
}  // namespace nonlinear_optimizer