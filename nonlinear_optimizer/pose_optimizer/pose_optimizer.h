#ifndef NONLINEAR_OPTIMIZER_POSE_OPTIMIZER_POSE_OPTIMIZER_H_
#define NONLINEAR_OPTIMIZER_POSE_OPTIMIZER_POSE_OPTIMIZER_H_

#include <cmath>
#include <limits>
#include <memory>
#include <unordered_set>

#include "Eigen/Dense"
#include "cost_function.h"

namespace nonlinear_optimizer {
namespace pose_optimizer {

template <int kDimTranslation, int kDimRotation>
class Problem {
 protected:
  static constexpr int kDimPose = kDimTranslation + kDimRotation;
  using CostFunctionPtr =
      std::shared_ptr<CostFunction<kDimTranslation, kDimRotation>>;
  using LossFunctionPtr = std::shared_ptr<LossFunction>;
  using ResidualBlockPtr =
      std::shared_ptr<ResidualBlock<kDimTranslation, kDimRotation>>;

 public:
  Problem() {}

  void AddResidualBlock(CostFunctionPtr* cost_function,
                        LossFunctionPtr* loss_function = nullptr) {
    ResidualBlockPtr residual_block =
        std::make_shared<ResidualBlock>(cost_function, loss_function);
    residual_block_set_.insert(residual_block);
  }

  const std::unordered_set<ResidualBlockPtr>& GetResidualBlocks() const {
    return residual_block_set_;
  }

 private:
  std::unordered_set<ResidualBlockPtr> residual_block_set_;
};

struct Options {
  int max_iterations = 100;
  struct {
    double parameter_tolerance = 1e-8;
    double gradient_tolerance = 1e-10;
    double function_tolerance = 1e-6;
  } convergence_handle;
  struct {
    double min_step_size = 1e-6;
    double max_step_size = 1.0;
  } step_size_handle;
  struct {
    double min_lambda = 1e-6;
    double max_lambda = 1e-2;
    double initial_lambda = 0.001;
    double lambda_increasing_factor = 2.0;
    double lambda_decreasing_factor = 0.6;
  } damping_handle;
};

struct Summary {};

template <int kDimTranslation, int kDimRotation>
class PoseOptimizer {
  static constexpr int kDimPose = kDimTranslation + kDimRotation;
  using HessianMatrix = Eigen::Matrix<double, kDimPose, kDimPose>;
  using GradientVector = Eigen::Matrix<double, kDimPose, 1>;

 public:
  PoseOptimizer() {}

  bool Solve(const Problem<kDimTranslation, kDimRotation>& problem,
             const Options& options, Pose* pose, Summary* summary = nullptr) {
    bool success = true;

    double lambda = options.damping_handle.initial_lambda;

    Pose optimized_pose = *pose;
    double previous_cost = std::numeric_limits<double>::max();
    int iteration = 0;
    for (; iteration < options.max_iterations; ++iteration) {
      HessianMatrix hessian{HessianMatrix::Zero()};
      GradientVector gradient{GradientVector::Zero()};

      double cost = 0.0;
      for (const auto& residual_block : problem.GetResidualBlocks()) {
        HessianMatrix local_hessian{HessianMatrix::Zero()};
        GradientVector local_gradient{GradientVector::Zero()};
        double local_cost = 0.0;
        residual_block->Evaluate(optimized_pose.rotation(),
                                 optimized_pose.translation(), &local_hessian,
                                 &local_gradient, &local_cost);
        AddLocalHessianOnlyUpperTriangle(local_hessian, &hessian);
        *gradient += local_gradient;
        cost += local_cost;
      }
      ReflectHessian(&hessian);

      // Damping hessian
      for (int k = 0; k < kDimPose; k++) hessian(k, k) *= 1.0 + lambda;

      // Solve the linear system hessian * delta = -gradient
      Eigen::LLT<HessianMatrix> llt(hessian);
      if (llt.info() != Eigen::Success) {
        success = false;
        break;
      }
      // Compute the step
      const Eigen::Matrix<double, kDimPose, 1> update_step =
          hessian.ldlt().solve(-gradient);

      // Update the pose
      const Eigen::Matrix<double, kDimTranslation, 1> delta_t =
          update_step.template block<kDimTranslation, 1>(0, 0);
      const Eigen::Matrix<double, kDimRotation, 1> delta_R =
          update_step.template block<kDimRotation, 1>(kDimTranslation, 0);

      // Check convergence
      if (update_step.norm() < options.convergence_handle.parameter_tolerance) {
        break;
      }
      if (gradient.norm() < options.convergence_handle.gradient_tolerance) {
        break;
      }
      if (std::abs(cost - previous_cost) <
          options.convergence_handle.function_tolerance) {
        break;
      }

      const auto& damping_handle = options.damping_handle;
      lambda *= (cost > previous_cost)
                    ? damping_handle.lambda_increasing_factor
                    : damping_handle.lambda_decreasing_factor;
      lambda = std::clamp(lambda, damping_handle.min_lambda,
                          damping_handle.max_lambda);

      if (kDimRotation == 1)
        optimized_pose.rotate(delta_R);
      else if (kDimRotation == 3)
        optimized_pose.rotate(ComputeQuaternion(delta_R));
      optimized_pose.translate(delta_t);
      // Rotation part should be projected to the SO(3) manifold

      previous_cost = cost;
    }

    *pose = optimized_pose;

    return success;
  }

 private:
  void AddLocalHessianOnlyUpperTriangle(const HessianMatrix& local_hessian,
                                        HessianMatrix* hessian_matrix) {
    for (int i = 0; i < kDimPose; ++i)
      for (int j = i; j < kDimPose; ++j)
        (*hessian_matrix)(i, j) += local_hessian(i, j);
  }

  void ReflectHessian(HessianMatrix* hessian_matrix) {
    for (int i = 0; i < kDimPose; ++i)
      for (int j = i + 1; j < kDimPose; ++j)
        (*hessian_matrix)(j, i) = (*hessian_matrix)(i, j);
  }

  Eigen::Quaterniond ComputeQuaternion(const Eigen::Vector3d& w) {
    Eigen::Quaterniond orientation{Eigen::Quaterniond::Identity()};
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
};

}  // namespace pose_optimizer
}  // namespace nonlinear_optimizer

#endif  // NONLINEAR_OPTIMIZER_POSE_OPTIMIZER_POSE_OPTIMIZER_H_