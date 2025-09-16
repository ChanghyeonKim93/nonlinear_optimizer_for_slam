#include "pose_optimizer.h"

namespace pose_optimizer {

PoseOptimizer::PoseOptimizer() {}

PoseOptimizer::~PoseOptimizer() {}

bool PoseOptimizer::Solve(Pose* pose) {
  constexpr double kMaxMu = 2.0;
  constexpr double kMinMu = 0.1;
  constexpr double kMaxTrRadius = 1e16;
  constexpr double kMinTrRadius = 1e-16;

  if (pose == nullptr) return false;

  Vec3 translation = pose->translation();
  Eigen::Quaterniond orientation(pose->rotation());

  Evaluation previous_evaluation;
  previous_evaluation.cost = std::numeric_limits<double>::max();

  double tr_radius = 1.0;
  double mu = 1e-3;
  bool is_previous_step_valid = true;
  int iteration = 0;
  for (; iteration < parameters_.max_iteration; ++iteration) {
    // Evaluate cost functions and Jacobians
    Evaluation evaluation;
    if (solver_type_ == SolverType::kGaussNewton || is_previous_step_valid)
      evaluation = Evaluate(translation, orientation, true);
    else
      evaluation = previous_evaluation;

    // Compute damping ratio satisfying the trust region radius constraint
    const auto& H = evaluation.H;
    const auto& g = evaluation.g;
    const Mat6x6 damped_H = H + mu * H.diagonal();
    Vec6 dx = -damped_H.ldlt().solve(g);
    const double step_norm = dx.norm();
    if (step_norm > tr_radius) {
      const double scale_factor = tr_radius / step_norm;
      dx *= scale_factor;
    }
    const Vec3 updated_translation = translation + dx.block<3, 1>(0, 0);
    const Orientation updated_orientation =
        orientation * GetQuaternion(dx.block<3, 1>(0, 3));

    if (solver_type_ == SolverType::kGaussNewton) {
      is_previous_step_valid = true;
      translation = updated_translation;
      orientation = updated_orientation;
    } else if (solver_type_ == SolverType::kLevenbergMarquardt) {
      // Compute new cost
      const Evaluation new_evaluation =
          Evaluate(updated_translation, updated_orientation, false);

      const double real_cost_change = new_evaluation.cost - evaluation.cost;
      const double estimated_cost_change = dx.dot(0.5 * H * dx + g);
      const double rho = real_cost_change / estimated_cost_change;
      if (real_cost_change < 0.0 && rho > 0.0) {
        // Accept the step
        is_previous_step_valid = true;
        translation = updated_translation;
        orientation = updated_orientation;
        if (rho > 0.75) {
          tr_radius = std::max(tr_radius, 3.0 * step_norm);
          mu *= 0.5;
        } else if (rho > 0.25) {
          tr_radius = std::max(tr_radius, 1.5 * step_norm);
          mu *= 0.7;
        } else {
          mu *= 0.9;
        }
        mu = std::clamp(mu, kMinMu, kMaxMu);
        tr_radius = std::clamp(tr_radius, kMinTrRadius, kMaxTrRadius);
      } else {
        // Reject the step
        is_previous_step_valid = false;
        tr_radius *= 0.25;
        mu *= 2.0;
        mu = std::clamp(mu, kMinMu, kMaxMu);
        tr_radius = std::clamp(tr_radius, kMinTrRadius, kMaxTrRadius);
        continue;
      }
    }
    // Check convergence
    if (dx.size() < 1e-7) break;
    if (std::abs(previous_evaluation.cost - evaluation.cost) < 1e-7) break;
    if (evaluation.g.norm() < 1e-7) break;

    // Update the previous step information
    previous_evaluation = evaluation;
  }

  pose->translation() = translation;
  pose->linear() = orientation.toRotationMatrix();

  return true;
}

Evaluation PoseOptimizer::Evaluate(const Vec3& translation,
                                   const Orientation& orientation,
                                   const bool compute_jacobians) {
  const auto& cost_weight = parameters_.cost_weight;
  Evaluation eval;
  AugmentEvaluation(EvaluateReprojectionErrorCost(translation, orientation,
                                                  compute_jacobians),
                    cost_weight.reprojection_error, &eval);
  AugmentEvaluation(EvaluateImuPreintegrationCost(translation, orientation,
                                                  compute_jacobians),
                    cost_weight.imu_preintegration, &eval);
  AugmentEvaluation(
      EvaluateTranslationPriorCost(translation, compute_jacobians),
      cost_weight.translation_prior, &eval);
  AugmentEvaluation(EvaluateRotationPriorCost(orientation, compute_jacobians),
                    cost_weight.rotation_prior, &eval);
  return eval;
}

Evaluation PoseOptimizer::EvaluateReprojectionErrorCost(
    const Vec3& translation, const Orientation& orientation,
    bool compute_jacobians) {
  Evaluation eval;
  (void)translation;
  (void)orientation;
  eval.cost;
  if (compute_jacobians) {
    eval.H;
    eval.g;
    ReflectHessian(&eval.H);
  }
  return eval;
}

Evaluation PoseOptimizer::EvaluateImuPreintegrationCost(
    const Vec3& translation, const Orientation& orientation,
    bool compute_jacobians) {
  Evaluation eval;
  (void)translation;
  (void)orientation;
  eval.cost;
  if (compute_jacobians) {
    eval.H;
    eval.g;
  }
  return eval;
}

Evaluation EvaluateTranslationPriorCost(const Vec3& translation,
                                        const bool compute_jacobians) {
  Evaluation eval;
  (void)translation;
  eval.cost;
  if (compute_jacobians) {
    eval.H;
    eval.g;
  }
  return eval;
}

Evaluation EvaluateRotationPriorCost(const Orientation& orientation,
                                     const bool compute_jacobians) {
  Evaluation eval;
  (void)orientation;
  eval.cost;
  if (compute_jacobians) {
    eval.H;
    eval.g;
  }
  return eval;
}

void PoseOptimizer::AugmentEvaluation(const Evaluation& evaluation,
                                      const double cost_weight,
                                      Evaluation* total_evaluation) {
  total_evaluation->H += cost_weight * evaluation.H;
  total_evaluation->g += cost_weight * evaluation.g;
  total_evaluation->cost += cost_weight * evaluation.cost;
}

Orientation PoseOptimizer::GetQuaternion(const Vec3& rotation_vector) {
  Orientation orientation;
  (void)rotation_vector;
  return orientation;
}

}  // namespace pose_optimizer