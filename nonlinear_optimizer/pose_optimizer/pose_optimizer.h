#ifndef POSE_OPTIMIZER_POSE_OPTIMIZER_H_
#define POSE_OPTIMIZER_POSE_OPTIMIZER_H_

#include <optional>

#include "Eigen/Dense"

#include "types.h"

namespace pose_optimizer {

enum class SolverType { kGaussNewton = 0, kLevenbergMarquardt };

struct Parameters {
  int max_iteration = 100;
  struct {
    double reprojection_error = 1.0;
    double imu_preintegration = 1.0;
    double translation_prior = 1.0;
    double rotation_prior = 1.0;
  } cost_weight;
};

struct Evaluation {
  double cost = 0.0;
  Mat6x6 H = Mat6x6::Zero();
  Vec6 g = Vec6::Zero();
};

class PoseOptimizer {
 public:
  PoseOptimizer();

  ~PoseOptimizer();

  bool Solve(Pose* pose);

 private:
  Evaluation Evaluate(const Vec3& translation, const Orientation& orientation,
                      const bool compute_jacobians = true);
  Evaluation EvaluateReprojectionErrorCost(const Vec3& translation,
                                           const Orientation& orientation,
                                           const bool compute_jacobians = true);
  Evaluation EvaluateImuPreintegrationCost(const Vec3& translation,
                                           const Orientation& orientation,
                                           const bool compute_jacobians = true);
  Evaluation EvaluateTranslationPriorCost(const Vec3& translation,
                                          const bool compute_jacobians = true);
  Evaluation EvaluateRotationPriorCost(const Orientation& orientation,
                                       const bool compute_jacobians = true);
  void AugmentEvaluation(const Evaluation& local_evaluation,
                         const double cost_weight,
                         Evaluation* total_evaluation);
  void ReflectHessian(Mat6x6* H);
  Orientation GetQuaternion(const Vec3& rotation_vector);

  const Parameters parameters_;
  SolverType solver_type_{SolverType::kLevenbergMarquardt};
};

}  // namespace pose_optimizer

#endif  // POSE_OPTIMIZER_POSE_OPTIMIZER_H_
