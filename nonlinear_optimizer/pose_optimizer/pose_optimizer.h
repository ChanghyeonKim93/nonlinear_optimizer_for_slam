#ifndef POSE_OPTIMIZER_POSE_OPTIMIZER_H_
#define POSE_OPTIMIZER_POSE_OPTIMIZER_H_

#include <optional>

#include "Eigen/Dense"

#include "types.h"

namespace pose_optimizer {

struct Parameters {
  int max_iteration = 100;
};

struct LocalResult {
  Mat6x6 H = Mat6x6::Zero();
  Vec6 g = Vec6::Zero();
  double cost = 0.0;
};

class PoseOptimizer {
 public:
  PoseOptimizer();

  ~PoseOptimizer();

  bool Solve(Pose* pose);

 private:
  LocalResult ComputeReprojectionError(const Pose& pose,
                                       const bool compute_jacobians);
  LocalResult ComputeImuPreintegrationError(const Pose& pose,
                                            const bool compute_jacobians);
  void ApplyLocalResult(const LocalResult& local_result, Mat6x6* H, Vec6* g,
                        double* cost);

  const Parameters parameters_;
};

}  // namespace pose_optimizer

#endif  // POSE_OPTIMIZER_POSE_OPTIMIZER_H_
