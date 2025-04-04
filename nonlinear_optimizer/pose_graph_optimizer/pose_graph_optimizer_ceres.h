#ifndef NONLINEAR_OPTIMIZER_POSE_GRAPH_OPTIMIZER_POSE_GRAPH_OPTIMIZER_CERES_H_
#define NONLINEAR_OPTIMIZER_POSE_GRAPH_OPTIMIZER_POSE_GRAPH_OPTIMIZER_CERES_H_

#include <vector>

#include "nonlinear_optimizer/pose_graph_optimizer/pose_graph_optimizer.h"
#include "nonlinear_optimizer/pose_graph_optimizer/types.h"
#include "nonlinear_optimizer/types.h"

namespace nonlinear_optimizer {
namespace pose_graph_optimizer {

class PoseGraphOptimizerCeres : public PoseGraphOptimizer {
 public:
  PoseGraphOptimizerCeres();

  ~PoseGraphOptimizerCeres();

  bool Solve(const Options& options) final;
};

}  // namespace pose_graph_optimizer
}  // namespace nonlinear_optimizer

#endif  // NONLINEAR_OPTIMIZER_POSE_GRAPH_OPTIMIZER_POSE_GRAPH_OPTIMIZER_CERES_H_