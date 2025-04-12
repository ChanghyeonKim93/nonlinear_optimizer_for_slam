#ifndef NONLINEAR_OPTIMIZER_POSE_GRAPH_OPTIMIZER_POSE_GRAPH_OPTIMIZER_ANALYTIC_H_
#define NONLINEAR_OPTIMIZER_POSE_GRAPH_OPTIMIZER_POSE_GRAPH_OPTIMIZER_ANALYTIC_H_

#include <vector>

#include "nonlinear_optimizer/pose_graph_optimizer/pose_graph_optimizer.h"
#include "nonlinear_optimizer/pose_graph_optimizer/types.h"
#include "nonlinear_optimizer/types.h"

namespace nonlinear_optimizer {
namespace pose_graph_optimizer {

class PoseGraphOptimizerAnalytic : public PoseGraphOptimizer {
 public:
  PoseGraphOptimizerAnalytic();

  ~PoseGraphOptimizerAnalytic();

  bool Solve(const Options& options) final;

 private:
  // Hessian
  std::unordered_map<int, Mat6x6f> block_hessian_;
  // Gradient

  std::unordered_map<int, Vec6f> block_gradient_;

  // Residual
  std::unordered_map<int, Vec6f> block_residual_;
};

}  // namespace pose_graph_optimizer
}  // namespace nonlinear_optimizer

#endif  // NONLINEAR_OPTIMIZER_POSE_GRAPH_OPTIMIZER_POSE_GRAPH_OPTIMIZER_ANALYTIC_H_