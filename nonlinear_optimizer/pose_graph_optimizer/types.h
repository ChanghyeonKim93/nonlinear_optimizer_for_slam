#ifndef NONLINEAR_OPTIMIZER_POSE_GRAPH_OPTIMIZER_TYPES_H_
#define NONLINEAR_OPTIMIZER_POSE_GRAPH_OPTIMIZER_TYPES_H_

#include <unordered_map>

#include "nonlinear_optimizer/types.h"

namespace nonlinear_optimizer {
namespace pose_graph_optimizer {

struct LoopConstraint {
  Pose reference_pose{Pose::Identity()};
  Pose query_pose{Pose::Identity()};
};

}  // namespace pose_graph_optimizer
}  // namespace nonlinear_optimizer

#endif  // NONLINEAR_OPTIMIZER_POSE_GRAPH_OPTIMIZER_TYPES_H_
