#ifndef NONLINEAR_OPTIMIZER_POSE_GRAPH_OPTIMIZER_TYPES_H_
#define NONLINEAR_OPTIMIZER_POSE_GRAPH_OPTIMIZER_TYPES_H_

#include <unordered_map>

#include "nonlinear_optimizer/types.h"

namespace nonlinear_optimizer {
namespace pose_graph_optimizer {

enum class ConstraintType { kNormal = 0, kLoop = 1 };

struct Constraint {
  int reference_pose_index{-1};
  int query_pose_index{-1};
  Pose relative_pose_from_reference_to_query{Pose::Identity()};
  ConstraintType type{ConstraintType::kNormal};
};

}  // namespace pose_graph_optimizer
}  // namespace nonlinear_optimizer

#endif  // NONLINEAR_OPTIMIZER_POSE_GRAPH_OPTIMIZER_TYPES_H_
