#include "nonlinear_optimizer/pose_graph_optimizer/pose_graph_optimizer_ceres.h"

#include "ceres/ceres.h"

#include "nonlinear_optimizer/ceres_loss_function.h"
#include "nonlinear_optimizer/pose_graph_optimizer/ceres_cost_functor.h"

namespace nonlinear_optimizer {
namespace pose_graph_optimizer {

PoseGraphOptimizerCeres::PoseGraphOptimizerCeres() {}

PoseGraphOptimizerCeres::~PoseGraphOptimizerCeres() {}

}  // namespace pose_graph_optimizer
}  // namespace nonlinear_optimizer