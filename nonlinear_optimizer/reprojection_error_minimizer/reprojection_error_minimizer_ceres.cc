#include "nonlinear_optimizer/reprojection_error_minimizer/reprojection_error_minimizer_ceres.h"

#include "ceres/ceres.h"

#include "nonlinear_optimizer/ceres_loss_function.h"
#include "nonlinear_optimizer/reprojection_error_minimizer/ceres_cost_functor.h"

namespace nonlinear_optimizer {
namespace reprojection_error_minimizer {

ReprojectionErrorMinimizerCeres::ReprojectionErrorMinimizerCeres() {}

ReprojectionErrorMinimizerCeres::~ReprojectionErrorMinimizerCeres() {}

bool ReprojectionErrorMinimizerCeres::Solve(
    const Options& options, const std::vector<Correspondence>& correspondences,
    const CameraIntrinsics& camera_intrinsics, Pose* pose) {
  constexpr double kC1{1.0};
  constexpr double kC2{1.0};

  const Pose initial_pose = *pose;

  double opt_trans[3] = {initial_pose.translation().x(),
                         initial_pose.translation().y(),
                         initial_pose.translation().z()};
  Orientation initial_orientation(initial_pose.rotation());
  double opt_orient[4] = {initial_orientation.w(), initial_orientation.x(),
                          initial_orientation.y(), initial_orientation.z()};

  ceres::LossFunction* loss_function = new ExponentialLoss(kC1, kC2);
  ceres::Problem problem;
  for (size_t index = 0; index < correspondences.size(); ++index) {
    ceres::CostFunction* cost_function =
        ReprojectionErrorWithExponentialLossCostFunctor::Create(
            correspondences.at(index), camera_intrinsics, kC1, kC2);
    problem.AddResidualBlock(cost_function, loss_function, opt_trans,
                             opt_orient);
  }

  ceres::Solver::Options ceres_options;
  ceres_options.linear_solver_type = ceres::DENSE_NORMAL_CHOLESKY;
  ceres_options.max_num_iterations = options.max_iterations + 100;
  //   ceres_options.function_tolerance =
  //       options.convergence_handle.function_tolerance;
  //   ceres_options.gradient_tolerance =
  //       options.convergence_handle.gradient_tolerance;
  //   ceres_options.parameter_tolerance =
  //       options.convergence_handle.parameter_tolerance;
  ceres::Solver::Summary summary;
  ceres::Solve(ceres_options, &problem, &summary);
  std::cerr << summary.BriefReport() << std::endl;

  pose->translation() << opt_trans[0], opt_trans[1], opt_trans[2];
  pose->linear() =
      Orientation(opt_orient[0], opt_orient[1], opt_orient[2], opt_orient[3])
          .toRotationMatrix();

  return true;
}

}  // namespace reprojection_error_minimizer
}  // namespace nonlinear_optimizer