#include "nonlinear_optimizer/mahalanobis_distance_minimizer/mahalanobis_distance_minimizer_ceres.h"

#include "ceres/ceres.h"

#include "nonlinear_optimizer/mahalanobis_distance_minimizer/ceres_cost_functor.h"

namespace nonlinear_optimizer {
namespace mahalanobis_distance_minimizer {

MahalanobisDistanceMinimizerCeres::MahalanobisDistanceMinimizerCeres() {}

MahalanobisDistanceMinimizerCeres::~MahalanobisDistanceMinimizerCeres() {}

bool MahalanobisDistanceMinimizerCeres::Solve(
    const Options& options, const std::vector<Correspondence>& correspondences,
    Pose* pose) {
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
        MahalanobisDistanceCostFunctor::Create(correspondences.at(index));
    problem.AddResidualBlock(cost_function, loss_function, opt_trans,
                             opt_orient);
  }

  ceres::Solver::Options ceres_options;
  ceres_options.linear_solver_type = ceres::DENSE_QR;
  ceres_options.max_num_iterations = options.max_iterations;
  ceres_options.function_tolerance =
      options.convergence_handle.function_tolerance;
  ceres_options.gradient_tolerance =
      options.convergence_handle.gradient_tolerance;
  ceres_options.parameter_tolerance =
      options.convergence_handle.parameter_tolerance;
  ceres::Solver::Summary summary;
  ceres::Solve(ceres_options, &problem, &summary);
  // std::cerr << summary.BriefReport() << std::endl;

  pose->translation() << opt_trans[0], opt_trans[1], opt_trans[2];
  pose->linear() =
      Orientation(opt_orient[0], opt_orient[1], opt_orient[2], opt_orient[3])
          .toRotationMatrix();

  return true;
}

bool MahalanobisDistanceMinimizerCeres::SolveByRedundantForEach(
    const std::vector<Correspondence>& correspondences, Pose* pose) {
  constexpr double kC1{1.0};
  constexpr double kC2{1.0};

  const Pose initial_pose = *pose;

  double opt_trans[3] = {initial_pose.translation().x(),
                         initial_pose.translation().y(),
                         initial_pose.translation().z()};
  Orientation initial_orientation(initial_pose.rotation());
  double opt_orient[4] = {initial_orientation.w(), initial_orientation.x(),
                          initial_orientation.y(), initial_orientation.z()};

  ceres::Problem problem;
  for (size_t index = 0; index < correspondences.size(); ++index) {
    ceres::CostFunction* cost_function =
        RedundantMahalanobisDistanceCostFunctorEach::Create(
            correspondences.at(index), kC1, kC2);
    problem.AddResidualBlock(cost_function, nullptr, opt_trans, opt_orient);
  }

  ceres::Solver::Options options;
  options.linear_solver_type = ceres::DENSE_QR;
  options.max_num_iterations = 30;
  ceres::Solver::Summary summary;
  ceres::Solve(options, &problem, &summary);
  // std::cerr << summary.BriefReport() << std::endl;

  pose->translation() << opt_trans[0], opt_trans[1], opt_trans[2];
  pose->linear() =
      Orientation(opt_orient[0], opt_orient[1], opt_orient[2], opt_orient[3])
          .toRotationMatrix();

  return true;
}

}  // namespace mahalanobis_distance_minimizer
}  // namespace nonlinear_optimizer