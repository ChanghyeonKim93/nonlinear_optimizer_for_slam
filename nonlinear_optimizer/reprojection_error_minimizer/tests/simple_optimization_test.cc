#include <iostream>
#include <random>

#include "Eigen/Dense"

#include "ceres/ceres.h"

#include "nonlinear_optimizer/reprojection_error_minimizer/ceres_cost_functor.h"
#include "nonlinear_optimizer/reprojection_error_minimizer/reprojection_error_minimizer.h"
#include "nonlinear_optimizer/reprojection_error_minimizer/reprojection_error_minimizer_analytic.h"
#include "nonlinear_optimizer/reprojection_error_minimizer/reprojection_error_minimizer_ceres.h"
#include "nonlinear_optimizer/time_checker.h"
#include "nonlinear_optimizer/types.h"

nonlinear_optimizer::Options options;

namespace nonlinear_optimizer {
namespace reprojection_error_minimizer {

std::vector<Vec3> GenerateReferencePoints();
std::vector<Vec3> WarpPoints(const std::vector<Vec3>& points, const Pose& pose);
std::vector<Vec2> ProjectToPixel(const std::vector<Vec3>& local_points,
                                 const CameraIntrinsics& camera_intrinsics);

Pose OptimizePoseCeres(const std::vector<Correspondence>& correspondences,
                       const CameraIntrinsics& camera_intrinsics,
                       const Pose& initial_pose);
Pose OptimizePoseAnalytic(const std::vector<Correspondence>& correspondences,
                          const CameraIntrinsics& camera_intrinsics,
                          const Pose& initial_pose);

}  // namespace reprojection_error_minimizer
}  // namespace nonlinear_optimizer

using namespace nonlinear_optimizer;
using namespace nonlinear_optimizer::reprojection_error_minimizer;

int main(int, char**) {
  CameraIntrinsics camera_intrinsics;
  camera_intrinsics.fx = 525.0;
  camera_intrinsics.fy = 525.0;
  camera_intrinsics.cx = 320.0;
  camera_intrinsics.cy = 240.0;
  camera_intrinsics.inv_fx = 1.0 / camera_intrinsics.fx;
  camera_intrinsics.inv_fy = 1.0 / camera_intrinsics.fy;
  camera_intrinsics.width = 640;
  camera_intrinsics.height = 480;

  // Make global points
  const auto reference_points = GenerateReferencePoints();
  std::cerr << "# points: " << reference_points.size() << std::endl;

  // Set true pose
  Pose true_pose{Pose::Identity()};
  true_pose.translation() = Vec3(-0.1, 0.123, -0.5);
  true_pose.linear() =
      Eigen::AngleAxisd(0.1, Vec3(0.0, 0.0, 1.0)).toRotationMatrix();

  const auto query_points = WarpPoints(reference_points, true_pose.inverse());
  const auto matched_pixels = ProjectToPixel(query_points, camera_intrinsics);
  std::vector<Correspondence> correspondences;
  for (size_t index = 0; index < query_points.size(); ++index) {
    Correspondence corr;
    corr.local_point = reference_points.at(index);
    corr.matched_pixel = matched_pixels.at(index);
    correspondences.push_back(corr);
  }

  // Optimize pose
  Pose initial_pose{Pose::Identity()};
  std::cerr << "Start OptimizedPoseOriginal" << std::endl;
  const auto opt_pose_ceres =
      OptimizePoseCeres(correspondences, camera_intrinsics, initial_pose)
          .inverse();

  std::cerr << "Start OptimizedPoseAnalytic" << std::endl;
  const auto opt_pose_analytic =
      OptimizePoseAnalytic(correspondences, camera_intrinsics, initial_pose)
          .inverse();

  std::cerr << "True pose: " << true_pose.translation().transpose() << " "
            << Eigen::Quaterniond(true_pose.linear()).coeffs().transpose()
            << std::endl;
  std::cerr << "Pose (ceres ): " << opt_pose_ceres.translation().transpose()
            << " "
            << Eigen::Quaterniond(opt_pose_ceres.linear()).coeffs().transpose()
            << std::endl;
  std::cerr
      << "Pose (analytic ): " << opt_pose_analytic.translation().transpose()
      << " "
      << Eigen::Quaterniond(opt_pose_analytic.linear()).coeffs().transpose()
      << std::endl;
  return 0;
}

namespace nonlinear_optimizer {
namespace reprojection_error_minimizer {

std::vector<Vec3> GenerateReferencePoints() {
  std::random_device rd;
  std::normal_distribution<double> nd_z(0.0, 0.1);

  const double z = 3.0;
  const double x_min = -1.5;
  const double x_max = 1.5;
  const double y_min = -1.0;
  const double y_max = 1.0;
  const double point_step = 0.1;

  std::vector<Eigen::Vector3d> points;
  double x, y;
  for (x = x_min; x <= x_max; x += point_step) {
    for (y = y_min; y <= y_max; y += point_step) {
      points.push_back(Vec3(x, y, z));
    }
  }

  return points;
}

std::vector<Vec3> WarpPoints(const std::vector<Vec3>& points,
                             const Pose& pose) {
  std::vector<Vec3> warped_points;
  warped_points.reserve(points.size());
  for (const auto& point : points) warped_points.push_back(pose * point);
  return warped_points;
}

std::vector<Vec2> ProjectToPixel(const std::vector<Vec3>& local_points,
                                 const CameraIntrinsics& camera_intrinsics) {
  std::vector<Vec2> projected_pixels;
  for (const auto& local_point : local_points) {
    Vec2 pixel;
    const double inv_z = 1.0 / local_point.z();
    pixel.x() =
        camera_intrinsics.fx * local_point.x() * inv_z + camera_intrinsics.cx;
    pixel.y() =
        camera_intrinsics.fy * local_point.y() * inv_z + camera_intrinsics.cy;
    projected_pixels.push_back(pixel);
  }
  return projected_pixels;
}

Pose OptimizePoseCeres(const std::vector<Correspondence>& correspondences,
                       const CameraIntrinsics& camera_intrinsics,
                       const Pose& initial_pose) {
  CHECK_EXEC_TIME_FROM_HERE

  double optimized_translation[3] = {initial_pose.translation().x(),
                                     initial_pose.translation().y(),
                                     initial_pose.translation().z()};
  auto initial_orientation = Orientation(initial_pose.rotation());
  double optimized_orientation[4] = {
      initial_orientation.w(), initial_orientation.x(), initial_orientation.y(),
      initial_orientation.z()};

  ceres::Problem problem;
  ceres::CostFunction* cost_function = nonlinear_optimizer::
      reprojection_error_minimizer::ReprojectionErrorCostFunctorBatch::Create(
          correspondences, camera_intrinsics, 1.0, 1.0);
  problem.AddResidualBlock(cost_function, nullptr, optimized_translation,
                           optimized_orientation);

  ceres::Solver::Options options;
  options.linear_solver_type = ceres::DENSE_QR;
  options.max_num_iterations = 300;
  ceres::Solver::Summary summary;
  ceres::Solve(options, &problem, &summary);
  std::cerr << "Summary: " << summary.BriefReport() << std::endl;

  Pose optimized_pose{Pose::Identity()};
  optimized_pose.translation() =
      Vec3(optimized_translation[0], optimized_translation[1],
           optimized_translation[2]);
  optimized_pose.linear() =
      Orientation(optimized_orientation[0], optimized_orientation[1],
                  optimized_orientation[2], optimized_orientation[3])
          .toRotationMatrix();

  return optimized_pose;
}

Pose OptimizePoseAnalytic(const std::vector<Correspondence>& correspondences,
                          const CameraIntrinsics& camera_intrinsics,
                          const Pose& initial_pose) {
  CHECK_EXEC_TIME_FROM_HERE

  double optimized_translation[3] = {initial_pose.translation().x(),
                                     initial_pose.translation().y(),
                                     initial_pose.translation().z()};
  auto initial_orientation = Orientation(initial_pose.rotation());
  double optimized_orientation[4] = {
      initial_orientation.w(), initial_orientation.x(), initial_orientation.y(),
      initial_orientation.z()};

  Pose optimized_pose{Pose::Identity()};
  optimized_pose.translation() =
      Vec3(optimized_translation[0], optimized_translation[1],
           optimized_translation[2]);
  optimized_pose.linear() =
      Orientation(optimized_orientation[0], optimized_orientation[1],
                  optimized_orientation[2], optimized_orientation[3])
          .toRotationMatrix();

  std::unique_ptr<nonlinear_optimizer::reprojection_error_minimizer::
                      ReprojectionErrorMinimizerAnalytic>
      optim =
          std::make_unique<nonlinear_optimizer::reprojection_error_minimizer::
                               ReprojectionErrorMinimizerAnalytic>();
  optim->SetLossFunction(
      std::make_shared<nonlinear_optimizer::ExponentialLossFunction>(1.0, 1.0));
  optim->Solve(options, correspondences, camera_intrinsics, &optimized_pose);

  return optimized_pose;
}

}  // namespace reprojection_error_minimizer
}  // namespace nonlinear_optimizer
