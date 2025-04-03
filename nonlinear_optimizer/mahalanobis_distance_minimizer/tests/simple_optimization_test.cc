#include <iostream>
#include <unordered_map>
#include <unordered_set>

#include "Eigen/Dense"

#include "ceres/ceres.h"
#include "flann/algorithms/dist.h"
#include "flann/algorithms/kdtree_single_index.h"
#include "flann/flann.hpp"

#include "nonlinear_optimizer/mahalanobis_distance_minimizer/ceres_cost_functor.h"
#include "nonlinear_optimizer/mahalanobis_distance_minimizer/mahalanobis_distance_minimizer.h"
#include "nonlinear_optimizer/mahalanobis_distance_minimizer/mahalanobis_distance_minimizer_analytic.h"
#include "nonlinear_optimizer/mahalanobis_distance_minimizer/mahalanobis_distance_minimizer_analytic_simd.h"
#include "nonlinear_optimizer/mahalanobis_distance_minimizer/mahalanobis_distance_minimizer_ceres.h"
#include "nonlinear_optimizer/time_checker.h"
#include "nonlinear_optimizer/types.h"

namespace nonlinear_optimizer {
namespace mahalanobis_distance_minimizer {

using VoxelKey = uint64_t;
using NdtMap = std::unordered_map<VoxelKey, NDT>;

std::vector<Vec3> GenerateGlobalPoints();
std::vector<Vec3> FilterPoints(const std::vector<Vec3>& points,
                               const double filter_voxel_size);
std::vector<Vec3> WarpPoints(const std::vector<Vec3>& points, const Pose& pose);
void UpdateNdtMap(const std::vector<Vec3>& points,
                  const double voxel_resolution, NdtMap* ndt_map);
VoxelKey ComputeVoxelKey(const Vec3& point,
                         const double inverse_voxel_resolution);
std::vector<Correspondence> MatchPointCloud(
    const NdtMap& ndt_map, const std::vector<Vec3>& local_points,
    const Pose& pose);
Pose OptimizePoseOriginal(const NdtMap& ndt_map,
                          const std::vector<Vec3>& local_points,
                          const Pose& pose);
Pose OptimizePoseSimplified(const NdtMap& ndt_map,
                            const std::vector<Vec3>& local_points,
                            const Pose& initial_pose);
Pose OptimizePoseRedundantEach(const NdtMap& ndt_map,
                               const std::vector<Vec3>& local_points,
                               const Pose& initial_pose);
Pose OptimizePoseAnalytic(const NdtMap& ndt_map,
                          const std::vector<Vec3>& local_points,
                          const Pose& initial_pose);
Pose OptimizePoseAnalyticSimd(const NdtMap& ndt_map,
                              const std::vector<Vec3>& local_points,
                              const Pose& initial_pose);
Pose OptimizePoseAnalyticSimdFloat(const NdtMap& ndt_map,
                                   const std::vector<Vec3>& local_points,
                                   const Pose& initial_pose);
Pose OptimizePoseAnalyticSimdFloatFAST1(const NdtMap& ndt_map,
                                        const std::vector<Vec3>& local_points,
                                        const Pose& initial_pose);
Pose OptimizePoseAnalyticSimdFloatFAST2(const NdtMap& ndt_map,
                                        const std::vector<Vec3>& local_points,
                                        const Pose& initial_pose);
Pose OptimizePoseAnalyticSimdUsingHelper(const NdtMap& ndt_map,
                                         const std::vector<Vec3>& local_points,
                                         const Pose& initial_pose);
Pose OptimizePoseAnalyticSimdUsingHelperFloat(
    const NdtMap& ndt_map, const std::vector<Vec3>& local_points,
    const Pose& initial_pose);

}  // namespace mahalanobis_distance_minimizer
}  // namespace nonlinear_optimizer

using namespace nonlinear_optimizer;
using namespace nonlinear_optimizer::mahalanobis_distance_minimizer;

int main(int, char**) {
  constexpr double kVoxelResolution{1.0};
  constexpr double kFilterVoxelResolution{0.1};

  // Make global points
  const auto points = GenerateGlobalPoints();
  std::cerr << "# points: " << points.size() << std::endl;

  // Create NDT map by global points
  NdtMap ndt_map;
  UpdateNdtMap(points, kVoxelResolution, &ndt_map);
  std::cerr << "Ndt map size: " << ndt_map.size() << std::endl;

  // Set true pose
  Pose true_pose{Pose::Identity()};
  true_pose.translation() = Vec3(-0.2, 0.123, 0.3);
  true_pose.linear() =
      Eigen::AngleAxisd(0.1, Vec3(0.0, 0.0, 1.0)).toRotationMatrix();

  // Create local points
  const auto filtered_points = FilterPoints(points, kFilterVoxelResolution);
  const auto local_points = WarpPoints(filtered_points, true_pose.inverse());

  // Optimize pose
  Pose initial_pose{Pose::Identity()};
  std::cerr << "Start OptimizedPoseOriginal" << std::endl;
  const auto opt_pose_ceres_redundant =
      OptimizePoseOriginal(ndt_map, local_points, initial_pose);
  std::cerr << "Start OptimizePoseSimplified" << std::endl;
  const auto opt_pose_ceres_simplified =
      OptimizePoseSimplified(ndt_map, local_points, initial_pose);
  std::cerr << "Start OptimizePoseRedundantEach" << std::endl;
  const auto optimized_pose3 =
      OptimizePoseRedundantEach(ndt_map, local_points, initial_pose);
  std::cerr << "Start OptimizePoseAnalytic" << std::endl;
  const auto opt_pose_analytic =
      OptimizePoseAnalytic(ndt_map, local_points, initial_pose);
  std::cerr << "Start OptimizePoseAnalyticSIMD" << std::endl;
  const auto opt_pose_analytic_simd =
      OptimizePoseAnalyticSimd(ndt_map, local_points, initial_pose);
  std::cerr << "Start OptimizePoseAnalyticSIMDFloat" << std::endl;
  const auto opt_pose_analytic_simd_float =
      OptimizePoseAnalyticSimdFloat(ndt_map, local_points, initial_pose);
  std::cerr << "Start OptimizePoseAnalyticSIMDFloatFAST" << std::endl;
  const auto opt_pose_analytic_simd_float_fast =
      OptimizePoseAnalyticSimdFloatFAST1(ndt_map, local_points, initial_pose);
  std::cerr << "Start OptimizePoseAnalyticSIMDFloatFAST2" << std::endl;
  const auto opt_pose_analytic_simd_float_fast2 =
      OptimizePoseAnalyticSimdFloatFAST2(ndt_map, local_points, initial_pose);
  std::cerr << "Start OptimizePoseAnalyticSimdUsingHelper" << std::endl;
  const auto opt_pose_analytic_simd_using_helper =
      OptimizePoseAnalyticSimdUsingHelper(ndt_map, local_points, initial_pose);
  std::cerr << "Start OptimizePoseAnalyticSimdUsingHelperFloat" << std::endl;
  const auto opt_pose_analytic_simd_using_helper_float =
      OptimizePoseAnalyticSimdUsingHelperFloat(ndt_map, local_points,
                                               initial_pose);

  std::cerr << "Pose (ceres redundant): "
            << opt_pose_ceres_redundant.translation().transpose() << " "
            << Eigen::Quaterniond(opt_pose_ceres_redundant.linear())
                   .coeffs()
                   .transpose()
            << std::endl;
  std::cerr << "Pose (ceres simplified): "
            << opt_pose_ceres_simplified.translation().transpose() << " "
            << Eigen::Quaterniond(opt_pose_ceres_simplified.linear())
                   .coeffs()
                   .transpose()
            << std::endl;
  std::cerr << "Pose (ceres redundant, each): "
            << optimized_pose3.translation().transpose() << " "
            << Eigen::Quaterniond(optimized_pose3.linear()).coeffs().transpose()
            << std::endl;
  std::cerr
      << "Pose (analytic): " << opt_pose_analytic.translation().transpose()
      << " "
      << Eigen::Quaterniond(opt_pose_analytic.linear()).coeffs().transpose()
      << std::endl;
  std::cerr << "Pose (analytic simd): "
            << opt_pose_analytic_simd.translation().transpose() << " "
            << Eigen::Quaterniond(opt_pose_analytic_simd.linear())
                   .coeffs()
                   .transpose()
            << std::endl;
  std::cerr << "Pose (analytic simd float): "
            << opt_pose_analytic_simd_float.translation().transpose() << " "
            << Eigen::Quaterniond(opt_pose_analytic_simd_float.linear())
                   .coeffs()
                   .transpose()
            << std::endl;
  std::cerr << "Pose (analytic simd float fast): "
            << opt_pose_analytic_simd_float_fast.translation().transpose()
            << " "
            << Eigen::Quaterniond(opt_pose_analytic_simd_float_fast.linear())
                   .coeffs()
                   .transpose()
            << std::endl;
  std::cerr << "Pose (analytic simd float fast2): "
            << opt_pose_analytic_simd_float_fast2.translation().transpose()
            << " "
            << Eigen::Quaterniond(opt_pose_analytic_simd_float_fast2.linear())
                   .coeffs()
                   .transpose()
            << std::endl;
  std::cerr << "Pose (analytic simd using helper): "
            << opt_pose_analytic_simd_using_helper.translation().transpose()
            << " "
            << Eigen::Quaterniond(opt_pose_analytic_simd_using_helper.linear())
                   .coeffs()
                   .transpose()
            << std::endl;
  std::cerr
      << "Pose (analytic simd using helper float): "
      << opt_pose_analytic_simd_using_helper_float.translation().transpose()
      << " "
      << Eigen::Quaterniond(opt_pose_analytic_simd_using_helper_float.linear())
             .coeffs()
             .transpose()
      << std::endl;
  std::cerr << "True pose: " << true_pose.translation().transpose() << " "
            << Eigen::Quaterniond(true_pose.linear()).coeffs().transpose()
            << std::endl;

  return 0;
}

namespace nonlinear_optimizer {
namespace mahalanobis_distance_minimizer {

std::vector<Eigen::Vector3d> GenerateGlobalPoints() {
  const double width = 5.0;
  const double length = 7.0;
  const double height = 2.5;
  const double point_step = 0.01;

  std::vector<Eigen::Vector3d> points;

  // floor
  double x, y, z;
  z = 0.0;
  for (x = -length / 2.0; x <= length / 2.0; x += point_step)
    for (y = -width / 2.0; y <= width / 2.0; y += point_step)
      points.push_back(Eigen::Vector3d(x, y, z));

  // left/right wall
  y = -width / 2.0;
  for (x = -length / 2.0; x <= length / 2.0; x += point_step) {
    for (z = 0.0; z <= height; z += point_step) {
      points.push_back(Eigen::Vector3d(x, y, z));
      points.push_back(Eigen::Vector3d(x, -y, z));
    }
  }

  // front/back wall
  x = -length / 2.0;
  for (y = -width / 2.0; y <= width / 2.0; y += point_step) {
    for (z = 0.0; z <= height; z += point_step) {
      points.push_back(Eigen::Vector3d(-x, y, z));
      points.push_back(Eigen::Vector3d(x, y, z));
    }
  }

  return points;
}

std::vector<Vec3> FilterPoints(const std::vector<Vec3>& points,
                               const double filter_voxel_size) {
  const double voxel_resolution{filter_voxel_size};
  const double inv_res = 1.0 / voxel_resolution;
  std::unordered_set<VoxelKey> filtered_voxel_key_set;
  std::vector<Vec3> filtered_points;
  filtered_points.reserve(points.size());
  for (const auto& point : points) {
    const auto voxel_key = ComputeVoxelKey(point, inv_res);
    if (filtered_voxel_key_set.find(voxel_key) !=
        filtered_voxel_key_set.end()) {
      continue;
    }
    filtered_voxel_key_set.insert(voxel_key);
    filtered_points.push_back(point);
  }
  return filtered_points;
}

std::vector<Vec3> WarpPoints(const std::vector<Vec3>& points,
                             const Pose& pose) {
  std::vector<Vec3> warped_points;
  warped_points.reserve(points.size());
  for (const auto& point : points) {
    Vec3 warped_point = pose * point;
    warped_points.push_back(warped_point);
  }
  return warped_points;
}

void UpdateNdtMap(const std::vector<Vec3>& points,
                  const double voxel_resolution, NdtMap* ndt_map) {
  const double inv_res = 1.0 / voxel_resolution;

  std::unordered_set<VoxelKey> updated_voxel_key_set;
  for (const auto& point : points) {
    const auto voxel_key = ComputeVoxelKey(point, inv_res);
    updated_voxel_key_set.insert(voxel_key);

    auto& ndt = (*ndt_map)[voxel_key];
    ++ndt.count;
    ndt.sum += point;
    ndt.moment += point * point.transpose();
  }

  for (const auto& voxel_key : updated_voxel_key_set) {
    auto& ndt = ndt_map->at(voxel_key);
    if (ndt.count < 5) {
      ndt.is_valid = false;
      continue;
    }

    ndt.mean = ndt.sum / ndt.count;
    const Mat3x3 covariance =
        ndt.moment / ndt.count - ndt.mean * ndt.mean.transpose();

    Eigen::SelfAdjointEigenSolver<Mat3x3> eigsol(covariance);
    if (eigsol.info() != Eigen::Success || eigsol.eigenvalues()(2) < 0.01) {
      ndt.is_valid = false;
      return;
    }

    const double min_eigval_ratio = 0.01;

    ndt.is_valid = true;
    Vec3 eigvals = eigsol.eigenvalues();
    eigvals(0) = std::max(eigvals(0), eigvals(2) * min_eigval_ratio);
    eigvals(1) = std::max(eigvals(1), eigvals(2) * min_eigval_ratio);

    ndt.sqrt_information =
        eigvals.cwiseInverse().cwiseSqrt().asDiagonal() * eigsol.eigenvectors();

    ndt.information = ndt.sqrt_information.transpose() * ndt.sqrt_information;
  }
}

VoxelKey ComputeVoxelKey(const Vec3& point,
                         const double inverse_voxel_resolution) {
  // Note: Cantor pairing function
  int x_key = std::floor(point.x() * inverse_voxel_resolution);
  int y_key = std::floor(point.y() * inverse_voxel_resolution);
  int z_key = std::floor(point.z() * inverse_voxel_resolution);
  x_key = x_key >= 0 ? 2 * x_key : -2 * x_key - 1;
  y_key = y_key >= 0 ? 2 * y_key : -2 * y_key - 1;
  z_key = z_key >= 0 ? 2 * z_key : -2 * z_key - 1;
  uint64_t xy_key = (x_key + y_key) * (x_key + y_key + 1) / 2 + y_key;
  uint64_t xyz_key = (xy_key + z_key) * (xy_key + z_key + 1) / 2 + z_key;
  return xyz_key;
}

std::vector<Correspondence> MatchPointCloud(
    const NdtMap& ndt_map, const std::vector<Vec3>& local_points,
    const Pose& pose) {
  constexpr int kNumNeighbors{2};
  constexpr double kSearchRadius{1.0};

  const auto points = WarpPoints(local_points, pose);

  // Generate kdtree of the voxel_key-ndt mean pair
  flann::KDTreeSingleIndex<flann::L2_Simple<double>> kdtree;
  std::vector<VoxelKey> voxel_keys;
  std::vector<Vec3> ndt_means;
  for (const auto& [voxel_key, ndt] : ndt_map) {
    if (!ndt.is_valid) continue;
    voxel_keys.push_back(voxel_key);
    ndt_means.push_back(ndt.mean);
  }
  double* ndt_means_array = const_cast<double*>(ndt_means.data()->data());
  kdtree.buildIndex(
      flann::Matrix<double>(ndt_means_array, ndt_means.size(), 3));

  // Find the nearest neighbors in the kdtree
  auto search_params = flann::SearchParams();
  search_params.max_neighbors = kNumNeighbors;
  search_params.eps = 0.0;

  std::vector<Correspondence> correspondences;
  correspondences.reserve(points.size());
  for (size_t index = 0; index < points.size(); ++index) {
    const auto& point = points.at(index);
    const auto& local_point = local_points.at(index);
    Correspondence corr;
    corr.point = local_point;

    std::vector<std::vector<size_t>> matched_ndt_mean_indices;
    std::vector<std::vector<double>> distances;
    flann::Matrix<double> flann_query(const_cast<double*>(point.data()), 1, 3);
    if (kdtree.radiusSearch(flann_query, matched_ndt_mean_indices, distances,
                            kSearchRadius, search_params)) {
      for (const auto& matched_ndt_mean_index : matched_ndt_mean_indices[0]) {
        corr.ndt = ndt_map.at(voxel_keys[matched_ndt_mean_index]);
        correspondences.push_back(corr);
      }
    }
  }
  return correspondences;
}

Pose OptimizePoseOriginal(const NdtMap& ndt_map,
                          const std::vector<Vec3>& local_points,
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
  int outer_iter = 0;
  for (; outer_iter < 10; ++outer_iter) {
    Pose last_optimized_pose{Pose::Identity()};
    last_optimized_pose.translation() =
        Vec3(optimized_translation[0], optimized_translation[1],
             optimized_translation[2]);
    last_optimized_pose.linear() =
        Eigen::Quaterniond(optimized_orientation[0], optimized_orientation[1],
                           optimized_orientation[2], optimized_orientation[3])
            .toRotationMatrix();

    // Find correspondences
    const auto correspondences =
        MatchPointCloud(ndt_map, local_points, last_optimized_pose);

    ceres::Problem problem;
    ceres::CostFunction* cost_function =
        nonlinear_optimizer::mahalanobis_distance_minimizer::
            RedundantMahalanobisDistanceCostFunctorBatch::Create(
                correspondences, 1.0, 1.0);
    problem.AddResidualBlock(cost_function, nullptr, optimized_translation,
                             optimized_orientation);

    ceres::Solver::Options options;
    options.linear_solver_type = ceres::DENSE_QR;
    options.max_num_iterations = 30;
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);
    // std::cerr << "Summary: " << summary.BriefReport() << std::endl;

    Pose current_optimized_pose{Pose::Identity()};
    current_optimized_pose.translation() =
        Vec3(optimized_translation[0], optimized_translation[1],
             optimized_translation[2]);
    current_optimized_pose.linear() =
        Orientation(optimized_orientation[0], optimized_orientation[1],
                    optimized_orientation[2], optimized_orientation[3])
            .toRotationMatrix();
    optimized_pose = current_optimized_pose;
    Pose pose_diff = current_optimized_pose.inverse() * last_optimized_pose;
    if (pose_diff.translation().norm() < 1e-5 &&
        Orientation(pose_diff.linear()).vec().norm() < 1e-5) {
      break;
    }
  }

  std::cerr << "outer_iter: " << outer_iter << std::endl;

  return optimized_pose;
}

Pose OptimizePoseRedundantEach(const NdtMap& ndt_map,
                               const std::vector<Vec3>& local_points,
                               const Pose& initial_pose) {
  CHECK_EXEC_TIME_FROM_HERE

  Pose optimized_pose = initial_pose;
  Pose last_optimized_pose = optimized_pose;
  int outer_iter = 0;
  for (; outer_iter < 10; ++outer_iter) {
    const auto correspondences =
        MatchPointCloud(ndt_map, local_points, optimized_pose);

    std::unique_ptr<nonlinear_optimizer::mahalanobis_distance_minimizer::
                        MahalanobisDistanceMinimizerCeres>
        optim = std::make_unique<
            nonlinear_optimizer::mahalanobis_distance_minimizer::
                MahalanobisDistanceMinimizerCeres>();

    optim->SolveByRedundantForEach(correspondences, &optimized_pose);

    Pose pose_diff = optimized_pose.inverse() * last_optimized_pose;
    if (pose_diff.translation().norm() < 1e-5 &&
        Orientation(pose_diff.linear()).vec().norm() < 1e-5) {
      break;
    }
    last_optimized_pose = optimized_pose;
  }
  std::cerr << "outer_iter: " << outer_iter << std::endl;

  return optimized_pose;
}

Pose OptimizePoseSimplified(const NdtMap& ndt_map,
                            const std::vector<Vec3>& local_points,
                            const Pose& initial_pose) {
  CHECK_EXEC_TIME_FROM_HERE

  Pose optimized_pose = initial_pose;
  Pose last_optimized_pose = optimized_pose;
  int outer_iter = 0;
  for (; outer_iter < 10; ++outer_iter) {
    const auto correspondences =
        MatchPointCloud(ndt_map, local_points, optimized_pose);

    std::unique_ptr<nonlinear_optimizer::mahalanobis_distance_minimizer::
                        MahalanobisDistanceMinimizer>
        optim = std::make_unique<
            nonlinear_optimizer::mahalanobis_distance_minimizer::
                MahalanobisDistanceMinimizerCeres>();

    optim->Solve(correspondences, &optimized_pose);

    Pose pose_diff = optimized_pose.inverse() * last_optimized_pose;
    if (pose_diff.translation().norm() < 1e-5 &&
        Orientation(pose_diff.linear()).vec().norm() < 1e-5) {
      break;
    }
    last_optimized_pose = optimized_pose;
  }
  std::cerr << "outer_iter: " << outer_iter << std::endl;

  return optimized_pose;
}

Pose OptimizePoseAnalytic(const NdtMap& ndt_map,
                          const std::vector<Vec3>& local_points,
                          const Pose& initial_pose) {
  CHECK_EXEC_TIME_FROM_HERE

  Pose optimized_pose = initial_pose;
  Pose last_optimized_pose = optimized_pose;
  int outer_iter = 0;
  for (; outer_iter < 10; ++outer_iter) {
    const auto correspondences =
        MatchPointCloud(ndt_map, local_points, optimized_pose);

    std::unique_ptr<nonlinear_optimizer::mahalanobis_distance_minimizer::
                        MahalanobisDistanceMinimizer>
        optim = std::make_unique<
            nonlinear_optimizer::mahalanobis_distance_minimizer::
                MahalanobisDistanceMinimizerAnalytic>();
    optim->SetLossFunction(
        std::make_shared<nonlinear_optimizer::ExponentialLossFunction>(1.0,
                                                                       1.0));
    optim->Solve(correspondences, &optimized_pose);

    Pose pose_diff = optimized_pose.inverse() * last_optimized_pose;
    if (pose_diff.translation().norm() < 1e-5 &&
        Orientation(pose_diff.linear()).vec().norm() < 1e-5) {
      break;
    }
    last_optimized_pose = optimized_pose;
  }
  std::cerr << "outer_iter: " << outer_iter << std::endl;

  return optimized_pose;
}

Pose OptimizePoseAnalyticSimd(const NdtMap& ndt_map,
                              const std::vector<Vec3>& local_points,
                              const Pose& initial_pose) {
  CHECK_EXEC_TIME_FROM_HERE

  Pose optimized_pose = initial_pose;
  Pose last_optimized_pose = optimized_pose;
  int outer_iter = 0;
  for (; outer_iter < 10; ++outer_iter) {
    const auto correspondences =
        MatchPointCloud(ndt_map, local_points, optimized_pose);

    std::unique_ptr<nonlinear_optimizer::mahalanobis_distance_minimizer::
                        MahalanobisDistanceMinimizer>
        optim = std::make_unique<
            nonlinear_optimizer::mahalanobis_distance_minimizer::
                MahalanobisDistanceMinimizerAnalyticSIMD>();
    optim->SetLossFunction(
        std::make_shared<nonlinear_optimizer::ExponentialLossFunction>(1.0,
                                                                       1.0));
    optim->Solve(correspondences, &optimized_pose);

    Pose pose_diff = optimized_pose.inverse() * last_optimized_pose;
    if (pose_diff.translation().norm() < 1e-5 &&
        Orientation(pose_diff.linear()).vec().norm() < 1e-5) {
      break;
    }
    last_optimized_pose = optimized_pose;
  }
  std::cerr << "outer_iter: " << outer_iter << std::endl;

  return optimized_pose;
}

Pose OptimizePoseAnalyticSimdFloat(const NdtMap& ndt_map,
                                   const std::vector<Vec3>& local_points,
                                   const Pose& initial_pose) {
  CHECK_EXEC_TIME_FROM_HERE

  Pose optimized_pose = initial_pose;
  Pose last_optimized_pose = optimized_pose;
  int outer_iter = 0;
  for (; outer_iter < 10; ++outer_iter) {
    const auto correspondences =
        MatchPointCloud(ndt_map, local_points, optimized_pose);

    std::unique_ptr<nonlinear_optimizer::mahalanobis_distance_minimizer::
                        MahalanobisDistanceMinimizerAnalyticSIMD>
        optim = std::make_unique<
            nonlinear_optimizer::mahalanobis_distance_minimizer::
                MahalanobisDistanceMinimizerAnalyticSIMD>();
    optim->SetLossFunction(
        std::make_shared<nonlinear_optimizer::ExponentialLossFunction>(1.0,
                                                                       1.0));
    optim->SolveFloat(correspondences, &optimized_pose);

    Pose pose_diff = optimized_pose.inverse() * last_optimized_pose;
    if (pose_diff.translation().norm() < 1e-5 &&
        Orientation(pose_diff.linear()).vec().norm() < 1e-5) {
      break;
    }
    last_optimized_pose = optimized_pose;
  }
  std::cerr << "outer_iter: " << outer_iter << std::endl;

  return optimized_pose;
}

Pose OptimizePoseAnalyticSimdFloatFAST1(const NdtMap& ndt_map,
                                        const std::vector<Vec3>& local_points,
                                        const Pose& initial_pose) {
  CHECK_EXEC_TIME_FROM_HERE

  Pose optimized_pose = initial_pose;
  Pose last_optimized_pose = optimized_pose;
  int outer_iter = 0;
  for (; outer_iter < 10; ++outer_iter) {
    const auto correspondences =
        MatchPointCloud(ndt_map, local_points, optimized_pose);

    std::unique_ptr<nonlinear_optimizer::mahalanobis_distance_minimizer::
                        MahalanobisDistanceMinimizerAnalyticSIMD>
        optim = std::make_unique<
            nonlinear_optimizer::mahalanobis_distance_minimizer::
                MahalanobisDistanceMinimizerAnalyticSIMD>();
    optim->SetLossFunction(
        std::make_shared<nonlinear_optimizer::ExponentialLossFunction>(1.0,
                                                                       1.0));
    optim->SolveFloat_FAST1(correspondences, &optimized_pose);

    Pose pose_diff = optimized_pose.inverse() * last_optimized_pose;
    if (pose_diff.translation().norm() < 1e-5 &&
        Orientation(pose_diff.linear()).vec().norm() < 1e-5) {
      break;
    }
    last_optimized_pose = optimized_pose;
  }
  std::cerr << "outer_iter: " << outer_iter << std::endl;

  return optimized_pose;
}

Pose OptimizePoseAnalyticSimdFloatFAST2(const NdtMap& ndt_map,
                                        const std::vector<Vec3>& local_points,
                                        const Pose& initial_pose) {
  CHECK_EXEC_TIME_FROM_HERE

  Pose optimized_pose = initial_pose;
  Pose last_optimized_pose = optimized_pose;
  int outer_iter = 0;
  for (; outer_iter < 10; ++outer_iter) {
    const auto correspondences =
        MatchPointCloud(ndt_map, local_points, optimized_pose);

    std::unique_ptr<nonlinear_optimizer::mahalanobis_distance_minimizer::
                        MahalanobisDistanceMinimizerAnalyticSIMD>
        optim = std::make_unique<
            nonlinear_optimizer::mahalanobis_distance_minimizer::
                MahalanobisDistanceMinimizerAnalyticSIMD>();
    optim->SetLossFunction(
        std::make_shared<nonlinear_optimizer::ExponentialLossFunction>(1.0,
                                                                       1.0));
    optim->SolveFloat_FAST2(correspondences, &optimized_pose);

    Pose pose_diff = optimized_pose.inverse() * last_optimized_pose;
    if (pose_diff.translation().norm() < 1e-5 &&
        Orientation(pose_diff.linear()).vec().norm() < 1e-5) {
      break;
    }
    last_optimized_pose = optimized_pose;
  }
  std::cerr << "outer_iter: " << outer_iter << std::endl;

  return optimized_pose;
}

Pose OptimizePoseAnalyticSimdUsingHelper(const NdtMap& ndt_map,
                                         const std::vector<Vec3>& local_points,
                                         const Pose& initial_pose) {
  CHECK_EXEC_TIME_FROM_HERE

  Pose optimized_pose = initial_pose;
  Pose last_optimized_pose = optimized_pose;
  int outer_iter = 0;
  for (; outer_iter < 10; ++outer_iter) {
    const auto correspondences =
        MatchPointCloud(ndt_map, local_points, optimized_pose);

    std::unique_ptr<nonlinear_optimizer::mahalanobis_distance_minimizer::
                        MahalanobisDistanceMinimizerAnalyticSIMD>
        optim = std::make_unique<
            nonlinear_optimizer::mahalanobis_distance_minimizer::
                MahalanobisDistanceMinimizerAnalyticSIMD>();
    optim->SetLossFunction(
        std::make_shared<nonlinear_optimizer::ExponentialLossFunction>(1.0,
                                                                       1.0));
    optim->SolveUsingHelper(correspondences, &optimized_pose);

    Pose pose_diff = optimized_pose.inverse() * last_optimized_pose;
    if (pose_diff.translation().norm() < 1e-5 &&
        Orientation(pose_diff.linear()).vec().norm() < 1e-5) {
      break;
    }
    last_optimized_pose = optimized_pose;
  }
  std::cerr << "outer_iter: " << outer_iter << std::endl;

  return optimized_pose;
}

Pose OptimizePoseAnalyticSimdUsingHelperFloat(
    const NdtMap& ndt_map, const std::vector<Vec3>& local_points,
    const Pose& initial_pose) {
  CHECK_EXEC_TIME_FROM_HERE

  Pose optimized_pose = initial_pose;
  Pose last_optimized_pose = optimized_pose;
  int outer_iter = 0;
  for (; outer_iter < 10; ++outer_iter) {
    const auto correspondences =
        MatchPointCloud(ndt_map, local_points, optimized_pose);

    std::unique_ptr<nonlinear_optimizer::mahalanobis_distance_minimizer::
                        MahalanobisDistanceMinimizerAnalyticSIMD>
        optim = std::make_unique<
            nonlinear_optimizer::mahalanobis_distance_minimizer::
                MahalanobisDistanceMinimizerAnalyticSIMD>();
    optim->SetLossFunction(
        std::make_shared<nonlinear_optimizer::ExponentialLossFunction>(1.0,
                                                                       1.0));
    optim->SolveUsingHelperFloat(correspondences, &optimized_pose);

    Pose pose_diff = optimized_pose.inverse() * last_optimized_pose;
    if (pose_diff.translation().norm() < 1e-5 &&
        Orientation(pose_diff.linear()).vec().norm() < 1e-5) {
      break;
    }
    last_optimized_pose = optimized_pose;
  }
  std::cerr << "outer_iter: " << outer_iter << std::endl;

  return optimized_pose;
}

}  // namespace mahalanobis_distance_minimizer
}  // namespace nonlinear_optimizer