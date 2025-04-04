#ifndef NONLINEAR_OPTIMIZER_REPROJECTION_ERROR_MINIMIZER_CERES_COST_FUNCTOR_H_
#define NONLINEAR_OPTIMIZER_REPROJECTION_ERROR_MINIMIZER_CERES_COST_FUNCTOR_H_

#include "types.h"

#include "Eigen/Dense"
#include "ceres/ceres.h"

namespace nonlinear_optimizer {
namespace reprojection_error_minimizer {

class ReprojectionErrorCostFunctorBatch {
 public:
  ReprojectionErrorCostFunctorBatch(
      const std::vector<Correspondence>& correspondences,
      const CameraIntrinsics& camera_intrinsics, const double c1,
      const double c2)
      : camera_intrinsics_(camera_intrinsics), c1_(c1), c2_(c2) {
    correspondences_ = correspondences;
  }

  template <typename T>
  bool operator()(const T* const translation_ptr, const T* const quaternion_ptr,
                  T* residual) const {
    double inv_fx = 1.0 / camera_intrinsics_.fx;
    double inv_fy = 1.0 / camera_intrinsics_.fy;
    double cx = camera_intrinsics_.cx;
    double cy = camera_intrinsics_.cy;
    Eigen::Matrix<T, 3, 1> translation(translation_ptr[0], translation_ptr[1],
                                       translation_ptr[2]);
    Eigen::Quaternion<T> rotation(quaternion_ptr[0], quaternion_ptr[1],
                                  quaternion_ptr[2], quaternion_ptr[3]);
    for (size_t i = 0; i < correspondences_.size(); ++i) {
      const auto& corr = correspondences_[i];
      Eigen::Matrix<T, 3, 1> warped_point =
          rotation * corr.local_point.cast<T>() + translation;
      if (warped_point(2) < T(0.0)) {
        residual[i] = T(0.0);
        continue;
      }
      Eigen::Matrix<T, 2, 1> projected_image_coordinate;
      projected_image_coordinate(0) = warped_point(0) / warped_point(2);
      projected_image_coordinate(1) = warped_point(1) / warped_point(2);
      Eigen::Matrix<double, 2, 1> matched_image_coordinate;
      matched_image_coordinate(0) = inv_fx * (corr.matched_pixel.x() - cx);
      matched_image_coordinate(1) = inv_fy * (corr.matched_pixel.y() - cy);
      Eigen::Matrix<T, 2, 1> reproj_error =
          projected_image_coordinate - matched_image_coordinate.cast<T>();
      T squared_reproj_error = reproj_error.transpose() * reproj_error;
      residual[i] = c1_ - c1_ * ceres::exp(-c2_ * squared_reproj_error);
    }
    return true;
  }

  static ceres::CostFunction* Create(
      const std::vector<Correspondence>& correspondences,
      const CameraIntrinsics& camera_intrinsics, const double c1,
      const double c2) {
    return new ceres::AutoDiffCostFunction<ReprojectionErrorCostFunctorBatch,
                                           ceres::DYNAMIC, 3, 4>(
        new ReprojectionErrorCostFunctorBatch(correspondences,
                                              camera_intrinsics, c1, c2),
        correspondences.size());
  }

 private:
  std::vector<Correspondence> correspondences_;
  const CameraIntrinsics camera_intrinsics_;
  const double c1_;
  const double c2_;
};

class ReprojectionErrorWithExponentialLossCostFunctor {
 public:
  ReprojectionErrorWithExponentialLossCostFunctor(
      const Correspondence& correspondence,
      const CameraIntrinsics& camera_intrinsics, const double c1,
      const double c2)
      : camera_intrinsics_(camera_intrinsics), c1_(c1), c2_(c2) {
    correspondence_ = correspondence;
  }

  template <typename T>
  bool operator()(const T* const translation_ptr, const T* const quaternion_ptr,
                  T* residual) const {
    Eigen::Matrix<T, 3, 1> translation(translation_ptr[0], translation_ptr[1],
                                       translation_ptr[2]);
    Eigen::Quaternion<T> rotation(quaternion_ptr[0], quaternion_ptr[1],
                                  quaternion_ptr[2], quaternion_ptr[3]);
    const auto& corr = correspondence_;
    Eigen::Matrix<T, 3, 1> warped_point =
        rotation * corr.local_point.cast<T>() + translation;

    if (warped_point(2) < 0.0) return false;

    T fx = T(camera_intrinsics_.fx);
    T fy = T(camera_intrinsics_.fy);
    T cx = T(camera_intrinsics_.cx);
    T cy = T(camera_intrinsics_.cy);
    Eigen::Matrix<T, 2, 1> projected_pixel;
    projected_pixel(0) = fx * warped_point(0) / warped_point(2) + cx;
    projected_pixel(1) = fy * warped_point(1) / warped_point(2) + cy;
    Eigen::Matrix<T, 2, 1> reproj_error =
        projected_pixel - corr.matched_pixel.cast<T>();
    T squared_reproj_error = reproj_error.transpose() * reproj_error;
    residual[0] = c1_ - c1_ * ceres::exp(-c2_ * squared_reproj_error);
    return true;
  }

  static ceres::CostFunction* Create(const Correspondence& correspondence,
                                     const CameraIntrinsics& camera_intrinsics,
                                     const double c1, const double c2) {
    return new ceres::AutoDiffCostFunction<
        ReprojectionErrorWithExponentialLossCostFunctor, 1, 3, 4>(
        new ReprojectionErrorWithExponentialLossCostFunctor(
            correspondence, camera_intrinsics, c1, c2));
  }

 private:
  Correspondence correspondence_;
  const CameraIntrinsics camera_intrinsics_;
  const double c1_;
  const double c2_;
};

// class ReprojectionErrorCostFunctor {
//  public:
//   ReprojectionErrorCostFunctor(const Correspondence& correspondence) {
//     correspondence_ = correspondence;
//   }

//   template <typename T>
//   bool operator()(const T* const translation_ptr, const T* const
//   quaternion_ptr,
//                   T* residual) const {
//     Eigen::Matrix<T, 3, 1> translation(translation_ptr[0],
//     translation_ptr[1],
//                                        translation_ptr[2]);
//     Eigen::Quaternion<T> rotation(quaternion_ptr[0], quaternion_ptr[1],
//                                   quaternion_ptr[2], quaternion_ptr[3]);
//     const auto& corr = correspondence_;
//     Eigen::Matrix<T, 3, 1> warped_point =
//         rotation * corr.point.cast<T>() + translation;
//     // Eigen::Matrix<T, 3, 1> e_i = warped_point - corr.ndt.mean.cast<T>();
//     // T squared_mahalanobis_dist =
//     //     e_i.transpose() * corr.ndt.information.cast<T>() * e_i;
//     // residual[0] = c1_ - c1_ * ceres::exp(-c2_ * squared_mahalanobis_dist);
//     return true;
//   }

//   static ceres::CostFunction* Create(const Correspondence& correspondence) {
//     return new ceres::AutoDiffCostFunction<ReprojectionErrorCostFunctor, 2,
//     3,
//                                            4>(
//         new ReprojectionErrorCostFunctor(correspondence));
//   }

//  private:
//   Correspondence correspondence_;
// };

}  // namespace reprojection_error_minimizer
}  // namespace nonlinear_optimizer

#endif  // NONLINEAR_OPTIMIZER_REPROJECTION_ERROR_MINIMIZER_CERES_COST_FUNCTOR_H_