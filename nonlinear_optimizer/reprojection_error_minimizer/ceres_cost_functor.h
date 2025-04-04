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
      const std::vector<Correspondence>& correspondences, const double c1,
      const double c2)
      : c1_(c1), c2_(c2) {
    correspondences_ = correspondences;
  }

  template <typename T>
  bool operator()(const T* const translation_ptr, const T* const quaternion_ptr,
                  T* residual) const {
    Eigen::Matrix<T, 3, 1> translation(translation_ptr[0], translation_ptr[1],
                                       translation_ptr[2]);
    Eigen::Quaternion<T> rotation(quaternion_ptr[0], quaternion_ptr[1],
                                  quaternion_ptr[2], quaternion_ptr[3]);
    for (size_t i = 0; i < correspondences_.size(); ++i) {
      const auto& corr = correspondences_[i];
      Eigen::Matrix<T, 3, 1> warped_point =
          rotation * corr.point.cast<T>() + translation;
      // Eigen::Matrix<T, 3, 1> e_i = warped_point - corr.ndt.mean.cast<T>();
      // T squared_mahalanobis_dist =
      //     e_i.transpose() * corr.ndt.information.cast<T>() * e_i;
      // residual[i] = c1_ - c1_ * ceres::exp(-c2_ * squared_mahalanobis_dist);
    }
    return true;
  }

  static ceres::CostFunction* Create(
      const std::vector<Correspondence>& correspondences, const double c1,
      const double c2) {
    return new ceres::AutoDiffCostFunction<ReprojectionErrorCostFunctorBatch,
                                           ceres::DYNAMIC, 3, 4>(
        new ReprojectionErrorCostFunctorBatch(correspondences, c1, c2),
        correspondences.size());
  }

 private:
  std::vector<Correspondence> correspondences_;
  const double c1_;
  const double c2_;
};

class ReprojectionErrorCostFunctor {
 public:
  ReprojectionErrorCostFunctor(const Correspondence& correspondence,
                               const double c1, const double c2)
      : c1_(c1), c2_(c2) {
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
        rotation * corr.point.cast<T>() + translation;
    // Eigen::Matrix<T, 3, 1> e_i = warped_point - corr.ndt.mean.cast<T>();
    // T squared_mahalanobis_dist =
    //     e_i.transpose() * corr.ndt.information.cast<T>() * e_i;
    // residual[0] = c1_ - c1_ * ceres::exp(-c2_ * squared_mahalanobis_dist);
    return true;
  }

  static ceres::CostFunction* Create(const Correspondence& correspondence,
                                     const double c1, const double c2) {
    return new ceres::AutoDiffCostFunction<ReprojectionErrorCostFunctor, 1, 3,
                                           4>(
        new ReprojectionErrorCostFunctor(correspondence, c1, c2));
  }

 private:
  Correspondence correspondence_;
  const double c1_;
  const double c2_;
};

}  // namespace reprojection_error_minimizer
}  // namespace nonlinear_optimizer

#endif  // NONLINEAR_OPTIMIZER_REPROJECTION_ERROR_MINIMIZER_CERES_COST_FUNCTOR_H_