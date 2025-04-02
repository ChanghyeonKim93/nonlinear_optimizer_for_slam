#include "nonlinear_optimizer/mahalanobis_distance_minimizer/mahalanobis_distance_minimizer_analytic_simd.h"

#include <iostream>

#include "nonlinear_optimizer/simd_helper.h"

namespace nonlinear_optimizer {
namespace mahalanobis_distance_minimizer {

MahalanobisDistanceMinimizerAnalyticSIMD::
    MahalanobisDistanceMinimizerAnalyticSIMD() {}

MahalanobisDistanceMinimizerAnalyticSIMD::
    ~MahalanobisDistanceMinimizerAnalyticSIMD() {}

bool MahalanobisDistanceMinimizerAnalyticSIMD::SolveUsingHelper(
    const std::vector<Correspondence>& correspondences, Pose* pose) {
  constexpr int max_iteration = 30;
  constexpr double min_lambda = 1e-6;
  constexpr double max_lambda = 1e-2;

  const Pose initial_pose = *pose;

  Vec3 optimized_translation{initial_pose.translation()};
  Orientation optimized_orientation(initial_pose.rotation());

  double lambda = 0.001;
  double previous_cost = std::numeric_limits<double>::max();
  int iteration = 0;
  for (; iteration < max_iteration; ++iteration) {
    const Mat3x3 opt_R = optimized_orientation.toRotationMatrix();
    const Vec3 opt_t = optimized_translation;
    simd::Matrix<3, 3> R__(opt_R);
    simd::Vector<3> t__(opt_t);

    simd::Vector<6> gradient__(Eigen::Matrix<double, 6, 1>::Zero());
    simd::Matrix<6, 6> hessian__(Eigen::Matrix<double, 6, 6>::Zero());
    simd::Scalar cost__(0.0);
    const size_t stride = simd::Scalar::GetDataStep();
    const int num_stride = correspondences.size() / stride;
    for (size_t point_idx = 0; point_idx < num_stride * stride;
         point_idx += stride) {
      simd::Vector<3> p__;
      simd::Vector<3> mu__;
      simd::Matrix<3, 3> sqrt_info__;

      const auto& corr1 = correspondences.at(point_idx);
      const auto& corr2 = correspondences.at(point_idx + 1);
      const auto& corr3 = correspondences.at(point_idx + 2);
      const auto& corr4 = correspondences.at(point_idx + 3);
      p__ =
          simd::Vector<3>({corr1.point, corr2.point, corr3.point, corr4.point});
      mu__ = simd::Vector<3>(
          {corr1.ndt.mean, corr2.ndt.mean, corr3.ndt.mean, corr4.ndt.mean});
      sqrt_info__ = simd::Matrix<3, 3>(
          {corr1.ndt.sqrt_information, corr2.ndt.sqrt_information,
           corr3.ndt.sqrt_information, corr4.ndt.sqrt_information});

      // clang-format off
      // pw = R*p + t
      const simd::Vector<3> pw__ = R__ * p__ + t__;

      // e_i = pw - mean
      const simd::Vector<3> e__ = pw__ - mu__;

      // r = sqrt_info * e
      const simd::Vector<3> r__ = sqrt_info__ * e__;

      simd::Matrix<3,3> minus_R_skewp__;
      minus_R_skewp__(0, 0) =  R__(0, 2) * p__(1) - R__(0, 1) * p__(2);
      minus_R_skewp__(1, 0) =  R__(1, 2) * p__(1) - R__(1, 1) * p__(2);
      minus_R_skewp__(2, 0) =  R__(2, 2) * p__(1) - R__(2, 1) * p__(2);
      minus_R_skewp__(0, 1) =  R__(0, 0) * p__(2) - R__(0, 2) * p__(0);
      minus_R_skewp__(1, 1) =  R__(1, 0) * p__(2) - R__(1, 2) * p__(0);
      minus_R_skewp__(2, 1) =  R__(2, 0) * p__(2) - R__(2, 2) * p__(0);
      minus_R_skewp__(0, 2) =  R__(0, 1) * p__(0) - R__(0, 0) * p__(1);
      minus_R_skewp__(1, 2) =  R__(1, 1) * p__(0) - R__(1, 0) * p__(1);
      minus_R_skewp__(2, 2) =  R__(2, 1) * p__(0) - R__(2, 0) * p__(1);
      const simd::Matrix<3,3> sqrt_info_minus_R_skewp_ = sqrt_info__ * minus_R_skewp__;

      simd::Matrix<3,6> J__;
      J__(0, 0) = sqrt_info__(0, 0); J__(0, 1) = sqrt_info__(0, 1); J__(0, 2) = sqrt_info__(0, 2);
      J__(1, 0) = sqrt_info__(1, 0); J__(1, 1) = sqrt_info__(1, 1); J__(1, 2) = sqrt_info__(1, 2);
      J__(2, 0) = sqrt_info__(2, 0); J__(2, 1) = sqrt_info__(2, 1); J__(2, 2) = sqrt_info__(2, 2);
      J__(0, 3) = sqrt_info_minus_R_skewp_(0, 0); J__(0, 4) = sqrt_info_minus_R_skewp_(0, 1); J__(0, 5) = sqrt_info_minus_R_skewp_(0, 2);
      J__(1, 3) = sqrt_info_minus_R_skewp_(1, 0); J__(1, 4) = sqrt_info_minus_R_skewp_(1, 1); J__(1, 5) = sqrt_info_minus_R_skewp_(1, 2);
      J__(2, 3) = sqrt_info_minus_R_skewp_(2, 0); J__(2, 4) = sqrt_info_minus_R_skewp_(2, 1); J__(2, 5) = sqrt_info_minus_R_skewp_(2, 2);
      // clang-format on

      // Compute loss and weight,
      // and add the local gradient and hessian to the global ones
      simd::Scalar sq_r__ = r__.GetNorm();
      simd::Scalar loss__(sq_r__);
      simd::Scalar weight__(1.0);
      if (loss_function_ != nullptr) {
        double sq_r_buf[4];
        sq_r__.StoreData(sq_r_buf);
        double loss_buf[4];
        double weight_buf[4];
        for (int k = 0; k < 4; ++k) {
          double loss_output[3] = {0.0, 0.0, 0.0};
          loss_function_->Evaluate(sq_r_buf[k], loss_output);
          loss_buf[k] = loss_output[0];
          weight_buf[k] = loss_output[1];
        }
        loss__ = simd::Scalar(loss_buf);
        weight__ = simd::Scalar(weight_buf);
      }

      // g(i) += (J(0,i)*r(0) + J(1,i)*r(1) + J(2,i)*r(2))
      gradient__ += (J__.transpose() * r__) * weight__;

      // H(i,j) = sum_{k} w * J(k,i) * J(k,j)
      for (int ii = 0; ii < 6; ++ii) {
        for (int jj = ii; jj < 6; ++jj) {
          hessian__(ii, jj) +=
              (weight__ * (J__(0, ii) * J__(0, jj) + J__(1, ii) * J__(1, jj) +
                           J__(2, ii) * J__(2, jj)));
        }
      }

      cost__ += loss__;
    }
    double buf[4];
    cost__.StoreData(buf);
    double cost = 0.0;
    cost += (buf[0] + buf[1] + buf[2] + buf[3]);

    Mat6x6 hessian{Mat6x6::Zero()};
    Vec6 gradient{Vec6::Zero()};
    for (int ii = 0; ii < 6; ++ii) {
      gradient__(ii).StoreData(buf);
      gradient(ii) += (buf[0] + buf[1] + buf[2] + buf[3]);
      for (int jj = ii; jj < 6; ++jj) {
        hessian__(ii, jj).StoreData(buf);
        hessian(ii, jj) += (buf[0] + buf[1] + buf[2] + buf[3]);
      }
    }

    // Reflect the hessian
    ReflectHessian(&hessian);

    // Damping hessian
    for (int k = 0; k < 6; k++) hessian(k, k) *= 1.0 + lambda;

    // Compute the step
    const Vec6 update_step = hessian.ldlt().solve(-gradient);

    // Update the pose
    const Vec3 delta_t = update_step.block<3, 1>(0, 0);
    const Vec3 delta_R = update_step.block<3, 1>(3, 0);
    optimized_translation += delta_t;
    optimized_orientation *= ComputeQuaternion(delta_R);
    optimized_orientation.normalize();

    // Check convergence
    if (update_step.norm() < 1e-7) {
      break;
    }
    if (gradient.norm() < 1e-7) {
      break;
    }

    if (cost > previous_cost) {
      lambda *= 2.0;
    } else {
      lambda *= 0.6;
    }
    lambda = std::clamp(lambda, min_lambda, max_lambda);
    previous_cost = cost;
  }
  std::cerr << "COST: " << previous_cost << ", iter: " << iteration
            << std::endl;

  pose->translation() = optimized_translation;
  pose->linear() = optimized_orientation.toRotationMatrix();

  return true;
}

bool MahalanobisDistanceMinimizerAnalyticSIMD::Solve(
    const std::vector<Correspondence>& correspondences, Pose* pose) {
  constexpr int max_iteration = 30;
  constexpr double min_lambda = 1e-6;
  constexpr double max_lambda = 1e-2;

  const Pose initial_pose = *pose;

  Vec3 optimized_translation{initial_pose.translation()};
  Orientation optimized_orientation(initial_pose.rotation());

  double lambda = 0.001;
  double previous_cost = std::numeric_limits<double>::max();
  int iteration = 0;
  for (; iteration < max_iteration; ++iteration) {
    const Mat3x3 opt_R = optimized_orientation.toRotationMatrix();
    const Vec3 opt_t = optimized_translation;

    simd::Scalar R__[3][3];
    simd::Scalar t__[3];
    for (int row = 0; row < 3; ++row) {
      t__[row] = simd::Scalar(opt_t(row));
      for (int col = 0; col < 3; ++col)
        R__[row][col] = simd::Scalar(opt_R(row, col));
    }

    simd::Scalar gradient__[6];
    simd::Scalar hessian__[6][6];
    simd::Scalar cost__;
    const size_t stride = simd::Scalar::GetDataStep();
    const int num_stride = correspondences.size() / stride;
    for (size_t point_idx = 0; point_idx < num_stride * stride;
         point_idx += stride) {
      double px_buf[stride];
      double py_buf[stride];
      double pz_buf[stride];
      double mx_buf[stride];
      double my_buf[stride];
      double mz_buf[stride];
      double sqrt_info_buf[3][3][stride];
      for (size_t k = 0; k < stride; ++k) {
        const auto& corr = correspondences.at(point_idx + k);
        px_buf[k] = corr.point(0);
        py_buf[k] = corr.point(1);
        pz_buf[k] = corr.point(2);
        mx_buf[k] = corr.ndt.mean(0);
        my_buf[k] = corr.ndt.mean(1);
        mz_buf[k] = corr.ndt.mean(2);
        sqrt_info_buf[0][0][k] = corr.ndt.sqrt_information(0, 0);
        sqrt_info_buf[0][1][k] = corr.ndt.sqrt_information(0, 1);
        sqrt_info_buf[0][2][k] = corr.ndt.sqrt_information(0, 2);
        sqrt_info_buf[1][0][k] = corr.ndt.sqrt_information(1, 0);
        sqrt_info_buf[1][1][k] = corr.ndt.sqrt_information(1, 1);
        sqrt_info_buf[1][2][k] = corr.ndt.sqrt_information(1, 2);
        sqrt_info_buf[2][0][k] = corr.ndt.sqrt_information(2, 0);
        sqrt_info_buf[2][1][k] = corr.ndt.sqrt_information(2, 1);
        sqrt_info_buf[2][2][k] = corr.ndt.sqrt_information(2, 2);
      }

      simd::Scalar p__[3];
      p__[0] = simd::Scalar(px_buf);
      p__[1] = simd::Scalar(py_buf);
      p__[2] = simd::Scalar(pz_buf);

      simd::Scalar mu__[3];
      mu__[0] = simd::Scalar(mx_buf);
      mu__[1] = simd::Scalar(my_buf);
      mu__[2] = simd::Scalar(mz_buf);

      simd::Scalar sqrt_info__[3][3];
      sqrt_info__[0][0] = simd::Scalar(sqrt_info_buf[0][0]);
      sqrt_info__[0][1] = simd::Scalar(sqrt_info_buf[0][1]);
      sqrt_info__[0][2] = simd::Scalar(sqrt_info_buf[0][2]);
      sqrt_info__[1][0] = simd::Scalar(sqrt_info_buf[1][0]);
      sqrt_info__[1][1] = simd::Scalar(sqrt_info_buf[1][1]);
      sqrt_info__[1][2] = simd::Scalar(sqrt_info_buf[1][2]);
      sqrt_info__[2][0] = simd::Scalar(sqrt_info_buf[2][0]);
      sqrt_info__[2][1] = simd::Scalar(sqrt_info_buf[2][1]);
      sqrt_info__[2][2] = simd::Scalar(sqrt_info_buf[2][2]);

      // clang-format off
      // pw = R*p + t
      simd::Scalar pw__[3];
      pw__[0] = R__[0][0] * p__[0] + R__[0][1] * p__[1] + R__[0][2] * p__[2] + t__[0];
      pw__[1] = R__[1][0] * p__[0] + R__[1][1] * p__[1] + R__[1][2] * p__[2] + t__[1];
      pw__[2] = R__[2][0] * p__[0] + R__[2][1] * p__[1] + R__[2][2] * p__[2] + t__[2];

      // e_i = pw - mean
      simd::Scalar e__[3];
      e__[0] = pw__[0] - mu__[0];
      e__[1] = pw__[1] - mu__[1];
      e__[2] = pw__[2] - mu__[2];
      // r = sqrt_info * e
      simd::Scalar r__[3];
      r__[0] = sqrt_info__[0][0] * e__[0] + sqrt_info__[0][1] * e__[1] + sqrt_info__[0][2] * e__[2];
      r__[1] = sqrt_info__[1][0] * e__[0] + sqrt_info__[1][1] * e__[1] + sqrt_info__[1][2] * e__[2];
      r__[2] = sqrt_info__[2][0] * e__[0] + sqrt_info__[2][1] * e__[1] + sqrt_info__[2][2] * e__[2];

      simd::Scalar J__[3][6];
      J__[0][0] = sqrt_info__[0][0]; J__[0][1] = sqrt_info__[0][1]; J__[0][2] = sqrt_info__[0][2];
      J__[1][0] = sqrt_info__[1][0]; J__[1][1] = sqrt_info__[1][1]; J__[1][2] = sqrt_info__[1][2];
      J__[2][0] = sqrt_info__[2][0]; J__[2][1] = sqrt_info__[2][1]; J__[2][2] = sqrt_info__[2][2];

      simd::Scalar minus_R_skewp__[3][3];
      minus_R_skewp__[0][0] =  R__[0][2] * p__[1] - R__[0][1] * p__[2];
      minus_R_skewp__[1][0] =  R__[1][2] * p__[1] - R__[1][1] * p__[2];
      minus_R_skewp__[2][0] =  R__[2][2] * p__[1] - R__[2][1] * p__[2];

      minus_R_skewp__[0][1] =  R__[0][0] * p__[2] - R__[0][2] * p__[0];
      minus_R_skewp__[1][1] =  R__[1][0] * p__[2] - R__[1][2] * p__[0];
      minus_R_skewp__[2][1] =  R__[2][0] * p__[2] - R__[2][2] * p__[0];
      
      minus_R_skewp__[0][2] =  R__[0][1] * p__[0] - R__[0][0] * p__[1];
      minus_R_skewp__[1][2] =  R__[1][1] * p__[0] - R__[1][0] * p__[1];
      minus_R_skewp__[2][2] =  R__[2][1] * p__[0] - R__[2][0] * p__[1];

      J__[0][3] = sqrt_info__[0][0] * minus_R_skewp__[0][0] + sqrt_info__[0][1] * minus_R_skewp__[1][0] + sqrt_info__[0][2] * minus_R_skewp__[2][0];
      J__[0][4] = sqrt_info__[0][0] * minus_R_skewp__[0][1] + sqrt_info__[0][1] * minus_R_skewp__[1][1] + sqrt_info__[0][2] * minus_R_skewp__[2][1];
      J__[0][5] = sqrt_info__[0][0] * minus_R_skewp__[0][2] + sqrt_info__[0][1] * minus_R_skewp__[1][2] + sqrt_info__[0][2] * minus_R_skewp__[2][2];
      J__[1][3] = sqrt_info__[1][0] * minus_R_skewp__[0][0] + sqrt_info__[1][1] * minus_R_skewp__[1][0] + sqrt_info__[1][2] * minus_R_skewp__[2][0];
      J__[1][4] = sqrt_info__[1][0] * minus_R_skewp__[0][1] + sqrt_info__[1][1] * minus_R_skewp__[1][1] + sqrt_info__[1][2] * minus_R_skewp__[2][1];
      J__[1][5] = sqrt_info__[1][0] * minus_R_skewp__[0][2] + sqrt_info__[1][1] * minus_R_skewp__[1][2] + sqrt_info__[1][2] * minus_R_skewp__[2][2];
      J__[2][3] = sqrt_info__[2][0] * minus_R_skewp__[0][0] + sqrt_info__[2][1] * minus_R_skewp__[1][0] + sqrt_info__[2][2] * minus_R_skewp__[2][0];
      J__[2][4] = sqrt_info__[2][0] * minus_R_skewp__[0][1] + sqrt_info__[2][1] * minus_R_skewp__[1][1] + sqrt_info__[2][2] * minus_R_skewp__[2][1];
      J__[2][5] = sqrt_info__[2][0] * minus_R_skewp__[0][2] + sqrt_info__[2][1] * minus_R_skewp__[1][2] + sqrt_info__[2][2] * minus_R_skewp__[2][2];
      // clang-format on

      // Compute loss and weight,
      // and add the local gradient and hessian to the global ones
      simd::Scalar sq_r__ = r__[0] * r__[0] + r__[1] * r__[1] + r__[2] * r__[2];
      simd::Scalar loss__(sq_r__);
      simd::Scalar weight__(1.0);
      if (loss_function_ != nullptr) {
        double sq_r_buf[4];
        sq_r__.StoreData(sq_r_buf);
        double loss_buf[4];
        double weight_buf[4];
        for (int k = 0; k < 4; ++k) {
          double loss_output[3] = {0.0, 0.0, 0.0};
          loss_function_->Evaluate(sq_r_buf[k], loss_output);
          loss_buf[k] = loss_output[0];
          weight_buf[k] = loss_output[1];
        }
        loss__ = simd::Scalar(loss_buf);
        weight__ = simd::Scalar(weight_buf);
      }

      // g(i) += (J(0,i)*r(0) + J(1,i)*r(1) + J(2,i)*r(2))
      for (int k = 0; k < 6; ++k)
        gradient__[k] += (weight__ * (J__[0][k] * r__[0] + J__[1][k] * r__[1] +
                                      J__[2][k] * r__[2]));

      // H(i,j) = sum_{k} w * J(k,i) * J(k,j)
      for (int ii = 0; ii < 6; ++ii) {
        for (int jj = ii; jj < 6; ++jj) {
          hessian__[ii][jj] +=
              (weight__ * (J__[0][ii] * J__[0][jj] + J__[1][ii] * J__[1][jj] +
                           J__[2][ii] * J__[2][jj]));
        }
      }

      cost__ += loss__;
    }
    double buf[4];
    cost__.StoreData(buf);
    double cost = 0.0;
    cost += (buf[0] + buf[1] + buf[2] + buf[3]);

    Mat6x6 hessian{Mat6x6::Zero()};
    Vec6 gradient{Vec6::Zero()};
    for (int ii = 0; ii < 6; ++ii) {
      gradient__[ii].StoreData(buf);
      gradient(ii) += (buf[0] + buf[1] + buf[2] + buf[3]);
      for (int jj = ii; jj < 6; ++jj) {
        hessian__[ii][jj].StoreData(buf);
        hessian(ii, jj) += (buf[0] + buf[1] + buf[2] + buf[3]);
      }
    }

    // Reflect the hessian
    ReflectHessian(&hessian);

    // Damping hessian
    for (int k = 0; k < 6; k++) hessian(k, k) *= 1.0 + lambda;

    // Compute the step
    const Vec6 update_step = hessian.ldlt().solve(-gradient);

    // Update the pose
    const Vec3 delta_t = update_step.block<3, 1>(0, 0);
    const Vec3 delta_R = update_step.block<3, 1>(3, 0);
    optimized_translation += delta_t;
    optimized_orientation *= ComputeQuaternion(delta_R);
    optimized_orientation.normalize();

    // Check convergence
    if (update_step.norm() < 1e-7) {
      break;
    }
    if (gradient.norm() < 1e-7) {
      break;
    }

    if (cost > previous_cost) {
      lambda *= 2.0;
    } else {
      lambda *= 0.6;
    }
    lambda = std::clamp(lambda, min_lambda, max_lambda);
    previous_cost = cost;
  }

  std::cerr << "COST: " << previous_cost << ", iter: " << iteration
            << std::endl;

  pose->translation() = optimized_translation;
  pose->linear() = optimized_orientation.toRotationMatrix();

  return true;
}

void MahalanobisDistanceMinimizerAnalyticSIMD::ComputeJacobianAndResidual(
    const Mat3x3& rotation, const Vec3& translation, const Correspondence& corr,
    Mat3x6* jacobian, Vec3* residual) {
  const auto& R = rotation;
  const auto& t = translation;
  const auto& p = corr.point;
  const auto& sqrt_information = corr.ndt.sqrt_information;

  Vec3 p_warped = R * p + t;
  Vec3 e_i = p_warped - corr.ndt.mean;

  // Compute the residual
  *residual = sqrt_information * e_i;  // residual

  static auto skew = [](const Vec3& v) {
    Mat3x3 skew{Mat3x3::Zero()};
    skew << 0.0, -v.z(), v.y(),  //
        v.z(), 0.0, -v.x(),      //
        -v.y(), v.x(), 0.0;
    return skew;
  };

  // Compute the Jacobian
  Mat3x3 R_skew_p = R * skew(p);
  //   Mat3x3 R_skew_p{Mat3x3::Zero()};
  //   R_skew_p(0, 0) = R(0, 1) * p(2) - R(0, 2) * p(1);
  //   R_skew_p(0, 1) = R(0, 2) * p(0) - R(0, 0) * p(2);
  //   R_skew_p(0, 2) = R(0, 0) * p(1) - R(0, 1) * p(0);
  //   R_skew_p(1, 0) = R(1, 1) * p(2) - R(1, 2) * p(1);
  //   R_skew_p(1, 1) = R(1, 2) * p(0) - R(1, 0) * p(2);
  //   R_skew_p(1, 2) = R(1, 0) * p(1) - R(1, 1) * p(0);
  //   R_skew_p(2, 0) = R(2, 1) * p(2) - R(2, 2) * p(1);
  //   R_skew_p(2, 1) = R(2, 2) * p(0) - R(2, 0) * p(2);
  //   R_skew_p(2, 2) = R(2, 0) * p(1) - R(2, 1) * p(0);
  (*jacobian).block<3, 3>(0, 0) = sqrt_information;
  (*jacobian).block<3, 3>(0, 3) = -sqrt_information * R_skew_p;
}

void MahalanobisDistanceMinimizerAnalyticSIMD::ComputeHessianOnlyUpperTriangle(
    const Mat3x6& jacobian, Mat6x6* local_hessian) {
  auto& H = *local_hessian;
  auto& J = jacobian;
  H.setZero();
  for (int row = 0; row < 6; ++row) {
    for (int col = row; col < 6; ++col) {
      for (int k = 0; k < 3; ++k) {
        H(row, col) += J(k, row) * J(k, col);
      }
    }
  }
}

void MahalanobisDistanceMinimizerAnalyticSIMD::MultiplyWeightOnlyUpperTriangle(
    const double weight, Mat6x6* local_hessian) {
  for (int row = 0; row < 6; ++row) {
    for (int col = row; col < 6; ++col) {
      (*local_hessian)(row, col) *= weight;
    }
  }
}

void MahalanobisDistanceMinimizerAnalyticSIMD::AddHessianOnlyUpperTriangle(
    const Mat6x6& local_hessian, Mat6x6* global_hessian) {
  auto& H = *global_hessian;
  for (int row = 0; row < 6; ++row) {
    for (int col = row; col < 6; ++col) {
      H(row, col) += local_hessian(row, col);
    }
  }
}

void MahalanobisDistanceMinimizerAnalyticSIMD::ReflectHessian(Mat6x6* hessian) {
  auto& H = *hessian;
  for (int row = 0; row < 6; ++row) {
    for (int col = row + 1; col < 6; ++col) {
      H(col, row) = H(row, col);
    }
  }
}

Orientation MahalanobisDistanceMinimizerAnalyticSIMD::ComputeQuaternion(
    const Vec3& w) {
  Orientation orientation{Orientation::Identity()};
  const double theta = w.norm();
  if (theta < 1e-6) {
    orientation.w() = 1.0;
    orientation.x() = 0.5 * w.x();
    orientation.y() = 0.5 * w.y();
    orientation.z() = 0.5 * w.z();
  } else {
    const double half_theta = theta * 0.5;
    const double sin_half_theta_divided_theta = std::sin(half_theta) / theta;
    orientation.w() = std::cos(half_theta);
    orientation.x() = sin_half_theta_divided_theta * w.x();
    orientation.y() = sin_half_theta_divided_theta * w.y();
    orientation.z() = sin_half_theta_divided_theta * w.z();
  }
  return orientation;
}

}  // namespace mahalanobis_distance_minimizer
}  // namespace nonlinear_optimizer