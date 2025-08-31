#ifndef NONLINEAR_OPTIMIZER_MAHALANOBIS_DISTANCE_MINIMIZER_MAHALANOBIS_DISTANCE_MINIMIZER_ANALYTIC_SIMD_H_
#define NONLINEAR_OPTIMIZER_MAHALANOBIS_DISTANCE_MINIMIZER_MAHALANOBIS_DISTANCE_MINIMIZER_ANALYTIC_SIMD_H_

#include <vector>

#include "nonlinear_optimizer/mahalanobis_distance_minimizer/mahalanobis_distance_minimizer.h"
#include "nonlinear_optimizer/mahalanobis_distance_minimizer/types.h"
#include "nonlinear_optimizer/types.h"

#include "simd_helper/simd_helper.h"

namespace nonlinear_optimizer {
namespace mahalanobis_distance_minimizer {

struct SOAData {
  simd::SOAContainer<3, 1> points;
  simd::SOAContainer<3, 1> means;
  simd::SOAContainer<3, 3> sqrt_infos;
};

class MahalanobisDistanceMinimizerAnalyticSIMD
    : public MahalanobisDistanceMinimizer {
 public:
  MahalanobisDistanceMinimizerAnalyticSIMD();

  ~MahalanobisDistanceMinimizerAnalyticSIMD();

  bool Solve(const Options& options,
             const std::vector<Correspondence>& correspondences,
             Pose* pose) final;

 private:
  PartialResult ComputeCostAndDerivatives(const Mat3x3& rotation,
                                          const Vec3& translation,
                                          const SOAData* soa_data,
                                          const size_t start_index,
                                          const size_t end_index);
  void AddHessianOnlyUpperTriangle(const Mat6x6& local_hessian,
                                   Mat6x6* global_hessian);
  void ReflectHessian(Mat6x6* hessian);
};

}  // namespace mahalanobis_distance_minimizer
}  // namespace nonlinear_optimizer

#endif  // NONLINEAR_OPTIMIZER_MAHALANOBIS_DISTANCE_MINIMIZER_MAHALANOBIS_DISTANCE_MINIMIZER_ANALYTIC_SIMD_H_