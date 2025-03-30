#ifndef NONLINEAR_OPTIMIZER_OPTIONS_H_
#define NONLINEAR_OPTIMIZER_OPTIONS_H_

namespace nonlinear_optimizer {

enum class MinimizerType {
  kGaussNewton = 0,
  kGradientDescent,
  kQuasiNewton,
  kLevenbergMarquardt
};

enum class LinearSolverType { kDenseQR = 0, kDenseCholesky, kSparseCholesky };

struct Options {
  int max_iterations{30};
  MinimizerType minimizer_type{MinimizerType::kGaussNewton};
  LinearSolverType linear_solver_type{LinearSolverType::kDenseQR};
  struct {
    double function_tolerance{1e-6};
    double gradient_tolerance{1e-6};
    double parameter_tolerance{1e-6};
  } convergence_handle;
  struct {
    double min_lambda{1e-6};
    double max_lambda{1e-2};
  } optimization_handle;
};

}  // namespace nonlinear_optimizer

#endif  // NONLINEAR_OPTIMIZER_OPTIONS_H_