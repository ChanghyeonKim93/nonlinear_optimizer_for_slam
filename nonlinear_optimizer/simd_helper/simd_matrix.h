#ifndef NONLINEAR_OPTIMIZER_SIMD_HELPER_SIMD_MATRIX_H_
#define NONLINEAR_OPTIMIZER_SIMD_HELPER_SIMD_MATRIX_H_

#include "nonlinear_optimizer/simd_helper/simd_vector.h"

namespace nonlinear_optimizer {
namespace simd {

/// @brief Matrix of SIMD data
/// @tparam kRow Matrix row size
/// @tparam kCol Matrix column size
template <int kRow, int kCol>
class MatrixF {
  const size_t kDataStep{_SIMD_DATA_STEP_FLOAT};
  using EigenMat = Eigen::Matrix<float, kRow, kCol>;

 public:
  MatrixF() {
    for (int row = 0; row < kRow; ++row)
      for (int col = 0; col < kCol; ++col)
        data_[row][col] = _mm256_setzero_ps();
  }
  ~MatrixF() {}

  explicit MatrixF(const EigenMat& single_matrix) {
    for (int row = 0; row < kRow; ++row)
      for (int col = 0; col < kCol; ++col)
        data_[row][col] = ScalarF(single_matrix(row, col));
  }
  explicit MatrixF(const std::vector<EigenMat>& multi_matrices) {
    if (multi_matrices.size() != kDataStep)
      throw std::runtime_error("Wrong number of data");
    float buf[8];
    for (int row = 0; row < kRow; ++row) {
      for (int col = 0; col < kCol; ++col) {
        for (size_t k = 0; k < kDataStep; ++k)
          buf[k] = multi_matrices[k](row, col);
        data_[row][col] = ScalarF(buf);
      }
    }
  }

  explicit MatrixF(const std::vector<float*>& multi_elements) {
    if (multi_elements.size() != kRow * kCol)
      throw std::runtime_error("Wrong number of data");
    for (int row = 0; row < kRow; ++row)
      for (int col = 0; col < kCol; ++col)
        data_[row][col] = ScalarF(multi_elements.at(row * kCol + col));
  }

  MatrixF(const MatrixF& rhs) {
    for (int row = 0; row < kRow; ++row)
      for (int col = 0; col < kCol; ++col)
        data_[row][col] = rhs.data_[row][col];
  }
  ScalarF& operator()(const int row, const int col) { return data_[row][col]; }
  const ScalarF& operator()(const int row, const int col) const {
    return data_[row][col];
  }
  MatrixF& operator=(const MatrixF& rhs) {
    for (int row = 0; row < kRow; ++row)
      for (int col = 0; col < kCol; ++col)
        data_[row][col] = rhs.data_[row][col];
    return *this;
  }
  MatrixF operator+(const MatrixF& rhs) const {
    MatrixF res;
    for (int row = 0; row < kRow; ++row)
      for (int col = 0; col < kCol; ++col)
        res.data_[row][col] = data_[row][col] + rhs.data_[row][col];
    return res;
  }
  MatrixF operator-(const MatrixF& rhs) const {
    MatrixF res;
    for (int row = 0; row < kRow; ++row)
      for (int col = 0; col < kCol; ++col)
        res.data_[row][col] = data_[row][col] - rhs.data_[row][col];
    return res;
  }
  MatrixF operator*(const float scalar) const {
    MatrixF res;
    for (int row = 0; row < kRow; ++row)
      for (int col = 0; col < kCol; ++col)
        res.data_[row][col] = data_[row][col] * scalar;
    return res;
  }
  MatrixF operator*(const ScalarF scalar) const {
    MatrixF res;
    for (int row = 0; row < kRow; ++row)
      for (int col = 0; col < kCol; ++col)
        res.data_[row][col] = data_[row][col] * scalar;
    return res;
  }
  template <int kRhsCol>
  inline MatrixF<kRow, kRhsCol> operator*(
      const MatrixF<kCol, kRhsCol>& matrix) const {
    MatrixF<kRow, kRhsCol> res;
    for (int row = 0; row < kRow; ++row)
      for (int col = 0; col < kRhsCol; ++col)
        for (int k = 0; k < kCol; ++k)
          res(row, col) += data_[row][k] * matrix(k, col);

    return res;
  }
  inline VectorF<kRow> operator*(const VectorF<kCol>& vector) const {
    VectorF<kRow> res(Eigen::Matrix<float, kRow, 1>::Zero());
    for (int row = 0; row < kRow; ++row)
      for (int col = 0; col < kCol; ++col)
        res.data_[row] += data_[row][col] * vector(col);
    return res;
  }
  MatrixF& operator+=(const MatrixF& rhs) {
    for (int row = 0; row < kRow; ++row)
      for (int col = 0; col < kCol; ++col)
        data_[row][col] += rhs.data_[row][col];
    return *this;
  }
  MatrixF& operator-=(const MatrixF& rhs) {
    for (int row = 0; row < kRow; ++row)
      for (int col = 0; col < kCol; ++col)
        data_[row][col] -= rhs.data_[row][col];
    return *this;
  }

  MatrixF<kCol, kRow> transpose() const {
    MatrixF<kCol, kRow> mat_trans;
    for (int row = 0; row < kRow; ++row)
      for (int col = 0; col < kCol; ++col)
        mat_trans(col, row) = data_[row][col];
    return mat_trans;
  }

  void StoreData(std::vector<EigenMat>* multi_matrices) const {
    if (multi_matrices->size() != kDataStep) multi_matrices->resize(kDataStep);
    float buf[8];
    for (int row = 0; row < kRow; ++row) {
      for (int col = 0; col < kCol; ++col) {
        data_[row][col].StoreData(buf);
        for (size_t k = 0; k < kDataStep; ++k)
          multi_matrices->at(k)(row, col) = buf[k];
      }
    }
  }

  friend std::ostream& operator<<(std::ostream& outputStream,
                                  const MatrixF& mat) {
    std::vector<EigenMat> multi_matrices;
    mat.StoreData(&multi_matrices);
    std::cout << "[["
              << "[" << multi_matrices[0] << "],\n"
              << "[" << multi_matrices[1] << "],\n"
              << "[" << multi_matrices[2] << "],\n"
              << "[" << multi_matrices[3] << "]]" << std::endl;
    return outputStream;
  }

 private:
  ScalarF data_[kRow][kCol];
};

}  // namespace simd
}  // namespace nonlinear_optimizer

#if defined(__amd64__) || defined(__x86_64__)

namespace nonlinear_optimizer {
namespace simd {

/// @brief Matrix of SIMD data
/// @tparam kRow Matrix row size
/// @tparam kCol Matrix column size
template <int kRow, int kCol>
class MatrixD {
  const size_t kDataStep{4};
  using EigenMat = Eigen::Matrix<double, kRow, kCol>;

 public:
  MatrixD() {
    for (int row = 0; row < kRow; ++row)
      for (int col = 0; col < kCol; ++col)
        data_[row][col] = _mm256_setzero_pd();
  }
  ~MatrixD() {}

  explicit MatrixD(const EigenMat& single_matrix) {
    for (int row = 0; row < kRow; ++row)
      for (int col = 0; col < kCol; ++col)
        data_[row][col] = ScalarD(single_matrix(row, col));
  }
  explicit MatrixD(const std::vector<EigenMat>& multi_matrices) {
    if (multi_matrices.size() != kDataStep)
      throw std::runtime_error("Wrong number of data");
    double buf[8];
    for (int row = 0; row < kRow; ++row) {
      for (int col = 0; col < kCol; ++col) {
        for (size_t k = 0; k < kDataStep; ++k)
          buf[k] = multi_matrices[k](row, col);
        data_[row][col] = ScalarD(buf);
      }
    }
  }
  MatrixD(const MatrixD& rhs) {
    for (int row = 0; row < kRow; ++row)
      for (int col = 0; col < kCol; ++col)
        data_[row][col] = rhs.data_[row][col];
  }
  ScalarD& operator()(const int row, const int col) { return data_[row][col]; }
  const ScalarD& operator()(const int row, const int col) const {
    return data_[row][col];
  }
  MatrixD& operator=(const MatrixD& rhs) {
    for (int row = 0; row < kRow; ++row)
      for (int col = 0; col < kCol; ++col)
        data_[row][col] = rhs.data_[row][col];
    return *this;
  }
  MatrixD operator+(const MatrixD& rhs) const {
    MatrixD res;
    for (int row = 0; row < kRow; ++row)
      for (int col = 0; col < kCol; ++col)
        res.data_[row][col] = data_[row][col] + rhs.data_[row][col];
    return res;
  }
  MatrixD operator-(const MatrixD& rhs) const {
    MatrixD res;
    for (int row = 0; row < kRow; ++row)
      for (int col = 0; col < kCol; ++col)
        res.data_[row][col] = data_[row][col] - rhs.data_[row][col];
    return res;
  }
  MatrixD operator*(const double scalar) const {
    MatrixD res;
    for (int row = 0; row < kRow; ++row)
      for (int col = 0; col < kCol; ++col)
        res.data_[row][col] = data_[row][col] * scalar;
    return res;
  }
  MatrixD operator*(const ScalarD scalar) const {
    MatrixD res;
    for (int row = 0; row < kRow; ++row)
      for (int col = 0; col < kCol; ++col)
        res.data_[row][col] = data_[row][col] * scalar;
    return res;
  }
  template <int kRhsCol>
  inline MatrixD<kRow, kRhsCol> operator*(
      const MatrixD<kCol, kRhsCol>& matrix) const {
    MatrixD<kRow, kRhsCol> res;
    if (kRow == 3 && kCol == 3 && kRhsCol == 3) {
      res(0, 0) += (data_[0][0] * matrix(0, 0) + data_[0][1] * matrix(1, 0) +
                    data_[0][2] * matrix(2, 0));
      res(0, 1) += (data_[0][0] * matrix(0, 1) + data_[0][1] * matrix(1, 1) +
                    data_[0][2] * matrix(2, 1));
      res(0, 2) += (data_[0][0] * matrix(0, 2) + data_[0][1] * matrix(1, 2) +
                    data_[0][2] * matrix(2, 2));
      res(1, 0) += (data_[1][0] * matrix(0, 0) + data_[1][1] * matrix(1, 0) +
                    data_[1][2] * matrix(2, 0));
      res(1, 1) += (data_[1][0] * matrix(0, 1) + data_[1][1] * matrix(1, 1) +
                    data_[1][2] * matrix(2, 1));
      res(1, 2) += (data_[1][0] * matrix(0, 2) + data_[1][1] * matrix(1, 2) +
                    data_[1][2] * matrix(2, 2));
      res(2, 0) += (data_[2][0] * matrix(0, 0) + data_[2][1] * matrix(1, 0) +
                    data_[2][2] * matrix(2, 0));
      res(2, 1) += (data_[2][0] * matrix(0, 1) + data_[2][1] * matrix(1, 1) +
                    data_[2][2] * matrix(2, 1));
      res(2, 2) += (data_[2][0] * matrix(0, 2) + data_[2][1] * matrix(1, 2) +
                    data_[2][2] * matrix(2, 2));
    } else {
      for (int row = 0; row < kRow; ++row)
        for (int col = 0; col < kRhsCol; ++col)
          for (int k = 0; k < kCol; ++k)
            res(row, col) += data_[row][k] * matrix(k, col);
    }

    return res;
  }
  inline VectorD<kRow> operator*(const VectorD<kCol>& vector) const {
    VectorD<kRow> res(Eigen::Matrix<double, kRow, 1>::Zero());
    if (kRow == 3 && kCol == 3) {
      res.data_[0] += data_[0][0] * vector(0) + data_[0][1] * vector(1) +
                      data_[0][2] * vector(2);
      res.data_[1] += data_[1][0] * vector(0) + data_[1][1] * vector(1) +
                      data_[1][2] * vector(2);
      res.data_[2] += data_[2][0] * vector(0) + data_[2][1] * vector(1) +
                      data_[2][2] * vector(2);
    } else {
      for (int row = 0; row < kRow; ++row)
        for (int col = 0; col < kCol; ++col)
          res.data_[row] += data_[row][col] * vector(col);
    }
    return res;
  }
  MatrixD& operator+=(const MatrixD& rhs) {
    for (int row = 0; row < kRow; ++row)
      for (int col = 0; col < kCol; ++col)
        data_[row][col] += rhs.data_[row][col];
    return *this;
  }
  MatrixD& operator-=(const MatrixD& rhs) {
    for (int row = 0; row < kRow; ++row)
      for (int col = 0; col < kCol; ++col)
        data_[row][col] -= rhs.data_[row][col];
    return *this;
  }

  MatrixD<kCol, kRow> transpose() const {
    MatrixD<kCol, kRow> mat_trans;
    for (int row = 0; row < kRow; ++row)
      for (int col = 0; col < kCol; ++col)
        mat_trans(col, row) = data_[row][col];
    return mat_trans;
  }

  void StoreData(std::vector<EigenMat>* multi_matrices) const {
    if (multi_matrices->size() != kDataStep) multi_matrices->resize(kDataStep);
    double buf[8];
    for (int row = 0; row < kRow; ++row) {
      for (int col = 0; col < kCol; ++col) {
        data_[row][col].StoreData(buf);
        for (size_t k = 0; k < kDataStep; ++k)
          multi_matrices->at(k)(row, col) = buf[k];
      }
    }
  }

  friend std::ostream& operator<<(std::ostream& outputStream,
                                  const MatrixD& mat) {
    std::vector<EigenMat> multi_matrices;
    mat.StoreData(&multi_matrices);
    std::cout << "[["
              << "[" << multi_matrices[0] << "],\n"
              << "[" << multi_matrices[1] << "],\n"
              << "[" << multi_matrices[2] << "],\n"
              << "[" << multi_matrices[3] << "]]" << std::endl;
    return outputStream;
  }

 private:
  ScalarD data_[kRow][kCol];
};

}  // namespace simd
}  // namespace nonlinear_optimizer

#endif

#endif  // NONLINEAR_OPTIMIZER_SIMD_HELPER_SIMD_MATRIX_H_