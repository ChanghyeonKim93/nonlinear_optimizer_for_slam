#ifndef SIMD_HELPER_V2_SIMD_MATRIX_H_
#define SIMD_HELPER_V2_SIMD_MATRIX_H_

#include "simd_helper_v2/simd_scalar.h"

#include "Eigen/Dense"

namespace simd {

template <int kRow, int kCol>
class MatrixBase {
 protected:
  using EigenMatrix = Eigen::Matrix<float, kRow, kCol>;

 public:
  // static member methods
  static inline size_t GetDataStride() { return __DATA_STRIDE; }

  static inline MatrixBase Zeros() { return MatrixBase(0.0f); }

  static inline MatrixBase Ones() { return MatrixBase(1.0f); }

  static inline MatrixBase Identity() {
    return MatrixBase(EigenMatrix::Identity());
  }

 public:
  // Initialization & Assignment operations
  MatrixBase() {
    for (int r = 0; r < kRow; ++r)
      for (int c = 0; c < kCol; ++c) data_[r][c] = Scalar(0.0f);
  }

  MatrixBase(const MatrixBase& rhs) {
    for (int r = 0; r < kRow; ++r)
      for (int c = 0; c < kCol; ++c) data_[r][c] = rhs.data_[r][c];
  }

  MatrixBase(const EigenMatrix& matrix) {
    for (int r = 0; r < kRow; ++r)
      for (int c = 0; c < kCol; ++c) data_[r][c] = Scalar(matrix(r, c));
  }

  MatrixBase(const std::vector<EigenMatrix>& matrices) {
    if (matrices.size() != GetDataStride())
      throw std::runtime_error("Wrong number of data.");

    float buf[GetDataStride()];
    for (int r = 0; r < kRow; ++r) {
      for (int c = 0; c < kCol; ++c) {
        for (size_t k = 0; k < simd::Scalar::GetDataStride(); ++k)
          buf[k] = matrices[k](r, c);
        data_[r][c] = Scalar(buf);
      }
    }
  }

  MatrixBase& operator=(const EigenMatrix& rhs) {
    for (int r = 0; r < kRow; ++r)
      for (int c = 0; c < kCol; ++c) data_[r][c] = Scalar(rhs(r, c));
    return *this;
  }

  MatrixBase& operator=(const MatrixBase& rhs) {
    for (int r = 0; r < kRow; ++r)
      for (int c = 0; c < kCol; ++c) data_[r][c] = rhs.data_[r][c];
    return *this;
  }

  // Accessor methods
  Scalar& operator()(const int r, const int c) { return data_[r][c]; }

  const Scalar& operator()(const int r, const int c) const {
    return data_[r][c];
  }

  // Arithmetic operations
  MatrixBase operator-() const { return MatrixBase(_s_sub(__zero, data_)); }

  // Arithmetic operations: element-wise operations
  MatrixBase operator+(const float rhs) const {
    MatrixBase res = *this;
    for (int r = 0; r < kRow; ++r)
      for (int c = 0; c < kCol; ++c) res.data_[r][c] += rhs;
    return res;
  }

  MatrixBase operator-(const float rhs) const {
    MatrixBase res = *this;
    for (int r = 0; r < kRow; ++r)
      for (int c = 0; c < kCol; ++c) res.data_[r][c] -= rhs;
    return res;
  }

  MatrixBase operator*(const float rhs) const {
    MatrixBase res = *this;
    for (int r = 0; r < kRow; ++r)
      for (int c = 0; c < kCol; ++c) res.data_[r][c] *= rhs;
    return res;
  }

  friend MatrixBase operator*(const float lhs, const MatrixBase& rhs) {
    return MatrixBase(_s_mul(rhs.data_, _s_set1(lhs)));
  }

  MatrixBase operator/(const float rhs) const {
    MatrixBase res = *this;
    for (int r = 0; r < kRow; ++r)
      for (int c = 0; c < kCol; ++c) res.data_[r][c] /= rhs;
    return res;
  }

  // Arithmetic operations: matrix-matrix operations
  MatrixBase operator+(const MatrixBase& rhs) const {
    MatrixBase res = *this;
    for (int r = 0; r < kRow; ++r)
      for (int c = 0; c < kCol; ++c) res.data_[r][c] += rhs.data_[r][c];
    return res;
  }

  MatrixBase operator-(const MatrixBase& rhs) const {
    MatrixBase res = *this;
    for (int r = 0; r < kRow; ++r)
      for (int c = 0; c < kCol; ++c) res.data_[r][c] -= rhs.data_[r][c];
    return res;
  }

  template <int kRhsCol>
  inline MatrixBase<kRow, kRhsCol> operator*(
      const MatrixBase<kCol, kRhsCol>& rhs) const {
    MatrixBase<kRow, kRhsCol> res;
    for (int r = 0; r < kRow; ++r)
      for (int c = 0; c < kRhsCol; ++c)
        for (int k = 0; k < kCol; ++k) res(r, c) += data_[r][k] * rhs(k, c);

    return res;
  }

  // Some matrix operations
  inline MatrixBase<kRow, kCol> transpose() const {
    MatrixBase<kCol, kRow> res;
    for (int r = 0; r < kRow; ++r)
      for (int c = 0; c < kCol; ++c) res(c, r) = data_[r][c];
    return res;
  }

  inline Scalar norm() const {
    Scalar res(0.0f);
    for (int r = 0; r < kRow; ++r)
      for (int c = 0; c < kCol; ++c) res += data_[r][c] * data_[r][c];
    return res;
  }

  void setZero() {
    for (int r = 0; r < kRow; ++r)
      for (int c = 0; c < kCol; ++c) data_[r][c] = Scalar(0.0f);
  }

  MatrixBase cwiseSqrt() {
    MatrixBase res{MatrixBase::Zeros()};
    for (int r = 0; r < kRow; ++r)
      for (int c = 0; c < kCol; ++c) res(r, c) = data_[r][c].sqrt();
  }

  MatrixBase cwiseSign() {
    MatrixBase res{MatrixBase::Zeros()};
    for (int r = 0; r < kRow; ++r)
      for (int c = 0; c < kCol; ++c) res(r, c) = data_[r][c].sign();
  }

  MatrixBase cwiseAbs() {
    MatrixBase res{MatrixBase::Zeros()};
    for (int r = 0; r < kRow; ++r)
      for (int c = 0; c < kCol; ++c) res(r, c) = data_[r][c].abs();
    return res;
  }

  // Store SIMD data to normal memory
  void StoreData(std::vector<EigenMatrix>* multi_matrices) const {
    const auto data_stride = GetDataStride();
    if (multi_matrices->size() != data_stride)
      multi_matrices->resize(data_stride);
    float buf[data_stride];
    for (int r = 0; r < kRow; ++r) {
      for (int c = 0; c < kCol; ++c) {
        data_[r][c].StoreData(buf);
        for (size_t k = 0; k < data_stride; ++k)
          multi_matrices->at(k)(r, c) = buf[k];
      }
    }
  }

  // Debug functions
  friend std::ostream& operator<<(std::ostream& outputStream,
                                  const MatrixBase& simd_mat) {
    static std::stringstream ss;
    ss.str("");
    std::vector<EigenMatrix> multi_matrices;
    simd_mat.StoreData(&multi_matrices);
    ss << "{";
    for (int i = 0; i < MatrixBase::GetDataStride(); ++i)
      ss << "[" << multi_matrices[i] << "]\n";
    ss << "}" << std::endl;
    std::cerr << ss.str();
    return outputStream;
  }

 protected:
  Scalar data_[kRow][kCol];
};

template <int kRow, int kCol>
class Matrix : public MatrixBase<kRow, kCol> {
  using EigenMatrix = typename MatrixBase<kRow, kCol>::EigenMatrix;

 public:
  Matrix() : MatrixBase<kRow, kCol>() {}

  Matrix(const Matrix& rhs) : MatrixBase<kRow, kCol>(rhs) {}

  Matrix(const EigenMatrix& matrix) : MatrixBase<kRow, kCol>(matrix) {}

  Matrix(const std::vector<EigenMatrix>& matrices)
      : MatrixBase<kRow, kCol>(matrices) {}
};

// Specialization for Nx1 matrix (== vector)
template <int kDim>
class Vector : public Matrix<kDim, 1> {
  using EigenMatrix = typename MatrixBase<kDim, 1>::EigenMatrix;

 public:
  Vector() : Matrix<kDim, 1>() {}

  Vector(const Vector& rhs) : Matrix<kDim, 1>(rhs) {}

  Vector(const Matrix<kDim, 1>& rhs) : Matrix<kDim, 1>(rhs) {}

  Vector(const EigenMatrix& matrix) : Matrix<kDim, 1>(matrix) {}

  Vector(const std::vector<EigenMatrix>& matrices)
      : Matrix<kDim, 1>(matrices) {}

  Scalar dot(const Matrix<kDim, 1>& rhs) const {
    Scalar res(0.0f);
    for (int i = 0; i < kDim; ++i) res += (*this)(i, 0) * rhs.data_[i][0];
    return res;
  }

  Scalar dot(const Matrix<1, kDim>& rhs) const {
    Scalar res(0.0f);
    for (int i = 0; i < kDim; ++i) res += (*this)(i, 0) * rhs.data_[0][i];
    return res;
  }

  Scalar dot(const Vector& rhs) const {
    Scalar res(0.0f);
    for (int i = 0; i < kDim; ++i) res += (*this)(i, 0) * rhs.data_(i);
    return res;
  }

  Scalar& operator()(const int i) const { return (*this)(i, 0); }
};

Scalar abs(const Scalar& input) { return input.abs(); }

Scalar sqrt(const Scalar& input) { return input.sqrt(); }

Scalar exp(const Scalar& input) { return input.exp(); }

}  // namespace simd

#endif  // SIMD_HELPER_V2_SIMD_MATRIX_H_