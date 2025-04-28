#ifndef NONLINEAR_OPTIMIZER_SIMD_HELPER_SIMD_HELPER_AMD_H_
#define NONLINEAR_OPTIMIZER_SIMD_HELPER_SIMD_HELPER_AMD_H_

#include <iostream>

#if defined(__amd64__) || defined(__x86_64__)

#include "immintrin.h"

// AMD CPU (Intel, AMD)
#define _SIMD_DATA_STEP_FLOAT 8
#define _SIMD_FLOAT __m256
#define _SIMD_SET1 _mm256_set1_ps
#define _SIMD_LOAD _mm256_load_ps
#define _SIMD_ADD _mm256_add_ps
#define _SIMD_SUB _mm256_sub_ps
#define _SIMD_MUL _mm256_mul_ps
#define _SIMD_DIV _mm256_div_ps
#define _SIMD_RCP _mm256_rcp_ps
#define _SIMD_STORE _mm256_store_ps

#define _SIMD_DATA_STEP_DOUBLE 4
#define _SIMD_DOUBLE __m256d
#define _SIMD_SET1_D _mm256_set1_pd
#define _SIMD_LOAD_D _mm256_load_pd
#define _SIMD_ADD_D _mm256_add_pd
#define _SIMD_SUB_D _mm256_sub_pd
#define _SIMD_MUL_D _mm256_mul_pd
#define _SIMD_RCP_D _mm256_div_pd
#define _SIMD_STORE_D _mm256_store_pd

namespace nonlinear_optimizer {
namespace simd {

class ScalarF {
 public:
  ScalarF() { data_ = _mm256_setzero_ps(); }

  explicit ScalarF(const float scalar) { data_ = _mm256_set1_ps(scalar); }

  explicit ScalarF(const float n1, const float n2, const float n3,
                   const float n4, const float n5, const float n6,
                   const float n7, const float n8) {
    data_ = _mm256_set_ps(n8, n7, n6, n5, n4, n3, n2, n1);
  }

  explicit ScalarF(const float* rhs) { data_ = _mm256_load_ps(rhs); }

  ScalarF(const __m256& rhs) { data_ = rhs; }

  ScalarF(const ScalarF& rhs) { data_ = rhs.data_; }

  ScalarF operator<(const float scalar) const {
    ScalarF comp_mask(
        _mm256_and_ps(_mm256_cmp_ps(data_, _mm256_set1_ps(scalar), _CMP_LT_OS),
                      _mm256_set1_ps(1.0f)));
    return comp_mask;
  }

  ScalarF operator<=(const float scalar) const {
    ScalarF comp_mask(
        _mm256_and_ps(_mm256_cmp_ps(data_, _mm256_set1_ps(scalar), _CMP_LE_OS),
                      _mm256_set1_ps(1.0f)));
    return comp_mask;
  }

  ScalarF operator>(const float scalar) const {
    ScalarF comp_mask(
        _mm256_and_ps(_mm256_cmp_ps(data_, _mm256_set1_ps(scalar), _CMP_GT_OS),
                      _mm256_set1_ps(1.0f)));
    return comp_mask;
  }

  ScalarF operator>=(const float scalar) const {
    // Convert mask to 0.0 or 1.0
    ScalarF comp_mask(
        _mm256_and_ps(_mm256_cmp_ps(data_, _mm256_set1_ps(scalar), _CMP_GE_OS),
                      _mm256_set1_ps(1.0f)));
    return comp_mask;
  }

  ScalarF operator<(const ScalarF& rhs) const {
    ScalarF comp_mask(_mm256_and_ps(_mm256_cmp_ps(data_, rhs.data_, _CMP_LT_OS),
                                    _mm256_set1_ps(1.0f)));
    return comp_mask;
  }

  ScalarF operator<=(const ScalarF& rhs) const {
    ScalarF comp_mask(_mm256_and_ps(_mm256_cmp_ps(data_, rhs.data_, _CMP_LE_OS),
                                    _mm256_set1_ps(1.0f)));
    return comp_mask;
  }

  ScalarF operator>(const ScalarF& rhs) const {
    ScalarF comp_mask(_mm256_and_ps(_mm256_cmp_ps(data_, rhs.data_, _CMP_GT_OS),
                                    _mm256_set1_ps(1.0f)));
    return comp_mask;
  }

  ScalarF operator>=(const ScalarF& rhs) const {
    // Convert mask to 0.0 or 1.0
    ScalarF comp_mask(_mm256_and_ps(_mm256_cmp_ps(data_, rhs.data_, _CMP_GE_OS),
                                    _mm256_set1_ps(1.0f)));
    return comp_mask;
  }

  ScalarF& operator=(const ScalarF& rhs) {
    data_ = rhs.data_;
    return *this;
  }

  ScalarF operator+(const float rhs) const {
    return ScalarF(_mm256_add_ps(data_, _mm256_set1_ps(rhs)));
  }

  ScalarF operator-() const {
    return ScalarF(_mm256_sub_ps(_mm256_set1_ps(0.0f), data_));
  }

  ScalarF operator-(const float rhs) const {
    return ScalarF(_mm256_sub_ps(data_, _mm256_set1_ps(rhs)));
  }

  ScalarF operator*(const float rhs) const {
    return ScalarF(_mm256_mul_ps(data_, _mm256_set1_ps(rhs)));
  }

  ScalarF operator/(const float rhs) const {
    return ScalarF(_mm256_div_ps(data_, _mm256_set1_ps(rhs)));
  }

  ScalarF operator+(const ScalarF& rhs) const {
    return ScalarF(_mm256_add_ps(data_, rhs.data_));
  }

  ScalarF operator-(const ScalarF& rhs) const {
    return ScalarF(_mm256_sub_ps(data_, rhs.data_));
  }

  ScalarF operator*(const ScalarF& rhs) const {
    return ScalarF(_mm256_mul_ps(data_, rhs.data_));
  }

  ScalarF operator/(const ScalarF& rhs) const {
    return ScalarF(_mm256_div_ps(data_, rhs.data_));
  }

  ScalarF& operator+=(const ScalarF& rhs) {
    data_ = _mm256_add_ps(data_, rhs.data_);
    return *this;
  }

  ScalarF& operator-=(const ScalarF& rhs) {
    data_ = _mm256_sub_ps(data_, rhs.data_);
    return *this;
  }

  ScalarF& operator*=(const ScalarF& rhs) {
    data_ = _mm256_mul_ps(data_, rhs.data_);
    return *this;
  }

  void StoreData(float* data) const { _mm256_store_ps(data, data_); }

  friend std::ostream& operator<<(std::ostream& outputStream,
                                  const ScalarF& scalar) {
    float multi_scalars[8];
    scalar.StoreData(multi_scalars);
    std::cout << "[["
              << "[" << multi_scalars[0] << "],\n"
              << "[" << multi_scalars[1] << "],\n"
              << "[" << multi_scalars[2] << "],\n"
              << "[" << multi_scalars[3] << "],\n"
              << "[" << multi_scalars[4] << "],\n"
              << "[" << multi_scalars[5] << "],\n"
              << "[" << multi_scalars[6] << "],\n"
              << "[" << multi_scalars[7] << "]]" << std::endl;
    return outputStream;
  }

  static size_t GetDataStep() { return _SIMD_DATA_STEP_FLOAT; }

 private:
  __m256 data_;
};

/// @brief Vector of Simd data. Consider four 3D vectors, v1, v2, v3, v4.
/// data_[0] = SimdDouble(v1.x(), v2.x(), v3.x(), v4.x());
/// data_[1] = SimdDouble(v1.y(), v2.y(), v3.y(), v4.y());
/// data_[2] = SimdDouble(v1.z(), v2.z(), v3.z(), v4.z());
/// @tparam kRow
template <int kRow>
class VectorF {
  const size_t kDataStep{8};
  using EigenVec = Eigen::Matrix<float, kRow, 1>;

 public:
  VectorF() {
    for (int row = 0; row < kRow; ++row) data_[row] = _mm256_set1_ps(0.0f);
  }
  ~VectorF() {}

  explicit VectorF(const EigenVec& single_vector) {
    for (int row = 0; row < kRow; ++row)
      data_[row] = ScalarF(single_vector(row));
  }
  explicit VectorF(const std::vector<EigenVec>& multi_vectors) {
    if (multi_vectors.size() != kDataStep)
      throw std::runtime_error("Wrong number of data");
    for (int row = 0; row < kRow; ++row) {
      float buf[kDataStep];
      for (size_t k = 0; k < kDataStep; ++k) buf[k] = multi_vectors[k](row);
      data_[row] = ScalarF(buf);
    }
  }
  explicit VectorF(const std::vector<float*>& multi_elements) {
    if (multi_elements.size() != kRow)
      throw std::runtime_error("Wrong number of data");
    for (int row = 0; row < kRow; ++row)
      data_[row] = ScalarF(multi_elements.at(row));
  }

  VectorF(const VectorF& rhs) {
    for (int row = 0; row < kRow; ++row) data_[row] = rhs.data_[row];
  }
  ScalarF& operator()(const int row) { return data_[row]; }
  const ScalarF& operator()(const int row) const { return data_[row]; }

  VectorF& operator=(const VectorF& rhs) {
    for (int row = 0; row < kRow; ++row) data_[row] = rhs.data_[row];
    return *this;
  }

  VectorF operator+(const VectorF& rhs) const {
    VectorF res;
    for (int row = 0; row < kRow; ++row)
      res.data_[row] = data_[row] + rhs.data_[row];
    return res;
  }
  VectorF operator-(const VectorF& rhs) const {
    VectorF res;
    for (int row = 0; row < kRow; ++row)
      res.data_[row] = data_[row] - rhs.data_[row];
    return res;
  }
  VectorF operator*(const float scalar) const {
    VectorF res;
    for (int row = 0; row < kRow; ++row) res.data_[row] = data_[row] * scalar;
    return res;
  }
  VectorF operator*(const ScalarF scalar) const {
    VectorF res;
    for (int row = 0; row < kRow; ++row) res.data_[row] = data_[row] * scalar;
    return res;
  }
  VectorF& operator+=(const VectorF& rhs) {
    for (int row = 0; row < kRow; ++row) data_[row] += rhs.data_[row];
    return *this;
  }
  VectorF& operator+=(const float scalar) {
    for (int row = 0; row < kRow; ++row) data_[row] += scalar;
    return *this;
  }
  VectorF& operator-=(const VectorF& rhs) {
    for (int row = 0; row < kRow; ++row) data_[row] -= rhs.data_[row];
    return *this;
  }
  VectorF& operator-=(const float scalar) {
    for (int row = 0; row < kRow; ++row) data_[row] -= scalar;
    return *this;
  }

  ScalarF GetNorm() const {
    ScalarF norm_values;
    for (int row = 0; row < kRow; ++row)
      norm_values += (data_[row] * data_[row]);
    return norm_values;
  }

  ScalarF ComputeDot(const VectorF& rhs) const {
    ScalarF res;
    for (int row = 0; row < kRow; ++row) res += (data_[row] * rhs.data_[row]);
    return res;
  }

  void StoreData(std::vector<EigenVec>* multi_vectors) const {
    if (multi_vectors->size() != kDataStep) multi_vectors->resize(kDataStep);
    for (int row = 0; row < kRow; ++row) {
      float buf[kDataStep];
      data_[row].StoreData(buf);
      for (size_t k = 0; k < kDataStep; ++k) multi_vectors->at(k)(row) = buf[k];
    }
  }

  friend std::ostream& operator<<(std::ostream& outputStream,
                                  const VectorF& vec) {
    std::vector<EigenVec> multi_vectors;
    vec.StoreData(&multi_vectors);
    std::cout << "["
              << "[" << multi_vectors[0] << "],\n"
              << "[" << multi_vectors[1] << "],\n"
              << "[" << multi_vectors[2] << "],\n"
              << "[" << multi_vectors[3] << "]]" << std::endl;
    return outputStream;
  }

 private:
  ScalarF data_[kRow];
  template <int kMatRow, int kMatCol>
  friend class MatrixF;
};

/// @brief Matrix of SIMD data
/// @tparam kRow Matrix row size
/// @tparam kCol Matrix column size
template <int kRow, int kCol>
class MatrixF {
  const size_t kDataStep{8};
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
    for (int row = 0; row < kRow; ++row) {
      for (int col = 0; col < kCol; ++col) {
        float buf[kDataStep];
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
    for (int row = 0; row < kRow; ++row) {
      for (int col = 0; col < kCol; ++col) {
        float buf[kDataStep];
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

class Scalar {
 public:
  Scalar() { data_ = _mm256_setzero_pd(); }
  explicit Scalar(const double scalar) { data_ = _mm256_set1_pd(scalar); }
  explicit Scalar(const double n1, const double n2, const double n3,
                  const double n4) {
    data_ = _mm256_set_pd(n4, n3, n2, n1);
  }
  explicit Scalar(const double* rhs) { data_ = _mm256_load_pd(rhs); }
  Scalar(const __m256d& rhs) { data_ = rhs; }
  Scalar(const Scalar& rhs) { data_ = rhs.data_; }

  Scalar operator<(const double scalar) const {
    Scalar comp_mask(
        _mm256_and_pd(_mm256_cmp_pd(data_, _mm256_set1_pd(scalar), _CMP_LT_OS),
                      _mm256_set1_pd(1.0)));
    return comp_mask;
  }
  Scalar operator<=(const double scalar) const {
    Scalar comp_mask(
        _mm256_and_pd(_mm256_cmp_pd(data_, _mm256_set1_pd(scalar), _CMP_LE_OS),
                      _mm256_set1_pd(1.0)));
    return comp_mask;
  }
  Scalar operator>(const double scalar) const {
    Scalar comp_mask(
        _mm256_and_pd(_mm256_cmp_pd(data_, _mm256_set1_pd(scalar), _CMP_GT_OS),
                      _mm256_set1_pd(1.0)));
    return comp_mask;
  }
  Scalar operator>=(const double scalar) const {
    // Convert mask to 0.0 or 1.0
    Scalar comp_mask(
        _mm256_and_pd(_mm256_cmp_pd(data_, _mm256_set1_pd(scalar), _CMP_GE_OS),
                      _mm256_set1_pd(1.0)));
    return comp_mask;
  }
  Scalar operator<(const Scalar& rhs) const {
    Scalar comp_mask(_mm256_and_pd(_mm256_cmp_pd(data_, rhs.data_, _CMP_LT_OS),
                                   _mm256_set1_pd(1.0)));
    return comp_mask;
  }
  Scalar operator<=(const Scalar& rhs) const {
    Scalar comp_mask(_mm256_and_pd(_mm256_cmp_pd(data_, rhs.data_, _CMP_LE_OS),
                                   _mm256_set1_pd(1.0)));
    return comp_mask;
  }
  Scalar operator>(const Scalar& rhs) const {
    Scalar comp_mask(_mm256_and_pd(_mm256_cmp_pd(data_, rhs.data_, _CMP_GT_OS),
                                   _mm256_set1_pd(1.0)));
    return comp_mask;
  }
  Scalar operator>=(const Scalar& rhs) const {
    // Convert mask to 0.0 or 1.0
    Scalar comp_mask(_mm256_and_pd(_mm256_cmp_pd(data_, rhs.data_, _CMP_GE_OS),
                                   _mm256_set1_pd(1.0)));
    return comp_mask;
  }
  Scalar& operator=(const Scalar& rhs) {
    data_ = rhs.data_;
    return *this;
  }
  Scalar operator+(const double rhs) const {
    return Scalar(_mm256_add_pd(data_, _mm256_set1_pd(rhs)));
  }
  Scalar operator-() const {
    return Scalar(_mm256_sub_pd(_mm256_set1_pd(0.0), data_));
  }
  Scalar operator-(const double rhs) const {
    return Scalar(_mm256_sub_pd(data_, _mm256_set1_pd(rhs)));
  }
  Scalar operator*(const double rhs) const {
    return Scalar(_mm256_mul_pd(data_, _mm256_set1_pd(rhs)));
  }
  Scalar operator/(const double rhs) const {
    return Scalar(_mm256_div_pd(data_, _mm256_set1_pd(rhs)));
  }
  Scalar operator+(const Scalar& rhs) const {
    return Scalar(_mm256_add_pd(data_, rhs.data_));
  }
  Scalar operator-(const Scalar& rhs) const {
    return Scalar(_mm256_sub_pd(data_, rhs.data_));
  }
  Scalar operator*(const Scalar& rhs) const {
    return Scalar(_mm256_mul_pd(data_, rhs.data_));
  }
  Scalar operator/(const Scalar& rhs) const {
    return Scalar(_mm256_div_pd(data_, rhs.data_));
  }
  Scalar& operator+=(const Scalar& rhs) {
    data_ = _mm256_add_pd(data_, rhs.data_);
    return *this;
  }
  Scalar& operator-=(const Scalar& rhs) {
    data_ = _mm256_sub_pd(data_, rhs.data_);
    return *this;
  }
  Scalar& operator*=(const Scalar& rhs) {
    data_ = _mm256_mul_pd(data_, rhs.data_);
    return *this;
  }

  void StoreData(double* data) const { _mm256_store_pd(data, data_); }

  friend std::ostream& operator<<(std::ostream& outputStream,
                                  const Scalar& scalar) {
    double multi_scalars[4];
    scalar.StoreData(multi_scalars);
    std::cout << "[["
              << "[" << multi_scalars[0] << "],\n"
              << "[" << multi_scalars[1] << "],\n"
              << "[" << multi_scalars[2] << "],\n"
              << "[" << multi_scalars[3] << "]]" << std::endl;
    return outputStream;
  }

  static size_t GetDataStep() { return _SIMD_DATA_STEP_DOUBLE; }

 private:
  __m256d data_;
};

/// @brief Vector of Simd data. Consider four 3D vectors, v1, v2, v3, v4.
/// data_[0] = SimdDouble(v1.x(), v2.x(), v3.x(), v4.x());
/// data_[1] = SimdDouble(v1.y(), v2.y(), v3.y(), v4.y());
/// data_[2] = SimdDouble(v1.z(), v2.z(), v3.z(), v4.z());
/// @tparam kRow
template <int kRow>
class Vector {
  const size_t kDataStep{4};
  using EigenVec = Eigen::Matrix<double, kRow, 1>;

 public:
  Vector() {
    for (int row = 0; row < kRow; ++row) data_[row] = _mm256_set1_pd(0.0);
  }
  ~Vector() {}

  explicit Vector(const EigenVec& single_vector) {
    for (int row = 0; row < kRow; ++row)
      data_[row] = Scalar(single_vector(row));
  }
  explicit Vector(const std::vector<EigenVec>& multi_vectors) {
    if (multi_vectors.size() != kDataStep)
      throw std::runtime_error("Wrong number of data");
    for (int row = 0; row < kRow; ++row) {
      double buf[kDataStep];
      for (size_t k = 0; k < kDataStep; ++k) buf[k] = multi_vectors[k](row);
      data_[row] = Scalar(buf);
    }
  }

  Vector(const Vector& rhs) {
    for (int row = 0; row < kRow; ++row) data_[row] = rhs.data_[row];
  }
  Scalar& operator()(const int row) { return data_[row]; }
  const Scalar& operator()(const int row) const { return data_[row]; }

  Vector& operator=(const Vector& rhs) {
    for (int row = 0; row < kRow; ++row) data_[row] = rhs.data_[row];
    return *this;
  }

  Vector operator+(const Vector& rhs) const {
    Vector res;
    for (int row = 0; row < kRow; ++row)
      res.data_[row] = data_[row] + rhs.data_[row];
    return res;
  }
  Vector operator-(const Vector& rhs) const {
    Vector res;
    for (int row = 0; row < kRow; ++row)
      res.data_[row] = data_[row] - rhs.data_[row];
    return res;
  }
  Vector operator*(const double scalar) const {
    Vector res;
    for (int row = 0; row < kRow; ++row) res.data_[row] = data_[row] * scalar;
    return res;
  }
  Vector operator*(const Scalar scalar) const {
    Vector res;
    for (int row = 0; row < kRow; ++row) res.data_[row] = data_[row] * scalar;
    return res;
  }
  Vector& operator+=(const Vector& rhs) {
    for (int row = 0; row < kRow; ++row) data_[row] += rhs.data_[row];
    return *this;
  }
  Vector& operator+=(const double scalar) {
    for (int row = 0; row < kRow; ++row) data_[row] += scalar;
    return *this;
  }
  Vector& operator-=(const Vector& rhs) {
    for (int row = 0; row < kRow; ++row) data_[row] -= rhs.data_[row];
    return *this;
  }
  Vector& operator-=(const double scalar) {
    for (int row = 0; row < kRow; ++row) data_[row] -= scalar;
    return *this;
  }

  Scalar GetNorm() const {
    Scalar norm_values;
    for (int row = 0; row < kRow; ++row)
      norm_values += (data_[row] * data_[row]);
    return norm_values;
  }

  Scalar ComputeDot(const Vector& rhs) const {
    Scalar res;
    for (int row = 0; row < kRow; ++row) res += (data_[row] * rhs.data_[row]);
    return res;
  }

  void StoreData(std::vector<EigenVec>* multi_vectors) const {
    if (multi_vectors->size() != kDataStep) multi_vectors->resize(kDataStep);
    for (int row = 0; row < kRow; ++row) {
      double buf[kDataStep];
      data_[row].StoreData(buf);
      for (size_t k = 0; k < kDataStep; ++k) multi_vectors->at(k)(row) = buf[k];
    }
  }

  friend std::ostream& operator<<(std::ostream& outputStream,
                                  const Vector& vec) {
    std::vector<EigenVec> multi_vectors;
    vec.StoreData(&multi_vectors);
    std::cout << "["
              << "[" << multi_vectors[0] << "],\n"
              << "[" << multi_vectors[1] << "],\n"
              << "[" << multi_vectors[2] << "],\n"
              << "[" << multi_vectors[3] << "]]" << std::endl;
    return outputStream;
  }

 private:
  Scalar data_[kRow];
  template <int kMatRow, int kMatCol>
  friend class Matrix;
};

/// @brief Matrix of SIMD data
/// @tparam kRow Matrix row size
/// @tparam kCol Matrix column size
template <int kRow, int kCol>
class Matrix {
  const size_t kDataStep{4};
  using EigenMat = Eigen::Matrix<double, kRow, kCol>;

 public:
  Matrix() {
    for (int row = 0; row < kRow; ++row)
      for (int col = 0; col < kCol; ++col)
        data_[row][col] = _mm256_setzero_pd();
  }
  ~Matrix() {}

  explicit Matrix(const EigenMat& single_matrix) {
    for (int row = 0; row < kRow; ++row)
      for (int col = 0; col < kCol; ++col)
        data_[row][col] = Scalar(single_matrix(row, col));
  }
  explicit Matrix(const std::vector<EigenMat>& multi_matrices) {
    if (multi_matrices.size() != kDataStep)
      throw std::runtime_error("Wrong number of data");
    for (int row = 0; row < kRow; ++row) {
      for (int col = 0; col < kCol; ++col) {
        double buf[kDataStep];
        for (size_t k = 0; k < kDataStep; ++k)
          buf[k] = multi_matrices[k](row, col);
        data_[row][col] = Scalar(buf);
      }
    }
  }
  Matrix(const Matrix& rhs) {
    for (int row = 0; row < kRow; ++row)
      for (int col = 0; col < kCol; ++col)
        data_[row][col] = rhs.data_[row][col];
  }
  Scalar& operator()(const int row, const int col) { return data_[row][col]; }
  const Scalar& operator()(const int row, const int col) const {
    return data_[row][col];
  }
  Matrix& operator=(const Matrix& rhs) {
    for (int row = 0; row < kRow; ++row)
      for (int col = 0; col < kCol; ++col)
        data_[row][col] = rhs.data_[row][col];
    return *this;
  }
  Matrix operator+(const Matrix& rhs) const {
    Matrix res;
    for (int row = 0; row < kRow; ++row)
      for (int col = 0; col < kCol; ++col)
        res.data_[row][col] = data_[row][col] + rhs.data_[row][col];
    return res;
  }
  Matrix operator-(const Matrix& rhs) const {
    Matrix res;
    for (int row = 0; row < kRow; ++row)
      for (int col = 0; col < kCol; ++col)
        res.data_[row][col] = data_[row][col] - rhs.data_[row][col];
    return res;
  }
  Matrix operator*(const double scalar) const {
    Matrix res;
    for (int row = 0; row < kRow; ++row)
      for (int col = 0; col < kCol; ++col)
        res.data_[row][col] = data_[row][col] * scalar;
    return res;
  }
  Matrix operator*(const Scalar scalar) const {
    Matrix res;
    for (int row = 0; row < kRow; ++row)
      for (int col = 0; col < kCol; ++col)
        res.data_[row][col] = data_[row][col] * scalar;
    return res;
  }
  template <int kRhsCol>
  inline Matrix<kRow, kRhsCol> operator*(
      const Matrix<kCol, kRhsCol>& matrix) const {
    Matrix<kRow, kRhsCol> res;
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
  inline Vector<kRow> operator*(const Vector<kCol>& vector) const {
    Vector<kRow> res(Eigen::Matrix<double, kRow, 1>::Zero());
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
  Matrix& operator+=(const Matrix& rhs) {
    for (int row = 0; row < kRow; ++row)
      for (int col = 0; col < kCol; ++col)
        data_[row][col] += rhs.data_[row][col];
    return *this;
  }
  Matrix& operator-=(const Matrix& rhs) {
    for (int row = 0; row < kRow; ++row)
      for (int col = 0; col < kCol; ++col)
        data_[row][col] -= rhs.data_[row][col];
    return *this;
  }

  Matrix<kCol, kRow> transpose() const {
    Matrix<kCol, kRow> mat_trans;
    for (int row = 0; row < kRow; ++row)
      for (int col = 0; col < kCol; ++col)
        mat_trans(col, row) = data_[row][col];
    return mat_trans;
  }

  void StoreData(std::vector<EigenMat>* multi_matrices) const {
    if (multi_matrices->size() != kDataStep) multi_matrices->resize(kDataStep);
    for (int row = 0; row < kRow; ++row) {
      for (int col = 0; col < kCol; ++col) {
        double buf[kDataStep];
        data_[row][col].StoreData(buf);
        for (size_t k = 0; k < kDataStep; ++k)
          multi_matrices->at(k)(row, col) = buf[k];
      }
    }
  }

  friend std::ostream& operator<<(std::ostream& outputStream,
                                  const Matrix& mat) {
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
  Scalar data_[kRow][kCol];
};

}  // namespace simd
}  // namespace nonlinear_optimizer

#endif  // NONLINEAR_OPTIMIZER_SIMD_HELPER_SIMD_HELPER_AMD_H_

#endif  // define