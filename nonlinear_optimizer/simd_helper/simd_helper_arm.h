#ifndef NONLINEAR_OPTIMIZER_SIMD_HELPER_SIMD_HELPER_ARM_H_
#define NONLINEAR_OPTIMIZER_SIMD_HELPER_SIMD_HELPER_ARM_H_

#include <iostream>

#if defined(__ARM_ARCH) || defined(__aarch64__)

// ARM-based CPU
// Required CMake flag: -o2 -ftree-vectorize -mtune=cortex-a72 (your cpu arch.
// model)

// Reference: https://arm-software.github.io/acle/neon_intrinsics/advsimd.html
#include "arm_neon.h"

#define _SIMD_DATA_STEP_FLOAT 4
#define _SIMD_FLOAT float32x4_t
#define _SIMD_SET1 vdupq_n_f32  // == vmovq_n_f32
#define _SIMD_LOAD vld1q_f32    // float32x4_t vld1q_f32(float32_t const *ptr)
#define _SIMD_RCP vrecpeq_f32   // float32x4_t vrecpeq_f32(float32x4_t a)
#define _SIMD_DIV \
  vdivq_f32  // float32x4_t vdivq_f32(float32x4_t a, float32x4_t b)
#define _SIMD_MUL \
  vmulq_f32  // float32x4_t vmulq_f32(float32x4_t a, float32x4_t b)
#define _SIMD_ADD \
  vaddq_f32  // float32x4_t vaddq_f32(float32x4_t a, float32x4_t b)
#define _SIMD_SUB \
  vsubq_f32  // float32x4_t vsubq_f32(float32x4_t a, float32x4_t b)
#define _SIMD_STORE \
  vst1q_f32  // void vst1q_f32(float32_t *ptr, float32x4_t val)
#define _SIMD_ROUND_FLOAT vrndaq_f32  // float32x4_t vrndq_f32(float32x4_t a)
#define _SIMD_SQRT_FLOAT vsqrtq_f32   // float32x4_t vsqrtq_f32(float32x4_t a)

namespace nonlinear_optimizer {

namespace simd {

class ScalarF {
 public:
  ScalarF() { data_ = vmovq_n_f32(0.0f); }

  explicit ScalarF(const float scalar) { data_ = vmovq_n_f32(scalar); }

  explicit ScalarF(const float n1, const float n2, const float n3,
                   const float n4) {
    data_ = vdupq_n_f32(0.0f);
    data_ = vsetq_lane_f32(n1, data_, 0);
    data_ = vsetq_lane_f32(n2, data_, 1);
    data_ = vsetq_lane_f32(n3, data_, 2);
    data_ = vsetq_lane_f32(n4, data_, 3);
  }

  explicit ScalarF(const float* rhs) { data_ = vld1q_f32(rhs); }

  ScalarF(const float32x4_t& rhs) { data_ = rhs; }

  ScalarF(const ScalarF& rhs) { data_ = rhs.data_; }

  // Note: vcgtq (a > b), vcgeq (a >= b), vcltq (a < b), vcleq (a <= b),
  // vceqq (a == b), vcneq (a != b)
  ScalarF operator<(const float scalar) const {
    ScalarF comp_mask(vbslq_f32(vcltq_f32(data_, vdupq_n_f32(scalar)),
                                vdupq_n_f32(1.0f), vdupq_n_f32(0.0f)));
    return comp_mask;
  }

  ScalarF operator<(const ScalarF& rhs) const {
    ScalarF comp_mask(vbslq_f32(vcltq_f32(data_, rhs.data_), vdupq_n_f32(1.0f),
                                vdupq_n_f32(0.0f)));
    return comp_mask;
  }

  ScalarF operator<=(const float scalar) const {
    ScalarF comp_mask(vbslq_f32(vcleq_f32(data_, vdupq_n_f32(scalar)),
                                vdupq_n_f32(1.0f), vdupq_n_f32(0.0f)));
    return comp_mask;
  }

  ScalarF operator<=(const ScalarF& rhs) const {
    ScalarF comp_mask(vbslq_f32(vcleq_f32(data_, rhs.data_), vdupq_n_f32(1.0f),
                                vdupq_n_f32(0.0f)));
    return comp_mask;
  }

  ScalarF operator>(const float scalar) const {
    ScalarF comp_mask(vbslq_f32(vcgtq_f32(data_, vdupq_n_f32(scalar)),
                                vdupq_n_f32(1.0f), vdupq_n_f32(0.0f)));
    return comp_mask;
  }

  ScalarF operator>(const ScalarF& rhs) const {
    ScalarF comp_mask(vbslq_f32(vcgtq_f32(data_, rhs.data_), vdupq_n_f32(1.0f),
                                vdupq_n_f32(0.0f)));
    return comp_mask;
  }

  ScalarF operator>=(const float scalar) const {
    ScalarF comp_mask(vbslq_f32(vcgeq_f32(data_, vdupq_n_f32(scalar)),
                                vdupq_n_f32(1.0f), vdupq_n_f32(0.0f)));
    return comp_mask;
  }

  ScalarF operator>=(const ScalarF& rhs) const {
    ScalarF comp_mask(vbslq_f32(vcgeq_f32(data_, rhs.data_), vdupq_n_f32(1.0f),
                                vdupq_n_f32(0.0f)));
    return comp_mask;
  }

  ScalarF& operator=(const ScalarF& rhs) {
    data_ = rhs.data_;
    return *this;
  }

  ScalarF operator+(const float rhs) const {
    return ScalarF(vaddq_f32(data_, vdupq_n_f32(rhs)));
  }

  ScalarF operator+(const ScalarF& rhs) const {
    return ScalarF(vaddq_f32(data_, rhs.data_));
  }

  ScalarF operator-() const {
    return ScalarF(vsubq_f32(vdupq_n_f32(0.0f), data_));
  }

  ScalarF operator-(const ScalarF& rhs) const {
    return ScalarF(vsubq_f32(data_, rhs.data_));
  }

  ScalarF operator-(const float rhs) const {
    return ScalarF(vsubq_f32(data_, vdupq_n_f32(rhs)));
  }

  ScalarF operator*(const float rhs) const {
    return ScalarF(vmulq_f32(data_, vdupq_n_f32(rhs)));
  }

  ScalarF operator*(const ScalarF& rhs) const {
    return ScalarF(vmulq_f32(data_, rhs.data_));
  }

  ScalarF operator/(const float rhs) const {
    return ScalarF(vdivq_f32(data_, vdupq_n_f32(rhs)));
  }

  ScalarF operator/(const ScalarF& rhs) const {
    return ScalarF(vdivq_f32(data_, rhs.data_));
  }

  ScalarF& operator+=(const ScalarF& rhs) {
    data_ = vaddq_f32(data_, rhs.data_);
    return *this;
  }

  ScalarF& operator-=(const ScalarF& rhs) {
    data_ = vsubq_f32(data_, rhs.data_);
    return *this;
  }

  ScalarF& operator*=(const ScalarF& rhs) {
    data_ = vmulq_f32(data_, rhs.data_);
    return *this;
  }

  void StoreData(float* data) const { vst1q_f32(data, data_); }

  friend std::ostream& operator<<(std::ostream& outputStream,
                                  const ScalarF& scalar) {
    float multi_scalars[3];
    scalar.StoreData(multi_scalars);
    std::cout << "[["
              << "[" << multi_scalars[0] << "],\n"
              << "[" << multi_scalars[1] << "],\n"
              << "[" << multi_scalars[2] << "],\n"
              << "[" << multi_scalars[3] << "]]" << std::endl;
    return outputStream;
  }

  static size_t GetDataStep() { return _SIMD_DATA_STEP_FLOAT; }

 private:
  float32x4_t data_;
};

/// @brief Vector of Simd data. Consider four 3D vectors, v1, v2, v3, v4.
/// data_[0] = SimdDouble(v1.x(), v2.x(), v3.x(), v4.x());
/// data_[1] = SimdDouble(v1.y(), v2.y(), v3.y(), v4.y());
/// data_[2] = SimdDouble(v1.z(), v2.z(), v3.z(), v4.z());
/// @tparam kRow
template <int kRow>
class VectorF {
  const size_t kDataStep{4};
  using EigenVec = Eigen::Matrix<float, kRow, 1>;

 public:
  VectorF() {
    for (int row = 0; row < kRow; ++row) data_[row] = vdupq_n_f32(0.0f);
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
  const size_t kDataStep{4};
  using EigenMat = Eigen::Matrix<float, kRow, kCol>;

 public:
  MatrixF() {
    for (int row = 0; row < kRow; ++row)
      for (int col = 0; col < kCol; ++col) data_[row][col] = vdupq_n_f32(0.0f);
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

}  // namespace simd
}  // namespace nonlinear_optimizer

#endif

#endif  // NONLINEAR_OPTIMIZER_SIMD_HELPER_SIMD_HELPER_ARM_H_