#ifndef NONLINEAR_OPTIMIZER_SIMD_HELPER_SIMD_VECTOR_H_
#define NONLINEAR_OPTIMIZER_SIMD_HELPER_SIMD_VECTOR_H_

#include "nonlinear_optimizer/simd_helper/simd_scalar_amd.h"
#include "nonlinear_optimizer/simd_helper/simd_scalar_arm.h"

namespace nonlinear_optimizer {
namespace simd {

/// @brief Vector of Simd data. Consider four 3D vectors, v1, v2, v3, v4.
/// data_[0] = SimdDouble(v1.x(), v2.x(), v3.x(), v4.x());
/// data_[1] = SimdDouble(v1.y(), v2.y(), v3.y(), v4.y());
/// data_[2] = SimdDouble(v1.z(), v2.z(), v3.z(), v4.z());
/// @tparam kRow
template <int kRow>
class VectorF {
  const size_t kDataStep{_SIMD_DATA_STEP_FLOAT};
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
    float buf[8];
    for (int row = 0; row < kRow; ++row) {
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
    float buf[8];
    for (int row = 0; row < kRow; ++row) {
      data_[row].StoreData(buf);
      for (size_t k = 0; k < kDataStep; ++k) multi_vectors->at(k)(row) = buf[k];
    }
  }

  friend std::ostream& operator<<(std::ostream& outputStream,
                                  const VectorF& vec) {
    std::vector<EigenVec> multi_vectors;
    vec.StoreData(&multi_vectors);
    std::cout << "[";
    for (int i = 0; i < _SIMD_DATA_STEP_FLOAT; ++i) {
      std::cout << "[" << multi_vectors[i] << "]";
      if (i != _SIMD_DATA_STEP_FLOAT - 1) std::cout << ",\n";
    }
    std::cout << "]" << std::endl;
    return outputStream;
  }

 private:
  ScalarF data_[kRow];
  template <int kMatRow, int kMatCol>
  friend class MatrixF;
};

}  // namespace simd
}  // namespace nonlinear_optimizer

#if defined(__amd64__) || defined(__x86_64__)

namespace nonlinear_optimizer {
namespace simd {

/// @brief Vector of Simd data. Consider four 3D vectors, v1, v2, v3, v4.
/// data_[0] = SimdDouble(v1.x(), v2.x(), v3.x(), v4.x());
/// data_[1] = SimdDouble(v1.y(), v2.y(), v3.y(), v4.y());
/// data_[2] = SimdDouble(v1.z(), v2.z(), v3.z(), v4.z());
/// @tparam kRow
template <int kRow>
class VectorD {
  const size_t kDataStep{4};
  using EigenVec = Eigen::Matrix<double, kRow, 1>;

 public:
  VectorD() {
    for (int row = 0; row < kRow; ++row) data_[row] = _mm256_set1_pd(0.0);
  }
  ~VectorD() {}

  explicit VectorD(const EigenVec& single_vector) {
    for (int row = 0; row < kRow; ++row)
      data_[row] = ScalarD(single_vector(row));
  }
  explicit VectorD(const std::vector<EigenVec>& multi_vectors) {
    if (multi_vectors.size() != kDataStep)
      throw std::runtime_error("Wrong number of data");
    double buf[8];
    for (int row = 0; row < kRow; ++row) {
      for (size_t k = 0; k < kDataStep; ++k) buf[k] = multi_vectors[k](row);
      data_[row] = ScalarD(buf);
    }
  }

  VectorD(const VectorD& rhs) {
    for (int row = 0; row < kRow; ++row) data_[row] = rhs.data_[row];
  }
  ScalarD& operator()(const int row) { return data_[row]; }
  const ScalarD& operator()(const int row) const { return data_[row]; }

  VectorD& operator=(const VectorD& rhs) {
    for (int row = 0; row < kRow; ++row) data_[row] = rhs.data_[row];
    return *this;
  }

  VectorD operator+(const VectorD& rhs) const {
    VectorD res;
    for (int row = 0; row < kRow; ++row)
      res.data_[row] = data_[row] + rhs.data_[row];
    return res;
  }
  VectorD operator-(const VectorD& rhs) const {
    VectorD res;
    for (int row = 0; row < kRow; ++row)
      res.data_[row] = data_[row] - rhs.data_[row];
    return res;
  }
  VectorD operator*(const double scalar) const {
    VectorD res;
    for (int row = 0; row < kRow; ++row) res.data_[row] = data_[row] * scalar;
    return res;
  }
  VectorD operator*(const ScalarD scalar) const {
    VectorD res;
    for (int row = 0; row < kRow; ++row) res.data_[row] = data_[row] * scalar;
    return res;
  }
  VectorD& operator+=(const VectorD& rhs) {
    for (int row = 0; row < kRow; ++row) data_[row] += rhs.data_[row];
    return *this;
  }
  VectorD& operator+=(const double scalar) {
    for (int row = 0; row < kRow; ++row) data_[row] += scalar;
    return *this;
  }
  VectorD& operator-=(const VectorD& rhs) {
    for (int row = 0; row < kRow; ++row) data_[row] -= rhs.data_[row];
    return *this;
  }
  VectorD& operator-=(const double scalar) {
    for (int row = 0; row < kRow; ++row) data_[row] -= scalar;
    return *this;
  }

  ScalarD GetNorm() const {
    ScalarD norm_values;
    for (int row = 0; row < kRow; ++row)
      norm_values += (data_[row] * data_[row]);
    return norm_values;
  }

  ScalarD ComputeDot(const VectorD& rhs) const {
    ScalarD res;
    for (int row = 0; row < kRow; ++row) res += (data_[row] * rhs.data_[row]);
    return res;
  }

  void StoreData(std::vector<EigenVec>* multi_vectors) const {
    if (multi_vectors->size() != kDataStep) multi_vectors->resize(kDataStep);
    double buf[8];
    for (int row = 0; row < kRow; ++row) {
      data_[row].StoreData(buf);
      for (size_t k = 0; k < kDataStep; ++k) multi_vectors->at(k)(row) = buf[k];
    }
  }

  friend std::ostream& operator<<(std::ostream& outputStream,
                                  const VectorD& vec) {
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
  ScalarD data_[kRow];
  template <int kMatRow, int kMatCol>
  friend class MatrixD;
};

}  // namespace simd
}  // namespace nonlinear_optimizer

#endif

#endif  // NONLINEAR_OPTIMIZER_SIMD_HELPER_SIMD_VECTOR_H_