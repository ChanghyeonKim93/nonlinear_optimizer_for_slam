#include <cassert>
#include <memory>

#include "gtest/gtest.h"

#include "nonlinear_optimizer/simd_helper.h"

namespace nonlinear_optimizer {

class SimdHelperTest : public ::testing::Test {
 protected:
  virtual void SetUp() {}
  virtual void TearDown() {}
};

TEST_F(SimdHelperTest, SimdDataLoadAndSaveTest) {
  // SIMD scalar data
  double value[4] = {1.23456, -0.45678, 4.16789, -1.42536};
  simd::Scalar v__(value);
  double buf[4];
  v__.StoreData(buf);
  for (int k = 0; k < 4; ++k) EXPECT_DOUBLE_EQ(value[k], buf[k]);
}

TEST_F(SimdHelperTest, MatrixVectorMultiplyTest) {
  Eigen::Matrix3d M1;
  M1 << 1, 2, 3, 4, 5, 6, 7, 8, 9;
  Eigen::Matrix3d M2;
  M2 << 1, -2, 3, -4, 5, -6, 7, -8, 9;
  Eigen::Matrix3d M3;
  M3 << -1, 2, -3, 4, -5, 6, -7, 8, -9;
  Eigen::Matrix3d M4;
  M4 << 9, -8, 7, -6, 5, -4, 3, -1, 2;
  Eigen::Vector3d v1(4, 6, 8);
  Eigen::Vector3d v2(-3, 1, 6);
  Eigen::Vector3d v3(9, 3, -5);
  Eigen::Vector3d v4(1, 6, -10);

  std::vector<Eigen::Vector3d> res_true;
  res_true.push_back(M1 * v1);
  res_true.push_back(M2 * v2);
  res_true.push_back(M3 * v3);
  res_true.push_back(M4 * v4);

  simd::Matrix<3, 3> M__({M1, M2, M3, M4});
  simd::Vector<3> v__({v1, v2, v3, v4});
  const auto Mv__ = M__ * v__;
  std::vector<Eigen::Vector3d> res;
  Mv__.StoreData(&res);
  for (int i = 0; i < 3; ++i)
    for (int k = 0; k < 4; ++k) EXPECT_DOUBLE_EQ(res_true[k](i), res[k](i));
}

TEST_F(SimdHelperTest, MemoryAlignmentTest) {
  const size_t alignment = 32;
  const size_t num_data = 10000;

  float* aligned_float = nullptr;
  aligned_float = reinterpret_cast<float*>(
      simd::GetAlignedMemory(num_data * sizeof(float)));
  EXPECT_NE(aligned_float, nullptr);
  bool is_aligned =
      (reinterpret_cast<std::uintptr_t>(aligned_float) % alignment == 0);
  EXPECT_TRUE(is_aligned);

  double* aligned_double = nullptr;
  aligned_double = reinterpret_cast<double*>(
      simd::GetAlignedMemory(num_data * sizeof(double)));
  EXPECT_NE(aligned_double, nullptr);
  is_aligned =
      (reinterpret_cast<std::uintptr_t>(aligned_double) % alignment == 0);
  EXPECT_TRUE(is_aligned);

  int* aligned_int = nullptr;
  aligned_int =
      reinterpret_cast<int*>(simd::GetAlignedMemory(num_data * sizeof(int)));
  EXPECT_NE(aligned_int, nullptr);
  is_aligned = (reinterpret_cast<std::uintptr_t>(aligned_int) % alignment == 0);
  EXPECT_TRUE(is_aligned);
}

}  // namespace nonlinear_optimizer
