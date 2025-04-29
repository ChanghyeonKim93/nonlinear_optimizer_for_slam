#include <iostream>
#include <random>
#include <vector>

#include "nonlinear_optimizer/simd_helper/simd_helper.h"
#include "nonlinear_optimizer/time_checker.h"

using namespace nonlinear_optimizer;

namespace {

std::random_device rd;
std::mt19937 gen(rd());

}  // namespace

std::vector<double> GenerateDoubleVector(const int size) {
  std::vector<double> vec(size);
  std::uniform_real_distribution<double> dis(-1.0, 1.0);
  for (int i = 0; i < size; ++i) vec[i] = dis(gen);
  return vec;
}
std::vector<float> GenerateFloatVector(const int size) {
  std::vector<float> vec(size);
  std::uniform_real_distribution<float> dis(-1.0, 1.0);
  for (int i = 0; i < size; ++i) vec[i] = dis(gen);
  return vec;
}

#define COMPUTE_COMPLEX_ARITHMETIC(a, b)                                  \
  ((a) + (b)) * ((a) + (b) * (b) + 1.0) / ((a) + (b) + 3.0) +             \
      ((b) * (b) * 2.0) * ((a) * (b) * (b) + 4.0) * ((a) / ((b) + 1.2)) + \
      ((b) * (b) * 3.0) * ((a) * (b) * (b) + 5.0) * ((a) / ((b) + 1.3)) + \
      ((b) * (b) * (b) * 3.0) * ((a) * (a) * (b) + 2.0) *                 \
          ((a) / ((b) + 5.5)) +                                           \
      ((b) * (b) * (b) * 4.0) * ((a) * (a) * (b) + 4.0) *                 \
          ((a) / ((b) + 5.1)) +                                           \
      ((b) * (b) * (b) * 3.0) * ((a) * (a) * (b) + 2.0) * ((a) / ((b) + 5.2))

void TestDoubleScalar(const std::vector<double>& d1,
                      const std::vector<double>& d2, std::vector<double>* rd) {
  CHECK_EXEC_TIME_FROM_HERE
  const int len = d1.size();
  for (int i = 0; i < len; ++i) {
    const double s1 = d1.at(i);
    const double s2 = d2.at(i);
    double rs = COMPUTE_COMPLEX_ARITHMETIC(s1, s2);
    rd->push_back(rs);
  }
}

void TestFloatScalar(const std::vector<float>& d1, const std::vector<float>& d2,
                     std::vector<float>* rd) {
  CHECK_EXEC_TIME_FROM_HERE
  const int len = d1.size();
  for (int i = 0; i < len; ++i) {
    const float s1 = d1.at(i);
    const float s2 = d2.at(i);
    float rs = COMPUTE_COMPLEX_ARITHMETIC(s1, s2);
    rd->push_back(rs);
  }
}

void TestDouble(const std::vector<double>& d1, const std::vector<double>& d2,
                std::vector<double>* rd) {
  CHECK_EXEC_TIME_FROM_HERE
  double res[4];
  const int len = d1.size();
  double d1_ptr[4];
  double d2_ptr[4];
  for (int i = 0; i < len; i += 4) {
    d1_ptr[0] = d1[i];
    d1_ptr[1] = d1[i + 1];
    d1_ptr[2] = d1[i + 2];
    d1_ptr[3] = d1[i + 3];
    d2_ptr[0] = d2[i];
    d2_ptr[1] = d2[i + 1];
    d2_ptr[2] = d2[i + 2];
    d2_ptr[3] = d2[i + 3];
    simd::ScalarD s1(d1_ptr);
    simd::ScalarD s2(d2_ptr);
    simd::ScalarD rs = COMPUTE_COMPLEX_ARITHMETIC(s1, s2);
    rs.StoreData(res);
    rd->push_back(res[0]);
    rd->push_back(res[1]);
    rd->push_back(res[2]);
    rd->push_back(res[3]);
  }
}

void TestFloat(const std::vector<float>& d1, const std::vector<float>& d2,
               std::vector<float>* rd) {
  CHECK_EXEC_TIME_FROM_HERE
  float res[8];
  const int len = d1.size();
  float d1_ptr[8];
  float d2_ptr[8];
  for (int i = 0; i < len; i += 8) {
    d1_ptr[0] = d1[i];
    d1_ptr[1] = d1[i + 1];
    d1_ptr[2] = d1[i + 2];
    d1_ptr[3] = d1[i + 3];
    d1_ptr[4] = d1[i + 4];
    d1_ptr[5] = d1[i + 5];
    d1_ptr[6] = d1[i + 6];
    d1_ptr[7] = d1[i + 7];
    d2_ptr[0] = d2[i];
    d2_ptr[1] = d2[i + 1];
    d2_ptr[2] = d2[i + 2];
    d2_ptr[3] = d2[i + 3];
    d2_ptr[4] = d2[i + 4];
    d2_ptr[5] = d2[i + 5];
    d2_ptr[6] = d2[i + 6];
    d2_ptr[7] = d2[i + 7];
    simd::ScalarF s1(d1_ptr);
    simd::ScalarF s2(d2_ptr);
    simd::ScalarF rs = COMPUTE_COMPLEX_ARITHMETIC(s1, s2);
    rs.StoreData(res);
    rd->push_back(res[0]);
    rd->push_back(res[1]);
    rd->push_back(res[2]);
    rd->push_back(res[3]);
    rd->push_back(res[4]);
    rd->push_back(res[5]);
    rd->push_back(res[6]);
    rd->push_back(res[7]);
  }
}

int main(int, char**) {
  // Double precision test
  constexpr int kSize = 10000000;
  std::vector<double> d1 = GenerateDoubleVector(kSize);
  std::vector<double> d2 = GenerateDoubleVector(kSize);
  std::vector<double> rd;
  TestDouble(d1, d2, &rd);

  std::vector<double> rd_scalar;
  TestDoubleScalar(d1, d2, &rd_scalar);

  std::vector<float> f1 = GenerateFloatVector(kSize);
  std::vector<float> f2 = GenerateFloatVector(kSize);
  std::vector<float> rf;
  TestFloat(f1, f2, &rf);

  std::vector<float> rf_scalar;
  TestFloatScalar(f1, f2, &rf_scalar);

  for (size_t i = 0; i < rd.size(); ++i) {
    if (fabs(rd[i] - rd_scalar[i]) > 1e-7) {
      std::cout << "Double SIMD test failed at index " << i << std::endl;
      return -1;
    }
  }
  for (size_t i = 0; i < rf.size(); ++i) {
    if (fabs(rf[i] - rf_scalar[i]) > 1e-3) {
      std::cout << "Float SIMD test failed at index " << i << " : "
                << fabs(rf[i] - rf_scalar[i]) << std::endl;
      return -1;
    }
  }

  Eigen::Matrix<double, 3, 1> v1(1, 2, 3);
  Eigen::Matrix<double, 3, 1> v2(4, 5, 6);
  Eigen::Matrix<double, 3, 1> v3(7, 8, 9);
  Eigen::Matrix<double, 3, 1> v4(10, 11, 12);
  simd::VectorD<3> va__({v1, v2, v1, v2});
  simd::VectorD<3> vb__({v3, v4, v4, v3});
  const auto vc__ = va__ + vb__;

  std::vector<Eigen::Matrix<double, 3, 1>> res_list;
  vc__.StoreData(&res_list);
  for (int k = 0; k < 4; ++k) {
    std::cerr << k << "-th: " << res_list.at(k).transpose() << std::endl;
  }

  Eigen::Matrix<double, 3, 3> R;
  R << 1, 2, 3, 4, 5, 6, 7, 8, 9;
  Eigen::Matrix<double, 3, 1> p;
  p << 2, 4, 6;

  simd::MatrixD<3, 3> R__(R);
  simd::VectorD<3> p__(p);

  const simd::VectorD<3> x__ = R__ * p__;
  std::cerr << "Sol: " << (R * p).transpose() << std::endl;
  std::vector<Eigen::Vector3d> res;
  x__.StoreData(&res);

  for (int k = 0; k < 4; ++k) {
    std::cerr << res.at(k).transpose() << std::endl;
  }

  return 0;
}