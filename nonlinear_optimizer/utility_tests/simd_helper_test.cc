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

  // SIMD vector data
}

}  // namespace nonlinear_optimizer
