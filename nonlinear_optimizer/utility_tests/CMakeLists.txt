project(${CMAKE_PROJECT_NAME}_utility_tests)
message("  -- [SUBPROJECT]: " ${PROJECT_NAME})

find_package(Eigen3 REQUIRED)
find_package(PkgConfig REQUIRED)

set(UTILITY_SRC ${CMAKE_SOURCE_DIR}/nonlinear_optimizer/time_checker.cc)

# tests
add_executable(
  ${PROJECT_NAME}_simd_helper_test
  simd_helper_test.cc ${UTILITY_SRC})
target_include_directories(
  ${PROJECT_NAME}_simd_helper_test
  PUBLIC ${CMAKE_SOURCE_DIR} # Include the parent CMake file's directory
         ${EIGEN3_INCLUDE_DIRS})