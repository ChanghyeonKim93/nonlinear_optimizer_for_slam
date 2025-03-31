project(${CMAKE_PROJECT_NAME}_utility_tests)
message("  -- [SUBPROJECT]: " ${PROJECT_NAME})

find_package(Eigen3 REQUIRED)
find_package(Threads REQUIRED)
find_package(PkgConfig REQUIRED)
find_package(GTest REQUIRED)

set(UTILITY_SRC ${CMAKE_SOURCE_DIR}/nonlinear_optimizer/time_checker.cc)

# tests
add_executable(
  ${PROJECT_NAME}_simd_helper_performance_test
  simd_helper_performance_test.cc ${UTILITY_SRC})
target_include_directories(
  ${PROJECT_NAME}_simd_helper_performance_test
  PUBLIC ${CMAKE_SOURCE_DIR} # Include the parent CMake file's directory
         ${EIGEN3_INCLUDE_DIRS})

add_executable(
  ${PROJECT_NAME}_simd_helper_test
  simd_helper_test.cc
)
target_include_directories(
  ${PROJECT_NAME}_simd_helper_test 
  PUBLIC ${CMAKE_SOURCE_DIR} ${EIGEN3_INCLUDE_DIRS} ${GTEST_INCLUDE_DIR}
)
target_link_libraries(
  ${PROJECT_NAME}_simd_helper_test
  PUBLIC ${GTEST_BOTH_LIBRARIES}
         Threads::Threads
)
add_test(NAME ${PROJECT_NAME}_simd_helper_test
         COMMAND ${PROJECT_NAME}_simd_helper_test
)