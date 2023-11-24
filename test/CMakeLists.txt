enable_testing()

add_executable(
  simd_access_test
  elementwise_test.cpp
  loop_test.cpp
  macro_test.cpp
  potential_operator_overload.cpp
  aos_test.cpp
)
target_link_libraries(
  simd_access_test
  GTest::gtest_main
#  -fsanitize=address,undefined
)

include(GoogleTest)
gtest_discover_tests(simd_access_test)
