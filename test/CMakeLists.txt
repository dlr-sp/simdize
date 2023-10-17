enable_testing()

add_executable(
  simdize
  simdize.cpp
)
target_link_libraries(
  simdize
  GTest::gtest_main
)

include(GoogleTest)
gtest_discover_tests(simdize)
