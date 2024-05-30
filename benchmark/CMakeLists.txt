add_executable(
  simd_access_benchmark
  loop_bm.cpp
)
target_link_libraries(
  simd_access_benchmark
  benchmark::benchmark
  benchmark::benchmark_main
#  -fsanitize=address,undefined
)

#include(googlebenchmark)
