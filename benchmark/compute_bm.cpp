
#include "benchmark/benchmark.h"
#include <experimental/bits/simd.h>
#include <vector>
#include <iostream>

#include "simd_access/simd_access.hpp"
#include "simd_access/simd_loop.hpp"

namespace sa = simd_access;
template<class T>
using fixed_simd = stdx::fixed_size_simd<T, stdx::native_simd<T>::size()>;

void HeatCPUAndStack(benchmark::State& state)
{
  fixed_simd<double> data[1000];
  for (int i = 0; i < 1000; ++i)
  {
    data[i] = 1.;
  }
  benchmark::DoNotOptimize(data);
  for (auto _ : state)
  {
    for (int i = 0; i < 1000; i += 4)
    {
      auto result = data[i] + data[i+1] * data[i+2] / data[i+3];
      benchmark::DoNotOptimize(result);
    }
  }
}

enum class Operation { Add, Sub, Mul, Div };

template<class T, Operation OP>
void SingleComputation(benchmark::State& state)
{
  T x, y;
  benchmark::DoNotOptimize(x);
  benchmark::DoNotOptimize(y);

  for (auto _ : state)
  {
    if constexpr (OP == Operation::Add)
    {
      auto result = x + y;
      benchmark::DoNotOptimize(result);
    }
    else if constexpr (OP == Operation::Sub)
    {
      auto result = x - y;
      benchmark::DoNotOptimize(result);
    }
    else if constexpr (OP == Operation::Mul)
    {
      auto result = x * y;
      benchmark::DoNotOptimize(result);
    }
    else if constexpr (OP == Operation::Div)
    {
      auto result = x / y;
      benchmark::DoNotOptimize(result);
    }
  }
}


BENCHMARK(HeatCPUAndStack);
BENCHMARK_TEMPLATE(SingleComputation, double, Operation::Add);
BENCHMARK_TEMPLATE(SingleComputation, fixed_simd<double>, Operation::Add);
BENCHMARK_TEMPLATE(SingleComputation, double, Operation::Mul);
BENCHMARK_TEMPLATE(SingleComputation, fixed_simd<double>, Operation::Mul);
BENCHMARK_TEMPLATE(SingleComputation, double, Operation::Div);
BENCHMARK_TEMPLATE(SingleComputation, fixed_simd<double>, Operation::Div);
