
#include "benchmark/benchmark.h"
#include <vector>
#include <iostream>

#include "simd_access/simd_access.hpp"
#include "simd_access/simd_loop.hpp"

namespace sa = simd_access;
template<class T>
using fixed_simd = stdx::fixed_size_simd<T, stdx::native_simd<T>::size()>;

template<class T>
void HeatCPUAndStack(benchmark::State& state)
{
  T data[1000];
  for (int i = 0; i < 1000; ++i)
  {
    data[i] = 1.;
  }
  for (auto _ : state)
  {
    benchmark::DoNotOptimize(data);
    for (int i = 0; i < 1000; i += 4)
    {
      auto result = data[i] + data[i+1] * data[i+2] / data[i+3];
      benchmark::DoNotOptimize(result);
    }
  }
}

template<class T, class OP>
void SingleComputation(benchmark::State& state)
{
  T result, data[2] = { 1., 1. };

  for (auto _ : state)
  {
    benchmark::DoNotOptimize(data);
    result = OP{}(data[0], data[1]);
    benchmark::DoNotOptimize(result);
  }
}


BENCHMARK_TEMPLATE(HeatCPUAndStack, double);
BENCHMARK_TEMPLATE(HeatCPUAndStack, fixed_simd<double>);
BENCHMARK_TEMPLATE(SingleComputation, double, std::plus<>);
BENCHMARK_TEMPLATE(SingleComputation, fixed_simd<double>, std::plus<>);
BENCHMARK_TEMPLATE(SingleComputation, double, std::multiplies<>);
BENCHMARK_TEMPLATE(SingleComputation, fixed_simd<double>, std::multiplies<>);
BENCHMARK_TEMPLATE(SingleComputation, double, std::divides<>);
BENCHMARK_TEMPLATE(SingleComputation, fixed_simd<double>, std::divides<>);
