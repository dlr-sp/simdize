
#include "benchmark/benchmark.h"
#include <experimental/bits/simd.h>
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

enum class Operation { Add, Sub, Mul, Div };

template<class T, Operation OP>
void SingleComputation(benchmark::State& state)
{
  T result, data[2] = { 1., 1. };

  for (auto _ : state)
  {
    benchmark::DoNotOptimize(data);
    if constexpr (OP == Operation::Add)
    {
      result = data[0] + data[1];
    }
    else if constexpr (OP == Operation::Sub)
    {
      result = data[0] - data[1];
    }
    else if constexpr (OP == Operation::Mul)
    {
      result = data[0] * data[1];
    }
    else if constexpr (OP == Operation::Div)
    {
      result = data[0] / data[1];
    }
    benchmark::DoNotOptimize(result);
  }
}


BENCHMARK_TEMPLATE(HeatCPUAndStack, double);
BENCHMARK_TEMPLATE(HeatCPUAndStack, fixed_simd<double>);
BENCHMARK_TEMPLATE(SingleComputation, double, Operation::Add);
BENCHMARK_TEMPLATE(SingleComputation, fixed_simd<double>, Operation::Add);
BENCHMARK_TEMPLATE(SingleComputation, double, Operation::Mul);
BENCHMARK_TEMPLATE(SingleComputation, fixed_simd<double>, Operation::Mul);
BENCHMARK_TEMPLATE(SingleComputation, double, Operation::Div);
BENCHMARK_TEMPLATE(SingleComputation, fixed_simd<double>, Operation::Div);
