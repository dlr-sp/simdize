
#include "benchmark/benchmark.h"
#include <experimental/bits/simd.h>
#include <vector>
#include <algorithm>
#include <numeric>
#include <random>

#include "simd_access/simd_access.hpp"
#include "simd_access/simd_loop.hpp"

template<class T>
void HeatCache(const std::vector<T>& testData)
{
  for (size_t i = 0, e = testData.size(); i < e; ++i)
  {
    auto result = testData[i];
    benchmark::DoNotOptimize(result);
  };
}

void Loop_ScatteredSimdReadAccess(benchmark::State& state)
{
  auto arraySize = state.range(0);
  constexpr size_t vec_size = stdx::native_simd<double>::size();
  std::vector<std::pair<double,double>> testData(arraySize, {1.0, 1.0});
  HeatCache(testData);
  auto dataPtr = testData.data();
  for (auto _ : state)
  {
    simd_access::loop<vec_size>(0, testData.size(), [&](auto i)
      {
        auto result = SIMD_ACCESS(dataPtr, i, .first).to_simd();
        benchmark::DoNotOptimize(result);
      }, simd_access::VectorResidualLoop);
  }
  state.SetBytesProcessed(arraySize * (sizeof(double)) * state.iterations());
}

void Loop_LinearSimdReadAccess(benchmark::State& state)
{
  auto arraySize = state.range(0);
  constexpr size_t vec_size = stdx::native_simd<double>::size();
  std::vector<double> testData(arraySize, 1.0);
  HeatCache(testData);
  auto dataPtr = testData.data();
  for (auto _ : state)
  {
    simd_access::loop<vec_size>(0, testData.size(), [&](auto i)
      {
        auto result = SIMD_ACCESS(dataPtr, i).to_simd();
        benchmark::DoNotOptimize(result);
      }, simd_access::VectorResidualLoop);
  }
  state.SetBytesProcessed(arraySize * (sizeof(double)) * state.iterations());
}

void Loop_LinearInlinedSimdReadAccess(benchmark::State& state)
{
  auto arraySize = state.range(0);
  constexpr size_t vec_size = stdx::native_simd<double>::size();
  std::vector<double> testData(arraySize, 1.0);
  HeatCache(testData);
  auto dataPtr = testData.data();
  for (auto _ : state)
  {
    for (size_t i = 0, e = testData.size(); i < e; i += vec_size)
    {
      auto result = stdx::fixed_size_simd<double, vec_size>(dataPtr + i, stdx::element_aligned);
      benchmark::DoNotOptimize(result);
    };
  }
  state.SetBytesProcessed(arraySize * (sizeof(double)) * state.iterations());
}

void Loop_LinearScalarReadAccess(benchmark::State& state)
{
  auto arraySize = state.range(0);
  constexpr size_t vec_size = stdx::native_simd<double>::size();
  std::vector<double> testData(arraySize, 1.0);
  HeatCache(testData);
  for (auto _ : state)
  {
    for (size_t i = 0, e = testData.size(); i < e; ++i)
    {
      auto result = testData[i];
      benchmark::DoNotOptimize(result);
    };
  }
  state.SetBytesProcessed(arraySize * (sizeof(double)) * state.iterations());
}

BENCHMARK(Loop_ScatteredSimdReadAccess)->Unit(benchmark::kMicrosecond)->UseRealTime()
  ->Arg(100)->Arg(4000)->ArgName("size");
BENCHMARK(Loop_LinearSimdReadAccess)->Unit(benchmark::kMicrosecond)->UseRealTime()
  ->Arg(100)->Arg(4000)->ArgName("size");
BENCHMARK(Loop_LinearInlinedSimdReadAccess)->Unit(benchmark::kMicrosecond)->UseRealTime()
  ->Arg(100)->Arg(4000)->ArgName("size");
BENCHMARK(Loop_LinearScalarReadAccess)->Unit(benchmark::kMicrosecond)->UseRealTime()
  ->Arg(100)->Arg(4000)->ArgName("size");
