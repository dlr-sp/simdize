
#include "benchmark/benchmark.h"
#include <vector>

template<class T>
void HeatCache(const std::vector<T>& testData)
{
  for (size_t i = 0, e = testData.size(); i < e; ++i)
  {
    auto result = testData[i];
    benchmark::DoNotOptimize(result);
  };
}

template<class OutputIt, typename SizeType, class Generator>
constexpr OutputIt GenerateNWithIndex(OutputIt first, SizeType count, Generator g)
{
  for (SizeType i = 0; i < count; ++i, ++first)
  {
    *first = g(i);
  }
  return first;
}
