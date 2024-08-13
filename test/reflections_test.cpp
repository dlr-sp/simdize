
#include <gtest/gtest.h>
#include <utility>
#include <vector>
#include <algorithm>
#include <numeric>
#include <random>

#include "simd_access/simd_access.hpp"

namespace {

template<class T>
struct TestStruct
{
  T x;
  T y[2];

  std::pair<T, T> GetPair() const
  {
    return std::pair(x, y[1]);
  }
};

struct TestData
{
  static constexpr size_t size = 103;
  std::vector<TestStruct<double>> v;

  explicit TestData() :
    v(size)
  {
    for (int i = 0; i < size; ++i)
    {
      v[i].x = i;
      v[i].y[0] = i + 1000;
      v[i].y[1] = i + 2000;
    }
  }
};

template<int SimdSize, class T>
inline auto simdized_value(const TestStruct<T>& t)
{
  using simd_access::simdized_value;
  return TestStruct<decltype(simdized_value<SimdSize>(t.x))>();
}

template<class DestType, class SrcType, class FN>
inline void simd_members(TestStruct<DestType>& d, const TestStruct<SrcType>& s, FN&& func)
{
  using simd_access::simd_members;
  simd_members(d.x, s.x, func);
  simd_members(d.y[0], s.y[0], func);
  simd_members(d.y[1], s.y[1], func);
}

}


TEST(Reflections, IndexedAccess)
{
  TestData src;
  constexpr size_t vec_size = stdx::native_simd<double>::size();

  simd_access::loop<vec_size>(0, 100, [&](auto i)
    {
      auto index = i.to_simd();
      auto ts = SIMD_ACCESS_V(src.v, index);
      for (int j = 0; j < vec_size; ++j)
      {
        EXPECT_EQ(ts.x[j], index[j]);
        EXPECT_EQ(ts.y[0][j], index[j] + 1000);
        EXPECT_EQ(ts.y[1][j], index[j] + 2000);
      }
    }, simd_access::VectorResidualLoop);
}

TEST(Reflections, RValueAccess)
{
  TestData src;
  constexpr size_t vec_size = stdx::native_simd<double>::size();

  {
    simd_access::index<vec_size> index{3};
    auto ts = SIMD_ACCESS(src.v, index, .GetPair());
    for (int j = 0; j < vec_size; ++j)
    {
      EXPECT_EQ(ts.first[j], j + 3);
      EXPECT_EQ(ts.second[j], j + 2003);
    }
  }

  {
    stdx::fixed_size_simd<size_t, vec_size> index;
    for (int i = 0; i < vec_size; ++i)
    {
      index[i] = vec_size - i + 3;
    }
    auto ts = SIMD_ACCESS(src.v, index, .GetPair());
    for (int i = 0; i < vec_size; ++i)
    {
      EXPECT_EQ(ts.first[i], vec_size - i + 3);
      EXPECT_EQ(ts.second[i], vec_size - i + 2003);
    }
  }

  {
    simd_access::index_array<vec_size> index;
    for (int i = 0; i < vec_size; ++i)
    {
      index.index_[i] = vec_size - i + 3;
    }
    auto ts = SIMD_ACCESS(src.v, index, .GetPair());
    for (int i = 0; i < vec_size; ++i)
    {
      EXPECT_EQ(ts.first[i], vec_size - i + 3);
      EXPECT_EQ(ts.second[i], vec_size - i + 2003);
    }
  }
}