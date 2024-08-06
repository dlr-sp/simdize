
#include <gtest/gtest.h>
#include <vector>
#include <algorithm>
#include <numeric>
#include <random>

#include "simd_access/simd_access.hpp"
#include "simd_access/simd_loop.hpp"

namespace {

template<class T>
struct TestStruct
{
  T x;
  T y[2];
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
  func(d.x, s.x);
  func(d.y[0], s.y[0]);
  func(d.y[1], s.y[1]);
}

template<int SimdSize, class T>
inline auto simdized_value(const std::vector<T>& v)
{
  using simd_access::simdized_value;
  std::vector<decltype(simdized_value<SimdSize>(std::declval<T>()))> result(v.size());
  return result;
}

template<class DestType, class SrcType, class FN>
inline void simd_members(std::vector<DestType>& d, const std::vector<SrcType>& s, FN&& func)
{
  for (auto e = d.size(), i = 0; i < e; ++i)
  {
    func(d[i], s[i]);
  }
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

