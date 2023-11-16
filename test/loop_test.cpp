
#include <gtest/gtest.h>
#include <vector>
#include <algorithm>
#include <numeric>
#include <random>

#include "simd_access.hpp"
#include "simd_loop.hpp"

namespace {

struct TestStruct
{
  double x;
  double y[1];
};

struct TestData
{
  static constexpr size_t size = 103;
  double a[size];
  double a_subarr[size][1];
  TestStruct s[size];
  std::vector<double> v;

  explicit TestData(bool jota) :
    v(size)
  {
    for (int i = 0; i < size; ++i)
    {
      a[i] = v[i] = a_subarr[i][0] = s[i].x = s[i].y[0] = jota ? i : 0;
    }
  }
};

}

TEST(Loop, LinearCopy)
{
  TestData src(true), dest(false);
  constexpr size_t vec_size = stdx::native_simd<double>::size();

  simd_access::loop<vec_size>(0, src.size, [&](auto i)
    {
      SIMD_ACCESS(dest.a, i) = SIMD_ACCESS(src.a, i) * 2;
      SIMD_ACCESS(dest.a_subarr, i, [0]) = SIMD_ACCESS(src.a_subarr, i, [0]) * 3;
    });

  for (int i = 0; i < src.size; ++i)
  {
    EXPECT_EQ(dest.a[i], i * 2);
    EXPECT_EQ(dest.a_subarr[i][0], i * 3);
  }
}

TEST(Loop, IndirectCopy)
{
  TestData src(true), dest(false);
  std::vector<int> indices(src.size);
  std::iota(indices.begin(), indices.end(), 0);
  std::mt19937 g(1);
  std::shuffle(indices.begin(), indices.end(), g);
  constexpr size_t vec_size = stdx::native_simd<double>::size();

  int linear_index = 0;
  simd_access::loop<vec_size>(indices, [&](auto i)
    {
      auto x = SIMD_ACCESS(src.a, i) * 1;
      simd_access::elementwise(x, [&](auto&& v)
      {
        dest.a[linear_index++] = v;
      });
    });

  for (int i = 0; i < src.size; ++i)
  {
    EXPECT_EQ(dest.a[i], indices[i]);
  }
}


