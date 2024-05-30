
#include <gtest/gtest.h>
#include <vector>
#include <algorithm>
#include <numeric>
#include <random>

#include "simd_access/simd_access.hpp"
#include "simd_access/simd_loop.hpp"

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
  simd_access::loop<vec_size>(indices.begin(), indices.end(), [&](auto i)
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

TEST(Loop, ResidualLoop)
{
  constexpr auto full_size = 64;
  constexpr auto dest_offset = 1001;
  constexpr size_t vec_size = stdx::native_simd<double>::size();
  constexpr size_t partial_size = full_size - vec_size + 1;
  double src[full_size], dest[full_size];
  auto fill_dest = [&]()
    { std::iota(dest, dest + full_size, dest_offset); };

  std::iota(src, src + full_size, 0);
  fill_dest();
  simd_access::loop<vec_size>(0, partial_size, [&](auto i)
    {
      SIMD_ACCESS(dest, i) = SIMD_ACCESS(src, i) * 2;
    });

  for (int i = 0; i < partial_size; ++i)
  {
    EXPECT_EQ(dest[i], i * 2);
  }
  for (int i = partial_size; i < full_size; ++i)
  {
    EXPECT_EQ(dest[i], dest_offset + i);
  }

  fill_dest();
  simd_access::loop<vec_size>(0, partial_size, [&](auto i)
    {
      SIMD_ACCESS(dest, i) = SIMD_ACCESS(src, i) * 2;
    }, simd_access::VectorResidualLoop);

  for (int i = 0; i < full_size; ++i)
  {
    EXPECT_EQ(dest[i], i * 2);
  }
}

