
#include <gtest/gtest.h>

#include "element_access.hpp"


TEST(Elementwise, SimpleRead)
{
  double source[5] = { 1, 2, 3, 4, 5 };
  double dest[5] = { };
  constexpr size_t vec_size = 4;
  stdx::fixed_size_simd<double, vec_size> x(source, stdx::element_aligned);

  int linear_index = 0;

  auto read_fn = [&](auto&& y)
    {
      dest[linear_index++] = y;
    };

  simd_access::elementwise(x, read_fn);
  simd_access::elementwise(source[4], read_fn);

  for (int i = 0; i < 5; ++i)
  {
    EXPECT_EQ(dest[i], i + 1);
  }
}

TEST(Elementwise, SimpleWrite)
{
  double source[5] = { 1, 2, 3, 4, 5 };
  constexpr size_t vec_size = 4;
  stdx::fixed_size_simd<double, vec_size> dest_v (.0);
  double dest_s = .0;

  int linear_index = 0;
  auto write_fn = [&](auto&& y)
    {
      simd_access::element_write(y) = source[linear_index++];
    };

  simd_access::elementwise(dest_v, write_fn);
  simd_access::elementwise(dest_s, write_fn);

  for (int i = 0; i < 4; ++i)
  {
    EXPECT_EQ(dest_v[i], i + 1);
  }
  EXPECT_EQ(dest_s, 5);
}

