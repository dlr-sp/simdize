
#include <gtest/gtest.h>

#include "simd_access/index.hpp"


TEST(Index, Linear)
{
  constexpr size_t vec_size = 4;
  simd_access::index<vec_size> index{3};
  auto value = index.to_simd();
  static_assert(index.size() == 4, "should be compile time size");

  for (int i = 0; i < vec_size; ++i)
  {
    EXPECT_EQ(index.scalar_index(i), i + 3);
    EXPECT_EQ(value[i], i + 3);
  }
}

TEST(Index, LinearTypeConvertion)
{
  constexpr size_t vec_size = 4;
  simd_access::index<vec_size, int> index{3};
  auto value = index.to_simd();
  static_assert(index.size() == 4, "should be compile time size");

  for (int i = 0; i < vec_size; ++i)
  {
    EXPECT_EQ(index.scalar_index(i), i + 3);
    EXPECT_EQ(value[i], i + 3);
  }
}
