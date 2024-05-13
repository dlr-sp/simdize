
#include <gtest/gtest.h>

#include "simd_access/simd_access.hpp"
#include "simd_access/simd_loop.hpp"

namespace {

template<class T>
struct Point
{
  T x, y;
};

template<class T>
Point<T> operator+(const Point<T>& p1, const Point<T>& p2)
{
  return Point<T>{p1.x + p2.x, p1.y + p2.y};
}

template<size_t ElementSize, template<class, int, class...> class Location, class T, int SimdSize, class... Args>
auto load(const Location<Point<T>, SimdSize, Args...>& location)
{
  return Point<stdx::fixed_size_simd<T, SimdSize>>
    {load<ElementSize>(location.template member_access<&Point<T>::x>()),
     load<ElementSize>(location.template member_access<&Point<T>::y>())};
}

template<size_t ElementSize, template<class, int, class...> class Location, class T, int SimdSize, class... Args>
void store(const Location<Point<T>, SimdSize, Args...>& location,
  const Point<stdx::fixed_size_simd<T, SimdSize>>& source)
{
  store<ElementSize>(location.template member_access<&Point<T>::x>(), source.x);
  store<ElementSize>(location.template member_access<&Point<T>::y>(), source.y);
}

}

TEST(AosTest, LinearAddition)
{
  static constexpr size_t size = 103;
  Point<double> src1[size], src2[size], dest[size];
  for (int i = 0; i < size; ++i)
  {
    src1[i].x = i;
    src2[i].x = i * 2;
    src1[i].y = i * 3;
    src2[i].y = i * 4;
  }

  constexpr size_t vec_size = stdx::native_simd<double>::size();

  simd_access::loop<vec_size>(0, size, [&](auto i)
    {
      SIMD_ACCESS(dest, i) = SIMD_ACCESS(src1, i) + SIMD_ACCESS(src2, i);
    });

  for (int i = 0; i < size; ++i)
  {
    EXPECT_EQ(dest[i].x, i * 3);
    EXPECT_EQ(dest[i].y, i * 7);
  }
}

TEST(AosTest, IndirectAddition)
{
  static constexpr size_t size = 10;
  Point<double> src1[size], src2[size], dest[size];
  for (int i = 0; i < size; ++i)
  {
    src1[i].x = i;
    src2[i].x = i * 2;
    src1[i].y = i * 3;
    src2[i].y = i * 4;
  }
  const int indices[10] = { 3, 2, 1, 3, 2, 1, 3, 2, 1, 0 };

  constexpr size_t vec_size = stdx::native_simd<double>::size();

  simd_access::loop_with_linear_index<vec_size>(indices, indices + size, [&](auto linear_index, auto i)
    {
      SIMD_ACCESS(dest, linear_index) = SIMD_ACCESS(src1, i) + SIMD_ACCESS(src2, i);
    });

  for (int i = 0; i < size; ++i)
  {
    auto expected_result = src1[indices[i]] + src2[indices[i]];
    EXPECT_EQ(dest[i].x, expected_result.x);
    EXPECT_EQ(dest[i].y, expected_result.y);
  }
}
