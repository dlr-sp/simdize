
#include <experimental/bits/simd.h>
#include <gtest/gtest.h>
#include <vector>

#include "simd_access/simd_access.hpp"

namespace {

struct TestStruct
{
  double x;
  double y[1];
};

struct TestData
{
  static constexpr size_t size = 10;
  double a[size];
  double a_subarr[size][1];
  TestStruct s[size];
  std::vector<double> v;

  TestData() :
    v(size)
  {
    for (int i = 0; i < size; ++i)
    {
      a[i] = v[i] = a_subarr[i][0] = s[i].x = s[i].y[0] = i;
    }
  }
};

}

TEST(Macro, UnvectorizedArrayAccess)
{
  TestData t;
  namespace sa = simd_access;

  for (int i = 0; i < TestData::size; ++i)
  {
    EXPECT_EQ(SIMD_ACCESS(t.a, i), i);
    EXPECT_EQ(SIMD_ACCESS(t.a_subarr, i, [0]), i);
    EXPECT_EQ(SIMD_ACCESS(t.s, i, .x), i);
    EXPECT_EQ(SIMD_ACCESS(t.s, i, .y[0]), i);
    EXPECT_EQ(SIMD_ACCESS(t.v, i), i);

    EXPECT_EQ(SIMD_ACCESS_V(t.a, i), i);
    EXPECT_EQ(SIMD_ACCESS_V(t.a_subarr, i, [0]), i);
    EXPECT_EQ(SIMD_ACCESS_V(t.s, i, .x), i);
    EXPECT_EQ(SIMD_ACCESS_V(t.s, i, .y[0]), i);
    EXPECT_EQ(SIMD_ACCESS_V(t.v, i), i);

    EXPECT_EQ(sa::sa(t.a, i), i);
    EXPECT_EQ(sa::sa(t.a_subarr, i)[0], i);
    EXPECT_EQ(sa::sa(t.v, i), i);
  }
}

TEST(Macro, DirectVectorizedArrayAccess)
{
  TestData t;
  namespace sa = simd_access;

  constexpr size_t vec_size = stdx::native_simd<double>::size();
  simd_access::index<vec_size> index{3};

  stdx::fixed_size_simd<double, vec_size> x_a = SIMD_ACCESS(t.a, index);
  stdx::fixed_size_simd<double, vec_size> x_a_arr = SIMD_ACCESS(t.a_subarr, index, [0]);
  stdx::fixed_size_simd<double, vec_size> x_s_x = SIMD_ACCESS(t.s, index, .x);
  stdx::fixed_size_simd<double, vec_size> x_s_y = SIMD_ACCESS(t.s, index, .y[0]);
  stdx::fixed_size_simd<double, vec_size> x_v = SIMD_ACCESS(t.v, index);

  stdx::fixed_size_simd<double, vec_size> x_sa_a = sa::sa(t.a, index);
  stdx::fixed_size_simd<double, vec_size> x_sa_arr = sa::sa(t.a_subarr, index)[0];
  stdx::fixed_size_simd<double, vec_size> x_sa_v = sa::sa(t.v, index);

  for (int i = 0; i < vec_size; ++i)
  {
    EXPECT_EQ(x_a[i], i + 3);
    EXPECT_EQ(x_a_arr[i], i + 3);
    EXPECT_EQ(x_s_x[i], i + 3);
    EXPECT_EQ(x_s_y[i], i + 3);
    EXPECT_EQ(x_v[i], i + 3);
    EXPECT_EQ(x_sa_a[i], i + 3);
    EXPECT_EQ(x_sa_arr[i], i + 3);
    EXPECT_EQ(x_sa_v[i], i + 3);
  }
}


TEST(Macro, IndirectVectorizedArrayAccess)
{
  TestData t;
  namespace sa = simd_access;

  constexpr size_t vec_size = stdx::native_simd<double>::size();
  simd_access::index_array<vec_size> index;
  for (int i = 0; i < vec_size; ++i)
  {
    index.index_[i] = vec_size - i + 3;
  }

  stdx::fixed_size_simd<double, vec_size> x_a = SIMD_ACCESS(t.a, index);
  stdx::fixed_size_simd<double, vec_size> x_a_arr = SIMD_ACCESS(t.a_subarr, index, [0]);
  stdx::fixed_size_simd<double, vec_size> x_s_x = SIMD_ACCESS(t.s, index, .x);
  stdx::fixed_size_simd<double, vec_size> x_s_y = SIMD_ACCESS(t.s, index, .y[0]);
  stdx::fixed_size_simd<double, vec_size> x_v = SIMD_ACCESS(t.v, index);

  stdx::fixed_size_simd<double, vec_size> x_sa_a = sa::sa(t.a, index);
  stdx::fixed_size_simd<double, vec_size> x_sa_arr = sa::sa(t.a_subarr, index)[0];
  stdx::fixed_size_simd<double, vec_size> x_sa_v = sa::sa(t.v, index);

  for (int i = 0; i < vec_size; ++i)
  {
    EXPECT_EQ(x_a[i], vec_size - i + 3);
    EXPECT_EQ(x_a_arr[i], vec_size - i + 3);
    EXPECT_EQ(x_s_x[i], vec_size - i + 3);
    EXPECT_EQ(x_s_y[i], vec_size - i + 3);
    EXPECT_EQ(x_v[i], vec_size - i + 3);
    EXPECT_EQ(x_sa_a[i], vec_size - i + 3);
    EXPECT_EQ(x_sa_arr[i], vec_size - i + 3);
    EXPECT_EQ(x_sa_v[i], vec_size - i + 3);
  }
}


TEST(Macro, SimdVectorizedArrayAccess)
{
  TestData t;
  namespace sa = simd_access;

  constexpr size_t vec_size = stdx::native_simd<double>::size();
  stdx::fixed_size_simd<size_t, vec_size> index;
  for (size_t i = 0; i < vec_size; ++i)
  {
    index[i] = vec_size - i + 3;
  }

  stdx::fixed_size_simd<double, vec_size> x_a = SIMD_ACCESS(t.a, index);
  stdx::fixed_size_simd<double, vec_size> x_a_arr = SIMD_ACCESS(t.a_subarr, index, [0]);
  stdx::fixed_size_simd<double, vec_size> x_s_x = SIMD_ACCESS(t.s, index, .x);
  stdx::fixed_size_simd<double, vec_size> x_s_y = SIMD_ACCESS(t.s, index, .y[0]);
  stdx::fixed_size_simd<double, vec_size> x_v = SIMD_ACCESS(t.v, index);

  stdx::fixed_size_simd<double, vec_size> x_sa_a = sa::sa(t.a, index);
  stdx::fixed_size_simd<double, vec_size> x_sa_arr = sa::sa(t.a_subarr, index)[0];
  stdx::fixed_size_simd<double, vec_size> x_sa_v = sa::sa(t.v, index);

  for (int i = 0; i < vec_size; ++i)
  {
    EXPECT_EQ(x_a[i], vec_size - i + 3);
    EXPECT_EQ(x_a_arr[i], vec_size - i + 3);
    EXPECT_EQ(x_s_x[i], vec_size - i + 3);
    EXPECT_EQ(x_s_y[i], vec_size - i + 3);
    EXPECT_EQ(x_v[i], vec_size - i + 3);
    EXPECT_EQ(x_sa_a[i], vec_size - i + 3);
    EXPECT_EQ(x_sa_arr[i], vec_size - i + 3);
    EXPECT_EQ(x_sa_v[i], vec_size - i + 3);
  }
}


TEST(Macro, DeducedSimdVectorizedArrayAccess)
{
  TestData t;
  constexpr size_t vec_size = stdx::native_simd<double>::size();
  simd_access::index<vec_size> index{5};

  auto deducedFunction = [](auto&& value)
  {
    for (int i = 0; i < vec_size; ++i)
    {
      EXPECT_EQ(value[i], i + 5);
    }
  };

  deducedFunction(SIMD_ACCESS_V(t.a, index));
  deducedFunction(SIMD_ACCESS_V(t.a_subarr, index, [0]));
  deducedFunction(SIMD_ACCESS_V(t.s, index, .x));
  deducedFunction(SIMD_ACCESS_V(t.s, index, .y[0]));
  deducedFunction(SIMD_ACCESS_V(t.v, index));
}