// See the file "LICENSE" for the full license governing this code.

/**
 * @file
 * @brief Functions looping over a given function in simd-style in a linear or indirect fashion.
 */

#ifndef SIMD_ACCESS_LOOP
#define SIMD_ACCESS_LOOP

#include <concepts>
#include "simd_access/index.hpp"

namespace simd_access
{

/**
 * Linear simd-ized iteration over a function. The function is first called with a simd index and the remainder
 * loop is called with an integral index.
 * @tparam SimdSize Vector size.
 * @param start Start of the iteration range [start, end).
 * @param end End of the iteration range [start, end).
 * @param fn Generic function to be called. Takes one argument, whose type is either `index<SimdSize, IntegralType>`
 *   or `IntegralType`.
 */
template<int SimdSize, auto ... Args>
void loop(std::integral auto start, std::integral auto end, auto&& fn)
{
  using IndexType = std::common_type_t<decltype(start), decltype(end)>;
  index<SimdSize, IndexType> simd_i{IndexType(start)};
  for (; simd_i.index_ + SimdSize < end + 1; simd_i.index_ += SimdSize)
  {
    if constexpr (sizeof...(Args) == 0)
    {
      fn(simd_i);
    }
    else
    {
      fn.template operator()<Args...>(simd_i);
    }
  }
  for (IndexType i = simd_i.index_; i < end; ++i)
  {
    if constexpr (sizeof...(Args) == 0)
    {
      fn(i);
    }
    else
    {
      fn.template operator()<Args...>(i);
    }
  }
}

/**
 * Simd-ized iteration over a function using indirect indexing. The function is first called with an index_array
 * and the remainder loop is called with an integral index.
 * @tparam SimdSize Vector size.
 * @tparam Args Optional additional template arguments passed to the function call operator.
 * @tparam IteratorType Deduced type of the random access iterator defining the range of indices.
 * @param start Inclusive start of the range of indices.
 * @param end Exclusive end of the range of indices.
 * @param fn Generic function to be called. Takes one argument, whose type is either
 *   `index_array<SimdSize, IteratorType>` or `IntegralType`.
 */
template<int SimdSize, auto ... Args, std::random_access_iterator IteratorType>
void loop(IteratorType start, const IteratorType& end, auto&& fn)
{
  index_array<SimdSize, IteratorType> simd_i{start};
  size_t i = 0, i_end = end - start;
  for (; i + SimdSize < i_end + 1; i += SimdSize, simd_i.index_ += SimdSize)
  {
    if constexpr (sizeof...(Args) == 0)
    {
      fn(simd_i);
    }
    else
    {
      fn.template operator()<Args...>(simd_i);
    }
  }
  for (; i < i_end; ++i)
  {
    if constexpr (sizeof...(Args) == 0)
    {
      fn(*(start + i));
    }
    else
    {
      fn.template operator()<Args...>(*(start + i));
    }
  }
}


/**
 * Simd-ized iteration over a function using indirect indexing. The function is first called with an index_array
 * and the remainder loop is called with an integral index.
 * @tparam SimdSize Vector size.
 * @tparam Args Optional additional template arguments passed to the function call operator.
 * @tparam IteratorType Deduced type of the random access iterator defining the range of indices.
 * @param start Inclusive start of the range of indices.
 * @param end Exclusive end of the range of indices.
 * @param fn Generic function to be called. Takes two arguments. The first is the linear index starting at 0, its
 *   type is either `index<SimdSize, size_t>` or `size_t`. The second argument is the indirect index, its type is
 *   either `index_array<SimdSize, IteratorType>` or `IntegralType`.
 */
template<int SimdSize, auto ... Args, std::random_access_iterator IteratorType>
void loop_with_linear_index(IteratorType start, const IteratorType& end, auto&& fn)
{
  index_array<SimdSize, IteratorType> simd_i{start};
  size_t i_end = end - start;
  index<SimdSize, size_t> i{0};
  for (; i.index_ + SimdSize < i_end + 1; i.index_ += SimdSize, simd_i.index_ += SimdSize)
  {
    if constexpr (sizeof...(Args) == 0)
    {
      fn(i, simd_i);
    }
    else
    {
      fn.template operator()<Args...>(i, simd_i);
    }
  }
  for (; i.index_ < i_end; ++i.index_)
  {
    if constexpr (sizeof...(Args) == 0)
    {
      fn(i.index_, *(start + i.index_));
    }
    else
    {
      fn.template operator()<Args...>(i.index_, *(start + i.index_));
    }
  }
}

} //namespace simd_access

#endif //SIMD_ACCESS_LOOP
