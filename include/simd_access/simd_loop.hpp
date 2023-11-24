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
 * @param indices Array of indices.
 * @param fn Generic function to be called. Takes one argument, whose type is either
 *   `index_array<SimdSize, const IntegralType*>` or `IntegralType`.
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

} //namespace simd_access

#endif //SIMD_ACCESS_LOOP
