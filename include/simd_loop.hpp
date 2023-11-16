// See the file "LICENSE" for the full license governing this code.

/**
 * @file
 * @brief Functions looping over a given function in simd-style in a linear or indirect fashion.
 */

#ifndef SIMD_ACCESS_LOOP
#define SIMD_ACCESS_LOOP

#include <concepts>
#include "index.hpp"

namespace simd_access
{

/**
 * Linear simd-ized iteration over a function. The function is first called with a simd index and the remainder
 * loop is called with an integral index.
 * @tparam SimdSize Vector size.
 * @tparam IntegralType Type of the range index.
 * @param start Start of the iteration range [start, end).
 * @param end End of the iteration range [start, end).
 * @param fn Generic function to be called. Takes one argument, whose type is either `index<SimdSize, IntegralType>`
 *   or `IntegralType`.
 */
template<int SimdSize>
void loop(std::integral auto start, std::integral auto end, auto&& fn)
{
  using IndexType = std::common_type_t<decltype(start), decltype(end)>;
  index<SimdSize, IndexType> simd_i{IndexType(start)};
  for (; simd_i.index_ + SimdSize < end + 1; simd_i.index_ += SimdSize)
  {
    fn(simd_i);
  }
  for (IndexType i = simd_i.index_; i < end; ++i)
  {
    fn(i);
  }
}

/**
 * Simd-ized iteration over a function using indirect indexing. The function is first called with an index_array
 * and the remainder loop is called with an integral index.
 * @tparam SimdSize Vector size.
 * @tparam IntegralType Type of the range index.
 * @param indices Array of indices.
 * @param fn Generic function to be called. Takes one argument, whose type is either
 *   `index_array<SimdSize, const IntegralType*>` or `IntegralType`.
 */
template<int SimdSize, std::integral IntegralType>
void loop(const std::vector<IntegralType>& indices, auto&& fn)
{
  index_array<SimdSize, const IntegralType*> simd_i{indices.data()};
  size_t i = 0, end = indices.size();
  for (; i + SimdSize < end + 1; i += SimdSize, simd_i.index_ += SimdSize)
  {
    fn(simd_i);
  }
  for (; i < end; ++i)
  {
    fn(indices[i]);
  }
}

} //namespace simd_access

#endif //SIMD_ACCESS_LOOP
