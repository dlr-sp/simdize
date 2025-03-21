// See the file "LICENSE" for the full license governing this code.

/**
 * @file
 * @brief Functions looping over a given function in simd-style in a linear or indirect fashion.
 */

#ifndef SIMD_ACCESS_LOOP
#define SIMD_ACCESS_LOOP

#include <concepts>
#include <type_traits>
#include "simd_access/index.hpp"

namespace simd_access
{

/// Type for scalar residual loop policy.
using ScalarResidualLoopT = std::integral_constant<int, 0>;
/// Type for vector residual loop policy.
using VectorResidualLoopT = std::integral_constant<int, 1>;
/// Value for scalar residual loop policy.
constexpr auto ScalarResidualLoop = ScalarResidualLoopT();
/// Value for vector residual loop policy.
constexpr auto VectorResidualLoop = VectorResidualLoopT();

/**
 * Linear simd-ized iteration over a function. The function is first called with a simd index and the remainder
 * loop is called with an integral index.
 * @tparam SimdSize Vector size.
 * @param start Start of the iteration range [start, end).
 * @param end End of the iteration range [start, end).
 * @param fn Generic function to be called. Takes one argument, whose type is either `index<SimdSize, IntegralType>`
 *   or `IntegralType`.
 * @param residualLoopPolicy Determines the execution policy of residual iterations. If `ScalarResidualLoop`, residual
 *   iterations are executed one by one. If `VectorResidualLoop`, residual iterations are executed vectorized. In that
 *   case the user is responsible for the handling of indices possbily extending the valid iteration range. Defaults
 *   to `ScalarResidualLoop`.
 */
template<int SimdSize, auto ... Args, typename ResidualLoopPolicyType = ScalarResidualLoopT>
inline void loop(std::integral auto start, std::integral auto end, auto&& fn,
  ResidualLoopPolicyType residualLoopPolicy = ScalarResidualLoop)
{
  using IndexType = std::common_type_t<decltype(start), decltype(end)>;
  index<SimdSize, IndexType> simd_i{IndexType(start)};
  constexpr auto endOffset = residualLoopPolicy == ScalarResidualLoop ? 1 : SimdSize;
  for (; simd_i.index_ + SimdSize < end + endOffset; simd_i.index_ += SimdSize)
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
  if constexpr (residualLoopPolicy == ScalarResidualLoop)
  {
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
}

/**
 * Linear simd-ized iteration over a function. The function is first called with an integral index until the
 * `alignTestFn` returns true for a specific index. From there on `alignTestFn` isn't called anymore and the function
 * is called with a simd index. The remainder loop is called with an integral index again.
 * @tparam SimdSize Vector size.
 * @param start Start of the iteration range [start, end).
 * @param end End of the iteration range [start, end).
 * @param alignTestFn Generic function to be called. Takes one scalar argument of the common type of `start` and `end`.
 *   Once it returns true, it isn't called anymore and the function starts to call `fn` with simd indices
 *   (including the index for which `alignTestFn` returned `true`).
 * @param fn Generic function to be called. Takes one argument, whose type is either `index<SimdSize, IntegralType>`
 *   or `IntegralType`.
 */
template<int SimdSize, auto ... Args>
inline void aligning_loop(std::integral auto start, std::integral auto end, auto&& alignTestFn, auto&& fn)
{
  using IndexType = std::common_type_t<decltype(start), decltype(end)>;
  index<SimdSize, IndexType> simd_i{IndexType(start)};
  for (; simd_i.index_ < end && !alignTestFn(simd_i.index_); ++simd_i.index_)
  {
    if constexpr (sizeof...(Args) == 0)
    {
      fn(simd_i.index_);
    }
    else
    {
      fn.template operator()<Args...>(simd_i.index_);
    }
  }
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
  for (; simd_i.index_ < end; ++simd_i.index_)
  {
    if constexpr (sizeof...(Args) == 0)
    {
      fn(simd_i.index_);
    }
    else
    {
      fn.template operator()<Args...>(simd_i.index_);
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
 * @param residualLoopPolicy Determines the execution policy of residual iterations. If `ScalarResidualLoop`, residual
 *   iterations are executed one by one. If `VectorResidualLoop`, residual iterations are executed vectorized. In that
 *   case the user is responsible for the handling of indices possbily extending the valid iteration range. Defaults
 *   to `ScalarResidualLoop`.
 */
template<int SimdSize, auto ... Args, std::random_access_iterator IteratorType,
  typename ResidualLoopPolicyType = ScalarResidualLoopT>
inline void loop(IteratorType start, const IteratorType& end, auto&& fn,
  ResidualLoopPolicyType residualLoopPolicy = ScalarResidualLoop)
{
  index_array<SimdSize, IteratorType> simd_i{start};
  size_t i = 0, i_end = end - start;
  constexpr auto endOffset = residualLoopPolicy == ScalarResidualLoop ? 1 : SimdSize;
  for (; i + SimdSize < i_end + endOffset; i += SimdSize, simd_i.index_ += SimdSize)
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
  if constexpr (residualLoopPolicy == ScalarResidualLoop)
  {
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
 * @param residualLoopPolicy Determines the execution policy of residual iterations. If `ScalarResidualLoop`, residual
 *   iterations are executed one by one. If `VectorResidualLoop`, residual iterations are executed vectorized. In that
 *   case the user is responsible for the handling of indices possbily extending the valid iteration range. Defaults
 *   to `ScalarResidualLoop`.
 */
template<int SimdSize, auto ... Args, std::random_access_iterator IteratorType,
  typename ResidualLoopPolicyType = ScalarResidualLoopT>
inline void loop_with_linear_index(IteratorType start, const IteratorType& end, auto&& fn,
  ResidualLoopPolicyType residualLoopPolicy = ScalarResidualLoop)
{
  index_array<SimdSize, IteratorType> simd_i{start};
  size_t i_end = end - start;
  constexpr auto endOffset = residualLoopPolicy == ScalarResidualLoop ? 1 : SimdSize;
  index<SimdSize, size_t> i{0};
  for (; i.index_ + SimdSize < i_end + endOffset; i.index_ += SimdSize, simd_i.index_ += SimdSize)
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
  if constexpr (residualLoopPolicy == ScalarResidualLoop)
  {
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
}

} //namespace simd_access

#endif //SIMD_ACCESS_LOOP
