// See the file "LICENSE" for the full license governing this code.

/**
 * @file
 * @brief Global functions to load and store simd values to using several addressing modes.
 */

#ifndef SIMD_LOAD_STORE
#define SIMD_LOAD_STORE

#include <array>
#include <experimental/simd>
namespace stdx = std::experimental;

#include "simd_access/location.hpp"

namespace simd_access
{

template<typename T> concept arithmetic = std::is_arithmetic_v<T>;
/**
 * Stores a simd value to a memory location defined by a base address and an linear index. The simd elements are
 * stored at the positions base, base+ElementSize, base+2*ElementSize, ...
 * @tparam ElementSize Size in bytes of the type of the simd-indexed element.
 * @tparam T Deduced type of a simd element.
 * @tparam SimdSize Deduced vector size of the simd type.
 * @param location Address of the memory location, at which the first simd element is stored.
 * @param source Simd value to be stored.
 */
template<size_t ElementSize, arithmetic T, int SimdSize>
inline void store(const linear_location<T, SimdSize>& location, const stdx::fixed_size_simd<T, SimdSize>& source)
{
  if constexpr (sizeof(T) == ElementSize)
  {
    source.copy_to(location.base_, stdx::element_aligned);
  }
  else
  {
    // scatter with constant pitch
    for (size_t i = 0; i < SimdSize; ++i)
    {
      *reinterpret_cast<T*>(reinterpret_cast<char*>(location.base_) + ElementSize * i) = source[i];
    }
  }
}

/**
 * Stores a simd value to a memory location defined by a base address and an indirect index. The simd elements are
 * stored at the positions base+indices[0]*ElementSize, base+indices[1]*ElementSize, ...
 * @tparam ElementSize Size in bytes of the type of the simd-indexed element.
 * @tparam T Deduced type of a simd element.
 * @tparam SimdSize Deduced vector size of the simd type.
 * @tparam ArrayType Deduced type of the array storing the indices.
 * @param location Address and indices of the memory location.
 * @param source Simd value to be stored.
 */
template<size_t ElementSize, arithmetic T, int SimdSize, class ArrayType>
inline void store(const indexed_location<T, SimdSize, ArrayType>& location,
  const stdx::fixed_size_simd<T, SimdSize>& source)
{
  // scatter with indirect indices
  for (size_t i = 0; i < SimdSize; ++i)
  {
    *reinterpret_cast<T*>(reinterpret_cast<char*>(location.base_) + ElementSize * location.indices_[i]) = source[i];
  }
}

/**
 * Loads a simd value from a memory location defined by a base address and an linear index. The simd elements to be
 * loaded are located at the positions base, base+ElementSize, base+2*ElementSize, ...
 * @tparam ElementSize Size in bytes of the type of the simd-indexed element.
 * @tparam T Type of a simd element.
 * @tparam SimdSize Vector size of the simd type.
 * @param location Address of the memory location, at which the first simd element is stored.
 * @return A simd value.
 */
template<size_t ElementSize, arithmetic T, int SimdSize>
inline auto load(const linear_location<T, SimdSize>& location)
{
  using ResultType = stdx::fixed_size_simd<std::remove_const_t<T>, SimdSize>;
  if constexpr (sizeof(T) == ElementSize)
  {
    return ResultType(location.base_, stdx::element_aligned);
  }
  else
  {
    // gather with constant pitch
    return ResultType([&](int i)
      {
        return *reinterpret_cast<const T*>(reinterpret_cast<const char*>(location.base_) + ElementSize * i);
      });
  }
}

/**
 * Loads a simd value from a memory location defined by a base address and an indirect index. The simd elements to be
 * loaded are stored at the positions base+indices[0]*ElementSize, base+indices[1]*ElementSize, ...
 * @tparam ElementSize Size in bytes of the type of the simd-indexed element.
 * @tparam T Deduced type of a simd element.
 * @tparam SimdSize Deduced vector size of the simd type.
 * @tparam ArrayType Deduced type of the array storing the indices.
 * @param location Address and indices of the memory location.
 * @return A simd value.
 */
template<size_t ElementSize, arithmetic T, int SimdSize, class ArrayType>
inline auto load(const indexed_location<T, SimdSize, ArrayType>& location)
{
  // gather with indirect indices
  return stdx::fixed_size_simd<std::remove_const_t<T>, SimdSize>([&](int i)
    {
      return *reinterpret_cast<const T*>
        (reinterpret_cast<const char*>(location.base_) + ElementSize * location.indices_[i]);
    });
}

} //namespace simd_access

#endif //SIMD_LOAD_STORE
