// See the file "LICENSE" for the full license governing this code.

/**
 * @file
 * @brief Global functions to load and store simd values to using several addressing modes.
 */

#ifndef SIMD_LOAD_STORE
#define SIMD_LOAD_STORE

#include "simd_access/base.hpp"
#include "simd_access/location.hpp"
#include "simd_access/index.hpp"

namespace simd_access
{

/**
 * Stores a simd value to a memory location defined by a base address and an linear index. The simd elements are
 * stored at the positions base, base+ElementSize, base+2*ElementSize, ...
 * @tparam ElementSize Size in bytes of the type of the simd-indexed element.
 * @tparam T Deduced type of a simd element.
 * @tparam SimdSize Deduced vector size of the simd type.
 * @param location Address of the memory location, at which the first simd element is stored.
 * @param source Simd value to be stored.
 */
template<size_t ElementSize, simd_arithmetic T, int SimdSize>
inline void store(const linear_location<T, SimdSize>& location, const stdx::fixed_size_simd<T, SimdSize>& source)
{
  if constexpr (sizeof(T) == ElementSize)
  {
    source.copy_to(location.base_, stdx::element_aligned);
  }
  else
  {
    // scatter with constant pitch
    for (int i = 0; i < SimdSize; ++i)
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
template<size_t ElementSize, simd_arithmetic T, int SimdSize, class ArrayType>
inline void store(const indexed_location<T, SimdSize, ArrayType>& location,
  const stdx::fixed_size_simd<T, SimdSize>& source)
{
  // scatter with indirect indices
  for (int i = 0; i < SimdSize; ++i)
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
 * @param location Address of the memory location, at which the first scalar element is stored.
 * @return A simd value.
 */
template<size_t ElementSize, simd_arithmetic T, int SimdSize>
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
template<size_t ElementSize, simd_arithmetic T, int SimdSize, class ArrayType>
inline auto load(const indexed_location<T, SimdSize, ArrayType>& location)
{
  // gather with indirect indices
  return stdx::fixed_size_simd<std::remove_const_t<T>, SimdSize>([&](int i)
    {
      return *reinterpret_cast<const T*>
        (reinterpret_cast<const char*>(location.base_) + ElementSize * location.indices_[i]);
    });
}

/**
 * Creates a simd value from rvalues returned by the operator[] applied to `base`.
 * @tparam BaseType Type of an simd element.
 * @tparam IndexType Deduced simd index type.
 * @param base Base object.
 * @param idx Index.
 * @return A simd value.
 */
template<simd_arithmetic BaseType, simd_index IndexType>
inline auto load_rvalue(auto&& base, const IndexType& idx)
{
  return stdx::fixed_size_simd<BaseType, IndexType::size()>([&](auto i) { return base[scalar_index(idx, i)]; });
}

/**
 * Creates a simd value from rvalues returned by the functor `subobject` applied to the range of elements defined by
 * the simd index `idx`.
 * @tparam BaseType Type of an simd element.
 * @tparam IndexType Deduced simd index type.
 * @param base Base object.
 * @param idx Index.
 * @param subobject Functor returning the sub-object.
 * @return A simd value.
 */
template<simd_arithmetic BaseType, simd_index IndexType>
inline auto load_rvalue(auto&& base, const IndexType& idx, auto&& subobject)
{
  return stdx::fixed_size_simd<BaseType, IndexType::size()>([&](auto i)
  {
    return subobject(base[scalar_index(idx, i)]);
  });
}

} //namespace simd_access

#endif //SIMD_LOAD_STORE
