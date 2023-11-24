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

namespace simd_access
{

template<int SimdSize, class IndexTyp = size_t>
struct index;
template<int SimdSize, class ArrayType = std::array<size_t, SimdSize>>
struct index_array;

/**
 * Stores a simd value to a memory location defined by a base address and an linear index. The simd elements are
 * stored at the positions base, base+ElementSize, base+2*ElementSize, ...
 * @tparam ElementSize Size in bytes of the type of the simd-indexed element.
 * @tparam T Deduced type of a simd element.
 * @tparam IndexType Deduced type of the index.
 * @tparam SimdSize Deduced vector size of the simd type.
 * @param base Address of the memory location, at which the first simd element is stored.
 * @param source Simd value to be stored.
 */
template<size_t ElementSize, class T, class IndexType, int SimdSize>
inline void store(T* base, index<SimdSize, IndexType>, const stdx::fixed_size_simd<T, SimdSize>& source)
{
  if constexpr (sizeof(T) == ElementSize)
  {
    source.copy_to(base, stdx::element_aligned);
  }
  else
  {
    // scatter with constant pitch
    for (size_t i = 0; i < SimdSize; ++i)
    {
      *reinterpret_cast<T*>(reinterpret_cast<char*>(base) + ElementSize * i) = source[i];
    }
  }
}

/**
 * Stores a simd value to a memory location defined by a base address and an indirect index. The simd elements are
 * stored at the positions base+indices[0]*ElementSize, base+indices[1]*ElementSize, ...
 * @tparam ElementSize Size in bytes of the type of the simd-indexed element.
 * @tparam T Deduced type of a simd element.
 * @tparam ArrayType Deduced type of the array storing the indices.
 * @tparam SimdSize Deduced vector size of the simd type.
 * @param base Address of the memory location, from which the indirect indexing starts.
 * @param indices Array of indices defining the actual positions.
 * @param source Simd value to be stored.
 */
template<size_t ElementSize, class T, class ArrayType, int SimdSize>
inline void store(T* base, const index_array<SimdSize>& indices, const stdx::fixed_size_simd<T, SimdSize>& source)
{
  // scatter with indirect indices
  for (size_t i = 0; i < SimdSize; ++i)
  {
    *reinterpret_cast<T*>(reinterpret_cast<char*>(base) + ElementSize * indices[i]) = source[i];
  }
}

/**
 * Loads a simd value from a memory location defined by a base address and an linear index. The simd elements to be
 * loaded are located at the positions base, base+ElementSize, base+2*ElementSize, ...
 * @tparam ElementSize Size in bytes of the type of the simd-indexed element.
 * @tparam T Type of a simd element.
 * @tparam IndexType Deduced type of the index.
 * @tparam SimdSize Vector size of the simd type.
 * @param base Address of the memory location, at which the first simd element is stored.
 * @return A simd value.
 */
template<size_t ElementSize, class T, class IndexType, int SimdSize>
inline auto load(T* base, index<SimdSize, IndexType>)
{
  using ResultType = stdx::fixed_size_simd<std::remove_const_t<T>, SimdSize>;
  if constexpr (sizeof(T) == ElementSize)
  {
    return ResultType(base, stdx::element_aligned);
  }
  else
  {
    // gather with constant pitch
    return ResultType([&](int i)
      {
        return *reinterpret_cast<const T*>(reinterpret_cast<const char*>(base) + ElementSize * i);
      });
  }
}

/**
 * Loads a simd value from a memory location defined by a base address and an indirect index. The simd elements to be
 * loaded are stored at the positions base+indices[0]*ElementSize, base+indices[1]*ElementSize, ...
 * @tparam ElementSize Size in bytes of the type of the simd-indexed element.
 * @tparam T Deduced type of a simd element.
 * @tparam ArrayType Deduced type of the array storing the indices.
 * @tparam SimdSize Deduced vector size of the simd type.
 * @param base Address of the memory location, from which the indirect indexing starts.
 * @param indices Array of indices defining the actual positions.
 * @return A simd value.
 */
template<size_t ElementSize, class T, class ArrayType, int SimdSize>
inline auto load(T* base, const index_array<SimdSize, ArrayType>& indices)
{
  // gather with indirect indices
  return stdx::fixed_size_simd<std::remove_const_t<T>, SimdSize>([&](int i)
    {
      return *reinterpret_cast<const T*>(reinterpret_cast<const char*>(base) + ElementSize * indices.index_[i]);
    });
}

} //namespace simd_access

#endif //SIMD_LOAD_STORE
