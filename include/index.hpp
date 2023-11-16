// See the file "LICENSE" for the full license governing this code.

/**
 * @file
 * @brief Classes representing simd indices to either a consecutive sequence of elements or indirect indexed elements.
 */

#ifndef SIMD_ACCESS_INDEX
#define SIMD_ACCESS_INDEX

#include <array>

#include "load_store.hpp"
#include "member_overload.hpp"

namespace simd_access
{

template<class T, class Index, size_t ElementSize>
class value_access;

/// Class representing a simd index to a consecutive sequence of elements.
/**
 * @tparam SimdSize Length of the simd sequence.
 * @tparam IndexType Type of the index.
 */
template<int SimdSize, class IndexType>
struct index
{
  /// Return the length of the simd sequence.
  /**
   * @return The length of the simd sequence.
   */
  static constexpr size_t size() { return SimdSize; }

  /// The index, at which the sequence starts.
  IndexType index_;

  /// A reverse overloaded operator[] for simdized array accesses, since global operator[] is not allowed (yet).
  /**
   * @tparam T Data type of the elements in the array.
   * @param data Pointer to the array.
   * @return A value_access representing a simd access expression to a consecutive sequence of elements in an array.
   */
  template<class T>
  auto operator[](T* data) const
  {
    return value_access<T, index<SimdSize, IndexType>, sizeof(T)>(data + index_, *this);
  }
};


/// Class representing a simd index to a indirect indexed elements in an array.
/**
 * @tparam SimdSize Length of the simd sequence.
 * @tparam ArrayType Type of the array, which stores the indices. Defaults to std::array<size_t, SimdSize>, but could
 *   be stdx::fixed_size_simd<size_t, SimdSize>. Also size_t can be exchanged for another integral type.
 *   It is also possible to specify e.g. `int*` and thus use a pointer to a location in a larger array.
 */
template<int SimdSize, class ArrayType>
struct index_array
{
  /// Return the length of the simd sequence.
  /**
   * @return The length of the simd sequence.
   */
  static constexpr size_t size() { return SimdSize; }

  /// The index array, the n'th entry defines the index of the n'th element in the simd type.
  ArrayType index_;

  /// A reverse overloaded operator[] for simdized array accesses, since global operator[] is not allowed (yet).
  /**
   * @tparam T Data type of the elements in the array.
   * @param data Pointer to the array.
   * @return A value_access representing a simd access expression to indirect indexed elements in an array.
   */
  template<class T>
  auto operator[](T* data) const
  {
    return value_access<T, index_array<SimdSize, ArrayType>, sizeof(T)>{data, *this};
  }
};

template<class PotentialIndexType>
concept is_index =
  requires(PotentialIndexType x) { []<int SimdSize, class IndexType>(index<SimdSize, IndexType>&){}(x); } ||
  requires(PotentialIndexType x) { []<int SimdSize, class ArrayType>(index_array<SimdSize, ArrayType>&){}(x); };

/// TODO: Introduce masked_index and masked_index_array to support e.g. residual masked loops.

} //namespace simd_access

#endif //SIMD_ACCESS_INDEX
