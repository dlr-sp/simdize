// See the file "LICENSE" for the full license governing this code.

/**
 * @file
 * @brief Classes representing simd indices to either a consecutive sequence of elements or indirect indexed elements.
 */

#ifndef SIMD_ACCESS_INDEX
#define SIMD_ACCESS_INDEX

#include "simd_access/base.hpp"
#include "simd_access/location.hpp"
#include <type_traits>

namespace simd_access
{

template<class Location, size_t ElementSize>
class value_access;

/// Class representing a simd index to a consecutive sequence of elements.
/**
 * @tparam SimdSize Length of the simd sequence.
 * @tparam IndexType Type of the index.
 */
template<int SimdSize, class IndexType = size_t>
struct index
{
  /// Return the length of the simd sequence.
  /**
   * @return The length of the simd sequence.
   */
  static constexpr int size() { return SimdSize; }

  /// Return the scalar index of a vector lane.
  /**
   * @param i Index in the vector must be in the range [0, SimdSize) .
   * @return The scalar index at vector lane i, i.e. index_ + i.
   */
  auto scalar_index(int i) const { return index_ + IndexType(i); }

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
    return value_access<linear_location<T, SimdSize>, sizeof(T)>(linear_location<T, SimdSize>{data + index_});
  }

  /// Transforms this to a simd value.
  /**
   * @return The value represented by this transformed to a simd value.
   */
  auto to_simd() const
  {
    return stdx::fixed_size_simd<IndexType, SimdSize>([this](auto i){ return IndexType(index_ + i); });
  }
};


/// Class representing a simd index to indirect indexed elements in an array.
/**
 * @tparam SimdSize Length of the simd sequence.
 * @tparam ArrayType Type of the array, which stores the indices. Defaults to std::array<size_t, SimdSize>, but could
 *   be stdx::fixed_size_simd<size_t, SimdSize>. Also size_t can be exchanged for another integral type.
 *   It is also possible to specify e.g. `int*` and thus use a pointer to a location in a larger array.
 */
template<int SimdSize, class ArrayType = std::array<size_t, SimdSize>>
struct index_array
{
  /// Return the length of the simd sequence.
  /**
   * @return The length of the simd sequence.
   */
  static constexpr int size() { return SimdSize; }

  /// Return the scalar index of a vector lane.
  /**
   * @param i Index in the vector must be in the range [0, SimdSize) .
   * @return The scalar index at vector lane i, i.e. index_[i].
   */
  auto scalar_index(int i) const { return index_[i]; }

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
    return value_access<indexed_location<T, SimdSize, ArrayType>, sizeof(T)>
      (indexed_location<T, SimdSize, ArrayType>{data, index_});
  }
};

template<class PotentialIndexType>
concept is_simd_index =
  (is_stdx_simd<PotentialIndexType> && std::is_integral_v<typename PotentialIndexType::value_type>) ||
  requires(std::remove_cvref_t<PotentialIndexType> x) { []<int SimdSize, class IndexType>(index<SimdSize, IndexType>&){}(x); } ||
  requires(std::remove_cvref_t<PotentialIndexType> x) { []<int SimdSize, class ArrayType>(index_array<SimdSize, ArrayType>&){}(x); };

/// TODO: Introduce masked_index and masked_index_array to support e.g. residual masked loops.

template<int SimdSize, class IndexType>
inline auto get_index(const index<SimdSize, IndexType>& idx, auto i)
{
  return idx.scalar_index(i);
}

template<int SimdSize, class ArrayType>
inline auto get_index(const index_array<SimdSize, ArrayType>& idx, auto i)
{
  return idx.scalar_index(i);
}

template<class IndexType, class Abi>
inline auto get_index(const stdx::simd<IndexType, Abi>& idx, auto i)
{
  return idx[i];
}

} //namespace simd_access

#endif //SIMD_ACCESS_INDEX
